//! One safetensors file: a JSON header, then tensor bytes.

use crate::offer::{Offer, Slice, Stored};
use graph::{Bf16, F16};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// A failed range request.
    Net(String),
    /// The file breaks the format's rules.
    Format(String),
    Missing(String),
    /// An element type this reader cannot widen to f32.
    DType(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "checkpoint: {e}"),
            Error::Net(m) => write!(f, "checkpoint over http: {m}"),
            Error::Format(m) => write!(f, "checkpoint format: {m}"),
            Error::Missing(name) => write!(f, "no tensor named {name}"),
            Error::DType(m) => write!(f, "unsupported element type: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

fn format(e: impl fmt::Display) -> Error {
    Error::Format(e.to_string())
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    /// Byte range, counted from the end of the header.
    data_offsets: [u64; 2],
}

/// Where the bytes live.
enum Backing {
    Disk(fs::File),
    /// Read by range requests.
    Web(String),
}

/// The header pad in exlr8 files: one request reads it.
const HEADER_PAD: u64 = 65_536;

pub struct File {
    backing: Backing,
    /// Where tensor bytes start.
    data: u64,
    entries: BTreeMap<String, Entry>,
}

/// How a safetensors dtype name lies in the file.
fn stored(dtype: &str) -> Stored {
    match dtype {
        "F32" => Stored::F32,
        "BF16" => Stored::Bf16,
        "F16" => Stored::F16,
        _ => Stored::Other,
    }
}

/// Bytes `start..start + len` of `url`.
fn fetch(url: &str, start: u64, len: u64) -> Result<Vec<u8>, Error> {
    let range = format!("bytes={start}-{}", start + len - 1);
    let net = |e: &dyn fmt::Display| Error::Net(format!("{url}: {e}"));
    let resp = ureq::get(url)
        .set("Range", &range)
        .call()
        .map_err(|e| net(&e))?;
    if resp.status() != 206 {
        return Err(net(&format_args!("status {} for {range}", resp.status())));
    }
    let mut bytes = Vec::with_capacity(len as usize);
    resp.into_reader().take(len).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != len {
        return Err(net(&format_args!("{} of {len} bytes", bytes.len())));
    }
    Ok(bytes)
}

impl File {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let mut file = fs::File::open(path)?;
        let size = file.metadata()?.len();
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len);
        if len > size.saturating_sub(8) {
            return Err(format(format_args!("a {len}-byte header in {size} bytes")));
        }
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header)?;
        Self::parse(Backing::Disk(file), &header, size)
    }

    /// A file served over http, one tensor per request.
    pub fn remote(url: &str) -> Result<Self, Error> {
        let first = fetch(url, 0, HEADER_PAD)?;
        let len = u64::from_le_bytes(first[..8].try_into().expect("8 bytes"));
        if len + 8 > HEADER_PAD {
            return Err(format(format_args!("a {len}-byte header past the pad")));
        }
        let header = &first[8..8 + len as usize];
        Self::parse(Backing::Web(url.into()), header, u64::MAX)
    }

    fn parse(backing: Backing, header: &[u8], size: u64) -> Result<Self, Error> {
        let len = header.len() as u64;
        let mut all: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(header).map_err(format)?;
        all.remove("__metadata__");
        let data = 8 + len;
        let mut entries = BTreeMap::new();
        for (name, value) in all {
            let e: Entry = serde_json::from_value(value).map_err(format)?;
            let [start, end] = e.data_offsets;
            if start > end || end > size - data {
                return Err(format(format_args!("{name} lies outside the file")));
            }
            entries.insert(name, e);
        }
        Ok(Self {
            backing,
            data,
            entries,
        })
    }

    /// Bytes `start..end` of the tensor area.
    fn read(&self, start: u64, end: u64) -> Result<Vec<u8>, Error> {
        match &self.backing {
            Backing::Disk(file) => {
                let mut bytes = vec![0u8; (end - start) as usize];
                let mut file = file;
                file.seek(SeekFrom::Start(self.data + start))?;
                file.read_exact(&mut bytes)?;
                Ok(bytes)
            }
            Backing::Web(url) => fetch(url, self.data + start, end - start),
        }
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// A tensor's bytes as stored.
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>, Error> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| Error::Missing(name.into()))?;
        let [start, end] = e.data_offsets;
        self.read(start, end)
    }

    /// A tensor's values, widened to f32.
    pub fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        let missing = || Error::Missing(name.into());
        let e = self.entries.get(name).ok_or_else(missing)?;
        let unknown = || Error::DType(format!("{name} is {}", e.dtype));
        let kind = stored(&e.dtype);
        let width = match kind {
            Stored::F32 => 4,
            Stored::Bf16 | Stored::F16 => 2,
            _ => return Err(unknown()),
        };
        let [start, end] = e.data_offsets;
        let count = e.shape.iter().product::<usize>() as u64;
        if end - start != count * width {
            return Err(format(format_args!("{name} has the wrong byte count")));
        }
        let bytes = self.read(start, end)?;
        let halves = || (bytes.as_chunks::<2>().0.iter()).map(|b| u16::from_le_bytes(*b));
        Ok(match kind {
            Stored::F32 => (bytes.as_chunks::<4>().0.iter())
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            Stored::Bf16 => halves().map(|h| f32::from(Bf16(h))).collect(),
            _ => halves().map(|h| f32::from(F16(h))).collect(),
        })
    }
}

impl Offer for File {
    fn slices(&self) -> Vec<Slice> {
        (self.entries.iter())
            .map(|(name, e)| Slice {
                name: name.clone(),
                shape: e.shape.clone(),
                stored: stored(&e.dtype),
            })
            .collect()
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        File::floats(self, name)
    }

    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        Err(Error::DType(format!("{name} holds floats, not a form")))
    }
}
