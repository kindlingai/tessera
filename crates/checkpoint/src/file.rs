//! One safetensors file: a JSON header, then tensor bytes.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
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

pub struct File {
    file: fs::File,
    /// Where tensor bytes start.
    data: u64,
    entries: BTreeMap<String, Entry>,
}

/// Bytes per element, for the types this reader widens.
fn width(dtype: &str) -> Option<u64> {
    match dtype {
        "F32" => Some(4),
        "BF16" => Some(2),
        _ => None,
    }
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
        let mut all: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&header).map_err(format)?;
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
            file,
            data,
            entries,
        })
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// A tensor's values, widened to f32.
    pub fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        let missing = || Error::Missing(name.into());
        let e = self.entries.get(name).ok_or_else(missing)?;
        let unknown = || Error::DType(format!("{name} is {}", e.dtype));
        let width = width(&e.dtype).ok_or_else(unknown)?;
        let [start, end] = e.data_offsets;
        let count = e.shape.iter().product::<usize>() as u64;
        if end - start != count * width {
            return Err(format(format_args!("{name} has the wrong byte count")));
        }
        let mut bytes = vec![0u8; (end - start) as usize];
        let mut file = &self.file;
        file.seek(SeekFrom::Start(self.data + start))?;
        file.read_exact(&mut bytes)?;
        let wide = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        // A bf16 is the top half of an f32.
        let half = |b: &[u8]| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
        Ok(match width {
            4 => bytes.chunks_exact(4).map(wide).collect(),
            _ => bytes.chunks_exact(2).map(half).collect(),
        })
    }
}
