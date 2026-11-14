//! A GGUF header and tensor bytes read on demand.

use crate::Error;
use crate::offer::{Offer, Slice, Stored};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum Meta {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Meta>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufType {
    F32,
    F16,
    Q4_0,
    Q8_0,
}

fn format(message: impl ToString) -> Error {
    Error::Format(message.to_string())
}

impl GgufType {
    fn parse(id: u32, name: &str) -> Result<Self, Error> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            _ => Err(Error::DType(format!("{name}: GGUF type {id}"))),
        }
    }

    fn bytes(self, elements: usize) -> Result<usize, Error> {
        let (block, bytes) = match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q8_0 => (32, 34),
        };
        if !elements.is_multiple_of(block) {
            return Err(format("incomplete GGUF block"));
        }
        (elements / block)
            .checked_mul(bytes)
            .ok_or_else(|| format("GGUF byte count overflow"))
    }
}

struct Reader {
    file: BufReader<File>,
    left: u64,
}

impl Reader {
    fn read(&mut self, bytes: &mut [u8]) -> Result<(), Error> {
        self.left = self
            .left
            .checked_sub(bytes.len() as u64)
            .ok_or_else(|| format("truncated GGUF header"))?;
        self.file.read_exact(bytes).map_err(format)
    }

    fn number<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let mut bytes = [0; N];
        self.read(&mut bytes)?;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.number()?))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.number()?))
    }

    fn count(&mut self, minimum: u64) -> Result<usize, Error> {
        let n = self.u64()?;
        if n > self.left / minimum {
            return Err(format("GGUF count exceeds file size"));
        }
        usize::try_from(n).map_err(format)
    }

    fn string(&mut self) -> Result<String, Error> {
        let n = self.count(1)?;
        let mut bytes = vec![0; n];
        self.read(&mut bytes)?;
        String::from_utf8(bytes).map_err(format)
    }

    fn value(&mut self, ty: u32, depth: usize) -> Result<Meta, Error> {
        Ok(match ty {
            0 => Meta::U8(self.number::<1>()?[0]),
            1 => Meta::I8(self.number::<1>()?[0] as i8),
            2 => Meta::U16(u16::from_le_bytes(self.number()?)),
            3 => Meta::I16(i16::from_le_bytes(self.number()?)),
            4 => Meta::U32(self.u32()?),
            5 => Meta::I32(i32::from_le_bytes(self.number()?)),
            6 => Meta::F32(f32::from_le_bytes(self.number()?)),
            7 => Meta::Bool(match self.number::<1>()?[0] {
                0 => false,
                1 => true,
                _ => return Err(format("invalid GGUF boolean")),
            }),
            8 => Meta::Str(self.string()?),
            9 => {
                if depth >= 64 {
                    return Err(format("GGUF arrays are nested too deeply"));
                }
                let elem = self.u32()?;
                let minimum = match elem {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    8 | 10..=12 => 8,
                    9 => 12,
                    _ => return Err(format("invalid GGUF array type")),
                };
                let n = self.count(minimum)?;
                Meta::Array(
                    (0..n)
                        .map(|_| self.value(elem, depth + 1))
                        .collect::<Result<_, _>>()?,
                )
            }
            10 => Meta::U64(self.u64()?),
            11 => Meta::I64(i64::from_le_bytes(self.number()?)),
            12 => Meta::F64(f64::from_le_bytes(self.number()?)),
            _ => return Err(format(format!("invalid GGUF metadata type {ty}"))),
        })
    }

    fn entry(&mut self) -> Result<(String, Entry), Error> {
        let name = self.string()?;
        let rank = self.u32()?;
        if u64::from(rank) > self.left / 8 {
            return Err(format("GGUF rank exceeds file size"));
        }
        let mut dims = Vec::new();
        let mut elements = 1usize;
        for _ in 0..rank {
            let dim = self.u64()?;
            elements = elements
                .checked_mul(usize::try_from(dim).map_err(format)?)
                .ok_or_else(|| format("GGUF dimensions overflow"))?;
            dims.push(dim);
        }
        let ty = GgufType::parse(self.u32()?, &name)?;
        let offset = self.u64()?;
        let bytes = ty.bytes(elements)?;
        Ok((
            name,
            Entry {
                dims,
                elements,
                ty,
                offset,
                bytes,
            },
        ))
    }
}

struct Entry {
    dims: Vec<u64>,
    elements: usize,
    ty: GgufType,
    offset: u64,
    bytes: usize,
}

pub struct Gguf {
    file: File,
    data: u64,
    metadata: BTreeMap<String, Meta>,
    entries: BTreeMap<String, Entry>,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        let mut reader = Reader {
            file: BufReader::new(file),
            left: size,
        };
        if reader.number::<4>()? != *b"GGUF" {
            return Err(format("invalid GGUF magic"));
        }
        if !matches!(reader.u32()?, 2 | 3) {
            return Err(format("unsupported GGUF version"));
        }
        let tensors = reader.count(24)?;
        let pairs = reader.count(12)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..pairs {
            let key = reader.string()?;
            let ty = reader.u32()?;
            if metadata.insert(key, reader.value(ty, 0)?).is_some() {
                return Err(format("duplicate GGUF metadata key"));
            }
        }
        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(Meta::U32(n)) => u64::from(*n),
            Some(Meta::U64(n)) => *n,
            _ => return Err(format("invalid GGUF alignment type")),
        };
        if !alignment.is_power_of_two() {
            return Err(format("invalid GGUF alignment"));
        }
        let mut entries = BTreeMap::new();
        for _ in 0..tensors {
            let (name, entry) = reader.entry()?;
            if entries.insert(name, entry).is_some() {
                return Err(format("duplicate GGUF tensor name"));
            }
        }
        let end = size - reader.left;
        let data = end
            .checked_add(alignment - 1)
            .map(|n| n & !(alignment - 1))
            .ok_or_else(|| format("GGUF alignment overflow"))?;
        let available = size
            .checked_sub(data)
            .ok_or_else(|| format("truncated GGUF data"))?;
        for (name, entry) in &entries {
            if entry
                .offset
                .checked_add(entry.bytes as u64)
                .is_none_or(|end| end > available)
            {
                return Err(format(format!("{name} lies outside the GGUF file")));
            }
        }
        Ok(Self {
            file: reader.file.into_inner(),
            data,
            metadata,
            entries,
        })
    }

    pub fn meta(&self, key: &str) -> Option<&Meta> {
        self.metadata.get(key)
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        match self.meta(key)? {
            Meta::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn u64(&self, key: &str) -> Option<u64> {
        match self.meta(key)? {
            Meta::U32(n) => Some(u64::from(*n)),
            Meta::U64(n) => Some(*n),
            Meta::I32(n) => u64::try_from(*n).ok(),
            Meta::I64(n) => u64::try_from(*n).ok(),
            _ => None,
        }
    }

    pub fn f32(&self, key: &str) -> Option<f32> {
        match self.meta(key)? {
            Meta::F32(n) => Some(*n),
            Meta::F64(n) => Some(*n as f32),
            _ => None,
        }
    }

    pub fn strs(&self, key: &str) -> Option<Vec<&str>> {
        let Meta::Array(values) = self.meta(key)? else {
            return None;
        };
        values
            .iter()
            .map(|v| match v {
                Meta::Str(s) => Some(s.as_str()),
                _ => None,
            })
            .collect()
    }

    pub fn u32s(&self, key: &str) -> Option<Vec<u32>> {
        let Meta::Array(values) = self.meta(key)? else {
            return None;
        };
        values
            .iter()
            .map(|v| match v {
                Meta::U32(n) => Some(*n),
                Meta::I32(n) => u32::try_from(*n).ok(),
                _ => None,
            })
            .collect()
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    fn entry(&self, name: &str) -> Result<&Entry, Error> {
        self.entries
            .get(name)
            .ok_or_else(|| Error::Missing(name.into()))
    }

    pub fn dims(&self, name: &str) -> Result<&[u64], Error> {
        Ok(&self.entry(name)?.dims)
    }

    pub fn elements(&self, name: &str) -> Result<usize, Error> {
        Ok(self.entry(name)?.elements)
    }

    pub fn tensor_type(&self, name: &str) -> Result<GgufType, Error> {
        Ok(self.entry(name)?.ty)
    }

    pub fn bytes(&self, name: &str) -> Result<(GgufType, usize, Vec<u8>), Error> {
        let entry = self.entry(name)?;
        let mut bytes = vec![0; entry.bytes];
        let mut file = &self.file;
        file.seek(SeekFrom::Start(self.data + entry.offset))?;
        file.read_exact(&mut bytes).map_err(format)?;
        Ok((entry.ty, entry.elements, bytes))
    }

    pub fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        match self.tensor_type(name)? {
            GgufType::F32 | GgufType::F16 => {}
            ty => return Err(Error::DType(format!("{name}: {ty:?} is packed"))),
        }
        let (ty, _, bytes) = self.bytes(name)?;
        Ok(match ty {
            GgufType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            _ => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from(graph::F16(u16::from_le_bytes(*b))))
                .collect(),
        })
    }
}

impl Offer for Gguf {
    /// GGUF puts the fastest axis first, a slice last.
    fn slices(&self) -> Vec<Slice> {
        let stored = |ty: GgufType| match ty {
            GgufType::F32 => Stored::F32,
            GgufType::F16 => Stored::F16,
            GgufType::Q8_0 => Stored::Form(&codegen::block::Q8_0),
            GgufType::Q4_0 => Stored::Form(&codegen::block::Q4_0),
        };
        (self.entries.iter())
            .map(|(name, e)| Slice {
                name: name.clone(),
                shape: e.dims.iter().rev().map(|d| *d as usize).collect(),
                stored: stored(e.ty),
            })
            .collect()
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        Gguf::floats(self, name)
    }

    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        match self.tensor_type(name)? {
            GgufType::F32 | GgufType::F16 => {
                Err(Error::DType(format!("{name} holds floats, not a form")))
            }
            _ => Ok(self.bytes(name)?.2),
        }
    }
}
