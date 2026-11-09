//! Host tensors and the interpreter's errors.

use graph::DType;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Data {
    F32(Vec<f32>),
    I32(Vec<i32>),
    U32(Vec<u32>),
}

/// Row-major, contiguous.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Data,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The graph needs something this interpreter lacks.
    Unsupported(String),
    /// Values disagree with the graph's types.
    Mismatch(String),
    Gpu(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Mismatch(m) => write!(f, "mismatch: {m}"),
            Error::Gpu(m) => write!(f, "gpu: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl Data {
    pub fn len(&self) -> usize {
        match self {
            Data::F32(v) => v.len(),
            Data::I32(v) => v.len(),
            Data::U32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> DType {
        match self {
            Data::F32(_) => DType::F32,
            Data::I32(_) => DType::I32,
            Data::U32(_) => DType::U32,
        }
    }

    /// Picks elements by index, keeping the element type.
    pub fn pick(&self, at: impl Iterator<Item = usize>) -> Data {
        match self {
            Data::F32(v) => Data::F32(at.map(|i| v[i]).collect()),
            Data::I32(v) => Data::I32(at.map(|i| v[i]).collect()),
            Data::U32(v) => Data::U32(at.map(|i| v[i]).collect()),
        }
    }

    pub fn cast(&self, to: DType) -> Result<Data, Error> {
        let f = |v: Vec<f64>| -> Result<Data, Error> {
            Ok(match to {
                DType::F32 => Data::F32(v.into_iter().map(|x| x as f32).collect()),
                DType::I32 => Data::I32(v.into_iter().map(|x| x as i32).collect()),
                DType::U32 => Data::U32(v.into_iter().map(|x| x as u32).collect()),
                other => return Err(Error::Unsupported(format!("cast to {other:?}"))),
            })
        };
        f(match self {
            Data::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Data::I32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Data::U32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        })
    }
}

impl Tensor {
    pub fn f32(shape: Vec<usize>, v: Vec<f32>) -> Tensor {
        Tensor {
            shape,
            data: Data::F32(v),
        }
    }

    pub fn floats(&self) -> Result<&[f32], Error> {
        match &self.data {
            Data::F32(v) => Ok(v),
            other => Err(Error::Mismatch(format!(
                "want f32, got {:?}",
                other.dtype()
            ))),
        }
    }

    /// Indices as usize, from either integer type.
    pub fn indices(&self) -> Result<Vec<usize>, Error> {
        match &self.data {
            Data::I32(v) => Ok(v.iter().map(|&i| i as usize).collect()),
            Data::U32(v) => Ok(v.iter().map(|&i| i as usize).collect()),
            Data::F32(_) => Err(Error::Mismatch("indices must be integers".into())),
        }
    }

    /// Size of the last axis, and how many such rows.
    pub fn rows(&self) -> (usize, usize) {
        let width = self.shape.last().copied().unwrap_or(1);
        let count = self.shape.iter().product::<usize>() / width.max(1);
        (count, width)
    }
}
