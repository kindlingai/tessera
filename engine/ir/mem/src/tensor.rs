//! An owned host value: a shape over packed bytes.

use crate::buffer::Buffer;
use crate::layout::Layout;
use crate::view::{View, ViewMut};
use crate::{Error, mismatch};
use graph::DType;
use std::fmt;

/// Row-major and contiguous, on the host.
pub struct Tensor {
    pub shape: Vec<usize>,
    dtype: DType,
    form: Option<&'static str>,
    /// Elements held, whether or not the shape agrees.
    count: usize,
    layout: Layout,
    buffer: Buffer,
}

fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // Plain numbers, read as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
}

impl Tensor {
    /// `bytes` as `dtype` elements, filling `shape`.
    pub fn from_bytes(shape: Vec<usize>, dtype: DType, bytes: &[u8]) -> Self {
        let mut out = Self::blank(shape, dtype, bytes.len());
        out.bytes_mut().copy_from_slice(bytes);
        out
    }

    /// Zeroed host memory for `bytes` of `dtype`, under `shape`.
    fn blank(shape: Vec<usize>, dtype: DType, bytes: usize) -> Self {
        let layout = Layout::contiguous(shape.clone());
        Self {
            shape,
            dtype,
            form: None,
            count: bytes / dtype.bytes(),
            layout,
            buffer: Buffer::host(0, bytes),
        }
    }

    /// Pre-packed bytes of registered form `form`, filling `shape`.
    pub fn packed(shape: Vec<usize>, form: &'static str, bytes: &[u8]) -> Self {
        let mut out = Self {
            count: shape.iter().product(),
            layout: Layout::contiguous(shape.clone()),
            shape,
            dtype: DType::F32,
            form: Some(form),
            buffer: Buffer::host(0, bytes.len()),
        };
        out.bytes_mut().copy_from_slice(bytes);
        out
    }

    pub fn f32(shape: Vec<usize>, v: Vec<f32>) -> Self {
        Self::from_bytes(shape, DType::F32, bytes_of(&v))
    }

    pub fn u32(shape: Vec<usize>, v: Vec<u32>) -> Self {
        Self::from_bytes(shape, DType::U32, bytes_of(&v))
    }

    pub fn i32(shape: Vec<usize>, v: Vec<i32>) -> Self {
        Self::from_bytes(shape, DType::I32, bytes_of(&v))
    }

    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let bytes = dtype.bytes() * shape.iter().product::<usize>();
        Self::blank(shape, dtype, bytes)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Whether the shape accounts for every element.
    pub fn filled(&self) -> bool {
        self.layout.len() == self.count
    }

    pub fn view(&self) -> View<'_> {
        View {
            form: self.form,
            buffer: &self.buffer,
            at: 0,
            dtype: self.dtype,
            layout: self.layout.clone(),
        }
    }

    pub fn view_mut(&mut self) -> ViewMut<'_> {
        assert!(
            self.form.is_none(),
            "packed weights have no typed mutable view"
        );
        ViewMut {
            buffer: &mut self.buffer,
            at: 0,
            dtype: self.dtype,
            layout: self.layout.clone(),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        let span = self
            .form
            .map_or(self.dtype.bytes() * self.count, |_| self.buffer.bytes());
        &self.buffer.backing().mapped().expect("host memory")[..span]
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        let span = self
            .form
            .map_or(self.dtype.bytes() * self.count, |_| self.buffer.bytes());
        &mut self.buffer.backing_mut().mapped_mut().expect("host memory")[..span]
    }

    pub fn floats(&self) -> Result<&[f32], Error> {
        self.view().as_f32()
    }

    /// Indices as usize, from either integer type.
    pub fn indices(&self) -> Result<Vec<usize>, Error> {
        self.view().indices()
    }

    /// Size of the last axis, and how many such rows.
    pub fn rows(&self) -> (usize, usize) {
        let width = self.shape.last().copied().unwrap_or(1);
        let count = self.len() / width.max(1);
        (count, width)
    }

    /// The same bytes under another shape.
    pub fn reshaped(&self, shape: Vec<usize>) -> Self {
        if let Some(form) = self.form {
            assert_eq!(shape.iter().product::<usize>(), self.count);
            return Self::packed(shape, form, self.bytes());
        }
        Self::from_bytes(shape, self.dtype, self.bytes())
    }

    /// Elements by index, keeping the element type.
    pub fn pick(&self, shape: Vec<usize>, at: impl Iterator<Item = usize>) -> Self {
        assert!(self.form.is_none(), "packed weights cannot select elements");
        let (all, size) = (self.bytes(), self.dtype.bytes());
        let mut out = Vec::with_capacity(size * shape.iter().product::<usize>());
        for i in at {
            out.extend_from_slice(&all[i * size..(i + 1) * size]);
        }
        Self::from_bytes(shape, self.dtype, &out)
    }

    pub fn cast(&self, to: DType) -> Result<Self, Error> {
        let wide: Vec<f64> = match self.dtype {
            DType::F32 => self.floats()?.iter().map(|x| f64::from(*x)).collect(),
            DType::I32 => self
                .view()
                .as_i32()?
                .iter()
                .map(|x| f64::from(*x))
                .collect(),
            DType::U32 => self
                .view()
                .as_u32()?
                .iter()
                .map(|x| f64::from(*x))
                .collect(),
            other => return mismatch(format_args!("cast from {other:?}")),
        };
        let shape = self.shape.clone();
        Ok(match to {
            DType::F32 => Self::f32(shape, wide.into_iter().map(|x| x as f32).collect()),
            DType::I32 => Self::i32(shape, wide.into_iter().map(|x| x as i32).collect()),
            DType::U32 => Self::u32(shape, wide.into_iter().map(|x| x as u32).collect()),
            other => return mismatch(format_args!("cast to {other:?}")),
        })
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        self.reshaped(self.shape.clone())
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.dtype == other.dtype
            && self.form == other.form
            && self.bytes() == other.bytes()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(form) = self.form {
            return write!(
                f,
                "Tensor {form} {:?} {} bytes",
                self.shape,
                self.bytes().len()
            );
        }
        write!(f, "Tensor {:?} {:?} ", self.dtype, self.shape)?;
        match self.dtype {
            DType::F32 => write!(f, "{:?}", self.floats().unwrap_or(&[])),
            DType::U32 | DType::I32 => write!(f, "{:?}", self.indices().unwrap_or_default()),
            _ => write!(f, "{} bytes", self.bytes().len()),
        }
    }
}
