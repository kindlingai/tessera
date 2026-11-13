//! Typed windows onto buffers, and copies between them.

use crate::buffer::Buffer;
use crate::layout::Layout;
use crate::tensor::Tensor;
use crate::{Error, mismatch};
use graph::{Bf16, DType};
use std::borrow::Cow;

/// A window `at` bytes into a buffer, with a layout.
#[derive(Clone)]
pub struct View<'a> {
    pub(crate) buffer: &'a Buffer,
    pub(crate) at: usize,
    pub(crate) dtype: DType,
    pub(crate) layout: Layout,
}

/// A window held alone. Writes go through it.
pub struct ViewMut<'a> {
    pub(crate) buffer: &'a mut Buffer,
    pub(crate) at: usize,
    pub(crate) dtype: DType,
    pub(crate) layout: Layout,
}

/// A typed slice of mapped bytes, when aligned.
fn typed<T>(bytes: &[u8]) -> Result<&[T], Error> {
    let size = size_of::<T>();
    let aligned = (bytes.as_ptr() as usize).is_multiple_of(align_of::<T>());
    if !aligned || !bytes.len().is_multiple_of(size) {
        return mismatch("a view off its element boundary");
    }
    // Plain numbers: any bytes are a valid value.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / size) })
}

fn typed_mut<T>(bytes: &mut [u8]) -> Result<&mut [T], Error> {
    let size = size_of::<T>();
    let aligned = (bytes.as_ptr() as usize).is_multiple_of(align_of::<T>());
    if !aligned || !bytes.len().is_multiple_of(size) {
        return mismatch("a view off its element boundary");
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len() / size) })
}

impl<'a> View<'a> {
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn len(&self) -> usize {
        self.layout.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    pub fn zone(&self) -> u32 {
        self.buffer.zone
    }

    /// Bytes from `at` to the furthest element, as stored.
    pub fn bytes(&self) -> Result<Cow<'a, [u8]>, Error> {
        let span = self.dtype.bytes() * self.layout.span();
        if self.at + span > self.buffer.bytes() {
            return mismatch("a view past the end of its buffer");
        }
        match self.buffer.backing().mapped() {
            Some(all) => Ok(Cow::Borrowed(&all[self.at..self.at + span])),
            None => {
                let mut out = vec![0u8; span];
                self.buffer.backing().download(self.at, &mut out)?;
                Ok(Cow::Owned(out))
            }
        }
    }

    /// The elements in row-major order, as contiguous bytes.
    pub fn packed(&self) -> Result<Cow<'a, [u8]>, Error> {
        let bytes = self.bytes()?;
        if self.layout.is_contiguous() {
            return Ok(bytes);
        }
        let size = self.dtype.bytes();
        let mut out = Vec::with_capacity(size * self.len());
        for at in self.layout.offsets() {
            out.extend_from_slice(&bytes[at * size..(at + 1) * size]);
        }
        Ok(Cow::Owned(out))
    }

    fn mapped(&self) -> Result<&'a [u8], Error> {
        if !self.layout.is_contiguous() {
            return mismatch("a strided view read in place");
        }
        let span = self.dtype.bytes() * self.len();
        match self.buffer.backing().mapped() {
            Some(all) if self.at + span <= all.len() => Ok(&all[self.at..self.at + span]),
            Some(_) => mismatch("a view past the end of its buffer"),
            None => mismatch(format_args!("zone {} is not mapped", self.zone())),
        }
    }

    fn expect(&self, dtype: DType) -> Result<(), Error> {
        match self.dtype == dtype {
            true => Ok(()),
            false => mismatch(format_args!("want {dtype:?}, got {:?}", self.dtype)),
        }
    }

    /// The floats in place: mapped, contiguous and f32.
    pub fn as_f32(&self) -> Result<&'a [f32], Error> {
        self.expect(DType::F32)?;
        typed(self.mapped()?)
    }

    pub fn as_u32(&self) -> Result<&'a [u32], Error> {
        self.expect(DType::U32)?;
        typed(self.mapped()?)
    }

    pub fn as_i32(&self) -> Result<&'a [i32], Error> {
        self.expect(DType::I32)?;
        typed(self.mapped()?)
    }

    /// The floats, wherever they lie. A bf16 widens.
    pub fn to_f32(&self) -> Result<Vec<f32>, Error> {
        let packed = self.packed()?;
        match self.dtype {
            DType::F32 => Ok(typed::<f32>(&packed)?.to_vec()),
            DType::Bf16 => Ok(typed::<u16>(&packed)?
                .iter()
                .map(|h| Bf16(*h).into())
                .collect()),
            other => mismatch(format_args!("floats from {other:?}")),
        }
    }

    pub fn to_u32(&self) -> Result<Vec<u32>, Error> {
        self.expect(DType::U32)?;
        Ok(typed::<u32>(&self.packed()?)?.to_vec())
    }

    /// Indices as usize, from either integer type.
    pub fn indices(&self) -> Result<Vec<usize>, Error> {
        let packed = self.packed()?;
        match self.dtype {
            DType::U32 => Ok(typed::<u32>(&packed)?.iter().map(|i| *i as usize).collect()),
            DType::I32 => Ok(typed::<i32>(&packed)?.iter().map(|i| *i as usize).collect()),
            other => mismatch(format_args!("{other:?} are not indices")),
        }
    }

    /// A packed copy on the host.
    pub fn to_tensor(&self) -> Result<Tensor, Error> {
        let mut out = Tensor::zeros(self.layout.shape.clone(), self.dtype);
        if !self.layout.is_contiguous() {
            out.bytes_mut().copy_from_slice(&self.packed()?);
            return Ok(out);
        }
        // Straight into the tensor: one copy.
        let span = self.dtype.bytes() * self.len();
        if self.at + span > self.buffer.bytes() {
            return mismatch("a view past the end of its buffer");
        }
        let backing = self.buffer.backing();
        match backing.mapped() {
            Some(all) => out
                .bytes_mut()
                .copy_from_slice(&all[self.at..self.at + span]),
            None => backing.download(self.at, out.bytes_mut())?,
        }
        Ok(out)
    }

    pub fn permute(&self, order: &[usize]) -> View<'a> {
        View {
            buffer: self.buffer,
            at: self.at,
            dtype: self.dtype,
            layout: self.layout.permute(order),
        }
    }

    pub fn slice(&self, axis: usize, start: usize, len: usize) -> View<'a> {
        let (layout, skip) = self.layout.slice(axis, start, len);
        View {
            buffer: self.buffer,
            at: self.at + skip * self.dtype.bytes(),
            dtype: self.dtype,
            layout,
        }
    }

    /// The same elements under another shape.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<View<'a>, Error> {
        let fits = self.layout.is_contiguous() && shape.iter().product::<usize>() == self.len();
        if !fits {
            return mismatch(format_args!(
                "{:?} does not reshape to {shape:?}",
                self.shape()
            ));
        }
        Ok(View {
            buffer: self.buffer,
            at: self.at,
            dtype: self.dtype,
            layout: Layout::contiguous(shape),
        })
    }
}

impl ViewMut<'_> {
    pub fn view(&self) -> View<'_> {
        View {
            buffer: self.buffer,
            at: self.at,
            dtype: self.dtype,
            layout: self.layout.clone(),
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.layout.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    fn mapped_mut(&mut self) -> Result<&mut [u8], Error> {
        if !self.layout.is_contiguous() {
            return mismatch("a strided view written in place");
        }
        let (at, span) = (self.at, self.dtype.bytes() * self.layout.len());
        let zone = self.buffer.zone;
        match self.buffer.backing_mut().mapped_mut() {
            Some(all) if at + span <= all.len() => Ok(&mut all[at..at + span]),
            Some(_) => mismatch("a view past the end of its buffer"),
            None => mismatch(format_args!("zone {zone} is not mapped")),
        }
    }

    pub fn as_f32_mut(&mut self) -> Result<&mut [f32], Error> {
        self.view().expect(DType::F32)?;
        typed_mut(self.mapped_mut()?)
    }

    pub fn as_u32_mut(&mut self) -> Result<&mut [u32], Error> {
        self.view().expect(DType::U32)?;
        typed_mut(self.mapped_mut()?)
    }

    /// Writes packed bytes of the view's type.
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        if !self.layout.is_contiguous() {
            return mismatch("a strided view written");
        }
        if bytes.len() != self.dtype.bytes() * self.len() {
            return mismatch(format_args!(
                "{} bytes for {} elements",
                bytes.len(),
                self.len()
            ));
        }
        let at = self.at;
        self.buffer.backing_mut().upload(at, bytes)
    }
}

/// Copies `src` into `dst` of the same type.
pub fn copy(src: View, dst: &mut ViewMut) -> Result<(), Error> {
    if src.len() != dst.len() || src.dtype != dst.dtype {
        let (s, d) = ((src.len(), src.dtype), (dst.len(), dst.dtype));
        return mismatch(format_args!("{s:?} into {d:?}"));
    }
    dst.write(&src.packed()?)
}
