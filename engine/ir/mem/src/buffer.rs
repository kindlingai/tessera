//! An allocation in one zone, behind whatever backend holds it.

use crate::layout::Layout;
use crate::view::{View, ViewMut};
use crate::{Error, mismatch};
use graph::DType;
use std::any::Any;

/// Memory of one zone, however the backend holds it.
pub trait Backing: Any {
    fn bytes(&self) -> usize;
    /// The bytes, when the host can address them.
    fn mapped(&self) -> Option<&[u8]>;
    fn mapped_mut(&mut self) -> Option<&mut [u8]>;
    /// Host to device, `at` bytes in. Synchronous.
    fn upload(&mut self, at: usize, bytes: &[u8]) -> Result<(), Error>;
    /// Device to host, `at` bytes in. Synchronous.
    fn download(&self, at: usize, out: &mut [u8]) -> Result<(), Error>;
    fn as_any(&self) -> &dyn Any;
}

/// Host memory in words. Four-byte views stay aligned.
pub struct Host(Vec<u32>, usize);

impl Host {
    /// Zeroed and touched: a device copy into fresh pages crawls.
    #[allow(clippy::slow_vector_initialization)]
    pub fn new(bytes: usize) -> Self {
        let mut words = Vec::with_capacity(bytes.div_ceil(4));
        words.resize(bytes.div_ceil(4), 0);
        Self(words, bytes)
    }
}

impl Backing for Host {
    fn bytes(&self) -> usize {
        self.1
    }

    fn mapped(&self) -> Option<&[u8]> {
        let (words, bytes) = (self.0.as_ptr().cast::<u8>(), self.bytes());
        // The words are plain data, read as bytes in place.
        Some(unsafe { std::slice::from_raw_parts(words, bytes) })
    }

    fn mapped_mut(&mut self) -> Option<&mut [u8]> {
        let (words, bytes) = (self.0.as_mut_ptr().cast::<u8>(), self.bytes());
        Some(unsafe { std::slice::from_raw_parts_mut(words, bytes) })
    }

    fn upload(&mut self, at: usize, bytes: &[u8]) -> Result<(), Error> {
        let all = self.mapped_mut().expect("host memory is mapped");
        let Some(span) = all.get_mut(at..at + bytes.len()) else {
            return mismatch("a write past the end of host memory");
        };
        span.copy_from_slice(bytes);
        Ok(())
    }

    fn download(&self, at: usize, out: &mut [u8]) -> Result<(), Error> {
        let all = self.mapped().expect("host memory is mapped");
        let Some(span) = all.get(at..at + out.len()) else {
            return mismatch("a read past the end of host memory");
        };
        out.copy_from_slice(span);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An allocation, and the zone that holds it.
pub struct Buffer {
    pub zone: u32,
    backing: Box<dyn Backing>,
}

impl Buffer {
    pub fn new(zone: u32, backing: impl Backing) -> Self {
        Self {
            zone,
            backing: Box::new(backing),
        }
    }

    /// Zeroed host memory.
    pub fn host(zone: u32, bytes: usize) -> Self {
        Self::new(zone, Host::new(bytes))
    }

    pub fn bytes(&self) -> usize {
        self.backing.bytes()
    }

    pub fn backing(&self) -> &dyn Backing {
        &*self.backing
    }

    pub fn backing_mut(&mut self) -> &mut dyn Backing {
        &mut *self.backing
    }

    /// A contiguous window `at` bytes in, shaped as `shape`.
    pub fn view(&self, at: usize, dtype: DType, shape: Vec<usize>) -> View<'_> {
        let layout = Layout::contiguous(shape);
        View {
            form: None,
            buffer: self,
            at,
            dtype,
            layout,
        }
    }

    pub fn view_mut(&mut self, at: usize, dtype: DType, shape: Vec<usize>) -> ViewMut<'_> {
        let layout = Layout::contiguous(shape);
        ViewMut {
            buffer: self,
            at,
            dtype,
            layout,
        }
    }
}
