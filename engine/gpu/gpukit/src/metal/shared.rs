//! A shared-storage Metal buffer, mapped on unified memory.

use mem::{Backing, Error};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::any::Any;

pub struct Shared(Retained<ProtocolObject<dyn MTLBuffer>>);

impl Shared {
    /// Zeroed, at least four bytes long.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Option<Self> {
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer = device.newBufferWithLength_options(bytes.max(4), shared)?;
        let all = buffer.length();
        unsafe { std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, all) };
        Some(Self(buffer))
    }

    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.0
    }
}

impl Backing for Shared {
    fn bytes(&self) -> usize {
        self.0.length()
    }

    fn mapped(&self) -> Option<&[u8]> {
        let at = self.0.contents().as_ptr().cast::<u8>();
        Some(unsafe { std::slice::from_raw_parts(at, self.bytes()) })
    }

    fn mapped_mut(&mut self) -> Option<&mut [u8]> {
        let at = self.0.contents().as_ptr().cast::<u8>();
        Some(unsafe { std::slice::from_raw_parts_mut(at, self.bytes()) })
    }

    fn upload(&mut self, at: usize, bytes: &[u8]) -> Result<(), Error> {
        let all = self.mapped_mut().expect("shared storage is mapped");
        let Some(span) = all.get_mut(at..at + bytes.len()) else {
            return Err(Error::Mismatch("a write past the buffer".into()));
        };
        span.copy_from_slice(bytes);
        Ok(())
    }

    fn download(&self, at: usize, out: &mut [u8]) -> Result<(), Error> {
        let all = self.mapped().expect("shared storage is mapped");
        let Some(span) = all.get(at..at + out.len()) else {
            return Err(Error::Mismatch("a read past the buffer".into()));
        };
        out.copy_from_slice(span);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
