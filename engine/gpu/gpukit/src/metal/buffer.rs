use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Size-bucketed `MTLBuffer` pool (`StorageModeShared` for unified memory).
pub struct BufferPool {
    free: Vec<(usize, Retained<ProtocolObject<dyn MTLBuffer>>)>,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool {
    pub fn new() -> Self {
        Self { free: Vec::new() }
    }

    pub fn allocate(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        size: usize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let buf = if let Some(idx) = self.free.iter().position(|(cap, _)| *cap >= size) {
            let (_, buf) = self.free.swap_remove(idx);
            buf
        } else {
            device.newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)?
        };
        // Zero so a partial write never exposes stale bytes.
        let len = buf.length();
        unsafe {
            std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, len);
        }
        Some(buf)
    }

    /// Drop all pooled buffers.
    pub fn clear(&mut self) {
        self.free.clear();
    }

    pub fn release(&mut self, capacity: usize, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        self.free.push((capacity, buffer));
    }

    /// Keep at most `max_buffers` pooled, freeing the rest.
    pub fn trim(&mut self, max_buffers: usize) {
        if self.free.len() > max_buffers {
            self.free.truncate(max_buffers);
        }
    }

    pub fn write_f32(buffer: &ProtocolObject<dyn MTLBuffer>, data: &[f32]) {
        let ptr = buffer.contents().as_ptr() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub fn read_f32(buffer: &ProtocolObject<dyn MTLBuffer>, out: &mut [f32]) {
        Self::read_f32_at(buffer, 0, out);
    }

    pub fn read_f32_at(
        buffer: &ProtocolObject<dyn MTLBuffer>,
        byte_offset: usize,
        out: &mut [f32],
    ) {
        let ptr = buffer.contents().as_ptr() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(byte_offset / 4), out.as_mut_ptr(), out.len());
        }
    }
}
