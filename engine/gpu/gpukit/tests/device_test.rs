//! Capture-time pinned host mapping guards across backends.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tessera_gpukit::backend::Backend;
use tessera_gpukit::device::{Caches, Device, DeviceBuffer};
use tessera_gpukit::{Error, Platform};
use tessera_mem::Backing;

macro_rules! mock_device {
    ($name:ident, $backend:expr) => {
        #[derive(Clone, Default)]
        struct $name(Arc<AtomicBool>);

        impl Device for $name {
            type Kernel = ();
            type Graph = ();
            const BACKEND: Backend = $backend;

            fn open() -> Result<Self, Error> {
                Ok(Self::default())
            }
            fn error(_: String) -> Error {
                Error::Gpu("mock")
            }
            fn caches() -> &'static Caches<Self> {
                unimplemented!()
            }
            fn name(&self) -> String {
                stringify!($name).into()
            }
            fn platform(&self) -> Platform {
                unimplemented!()
            }
            fn alloc(&self, _: usize) -> Result<u64, Error> {
                Ok(4096)
            }
            fn free(&self, _: u64) {}
            fn pin(&self, size: usize) -> Result<(u64, u64), Error> {
                assert_eq!(size, 16);
                let ptr = Box::into_raw(Box::new([0u8; 16])) as u64;
                Ok((ptr, ptr))
            }
            fn unpin(&self, host: u64) {
                unsafe { drop(Box::from_raw(host as *mut [u8; 16])) };
            }
            fn upload(&self, _: u64, _: &[u8]) -> Result<(), Error> {
                Err(Error::Gpu("blocked during capture"))
            }
            fn download(&self, _: u64, _: &mut [u8]) -> Result<(), Error> {
                Err(Error::Gpu("blocked during capture"))
            }
            fn zero(&self, _: u64, _: usize) -> Result<(), Error> {
                Ok(())
            }
            fn compile(&self, _: &str, _: &str) -> Result<(), Error> {
                Ok(())
            }
            fn finish(&self) -> Result<Option<f64>, Error> {
                Ok(None)
            }
            fn capturing(&self) -> bool {
                self.0.load(Ordering::SeqCst)
            }
        }
    };
}

mock_device!(CudaMock, Backend::Cuda);
mock_device!(RocmMock, Backend::Rocm);
mock_device!(MetalMock, Backend::Metal);

/// CUDA and ROCm keep pinned mappings while Metal blocks them.
#[test]
fn pinned_capture_scope() {
    for (name, ctx_flag, mut pin, expect_mapped) in [
        (
            "cuda",
            {
                let c = CudaMock::default();
                let f = c.0.clone();
                let p = DeviceBuffer::pinned(&c, 16).unwrap();
                (f, Box::new(p) as Box<dyn Backing>)
            },
            true,
        ),
        (
            "rocm",
            {
                let c = RocmMock::default();
                let f = c.0.clone();
                let p = DeviceBuffer::pinned(&c, 16).unwrap();
                (f, Box::new(p) as Box<dyn Backing>)
            },
            true,
        ),
        (
            "metal",
            {
                let c = MetalMock::default();
                let f = c.0.clone();
                let p = DeviceBuffer::pinned(&c, 16).unwrap();
                (f, Box::new(p) as Box<dyn Backing>)
            },
            false,
        ),
    ]
    .into_iter()
    .map(|(n, (f, p), m)| (n, f, p, m))
    {
        pin.upload(0, &[5u8; 4]).expect(name);
        assert!(pin.mapped().is_some(), "{name}: mapped before capture");
        ctx_flag.store(true, Ordering::SeqCst);
        let mut out = [0u8; 4];
        if expect_mapped {
            assert!(pin.mapped().is_some(), "{name}: mapped during capture");
            assert!(pin.mapped_mut().is_some(), "{name}: mapped_mut");
            pin.upload(0, &[9u8; 4]).expect(name);
            pin.download(0, &mut out).expect(name);
            assert_eq!(out, [9u8; 4], "{name}: pinned write/read");
        } else {
            assert!(pin.mapped().is_none(), "{name}: blocked during capture");
            assert!(pin.mapped_mut().is_none(), "{name}: mapped_mut blocked");
            assert!(pin.upload(0, &[9u8; 4]).is_err(), "{name}: upload err");
            assert!(pin.download(0, &mut out).is_err(), "{name}: download err");
        }
        ctx_flag.store(false, Ordering::SeqCst);
        assert!(pin.mapped().is_some(), "{name}: mapped after capture");
        pin.download(0, &mut out).expect(name);
        let want = if expect_mapped { [9u8; 4] } else { [5u8; 4] };
        assert_eq!(out, want, "{name}: post-capture bytes");
    }
}
