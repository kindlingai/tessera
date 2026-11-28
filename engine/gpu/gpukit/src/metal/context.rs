//! A Metal device on the shared device traits.

use crate::backend::Backend;
use crate::device::{Caches, Device};
use crate::launch::{Bound, KernelArgs, Launch};
use crate::metal::expand::expand;
use crate::metal::pipeline_cache::Pipelines;
use crate::pending::{Pending, PendingOps};
use crate::platform::{Asked, Platform};
use crate::{Error, Limits};
use objc2::msg_send;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLGPUFamily, MTLLibrary,
    MTLPipelineOption, MTLResourceOptions, MTLSize,
};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Buffers by the address `alloc` gave them.
#[derive(Default)]
struct Allocs {
    next: u64,
    by_base: BTreeMap<u64, Buf>,
    by_host: BTreeMap<u64, u64>,
}

/// Address space in pages, so offsets never cross buffers.
const PAGE: u64 = 4096;

impl Allocs {
    fn add(&mut self, buffer: Buf) -> u64 {
        let base = self.next.max(PAGE);
        self.next = base + (buffer.length() as u64).next_multiple_of(PAGE) + PAGE;
        self.by_base.insert(base, buffer);
        base
    }

    /// The buffer holding `ptr`, and how far in it lies.
    fn find(&self, ptr: u64) -> Result<(&Buf, usize), Error> {
        let (base, buffer) = self
            .by_base
            .range(..=ptr)
            .next_back()
            .filter(|(base, b)| ptr < *base + b.length() as u64)
            .ok_or_else(|| Error::Metal(format!("no buffer at {ptr:#x}")))?;
        Ok((buffer, (ptr - base) as usize))
    }
}

type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type CompEnc = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type MetalPending = Pending<CmdBuf, CompEnc>;

struct MetalOps<'a> {
    queue: &'a ProtocolObject<dyn MTLCommandQueue>,
}

impl PendingOps for MetalOps<'_> {
    type Cmd = CmdBuf;
    type Enc = CompEnc;

    fn new_cmd(&mut self) -> Result<CmdBuf, Error> {
        autoreleasepool(|_| {
            self.queue
                .commandBuffer()
                .ok_or(Error::Gpu("command buffer"))
        })
    }

    fn open_enc(&mut self, cmd: &CmdBuf, concurrent: bool) -> Result<CompEnc, Error> {
        let kind = match concurrent {
            true => MTLDispatchType::Concurrent,
            false => MTLDispatchType::Serial,
        };
        autoreleasepool(|_| {
            cmd.computeCommandEncoderWithDispatchType(kind)
                .ok_or(Error::Gpu("encoder"))
        })
    }

    fn end_enc(&mut self, enc: CompEnc) {
        autoreleasepool(|_| enc.endEncoding());
    }

    fn commit(&mut self, cmd: CmdBuf) -> f64 {
        autoreleasepool(|_| {
            cmd.commit();
            cmd.waitUntilCompleted();
            // The GPU's own clock leaves out the scheduling wait.
            let raw = &*cmd;
            let (start, end): (f64, f64) =
                unsafe { (msg_send![raw, GPUStartTime], msg_send![raw, GPUEndTime]) };
            end - start
        })
    }
}

thread_local! {
    /// This thread's open command buffer. Encoders are not shared.
    static PENDING: RefCell<Option<MetalPending>> = const { RefCell::new(None) };
    /// Whether this thread encodes launches concurrently with explicit barriers.
    static CONCURRENT: Cell<bool> = const { Cell::new(false) };
}

struct Inner {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Arc<Pipelines>,
    allocs: Mutex<Allocs>,
    msl_bfloat: OnceLock<bool>,
}

// SAFETY: Metal objects are thread-safe, and the buffers are locked.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// A Metal device, command queue and compiled pipelines.
#[derive(Clone)]
pub struct Context(Arc<Inner>);

/// A compiled kernel, bound to a device.
#[derive(Clone)]
pub struct ComputePipeline {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Typical GPU core count and memory bandwidth for Apple Silicon.
pub(super) fn apple_specs(name: &str) -> (usize, f64) {
    const SPECS: &[(&str, usize, f64)] = &[
        ("M5 Ultra", 80, 1228e9),
        ("M5 Max", 40, 614e9),
        ("M5 Pro", 20, 307e9),
        ("M5", 10, 153e9),
        ("M4 Ultra", 64, 819e9),
        ("M4 Max", 40, 546e9),
        ("M4 Pro", 20, 273e9),
        ("M4", 10, 120e9),
        ("M3 Ultra", 60, 800e9),
        ("M3 Max", 30, 300e9),
        ("M3 Pro", 14, 150e9),
        ("M3", 10, 100e9),
        ("M2 Ultra", 60, 800e9),
        ("M2 Max", 30, 400e9),
        ("M2 Pro", 16, 200e9),
        ("M2", 10, 100e9),
        ("M1 Ultra", 48, 800e9),
        ("M1 Max", 24, 400e9),
        ("M1 Pro", 14, 200e9),
        ("M1", 8, 68.25e9),
    ];
    SPECS
        .iter()
        .find(|(tag, _, _)| name.contains(tag))
        .map_or((0, 0.0), |&(_, units, bw)| (units, bw))
}

fn compile_error(err: Retained<NSError>) -> Error {
    Error::Compile(err.localizedDescription().to_string())
}

impl Context {
    /// Opens the default device.
    pub fn new() -> Result<Self, Error> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::Gpu("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or(Error::Gpu("failed to create Metal command queue"))?;
        Ok(Self(Arc::new(Inner {
            device,
            queue,
            pipelines: Pipelines::shared(),
            allocs: Mutex::new(Allocs::default()),
            msl_bfloat: OnceLock::new(),
        })))
    }

    /// Whether the compiler supports MSL 3.1 bfloat tiles.
    fn supports_msl_bfloat(&self) -> bool {
        const PROBE: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void p(device bfloat *a [[buffer(0)]]) {
    simdgroup_bfloat8x8 t;
    simdgroup_load(t, a, 8);
    simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_multiply_accumulate(c, t, t, c);
}
";
        *self.0.msl_bfloat.get_or_init(|| {
            let Ok(lib) = self.compile_library(PROBE) else {
                return false;
            };
            let Some(func) = lib.newFunctionWithName(&NSString::from_str("p")) else {
                return false;
            };
            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(Some(&func));
            self.0
                .device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &desc,
                    MTLPipelineOption::empty(),
                    None,
                )
                .is_ok()
        })
    }

    /// Apple GPUs run 32 lanes together. The rest is asked.
    pub fn limits(&self) -> Limits {
        let device = &self.0.device;
        Limits {
            lanes: 32,
            group_bytes: device.maxThreadgroupMemoryLength(),
            dynamic_group_bytes: 0,
            group_threads: device.maxThreadsPerThreadgroup().width,
            compiler: true,
            pdl: false,
            msl_bfloat: self.supports_msl_bfloat(),
        }
    }

    /// Compile a source string (quoted includes expanded) to a library.
    pub fn compile_library(
        &self,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, Error> {
        let ns_source = NSString::from_str(&expand(source, &[]));
        self.0
            .device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(compile_error)
    }

    /// Compile one kernel, cached by entry and text.
    pub fn compile_kernel(&self, source: &str, entry: &str) -> Result<ComputePipeline, Error> {
        let library = self.compile_library(source)?;
        // Two texts may share an entry. The label holds both.
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let label = format!("{entry}#{:016x}", hasher.finish());
        self.pipeline(&library, entry, &label)
    }

    /// `entry` in `library`, cached under `label`.
    fn pipeline(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        entry: &str,
        label: &str,
    ) -> Result<ComputePipeline, Error> {
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName(&name)
            .ok_or(Error::Gpu("Metal kernel entry not found"))?;
        let pipeline = self
            .0
            .pipelines
            .compile_compute(&self.0.device, &function, label)?;
        Ok(ComputePipeline { pipeline })
    }

    /// Bytes `at` in the buffer holding `ptr`, `len` long.
    fn span(&self, ptr: u64, len: usize) -> Result<NonNull<u8>, Error> {
        let allocs = lock(&self.0.allocs);
        let (buffer, at) = allocs.find(ptr)?;
        if at + len > buffer.length() {
            return Err(Error::Metal(format!("{len} bytes past the buffer")));
        }
        // The buffer stays alive while its address is held.
        Ok(unsafe { buffer.contents().cast::<u8>().add(at) })
    }

    /// Allocates a zeroed shared Metal buffer.
    fn make_buffer(&self, size: usize) -> Result<Buf, Error> {
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer = self
            .0
            .device
            .newBufferWithLength_options(size.max(4), shared)
            .ok_or(Error::Gpu("buffer allocation"))?;
        let all = buffer.length();
        unsafe { std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, all) };
        Ok(buffer)
    }

    fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    fn ops(&self) -> MetalOps<'_> {
        MetalOps {
            queue: &self.0.queue,
        }
    }

    /// This thread's open encoder, or a new command buffer's.
    fn encoder(&self, pending: &mut Option<MetalPending>) -> Result<CompEnc, Error> {
        let want = CONCURRENT.with(Cell::get);
        Pending::acquire(pending, self.id(), want, &mut self.ops())
    }

    fn fence_concurrent(&self) {
        PENDING.with(|p| {
            if let Some(pending) = p.borrow_mut().as_mut()
                && pending.owner == self.id()
            {
                pending.fence(|enc| enc.memoryBarrierWithScope(MTLBarrierScope::Buffers));
            }
        });
    }
}

impl Launch for Context {
    type Kernel = ComputePipeline;

    fn launch(
        &self,
        kernel: &ComputePipeline,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        args: &mut KernelArgs,
    ) -> Result<(), Error> {
        let enc = PENDING.with(|p| self.encoder(&mut p.borrow_mut()))?;
        enc.setComputePipelineState(&kernel.pipeline);
        let allocs = lock(&self.0.allocs);
        for (index, arg) in args.bound().enumerate() {
            match arg {
                Bound::Pointer(ptr) => {
                    let (buffer, at) = allocs.find(ptr)?;
                    unsafe { enc.setBuffer_offset_atIndex(Some(buffer), at, index) };
                }
                Bound::Bytes(bytes) => {
                    let at = NonNull::from(bytes).cast();
                    unsafe { enc.setBytes_length_atIndex(at, bytes.len(), index) };
                }
            }
        }
        if shared_bytes > 0 {
            unsafe { enc.setThreadgroupMemoryLength_atIndex(shared_bytes as usize, 0) };
        }
        let size = |(x, y, z): (u32, u32, u32)| MTLSize {
            width: x as usize,
            height: y as usize,
            depth: z as usize,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(grid), size(block));
        PENDING.with(|p| {
            if let Some(pending) = p.borrow_mut().as_mut()
                && pending.owner == self.id()
            {
                pending.dirty = true;
            }
        });
        Ok(())
    }
}

impl Device for Context {
    type Kernel = ComputePipeline;
    type Graph = ();

    const BACKEND: Backend = Backend::Metal;

    fn open() -> Result<Self, Error> {
        Self::new()
    }

    fn error(msg: String) -> Error {
        Error::Metal(msg)
    }

    fn caches() -> &'static Caches<Self> {
        static CACHES: Caches<Context> = Caches::new();
        &CACHES
    }

    /// The device's marketing name.
    fn name(&self) -> String {
        self.0.device.name().to_string()
    }

    fn platform(&self) -> Platform {
        let device = &self.0.device;
        let families = [
            (MTLGPUFamily::Apple9, "apple9"),
            (MTLGPUFamily::Apple8, "apple8"),
            (MTLGPUFamily::Apple7, "apple7"),
        ];
        let newest = families.iter().find(|(f, _)| device.supportsFamily(*f));
        // The architecture names the chip's GPU, as `applegpu_g15p`.
        let arch = match device.architecture().name().to_string() {
            name if name.starts_with("applegpu_") => name,
            _ => newest.map_or("apple", |(_, n)| n).into(),
        };
        let name = self.name();
        let (units, bandwidth) = apple_specs(&name);
        Platform::new(Asked {
            backend: crate::backend::Backend::Metal.name(),
            arch,
            name,
            driver: String::new(),
            limits: self.limits(),
            units,
            bandwidth,
            integrated: true,
            memory: device.recommendedMaxWorkingSetSize(),
        })
    }

    /// Shared storage, zeroed, at least four bytes long.
    fn alloc(&self, size: usize) -> Result<u64, Error> {
        let buffer = self.make_buffer(size)?;
        Ok(lock(&self.0.allocs).add(buffer))
    }

    fn free(&self, ptr: u64) {
        lock(&self.0.allocs).by_base.remove(&ptr);
    }

    fn pin(&self, size: usize) -> Result<(u64, u64), Error> {
        let buffer = self.make_buffer(size)?;
        let host = buffer.contents().as_ptr() as u64;
        let mut allocs = lock(&self.0.allocs);
        let base = allocs.add(buffer);
        allocs.by_host.insert(host, base);
        Ok((host, base))
    }

    fn unpin(&self, host: u64) {
        let mut allocs = lock(&self.0.allocs);
        if let Some(base) = allocs.by_host.remove(&host) {
            allocs.by_base.remove(&base);
        }
    }

    fn unified(&self) -> bool {
        self.0.device.hasUnifiedMemory()
    }

    fn free_memory(&self) -> Option<u64> {
        let device = &self.0.device;
        let used = device.currentAllocatedSize() as u64;
        Some(device.recommendedMaxWorkingSetSize().saturating_sub(used))
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        self.finish()?;
        let at = self.span(ptr, bytes.len())?;
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.as_ptr(), bytes.len()) };
        Ok(())
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        self.finish()?;
        let at = self.span(ptr, bytes.len())?;
        unsafe { std::ptr::copy_nonoverlapping(at.as_ptr(), bytes.as_mut_ptr(), bytes.len()) };
        Ok(())
    }

    fn copy_d2d(&self, dst: u64, src: u64, size: usize) -> Result<(), Error> {
        if size == 0 || dst == src {
            return Ok(());
        }
        self.finish()?;
        let from = self.span(src, size)?;
        let to = self.span(dst, size)?;
        unsafe { std::ptr::copy(from.as_ptr(), to.as_ptr(), size) };
        Ok(())
    }

    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error> {
        self.finish()?;
        let at = self.span(ptr, size)?;
        unsafe { std::ptr::write_bytes(at.as_ptr(), 0, size) };
        Ok(())
    }

    fn compile(&self, source: &str, entry: &str) -> Result<ComputePipeline, Error> {
        self.compile_kernel(source, entry)
    }

    /// Shared storage, once this thread's launches finish.
    fn mapped(&self, ptr: u64, size: usize) -> Option<NonNull<u8>> {
        self.finish().ok()?;
        self.span(ptr, size).ok()
    }

    /// Commits this thread's command buffer. Returns its GPU seconds.
    fn finish(&self) -> Result<Option<f64>, Error> {
        let taken = PENDING.with(|p| {
            let mut p = p.borrow_mut();
            match &*p {
                Some(pending) if pending.owner == self.id() => p.take(),
                _ => None,
            }
        });
        let mut ops = self.ops();
        Ok(taken.map(|p| p.finish(&mut ops)))
    }

    fn has_barriers(&self) -> bool {
        true
    }

    fn set_concurrent(&self, on: bool) {
        if !on {
            self.fence_concurrent();
        }
        CONCURRENT.with(|c| c.set(on));
    }

    fn barrier(&self) -> Result<(), Error> {
        self.fence_concurrent();
        Ok(())
    }
}
