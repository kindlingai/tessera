//! A Metal device on the shared device traits.

use crate::backend::Backend;
use crate::device::{Caches, Device};
use crate::launch::{Bound, KernelArgs, Launch};
use crate::metal::expand::expand;
use crate::metal::graph::{Buf, Graph, GraphData, Op, Recording};
use crate::metal::pipeline_cache::Pipelines;
use crate::pending::{Pending, PendingCmd, PendingEnc, PendingOps};
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

/// Queue command buffer capacity for chunked graph replays.
const MAX_COMMAND_BUFFERS: usize = 256;
/// Dispatches before the first barrier flush wakes the GPU.
const FIRST_CHUNK: usize = 64;
/// Dispatches per subsequent barrier-aligned chunk during graph replay.
const NEXT_CHUNK: usize = 512;

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
type MetalPending = Pending<CmdBuf, CompEnc, Arc<GraphData>>;

impl PendingCmd for CmdBuf {
    fn wait_done(&self) {
        autoreleasepool(|_| self.waitUntilCompleted());
    }
}

impl PendingEnc for CompEnc {
    fn end_open(&self) {
        autoreleasepool(|_| self.endEncoding());
    }
}

struct MetalOps<'a> {
    queue: &'a ProtocolObject<dyn MTLCommandQueue>,
}

impl PendingOps for MetalOps<'_> {
    type Cmd = CmdBuf;
    type Enc = CompEnc;

    fn new_cmd(&mut self, unretained: bool) -> Result<CmdBuf, Error> {
        autoreleasepool(|_| {
            let cmd = match unretained {
                true => self.queue.commandBufferWithUnretainedReferences(),
                false => self.queue.commandBuffer(),
            };
            cmd.ok_or(Error::Gpu("command buffer"))
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
        enc.end_open();
    }

    fn commit(&mut self, cmd: &CmdBuf) {
        autoreleasepool(|_| cmd.commit());
    }

    fn wait(&mut self, cmd: &CmdBuf) -> (f64, f64) {
        autoreleasepool(|_| {
            cmd.waitUntilCompleted();
            // The GPU's own clock leaves out the scheduling wait.
            let raw = &**cmd;
            unsafe { (msg_send![raw, GPUStartTime], msg_send![raw, GPUEndTime]) }
        })
    }
}

thread_local! {
    /// This thread's open command buffer. Encoders are not shared.
    static PENDING: RefCell<Option<MetalPending>> = const { RefCell::new(None) };
    /// Whether this thread encodes launches concurrently with explicit barriers.
    static CONCURRENT: Cell<bool> = const { Cell::new(false) };
    /// Active Metal graph recording on this thread, if any.
    static CAPTURING: RefCell<Option<Recording>> = const { RefCell::new(None) };
}

struct CaptureGuard;

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        CAPTURING.with(|c| {
            c.borrow_mut().take();
        });
    }
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
            .newCommandQueueWithMaxCommandBufferCount(MAX_COMMAND_BUFFERS)
            .or_else(|| device.newCommandQueue())
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

    fn reject_capture(&self, what: &str) -> Result<(), Error> {
        if self.capturing() {
            return Err(Error::Metal(format!(
                "cannot {what} during Metal graph capture"
            )));
        }
        Ok(())
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

    fn replay_graph(&self, graph: &Graph) -> Result<(), Error> {
        let data = &graph.0;
        if data.owner != self.id() {
            return Err(Error::Metal(
                "graph belongs to another Metal context".into(),
            ));
        }
        self.reject_capture("replay a graph")?;
        let mut ops = self.ops();
        autoreleasepool(|_| {
            PENDING.with(|slot| {
                let mut pending = slot.borrow_mut();
                if let Some(p) = &*pending
                    && p.owner != self.id()
                {
                    let _ = pending.take().map(|other| other.finish(&mut ops));
                }
                let mut want = data.initial_concurrent;
                let mut attached = false;
                let mut last_pipe = 0usize;
                let mut chunk_limit = match &*pending {
                    Some(p) if !p.flushed.is_empty() || p.dispatches >= FIRST_CHUNK => NEXT_CHUNK,
                    _ => FIRST_CHUNK,
                };
                for (idx, op) in data.ops.iter().enumerate() {
                    match op {
                        Op::Concurrent(on) => want = *on,
                        Op::Barrier => {
                            if let Some(p) = &mut *pending
                                && want
                                && p.dispatches > 0
                            {
                                let more = data.last_dispatch.is_some_and(|last| idx < last);
                                if p.dispatches >= chunk_limit && more {
                                    // Committing the chunk acts as a full device barrier.
                                    p.rotate(want, true, &mut ops)?;
                                    last_pipe = 0;
                                    chunk_limit = NEXT_CHUNK;
                                } else {
                                    p.fence(|enc| {
                                        enc.memoryBarrierWithScope(MTLBarrierScope::Buffers)
                                    });
                                }
                            }
                        }
                        Op::Dispatch {
                            pipeline,
                            grid,
                            block,
                            shared,
                            bufs,
                            bytes,
                        } => {
                            let p = match &mut *pending {
                                Some(p) => p,
                                None => {
                                    pending.insert(Pending::open(self.id(), want, true, &mut ops)?)
                                }
                            };
                            if !attached {
                                p.keep.push(Arc::clone(data));
                                attached = true;
                            }
                            if p.dispatches >= chunk_limit * 2 {
                                p.rotate(want, true, &mut ops)?;
                                last_pipe = 0;
                                chunk_limit = NEXT_CHUNK;
                            } else if p.concurrent != want || p.enc.is_none() {
                                let _ = p.ensure_mode(want, &mut ops)?;
                                last_pipe = 0;
                            }
                            let enc = p.enc.as_ref().expect("opened encoder");
                            let pid = (&**pipeline as *const _) as usize;
                            if last_pipe != pid {
                                enc.setComputePipelineState(pipeline);
                                last_pipe = pid;
                            }
                            for arg in &data.bufs[bufs.start as usize..bufs.end as usize] {
                                unsafe {
                                    enc.setBuffer_offset_atIndex(
                                        Some(&*arg.buf),
                                        arg.offset,
                                        arg.index as usize,
                                    );
                                }
                            }
                            for arg in &data.byte_args[bytes.start as usize..bytes.end as usize] {
                                let start = arg.start as usize;
                                let slice = &data.bytes[start..start + arg.len as usize];
                                let at = NonNull::from(slice).cast();
                                unsafe {
                                    enc.setBytes_length_atIndex(
                                        at,
                                        slice.len(),
                                        arg.index as usize,
                                    );
                                }
                            }
                            if *shared > 0 {
                                unsafe {
                                    enc.setThreadgroupMemoryLength_atIndex(*shared as usize, 0);
                                }
                            }
                            enc.dispatchThreadgroups_threadsPerThreadgroup(*grid, *block);
                            p.mark_dispatch();
                        }
                    }
                }
                Ok(())
            })
        })
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
        let size = |(x, y, z): (u32, u32, u32)| MTLSize {
            width: x as usize,
            height: y as usize,
            depth: z as usize,
        };
        let recorded = CAPTURING.with(|slot| {
            let mut slot = slot.borrow_mut();
            let rec = slot.as_mut().filter(|r| r.owner() == self.id())?;
            let allocs = lock(&self.0.allocs);
            Some(rec.dispatch(
                kernel.pipeline.clone(),
                size(grid),
                size(block),
                shared_bytes,
                args,
                |ptr| allocs.find(ptr).map(|(b, off)| (b.clone(), off)),
            ))
        });
        if let Some(res) = recorded {
            return res;
        }
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
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(grid), size(block));
        PENDING.with(|p| {
            if let Some(pending) = p.borrow_mut().as_mut()
                && pending.owner == self.id()
            {
                pending.mark_dispatch();
            }
        });
        Ok(())
    }
}

impl Device for Context {
    type Kernel = ComputePipeline;
    type Graph = Graph;

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
        self.reject_capture("upload")?;
        self.finish()?;
        let at = self.span(ptr, bytes.len())?;
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.as_ptr(), bytes.len()) };
        Ok(())
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        self.reject_capture("download")?;
        self.finish()?;
        let at = self.span(ptr, bytes.len())?;
        unsafe { std::ptr::copy_nonoverlapping(at.as_ptr(), bytes.as_mut_ptr(), bytes.len()) };
        Ok(())
    }

    fn copy_d2d(&self, dst: u64, src: u64, size: usize) -> Result<(), Error> {
        self.reject_capture("copy_d2d")?;
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
        self.reject_capture("zero")?;
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
        if self.capturing() {
            return None;
        }
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
        let recorded = CAPTURING.with(|slot| {
            let mut slot = slot.borrow_mut();
            let Some(rec) = slot.as_mut().filter(|r| r.owner() == self.id()) else {
                return false;
            };
            rec.set_concurrent(on);
            true
        });
        if !recorded {
            if !on {
                self.fence_concurrent();
            }
            CONCURRENT.with(|c| c.set(on));
        }
    }

    fn barrier(&self) -> Result<(), Error> {
        let recorded = CAPTURING.with(|slot| {
            let mut slot = slot.borrow_mut();
            let Some(rec) = slot.as_mut().filter(|r| r.owner() == self.id()) else {
                return false;
            };
            rec.barrier();
            true
        });
        if !recorded {
            self.fence_concurrent();
        }
        Ok(())
    }

    fn graphs(&self) -> bool {
        true
    }

    fn capturing(&self) -> bool {
        CAPTURING.with(|slot| {
            slot.borrow()
                .as_ref()
                .is_some_and(|r| r.owner() == self.id())
        })
    }

    fn capture<E: From<Error>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<Self::Graph, E> {
        let already = CAPTURING.with(|slot| slot.borrow().is_some());
        if already {
            return Err(Error::Metal("nested Metal graph capture".into()).into());
        }
        let initial = CONCURRENT.with(Cell::get);
        CAPTURING.with(|slot| {
            *slot.borrow_mut() = Some(Recording::new(self.id(), initial));
        });
        let _guard = CaptureGuard;
        record()?;
        let rec = CAPTURING
            .with(|slot| slot.borrow_mut().take())
            .ok_or_else(|| Error::Metal("Metal graph capture lost".into()))?;
        Ok(rec.finish())
    }

    fn replay(&self, graph: &Self::Graph) -> Result<(), Error> {
        self.replay_graph(graph)
    }
}
