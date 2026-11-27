//! Occupancy attrs, scoped GraphKeys, targeted invalidation, overrides, caching and sizer.

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard};
use tessera_codegen::ir::Body;
use tessera_engine_comm::{Error as CommError, Transport};
use tessera_engine_settings::{Backend, ByBackend, Memory, Settings};
use tessera_gpukit::device::{
    Caches, Device, KernelAttrs, KernelStats, cached_context, cached_prefixed, cached_source,
};
use tessera_gpukit::launch::{Bound, KernelArgs, Launch};
use tessera_gpukit::{Error as GpuError, Limits, Platform};
use tessera_graph::DimName::{self, Context};
use tessera_lower::kernels::AllReduce;
use tessera_lower::{
    Arg, COMMIT, Extent, Kernel, KernelRef, Launch as Call, Launching, Needs, Plan, Resident,
    Shape, Slot, Source, Step,
};
use tessera_mem::DType;
use tessera_run::testing::{device, host};
use tessera_run::{Error as RunError, Grid, Item, KeptDesc, Layout, Rank, Run, Sum, sizer};

#[derive(Default)]
struct MockState {
    events: Vec<String>,
    fail_alloc: Option<usize>,
    fail_finish: usize,
    grid_running: bool,
    attr_calls: Vec<(u32, u32)>,
    recording: Option<Vec<u64>>,
    mem: HashMap<u64, Vec<u8>>,
    next_ptr: u64,
    modules: HashSet<u64>,
    next_mod: u64,
    allocs: usize,
    frees: usize,
    compiles: usize,
    captures: usize,
    replays: usize,
}

#[derive(Clone)]
struct TinyDevice(Arc<Mutex<MockState>>);

struct TinyModule {
    state: Arc<Mutex<MockState>>,
    id: u64,
}

impl Drop for TinyModule {
    fn drop(&mut self) {
        let mut s = self.state.lock().unwrap();
        assert!(!s.grid_running, "dropped module while grid ran");
        assert!(s.modules.remove(&self.id), "double module drop {}", self.id);
        s.events.push(format!("drop_module:{}", self.id));
    }
}

#[derive(Clone)]
struct TinyKernel(Arc<TinyModule>);

struct TinyGraph {
    state: Arc<Mutex<MockState>>,
    ptrs: Vec<u64>,
}

impl Drop for TinyGraph {
    fn drop(&mut self) {
        let mut s = self.state.lock().unwrap();
        assert!(!s.grid_running, "dropped graph while grid ran");
        let first = self.ptrs.first().copied().unwrap_or(0);
        s.events.push(format!("drop_graph:{first:#x}"));
    }
}

impl MockState {
    fn alloc_buf(&mut self, size: usize) -> Result<u64, GpuError> {
        assert!(!self.grid_running, "alloc while grid ran");
        if let Some(left) = self.fail_alloc {
            if left == 0 {
                self.fail_alloc = None;
                self.events.push("alloc_err".into());
                return Err(GpuError::Cuda("oom".into()));
            }
            self.fail_alloc = Some(left - 1);
        }
        let ptr = self.next_ptr;
        self.next_ptr += (size.max(4096) as u64).next_multiple_of(4096) + 4096;
        self.mem.insert(ptr, vec![0u8; size.max(1)]);
        self.allocs += 1;
        self.events.push(format!("alloc:{ptr:#x}"));
        Ok(ptr)
    }

    fn span_mut(&mut self, ptr: u64, len: usize) -> &mut [u8] {
        for (&base, buf) in &mut self.mem {
            if ptr >= base && (ptr - base) as usize + len <= buf.len() {
                let off = (ptr - base) as usize;
                return &mut buf[off..off + len];
            }
        }
        panic!("access outside live buffer at {ptr:#x} len {len}");
    }

    fn free_buf(&mut self, ptr: u64) {
        assert!(!self.grid_running, "free while grid ran");
        assert!(self.mem.remove(&ptr).is_some(), "double free {ptr:#x}");
        self.frees += 1;
        self.events.push(format!("free:{ptr:#x}"));
    }

    fn finish_stream(&mut self) -> Result<Option<f64>, GpuError> {
        if self.fail_finish > 0 {
            self.fail_finish -= 1;
            self.events.push("finish_err".into());
            return Err(GpuError::Cuda("finish failed".into()));
        }
        self.events.push("finish_ok".into());
        Ok(None)
    }

    fn compile_module(&mut self) -> u64 {
        assert!(!self.grid_running, "compile while grid ran");
        self.compiles += 1;
        self.next_mod += 1;
        self.modules.insert(self.next_mod);
        self.next_mod
    }

    fn occupancy(&mut self, threads: u32, shared: u32) -> u32 {
        self.attr_calls.push((threads, shared));
        let total = 1024u32.saturating_add(shared);
        (98_304 / total.max(1)).clamp(1, 6)
    }

    fn check_replay(&mut self, ptrs: &[u64]) {
        for &ptr in ptrs {
            let live = self
                .mem
                .iter()
                .any(|(&b, m)| ptr >= b && ptr < b + m.len() as u64);
            assert!(live, "replayed stale pointer {ptr:#x}");
        }
        self.replays += 1;
    }
}

impl TinyDevice {
    fn state(&self) -> MutexGuard<'_, MockState> {
        self.0.lock().unwrap()
    }

    fn reset(&self) {
        let mut s = self.state();
        (s.allocs, s.frees, s.compiles, s.captures, s.replays) = (0, 0, 0, 0, 0);
        (s.recording, s.grid_running, s.fail_alloc, s.fail_finish) = (None, false, None, 0);
        s.attr_calls.clear();
        s.events.clear();
    }

    fn take_events(&self) -> Vec<String> {
        std::mem::take(&mut self.state().events)
    }

    fn stats_snapshot(&self) -> (usize, usize, usize, usize, usize) {
        let s = self.state();
        (s.allocs, s.frees, s.compiles, s.captures, s.replays)
    }
}

impl Launch for TinyDevice {
    type Kernel = TinyKernel;

    fn launch(
        &self,
        kernel: &TinyKernel,
        _: (u32, u32, u32),
        _: (u32, u32, u32),
        _: u32,
        args: &mut KernelArgs,
    ) -> Result<(), GpuError> {
        let mut s = self.state();
        assert!(s.modules.contains(&kernel.0.id), "launched unloaded module");
        if let Some(rec) = &mut s.recording {
            rec.extend(args.bound().filter_map(|b| match b {
                Bound::Pointer(p) => Some(p),
                Bound::Bytes(_) => None,
            }));
        }
        Ok(())
    }
}

impl Device for TinyDevice {
    type Kernel = TinyKernel;
    type Graph = TinyGraph;
    const BACKEND: Backend = Backend::Cuda;

    fn compile(&self, _: &str, _: &str) -> Result<TinyKernel, GpuError> {
        let id = self.state().compile_module();
        Ok(TinyKernel(Arc::new(TinyModule {
            state: self.0.clone(),
            id,
        })))
    }
    fn resident(&self, _: &TinyKernel, threads: u32, shared: u32) -> Result<u32, GpuError> {
        Ok(self.state().occupancy(threads, shared))
    }
    fn stats(&self, _: &TinyKernel) -> Option<KernelStats> {
        Some(KernelStats {
            regs: 32,
            shared_bytes: 1024,
            local_bytes: 0,
            max_threads: 512,
        })
    }
    fn replay(&self, graph: &TinyGraph) -> Result<(), GpuError> {
        self.state().check_replay(&graph.ptrs);
        Ok(())
    }
    fn capture<E: From<GpuError>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<TinyGraph, E> {
        {
            let mut s = self.state();
            assert!(!s.grid_running, "capture while grid ran");
            s.recording = Some(Vec::new());
        }
        let res = record();
        let mut s = self.state();
        let ptrs = s.recording.take().unwrap_or_default();
        res?;
        s.captures += 1;
        Ok(TinyGraph {
            state: self.0.clone(),
            ptrs,
        })
    }
    fn alloc(&self, size: usize) -> Result<u64, GpuError> {
        self.state().alloc_buf(size)
    }
    fn free(&self, ptr: u64) {
        self.state().free_buf(ptr);
    }
    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), GpuError> {
        self.state()
            .span_mut(ptr, bytes.len())
            .copy_from_slice(bytes);
        Ok(())
    }
    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), GpuError> {
        bytes.copy_from_slice(self.state().span_mut(ptr, bytes.len()));
        Ok(())
    }
    fn zero(&self, ptr: u64, len: usize) -> Result<(), GpuError> {
        let mut s = self.state();
        assert!(!s.grid_running, "zero while grid ran");
        s.span_mut(ptr, len).fill(0);
        Ok(())
    }
    fn finish(&self) -> Result<Option<f64>, GpuError> {
        self.state().finish_stream()
    }
    fn graphs(&self) -> bool {
        true
    }
    fn platform(&self) -> Platform {
        Platform::assumed(Limits::assumed())
    }
    fn name(&self) -> String {
        "tiny".into()
    }
    fn caches() -> &'static Caches<Self> {
        static CACHE: Caches<TinyDevice> = Caches::new();
        &CACHE
    }
    fn error(msg: String) -> GpuError {
        GpuError::Cuda(msg)
    }
    fn open() -> Result<Self, GpuError> {
        let state = MockState {
            next_ptr: 0x10_0000,
            ..Default::default()
        };
        Ok(Self(Arc::new(Mutex::new(state))))
    }
}

#[derive(Debug)]
struct FixedKernel {
    name: &'static str,
    entry: &'static str,
    source: &'static str,
    threads: usize,
    dyn_shared: usize,
    resident: Option<Resident>,
    dim: Option<DimName>,
    made: bool,
}

impl Kernel for FixedKernel {
    fn name(&self) -> &'static str {
        self.name
    }

    fn source(&self) -> Source {
        if self.made {
            return Source::Made(Body {
                entry: self.entry.into(),
                params: Vec::new(),
                stmts: Vec::new(),
                ret: None,
                helpers: Vec::new(),
            });
        }
        Source::Fixed {
            text: ByBackend::from_fn(|_| self.source),
            entry: self.entry,
        }
    }

    fn launches(&self, step: &Step) -> Vec<Call> {
        let mut args = vec![step.read(0), step.out()];
        if let Some(name) = self.dim {
            args.push(Arg::size(Extent::Dim {
                name,
                mul: 1,
                add: 0,
                most: 64,
            }));
        }
        let shape = Shape::Grid {
            across: 1,
            down: 1,
            threads: self.threads,
        };
        vec![Call { args, shape }]
    }

    fn needs(&self) -> Needs {
        Needs {
            dynamic_shared: self.dyn_shared,
            ..Needs::text()
        }
    }

    fn resident(&self) -> Option<Resident> {
        self.resident.clone()
    }
}

struct NoopGrid;

impl Grid for NoopGrid {
    fn open(&mut self, _: &KernelRef, _: Layout) -> Result<(), RunError> {
        Ok(())
    }

    fn launch(
        &mut self,
        _: &KernelRef,
        _: &[Item],
        _: &[KeptDesc],
        _: &[u32],
    ) -> Result<(), RunError> {
        Ok(())
    }
}

struct NullTransport;

impl Transport for NullTransport {
    fn rank(&self) -> u32 {
        0
    }

    fn ranks(&self) -> u32 {
        2
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, CommError> {
        Ok(vec![mine.to_vec(), mine.to_vec()])
    }
}

type PauseLog = Arc<Mutex<(usize, usize)>>;
type PauseFaults = Arc<Mutex<VecDeque<bool>>>;

struct TinySum {
    ctx: &'static TinyDevice,
    running: bool,
    log: PauseLog,
    fail_pause: PauseFaults,
}

impl TinySum {
    fn with_faults(ctx: &'static TinyDevice) -> (Self, PauseLog, PauseFaults) {
        let (log, fail_pause) = (
            Arc::new(Mutex::new((0, 0))),
            Arc::new(Mutex::new(VecDeque::new())),
        );
        let sum = Self {
            ctx,
            running: false,
            log: log.clone(),
            fail_pause: fail_pause.clone(),
        };
        (sum, log, fail_pause)
    }

    fn set_running(&mut self, on: bool) {
        if self.running != on {
            self.running = on;
            self.ctx.state().grid_running = on;
            let mut log = self.log.lock().unwrap();
            if on {
                log.0 += 1;
            } else {
                log.1 += 1;
            }
        }
    }
}

impl Drop for TinySum {
    fn drop(&mut self) {
        self.set_running(false);
    }
}

impl Sum for TinySum {
    fn sum(&mut self, _: u64, _: u64, len: usize, _: DType) -> Result<(), RunError> {
        if len == 0 {
            self.set_running(true);
        }
        Ok(())
    }

    fn capturable(&self) -> bool {
        true
    }

    fn pause(&mut self) -> Result<(), RunError> {
        if self.fail_pause.lock().unwrap().pop_front() == Some(true) {
            self.ctx.state().events.push("pause_err".into());
            return Err(RunError::Grid("pause failed".into()));
        }
        self.set_running(false);
        self.ctx.state().events.push("pause_ok".into());
        Ok(())
    }
}

static LOCK: Mutex<()> = Mutex::new(());

fn lock_ctx() -> (MutexGuard<'static, ()>, &'static TinyDevice) {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = cached_context::<TinyDevice>().unwrap();
    ctx.reset();
    (guard, ctx)
}

fn step_of(kernel: KernelRef, rule: &'static str, read: Slot, write: Slot, lane: u8) -> Step {
    Step {
        kernel,
        rule,
        site: None,
        reads: vec![read],
        write,
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane,
        table: None,
        index_table: None,
    }
}

fn make_step(kernel: FixedKernel, buf: u32, len: usize) -> Step {
    let slot = Slot::floats(buf, len);
    step_of(KernelRef::new(kernel), "test", slot, slot, 0)
}

fn make_plan(buffers: Vec<usize>, steps: Vec<Step>) -> Plan {
    let zones = vec![0; buffers.len()];
    Plan {
        id: Plan::fresh(),
        buffers,
        zones,
        steps,
        ..Plan::default()
    }
}

fn plain_kernel(name: &'static str, entry: &'static str, dim: Option<DimName>) -> FixedKernel {
    FixedKernel {
        name,
        entry,
        source: "__global__ void plain_k() {}",
        threads: 128,
        dyn_shared: 0,
        resident: None,
        dim,
        made: false,
    }
}

fn made_kernel(name: &'static str, entry: &'static str) -> FixedKernel {
    FixedKernel {
        made: true,
        ..plain_kernel(name, entry, None)
    }
}

fn made_plan(bytes: usize, name: &'static str, entry: &'static str) -> Plan {
    make_plan(
        vec![bytes],
        vec![make_step(made_kernel(name, entry), 0, 16)],
    )
}

fn one_plan(bytes: usize, name: &'static str, entry: &'static str) -> Plan {
    make_plan(
        vec![bytes],
        vec![make_step(plain_kernel(name, entry, None), 0, 16)],
    )
}

fn open_runner(plan: Plan, graphs: bool) -> Box<dyn Run> {
    let settings = Settings {
        memory: Memory::Off,
        ..host()
    };
    let mut runner = device::<TinyDevice>(plan, &settings).unwrap();
    runner.launching(Launching { graphs, pdl: false });
    runner
}

#[test]
fn attrs_use_shared() {
    let (_g, ctx) = lock_ctx();
    let dyn_k = FixedKernel {
        threads: 256,
        dyn_shared: 49_152,
        ..plain_kernel("dyn_smem", "dyn_smem_k", None)
    };
    let res_k = FixedKernel {
        threads: 64,
        resident: Some(Resident {
            threads: 128,
            shared: 16_384,
            scratch: 0,
            lanes: Vec::new(),
        }),
        ..plain_kernel("res_smem", "res_smem_k", None)
    };
    let steps = vec![make_step(dyn_k, 0, 16), make_step(res_k, 0, 16)];
    let mut runner = open_runner(make_plan(vec![64], steps), false);
    assert_eq!(ctx.state().attr_calls, [(256, 49_152), (128, 16_384)]);
    runner.persist(Box::new(NoopGrid)).unwrap();
    runner.profiling(true);
    runner.submit(0..2).unwrap();
    let prof = runner.take_profile().unwrap();
    for (k, shared_bytes, ctas_per_sm) in [
        ("dyn_smem [dyn_smem_k | test]", 1024 + 49_152, 1),
        ("res_smem [res_smem_k | test]", 1024 + 16_384, 5),
    ] {
        let want = KernelAttrs {
            regs: 32,
            local_bytes: 0,
            shared_bytes,
            ctas_per_sm,
        };
        assert_eq!(prof.spent[k].attrs, Some(want));
    }
}

#[test]
fn commit_keeps_graph() {
    let (_g, ctx) = lock_ctx();
    let fwd = make_step(plain_kernel("fwd", "fwd_k", Some(Context)), 0, 16);
    let cmt = make_step(plain_kernel("cmt", "cmt_k", Some(COMMIT)), 0, 16);
    let mut runner = open_runner(make_plan(vec![64], vec![fwd, cmt]), true);
    runner.bind(Context, 16);
    for c in 1..=8 {
        runner.bind(COMMIT, c);
        runner.submit(0..1).unwrap();
    }
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (1, 8));
    for c in [32, 16] {
        runner.bind(Context, c);
        runner.submit(0..1).unwrap();
    }
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (2, 10));
    for c in [1, 2, 1] {
        runner.bind(COMMIT, c);
        runner.submit(1..2).unwrap();
    }
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (4, 13));
}

#[test]
fn growth_keeps_unrelated() {
    let (_g, ctx) = lock_ctx();
    let s0 = || make_step(plain_kernel("k0", "k0", None), 0, 16);
    let s1 = || make_step(plain_kernel("k1", "k1", None), 1, 16);
    let plan_a = make_plan(vec![256], vec![s0()]);
    let plan_b = make_plan(vec![256, 256], vec![s0(), s1()]);
    let plan_b_big = make_plan(vec![256, 65_536], vec![s0(), s1()]);
    let mut runner = open_runner(plan_a.clone(), true);
    runner.submit(0..1).unwrap();
    runner.replan(&plan_b).unwrap();
    runner.submit(0..2).unwrap();
    runner.replan(&plan_b_big).unwrap();
    runner.submit(0..2).unwrap();
    assert_eq!((ctx.stats_snapshot().1, ctx.stats_snapshot().3), (1, 3));
    // Returning to plan A reuses its graph without recapturing.
    runner.replan(&plan_a).unwrap();
    runner.submit(0..1).unwrap();
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (3, 4));
    // Plan B lost its graph when scratch slot 1 retired.
    runner.replan(&plan_b).unwrap();
    runner.submit(0..2).unwrap();
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (4, 5));
}

fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let dir = tessera_testkit::scratch(&format!("{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn unchanged_overrides_stay() {
    let (_g, ctx) = lock_ctx();
    let dir = scratch_dir("pause-overrides");
    let (mut rank, log, _) = open_rank(ctx, ranked_plan(vec![64, 64], "ov_k"), Some(dir.clone()));
    for _ in 0..6 {
        rank.submit(0..2).unwrap();
    }
    let cr = || {
        let s = ctx.stats_snapshot();
        (s.3, s.4)
    };
    assert_eq!((*log.lock().unwrap(), cr()), ((1, 0), (1, 6)));
    let file = dir.join("ov_k.cu");
    std::fs::write(&file, "__global__ void ov_k() { /* edited */ }").unwrap();
    rank.submit(0..2).unwrap();
    rank.submit(0..2).unwrap();
    assert_eq!((*log.lock().unwrap(), cr()), ((2, 1), (2, 8)));
    std::fs::remove_file(&file).unwrap();
    rank.submit(0..2).unwrap();
    assert_eq!((*log.lock().unwrap(), cr()), ((3, 2), (3, 9)));
    drop(rank);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn marked_kernel_cached() {
    let (_g, ctx) = lock_ctx();
    let marked = || FixedKernel {
        source: "// TS_MARK()\n__global__ void mk_entry() {}",
        threads: 64,
        ..plain_kernel("mk", "mk_entry", None)
    };
    let _r1 = open_runner(make_plan(vec![64], vec![make_step(marked(), 0, 16)]), true);
    let _r2 = open_runner(make_plan(vec![64], vec![make_step(marked(), 0, 16)]), true);
    assert_eq!(ctx.stats_snapshot().2, 1);
    let (src, p1, p2) = (
        "__global__ void pref_k() {}",
        "#define A 1\n",
        "#define A 2\n",
    );
    cached_prefixed::<TinyDevice>(p1, src, "pref_k").unwrap();
    cached_prefixed::<TinyDevice>(p1, src, "pref_k").unwrap();
    assert_eq!(ctx.stats_snapshot().2, 2);
    cached_prefixed::<TinyDevice>(p2, src, "pref_k").unwrap();
    assert_eq!(ctx.stats_snapshot().2, 3);
    cached_source::<TinyDevice>(src, "pref_k").unwrap();
    cached_prefixed::<TinyDevice>("", src, "pref_k").unwrap();
    assert_eq!(ctx.stats_snapshot().2, 4);
}

#[test]
fn cold_plan_timers() {
    let (_g, _ctx) = lock_ctx();
    let mut runner = open_runner(one_plan(64, "a", "a_k"), true);
    runner.profiling(true);
    runner.replan(&one_plan(65_536, "b", "b_k")).unwrap();
    runner.submit(0..1).unwrap();
    let cold = runner.take_profile().unwrap();
    runner.submit(0..1).unwrap();
    let warm = runner.take_profile().unwrap();
    for key in ["plan: alloc", "plan: compile", "plan: capture"] {
        assert_eq!(cold.spent[key].count, 1);
        assert!(!warm.spent.contains_key(key));
    }
}

#[test]
fn sizer_scope_isolated() {
    assert!(!sizer::counting());
    let guard = sizer::sizing();
    assert!(sizer::counting() && !std::thread::spawn(sizer::counting).join().unwrap());
    let scope = sizer::current_scope().expect("active scope");
    let plan = one_plan(1024, "s", "s_k");
    std::thread::spawn(move || {
        scope.run(|| assert!(sizer::counting() && tessera_run::open(plan, &host()).is_ok()))
    })
    .join()
    .unwrap();
    assert!(guard.sized().scratch >= 1024);
    drop(guard);
    assert!(!sizer::counting());
}

fn open_rank(
    ctx: &'static TinyDevice,
    plan: Plan,
    kernels: Option<std::path::PathBuf>,
) -> (Rank<NullTransport>, PauseLog, PauseFaults) {
    let settings = Settings {
        memory: Memory::Off,
        kernels,
        ..host()
    };
    let (sum, log, fails) = TinySum::with_faults(ctx);
    let mut rank = Rank::open_device::<TinyDevice>(plan, &settings, NullTransport)
        .unwrap()
        .with_grid(Some(Box::new(sum)))
        .unwrap();
    rank.launching(Launching {
        graphs: true,
        pdl: false,
    });
    (rank, log, fails)
}

fn ranked_kernel(buffers: Vec<usize>, k: FixedKernel) -> Plan {
    let s0 = make_step(k, 0, 16);
    let ar = KernelRef::new(AllReduce {
        len: 16,
        wire: DType::F32,
    });
    let s1 = step_of(
        ar,
        "all_reduce",
        Slot::floats(0, 16),
        Slot::floats(1, 16),
        1,
    );
    let mut plan = make_plan(buffers, vec![s0, s1]);
    (plan.rank, plan.ranks) = (0, 2);
    plan
}

fn ranked_plan(buffers: Vec<usize>, entry: &'static str) -> Plan {
    ranked_kernel(buffers, plain_kernel("ov", entry, None))
}

fn ranked_made(buffers: Vec<usize>, entry: &'static str) -> Plan {
    ranked_kernel(buffers, made_kernel("rm", entry))
}

fn pos(events: &[String], needle: &str) -> usize {
    events
        .iter()
        .position(|e| e == needle || e.starts_with(needle))
        .unwrap_or_else(|| panic!("missing `{needle}` in {events:?}"))
}

fn assert_retained(ev: &[String]) {
    assert!(!ev.iter().any(|e| {
        e.starts_with("drop_graph:") || e.starts_with("free:") || e.starts_with("drop_module:")
    }));
}

fn assert_retired_order(ev: &[String], ptr: u64) {
    let free = format!("free:{ptr:#x}");
    assert!(pos(ev, "finish_ok") < pos(ev, "drop_graph:"));
    assert!(pos(ev, "drop_graph:") < pos(ev, &free));
}

fn assert_dirty(r: &mut dyn Run, slot: Slot, data: &tessera_mem::Tensor) {
    assert_eq!(r.address(slot), None);
    assert!(matches!(r.read(slot), Err(RunError::Inconsistent)));
    assert!(matches!(
        r.write(slot, data.view()),
        Err(RunError::Inconsistent)
    ));
}

#[test]
fn retire_partial_bind() {
    let (_g, ctx) = lock_ctx();
    let [p1, p2] = [64, 65_536].map(|n| {
        make_plan(
            vec![n, n],
            vec![
                make_step(plain_kernel("a", "a_k", None), 0, 16),
                make_step(plain_kernel("b", "b_k", None), 1, 16),
            ],
        )
    });
    let mut runner = open_runner(p1.clone(), true);
    runner.submit(0..2).unwrap();
    let old0 = runner.address(Slot::floats(0, 16)).unwrap();
    ctx.take_events();
    // Grow slot 0, then fail slot 1 allocation.
    ctx.state().fail_alloc = Some(1);
    assert!(runner.replan(&p2).is_err());
    let ev = ctx.take_events();
    assert!(pos(&ev, "alloc_err") < pos(&ev, "finish_ok"));
    assert_retired_order(&ev, old0);
    assert!(matches!(runner.submit(0..2), Err(RunError::Inconsistent)));
    let data = tessera_mem::Tensor::f32(vec![16], vec![1.5; 16]);
    let slots = [Slot::floats(0, 16), Slot::floats(1, 16)];
    slots
        .into_iter()
        .for_each(|s| assert_dirty(&mut *runner, s, &data));
    runner.replan(&p1).unwrap();
    for slot in slots {
        assert!(runner.address(slot).is_some());
        runner.write(slot, data.view()).unwrap();
        assert_eq!(runner.read(slot).unwrap().floats().unwrap(), &[1.5; 16]);
    }
    runner.submit(0..2).unwrap();
}

#[test]
fn retire_pause_error() {
    let (_g, ctx) = lock_ctx();
    let (p1, p2) = (
        ranked_plan(vec![64, 64], "rp_k"),
        ranked_plan(vec![65_536, 64], "rp_k"),
    );
    let (mut rank, _, fails) = open_rank(ctx, p1, None);
    rank.submit(0..2).unwrap();
    let old0 = rank.address(Slot::floats(0, 16)).unwrap();
    ctx.take_events();
    // First pause succeeds, then retire pause fails.
    fails.lock().unwrap().extend([false, true]);
    assert!(rank.replan(&p2).is_err());
    let ev = ctx.take_events();
    assert!(ev.iter().any(|e| e == "pause_err"));
    assert_retained(&ev);
    assert!(ctx.state().mem.contains_key(&old0));
    assert!(matches!(rank.submit(0..2), Err(RunError::Inconsistent)));
    let data = tessera_mem::Tensor::f32(vec![16], vec![2.5; 16]);
    let s0 = Slot::floats(0, 16);
    assert_dirty(&mut rank, s0, &data);
    // Retry replan once pause recovers.
    rank.replan(&p2).unwrap();
    let ev = ctx.take_events();
    assert!(pos(&ev, "pause_ok") < pos(&ev, "finish_ok"));
    assert_retired_order(&ev, old0);
    assert!(!ctx.state().mem.contains_key(&old0));
    rank.write(s0, data.view()).unwrap();
    assert_eq!(rank.read(s0).unwrap().floats().unwrap(), &[2.5; 16]);
    rank.submit(0..2).unwrap();
}

#[test]
fn retire_finish_error() {
    let (_g, ctx) = lock_ctx();
    let mut runner = open_runner(one_plan(64, "f", "f_k"), true);
    let p2 = one_plan(65_536, "f", "f_k");
    runner.submit(0..1).unwrap();
    let old0 = runner.address(Slot::floats(0, 16)).unwrap();
    ctx.take_events();
    ctx.state().fail_finish = 1;
    assert!(runner.replan(&p2).is_err());
    let ev = ctx.take_events();
    assert!(ev.iter().any(|e| e == "finish_err"));
    assert_retained(&ev);
    assert!(ctx.state().mem.contains_key(&old0));
    assert!(matches!(runner.submit(0..1), Err(RunError::Inconsistent)));
    // Retry once stream synchronization succeeds.
    runner.replan(&p2).unwrap();
    let ev = ctx.take_events();
    assert_retired_order(&ev, old0);
    assert!(!ctx.state().mem.contains_key(&old0));
    runner.submit(0..1).unwrap();
}

#[test]
fn override_pause_retry() {
    let (_g, ctx) = lock_ctx();
    let dir = scratch_dir("pause-retry");
    let (mut rank, log, fails) =
        open_rank(ctx, ranked_plan(vec![64, 64], "pr_k"), Some(dir.clone()));
    rank.submit(0..2).unwrap();
    let (_, _, c0, cap0, _) = ctx.stats_snapshot();
    let cc = || {
        let s = ctx.stats_snapshot();
        (s.2, s.3)
    };
    // Edit file, fail pause once, retry without touching disk.
    let file = dir.join("pr_k.cu");
    std::fs::write(&file, "__global__ void pr_k() { /* edit */ }").unwrap();
    fails.lock().unwrap().push_back(true);
    assert!(rank.submit(0..2).is_err());
    assert_eq!(cc(), (c0, cap0));
    rank.submit(0..2).unwrap();
    assert_eq!((cc(), *log.lock().unwrap()), ((c0 + 1, cap0 + 1), (2, 1)));
    // Remove file, fail pause once, retry without touching disk.
    std::fs::remove_file(&file).unwrap();
    fails.lock().unwrap().push_back(true);
    assert!(rank.submit(0..2).is_err());
    assert_eq!(cc().1, cap0 + 1);
    rank.submit(0..2).unwrap();
    assert_eq!((cc().1, *log.lock().unwrap()), (cap0 + 2, (3, 2)));
    drop(rank);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sizer_guard_order() {
    trait NotSend {
        const IS_SEND: bool = false;
    }
    struct Check<T: ?Sized>(PhantomData<T>);
    impl<T: ?Sized> NotSend for Check<T> {}
    impl<T: Send + ?Sized> Check<T> {
        const IS_SEND: bool = true;
    }
    const { assert!(!Check::<sizer::SizerGuard>::IS_SEND && Check::<sizer::Scope>::IS_SEND) };
    // Out-of-order drop: dropping outer never clobbers or resurrects.
    let (outer, inner) = (sizer::sizing(), sizer::sizing());
    assert!(tessera_run::open(one_plan(2048, "s", "s_k"), &host()).is_ok());
    assert!(inner.sized().scratch >= 2048 && outer.sized().scratch == 0);
    drop(outer);
    assert!(sizer::counting() && tessera_run::open(one_plan(4096, "s", "s_k"), &host()).is_ok());
    assert!(inner.sized().scratch >= 4096);
    drop(inner);
    assert!(!sizer::counting());
    // Normal LIFO drop restores outer until it drops.
    let (outer, inner) = (sizer::sizing(), sizer::sizing());
    drop(inner);
    assert!(sizer::counting() && tessera_run::open(one_plan(1024, "s", "s_k"), &host()).is_ok());
    assert!(outer.sized().scratch >= 1024);
    drop(outer);
    assert!(!sizer::counting());
}

#[test]
fn drop_retains_unquiesced() {
    let (_g, ctx) = lock_ctx();
    let live_mod = |id| ctx.state().modules.contains(&id);
    // Normal teardown releases non-process-cached modules on runner and rank.
    let mut ok = open_runner(made_plan(64, "ok_m", "ok_k"), true);
    let ok_mod = ctx.state().next_mod;
    ok.submit(0..1).unwrap();
    assert!(live_mod(ok_mod));
    drop(ok);
    assert!(!live_mod(ok_mod));
    let (mut ok_r, _, _) = open_rank(ctx, ranked_made(vec![64, 64], "rok_k"), None);
    let ok_rm = ctx.state().next_mod;
    ok_r.submit(0..2).unwrap();
    assert!(live_mod(ok_rm));
    drop(ok_r);
    assert!(!live_mod(ok_rm));
    // Failed finish retains active, cached-plan, and buffer owners.
    let mut runner = open_runner(made_plan(64, "d0", "d0_k"), true);
    let m0 = ctx.state().next_mod;
    runner.submit(0..1).unwrap();
    runner.replan(&made_plan(64, "d1", "d1_k")).unwrap();
    let m1 = ctx.state().next_mod;
    runner.submit(0..1).unwrap();
    let old0 = runner.address(Slot::floats(0, 16)).unwrap();
    ctx.take_events();
    ctx.state().fail_finish = 2;
    assert!(runner.replan(&made_plan(65_536, "d1", "d1_k")).is_err());
    drop(runner);
    let ev = ctx.take_events();
    assert_eq!(ev.iter().filter(|e| *e == "finish_err").count(), 2);
    assert_retained(&ev);
    assert!(ctx.state().mem.contains_key(&old0) && live_mod(m0) && live_mod(m1));
    // Failed rank pause retains planned and override modules.
    let dir = scratch_dir("drop-retain");
    let (mut rank, _, fails) = open_rank(ctx, ranked_made(vec![64, 64], "dp_k"), Some(dir.clone()));
    let rm0 = ctx.state().next_mod;
    std::fs::write(dir.join("dp_k.cu"), "__global__ void dp_k() { /* ov */ }").unwrap();
    rank.submit(0..2).unwrap();
    let ov_mod = ctx.state().next_mod;
    assert_ne!(rm0, ov_mod);
    let r_old = rank.address(Slot::floats(0, 16)).unwrap();
    ctx.take_events();
    fails.lock().unwrap().extend([false, true, true]);
    assert!(rank.replan(&ranked_made(vec![65_536, 64], "dp_k")).is_err());
    drop(rank);
    let ev = ctx.take_events();
    assert_eq!(ev.iter().filter(|e| *e == "pause_err").count(), 2);
    assert_retained(&ev);
    assert!(ctx.state().mem.contains_key(&r_old) && live_mod(rm0) && live_mod(ov_mod));
    let _ = std::fs::remove_dir_all(&dir);
}
