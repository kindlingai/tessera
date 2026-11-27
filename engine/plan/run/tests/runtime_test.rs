//! Occupancy attrs, scoped GraphKeys, targeted invalidation, overrides, caching and sizer.

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard};
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
use tessera_run::{Error as RunError, Grid, Item, KeptDesc, Layout, Marks, Rank, Run, Sum, sizer};

#[derive(Default)]
struct MockState {
    next_ptr: u64,
    live: HashMap<u64, usize>,
    allocs: usize,
    frees: usize,
    compiles: usize,
    captures: usize,
    replays: usize,
    recording: Option<Vec<u64>>,
    attr_calls: Vec<(u32, u32)>,
    grid_running: bool,
    fail_alloc: Option<usize>,
    fail_finish: usize,
    events: Vec<String>,
}

#[derive(Clone)]
struct TinyDevice(Arc<Mutex<MockState>>);

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
    fn open_state() -> Self {
        Self {
            next_ptr: 0x10_0000,
            ..Self::default()
        }
    }

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
        self.live.insert(ptr, size.max(1));
        self.allocs += 1;
        self.events.push(format!("alloc:{ptr:#x}"));
        Ok(ptr)
    }

    fn free_buf(&mut self, ptr: u64) {
        assert!(!self.grid_running, "free while grid ran");
        assert!(self.live.remove(&ptr).is_some(), "double free {ptr:#x}");
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

    fn count_compile(&mut self) {
        assert!(!self.grid_running, "compile while grid ran");
        self.compiles += 1;
    }

    fn occupancy(&mut self, threads: u32, shared: u32) -> u32 {
        self.attr_calls.push((threads, shared));
        let total = 1024u32.saturating_add(shared);
        (98_304 / total.max(1)).clamp(1, 6)
    }

    fn check_replay(&mut self, ptrs: &[u64]) {
        for &ptr in ptrs {
            let live = self
                .live
                .iter()
                .any(|(&base, &len)| ptr >= base && ptr < base + len as u64);
            assert!(live, "replayed stale pointer {ptr:#x}");
        }
        self.replays += 1;
    }
}

impl TinyDevice {
    fn reset(&self) {
        let mut s = self.0.lock().unwrap();
        (s.allocs, s.frees, s.compiles, s.captures, s.replays) = (0, 0, 0, 0, 0);
        s.recording = None;
        s.attr_calls.clear();
        s.grid_running = false;
        s.fail_alloc = None;
        s.fail_finish = 0;
        s.events.clear();
    }

    fn take_events(&self) -> Vec<String> {
        std::mem::take(&mut self.0.lock().unwrap().events)
    }

    fn fail_alloc(&self, after: Option<usize>) {
        self.0.lock().unwrap().fail_alloc = after;
    }

    fn fail_finish(&self, count: usize) {
        self.0.lock().unwrap().fail_finish = count;
    }

    fn is_live(&self, ptr: u64) -> bool {
        self.0.lock().unwrap().live.contains_key(&ptr)
    }

    fn stats_snapshot(&self) -> (usize, usize, usize, usize, usize) {
        let s = self.0.lock().unwrap();
        (s.allocs, s.frees, s.compiles, s.captures, s.replays)
    }

    fn attr_calls(&self) -> Vec<(u32, u32)> {
        self.0.lock().unwrap().attr_calls.clone()
    }
}

impl Launch for TinyDevice {
    type Kernel = ();

    fn launch(
        &self,
        _: &(),
        _: (u32, u32, u32),
        _: (u32, u32, u32),
        _: u32,
        args: &mut KernelArgs,
    ) -> Result<(), GpuError> {
        let mut s = self.0.lock().unwrap();
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
    type Kernel = ();
    type Graph = TinyGraph;
    const BACKEND: Backend = Backend::Cuda;

    fn open() -> Result<Self, GpuError> {
        Ok(Self(Arc::new(Mutex::new(MockState::open_state()))))
    }
    fn error(msg: String) -> GpuError {
        GpuError::Cuda(msg)
    }
    fn caches() -> &'static Caches<Self> {
        static CACHE: Caches<TinyDevice> = Caches::new();
        &CACHE
    }
    fn name(&self) -> String {
        "tiny".into()
    }
    fn platform(&self) -> Platform {
        Platform::assumed(Limits::assumed())
    }
    fn graphs(&self) -> bool {
        true
    }
    fn finish(&self) -> Result<Option<f64>, GpuError> {
        self.0.lock().unwrap().finish_stream()
    }
    fn upload(&self, _: u64, _: &[u8]) -> Result<(), GpuError> {
        Ok(())
    }
    fn download(&self, _: u64, bytes: &mut [u8]) -> Result<(), GpuError> {
        bytes.fill(0);
        Ok(())
    }
    fn zero(&self, _: u64, _: usize) -> Result<(), GpuError> {
        assert!(!self.0.lock().unwrap().grid_running, "zero while grid ran");
        Ok(())
    }
    fn alloc(&self, size: usize) -> Result<u64, GpuError> {
        self.0.lock().unwrap().alloc_buf(size)
    }
    fn free(&self, ptr: u64) {
        self.0.lock().unwrap().free_buf(ptr);
    }
    fn compile(&self, _: &str, _: &str) -> Result<(), GpuError> {
        self.0.lock().unwrap().count_compile();
        Ok(())
    }
    fn resident(&self, _: &(), threads: u32, shared: u32) -> Result<u32, GpuError> {
        Ok(self.0.lock().unwrap().occupancy(threads, shared))
    }
    fn stats(&self, _: &()) -> Option<KernelStats> {
        Some(KernelStats {
            regs: 32,
            shared_bytes: 1024,
            local_bytes: 0,
            max_threads: 512,
        })
    }
    fn replay(&self, graph: &TinyGraph) -> Result<(), GpuError> {
        self.0.lock().unwrap().check_replay(&graph.ptrs);
        Ok(())
    }
    fn capture<E: From<GpuError>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<TinyGraph, E> {
        {
            let mut s = self.0.lock().unwrap();
            assert!(!s.grid_running, "capture while grid ran");
            s.recording = Some(Vec::new());
        }
        let res = record();
        let mut s = self.0.lock().unwrap();
        let ptrs = s.recording.take().unwrap_or_default();
        res?;
        s.captures += 1;
        Ok(TinyGraph {
            state: self.0.clone(),
            ptrs,
        })
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
}

impl Kernel for FixedKernel {
    fn name(&self) -> &'static str {
        self.name
    }

    fn source(&self) -> Source {
        Source::Fixed {
            text: ByBackend::from_fn(|_| self.source),
            entry: self.entry,
        }
    }

    fn launches(&self, step: &Step) -> Vec<Call> {
        let mut args = vec![step.read(0), step.out()];
        if let Some(name) = self.dim {
            let ext = Extent::Dim {
                name,
                mul: 1,
                add: 0,
                most: 64,
            };
            args.push(Arg::size(ext));
        }
        vec![Call {
            args,
            shape: Shape::Grid {
                across: 1,
                down: 1,
                threads: self.threads,
            },
        }]
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

    fn profiling(&mut self, _: bool) {}

    fn replace(&mut self, _: &str, _: Option<String>) {}

    fn take_marks(&mut self) -> Vec<(String, Marks)> {
        Vec::new()
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
    fn new(ctx: &'static TinyDevice) -> (Self, PauseLog) {
        let (sum, log, _) = Self::with_faults(ctx);
        (sum, log)
    }

    fn with_faults(ctx: &'static TinyDevice) -> (Self, PauseLog, PauseFaults) {
        let log = Arc::new(Mutex::new((0, 0)));
        let fail_pause = Arc::new(Mutex::new(VecDeque::new()));
        (
            Self {
                ctx,
                running: false,
                log: log.clone(),
                fail_pause: fail_pause.clone(),
            },
            log,
            fail_pause,
        )
    }

    fn start(&mut self) {
        if !self.running {
            self.running = true;
            self.ctx.0.lock().unwrap().grid_running = true;
            self.log.lock().unwrap().0 += 1;
        }
    }

    fn stop(&mut self) {
        if std::mem::take(&mut self.running) {
            self.ctx.0.lock().unwrap().grid_running = false;
            self.log.lock().unwrap().1 += 1;
        }
    }
}

impl Drop for TinySum {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Sum for TinySum {
    fn sum(&mut self, _: u64, _: u64, len: usize, _: DType) -> Result<(), RunError> {
        if len == 0 {
            self.start();
        }
        Ok(())
    }

    fn capturable(&self) -> bool {
        true
    }

    fn pause(&mut self) -> Result<(), RunError> {
        if self.fail_pause.lock().unwrap().pop_front() == Some(true) {
            self.ctx.0.lock().unwrap().events.push("pause_err".into());
            return Err(RunError::Grid("pause failed".into()));
        }
        self.stop();
        self.ctx.0.lock().unwrap().events.push("pause_ok".into());
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

fn make_step(kernel: FixedKernel, buf: u32, len: usize) -> Step {
    let slot = Slot::floats(buf, len);
    Step {
        kernel: KernelRef::new(kernel),
        rule: "test",
        reads: vec![slot],
        write: slot,
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane: 0,
        table: None,
        index_table: None,
    }
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
    }
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
        name: "dyn_smem",
        entry: "dyn_smem_k",
        source: "__global__ void dyn_smem_k() {}",
        threads: 256,
        dyn_shared: 49_152,
        resident: None,
        dim: None,
    };
    let res_k = FixedKernel {
        name: "res_smem",
        entry: "res_smem_k",
        source: "__global__ void res_smem_k() {}",
        threads: 64,
        dyn_shared: 0,
        resident: Some(Resident {
            threads: 128,
            shared: 16_384,
            scratch: 0,
            lanes: Vec::new(),
        }),
        dim: None,
    };
    let plan = make_plan(
        vec![64],
        vec![make_step(dyn_k, 0, 16), make_step(res_k, 0, 16)],
    );
    let mut runner = open_runner(plan, false);
    assert_eq!(ctx.attr_calls(), [(256, 49_152), (128, 16_384)]);
    runner.persist(Box::new(NoopGrid)).unwrap();
    runner.profiling(true);
    runner.submit(0..2).unwrap();
    let prof = runner.take_profile().unwrap();
    assert_eq!(
        prof.spent["dyn_smem"].attrs,
        Some(KernelAttrs {
            regs: 32,
            local_bytes: 0,
            shared_bytes: 1024 + 49_152,
            ctas_per_sm: 1,
        })
    );
    assert_eq!(
        prof.spent["res_smem"].attrs,
        Some(KernelAttrs {
            regs: 32,
            local_bytes: 0,
            shared_bytes: 1024 + 16_384,
            ctas_per_sm: 5,
        })
    );
}

#[test]
fn commit_keeps_graph() {
    let (_g, ctx) = lock_ctx();
    let fwd = make_step(plain_kernel("fwd", "fwd_k", Some(Context)), 0, 16);
    let cmt = make_step(plain_kernel("cmt", "cmt_k", Some(COMMIT)), 0, 16);
    let plan = make_plan(vec![64], vec![fwd, cmt]);
    let mut runner = open_runner(plan, true);
    runner.bind(Context, 16);
    for c in 1..=8 {
        runner.bind(COMMIT, c);
        runner.submit(0..1).unwrap();
    }
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (1, 8));
    runner.bind(Context, 32);
    runner.submit(0..1).unwrap();
    runner.bind(Context, 16);
    runner.submit(0..1).unwrap();
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (2, 10));
    runner.bind(COMMIT, 1);
    runner.submit(1..2).unwrap();
    runner.bind(COMMIT, 2);
    runner.submit(1..2).unwrap();
    runner.bind(COMMIT, 1);
    runner.submit(1..2).unwrap();
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

#[test]
fn unchanged_overrides_stay() {
    let (_g, ctx) = lock_ctx();
    let dir = tessera_testkit::scratch(&format!("pause-overrides-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let s0 = make_step(plain_kernel("ov", "ov_k", None), 0, 4);
    let ar = Step {
        kernel: KernelRef::new(AllReduce {
            len: 4,
            wire: DType::F32,
        }),
        rule: "all_reduce",
        reads: vec![Slot::floats(0, 4)],
        write: Slot::floats(1, 4),
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane: 1,
        table: None,
        index_table: None,
    };
    let mut plan = make_plan(vec![64, 64], vec![s0, ar]);
    (plan.rank, plan.ranks) = (0, 2);
    let settings = Settings {
        memory: Memory::Off,
        kernels: Some(dir.clone()),
        ..host()
    };
    let (sum, log) = TinySum::new(ctx);
    let mut rank = Rank::open_device::<TinyDevice>(plan, &settings, NullTransport)
        .unwrap()
        .with_grid(Some(Box::new(sum)))
        .unwrap();
    rank.launching(Launching {
        graphs: true,
        pdl: false,
    });
    for _ in 0..6 {
        rank.submit(0..2).unwrap();
    }
    assert_eq!(*log.lock().unwrap(), (1, 0));
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (1, 6));
    let file = dir.join("ov_k.cu");
    std::fs::write(&file, "__global__ void ov_k() { /* edited */ }").unwrap();
    rank.submit(0..2).unwrap();
    rank.submit(0..2).unwrap();
    assert_eq!(*log.lock().unwrap(), (2, 1));
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (2, 8));
    std::fs::remove_file(&file).unwrap();
    rank.submit(0..2).unwrap();
    assert_eq!(*log.lock().unwrap(), (3, 2));
    assert_eq!((ctx.stats_snapshot().3, ctx.stats_snapshot().4), (3, 9));
    drop(rank);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn marked_kernel_cached() {
    let (_g, ctx) = lock_ctx();
    let marked = || FixedKernel {
        name: "mk",
        entry: "mk_entry",
        source: "// TS_MARK()\n__global__ void mk_entry() {}",
        threads: 64,
        dyn_shared: 0,
        resident: None,
        dim: None,
    };
    let _r1 = open_runner(make_plan(vec![64], vec![make_step(marked(), 0, 16)]), true);
    let _r2 = open_runner(make_plan(vec![64], vec![make_step(marked(), 0, 16)]), true);
    assert_eq!(ctx.stats_snapshot().2, 1);
    const SRC: &str = "__global__ void pref_k() {}";
    const P1: &str = "#define A 1\n";
    const P2: &str = "#define A 2\n";
    cached_prefixed::<TinyDevice>(P1, SRC, "pref_k").unwrap();
    cached_prefixed::<TinyDevice>(P1, SRC, "pref_k").unwrap();
    assert_eq!(ctx.stats_snapshot().2, 2);
    cached_prefixed::<TinyDevice>(P2, SRC, "pref_k").unwrap();
    assert_eq!(ctx.stats_snapshot().2, 3);
    cached_source::<TinyDevice>(SRC, "pref_k").unwrap();
    cached_prefixed::<TinyDevice>("", SRC, "pref_k").unwrap();
    assert_eq!(ctx.stats_snapshot().2, 4);
}

#[test]
fn cold_plan_timers() {
    let (_g, _ctx) = lock_ctx();
    let p1 = make_plan(
        vec![64],
        vec![make_step(plain_kernel("a", "a_k", None), 0, 16)],
    );
    let p2 = make_plan(
        vec![65_536],
        vec![make_step(plain_kernel("b", "b_k", None), 0, 16)],
    );
    let mut runner = open_runner(p1, true);
    runner.profiling(true);
    runner.replan(&p2).unwrap();
    runner.submit(0..1).unwrap();
    let cold = runner.take_profile().unwrap();
    assert_eq!(cold.spent["plan: alloc"].count, 1);
    assert_eq!(cold.spent["plan: compile"].count, 1);
    assert_eq!(cold.spent["plan: capture"].count, 1);
    runner.submit(0..1).unwrap();
    let warm = runner.take_profile().unwrap();
    assert!(!warm.spent.contains_key("plan: alloc"));
    assert!(!warm.spent.contains_key("plan: compile"));
    assert!(!warm.spent.contains_key("plan: capture"));
}

#[test]
fn sizer_scope_isolated() {
    assert!(!sizer::counting());
    let guard = sizer::sizing();
    assert!(sizer::counting());
    let isolated = std::thread::spawn(sizer::counting).join().unwrap();
    assert!(!isolated);
    let scope = sizer::current_scope().expect("active scope");
    let plan = make_plan(
        vec![1024],
        vec![make_step(plain_kernel("s", "s_k", None), 0, 16)],
    );
    std::thread::spawn(move || {
        scope.run(|| {
            assert!(sizer::counting());
            let _run = tessera_run::open(plan, &host()).unwrap();
        });
    })
    .join()
    .unwrap();
    assert!(guard.sized().scratch >= 1024);
    drop(guard);
    assert!(!sizer::counting());
}

fn ranked_plan(buffers: Vec<usize>, entry: &'static str) -> Plan {
    let s0 = make_step(plain_kernel("ov", entry, None), 0, 16);
    let ar = Step {
        kernel: KernelRef::new(AllReduce {
            len: 16,
            wire: DType::F32,
        }),
        rule: "all_reduce",
        reads: vec![Slot::floats(0, 16)],
        write: Slot::floats(1, 16),
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane: 1,
        table: None,
        index_table: None,
    };
    let mut plan = make_plan(buffers, vec![s0, ar]);
    (plan.rank, plan.ranks) = (0, 2);
    plan
}

fn pos(events: &[String], needle: &str) -> usize {
    events
        .iter()
        .position(|e| e == needle || e.starts_with(needle))
        .unwrap_or_else(|| panic!("missing `{needle}` in {events:?}"))
}

#[test]
fn retire_partial_bind() {
    let (_g, ctx) = lock_ctx();
    let p1 = make_plan(
        vec![64, 64],
        vec![
            make_step(plain_kernel("a", "a_k", None), 0, 16),
            make_step(plain_kernel("b", "b_k", None), 1, 16),
        ],
    );
    let p2 = make_plan(
        vec![65_536, 65_536],
        vec![
            make_step(plain_kernel("a", "a_k", None), 0, 16),
            make_step(plain_kernel("b", "b_k", None), 1, 16),
        ],
    );
    let mut runner = open_runner(p1.clone(), true);
    runner.submit(0..2).unwrap();
    let old0 = runner.address(Slot::floats(0, 16)).unwrap();
    ctx.take_events();
    // Grow slot 0, then fail slot 1 allocation.
    ctx.fail_alloc(Some(1));
    assert!(runner.replan(&p2).is_err());
    let ev = ctx.take_events();
    let free0 = format!("free:{old0:#x}");
    assert!(pos(&ev, "alloc_err") < pos(&ev, "finish_ok"));
    assert!(pos(&ev, "finish_ok") < pos(&ev, "drop_graph:"));
    assert!(pos(&ev, "drop_graph:") < pos(&ev, &free0));
    assert!(matches!(runner.submit(0..2), Err(RunError::Inconsistent)));
    runner.replan(&p1).unwrap();
    runner.submit(0..2).unwrap();
}

#[test]
fn retire_pause_error() {
    let (_g, ctx) = lock_ctx();
    let p1 = ranked_plan(vec![64, 64], "rp_k");
    let p2 = ranked_plan(vec![65_536, 64], "rp_k");
    let settings = Settings {
        memory: Memory::Off,
        ..host()
    };
    let (sum, _, fails) = TinySum::with_faults(ctx);
    let mut rank = Rank::open_device::<TinyDevice>(p1, &settings, NullTransport)
        .unwrap()
        .with_grid(Some(Box::new(sum)))
        .unwrap();
    rank.launching(Launching {
        graphs: true,
        pdl: false,
    });
    rank.submit(0..2).unwrap();
    let old0 = rank.address(Slot::floats(0, 16)).unwrap();
    ctx.take_events();
    // First pause succeeds, then retire pause fails.
    fails.lock().unwrap().extend([false, true]);
    assert!(rank.replan(&p2).is_err());
    let ev = ctx.take_events();
    assert!(ev.iter().any(|e| e == "pause_err"));
    assert!(!ev.iter().any(|e| e.starts_with("drop_graph:")));
    assert!(!ev.iter().any(|e| e.starts_with("free:")));
    assert!(ctx.is_live(old0));
    assert!(matches!(rank.submit(0..2), Err(RunError::Inconsistent)));
    // Retry replan once pause recovers.
    rank.replan(&p2).unwrap();
    let ev = ctx.take_events();
    let free0 = format!("free:{old0:#x}");
    assert!(pos(&ev, "pause_ok") < pos(&ev, "finish_ok"));
    assert!(pos(&ev, "finish_ok") < pos(&ev, "drop_graph:"));
    assert!(pos(&ev, "drop_graph:") < pos(&ev, &free0));
    assert!(!ctx.is_live(old0));
    rank.submit(0..2).unwrap();
}

#[test]
fn retire_finish_error() {
    let (_g, ctx) = lock_ctx();
    let p1 = make_plan(
        vec![64],
        vec![make_step(plain_kernel("f", "f_k", None), 0, 16)],
    );
    let p2 = make_plan(
        vec![65_536],
        vec![make_step(plain_kernel("f", "f_k", None), 0, 16)],
    );
    let mut runner = open_runner(p1, true);
    runner.submit(0..1).unwrap();
    let old0 = runner.address(Slot::floats(0, 16)).unwrap();
    ctx.take_events();
    ctx.fail_finish(1);
    assert!(runner.replan(&p2).is_err());
    let ev = ctx.take_events();
    assert!(ev.iter().any(|e| e == "finish_err"));
    assert!(!ev.iter().any(|e| e.starts_with("drop_graph:")));
    assert!(!ev.iter().any(|e| e.starts_with("free:")));
    assert!(ctx.is_live(old0));
    assert!(matches!(runner.submit(0..1), Err(RunError::Inconsistent)));
    // Retry once stream synchronization succeeds.
    runner.replan(&p2).unwrap();
    let ev = ctx.take_events();
    let free0 = format!("free:{old0:#x}");
    assert!(pos(&ev, "finish_ok") < pos(&ev, "drop_graph:"));
    assert!(pos(&ev, "drop_graph:") < pos(&ev, &free0));
    assert!(!ctx.is_live(old0));
    runner.submit(0..1).unwrap();
}

#[test]
fn override_pause_retry() {
    let (_g, ctx) = lock_ctx();
    let dir = tessera_testkit::scratch(&format!("pause-retry-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let plan = ranked_plan(vec![64, 64], "pr_k");
    let settings = Settings {
        memory: Memory::Off,
        kernels: Some(dir.clone()),
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
    rank.submit(0..2).unwrap();
    let (_, _, c0, cap0, _) = ctx.stats_snapshot();
    // Edit file, fail pause once, retry without touching disk.
    let file = dir.join("pr_k.cu");
    std::fs::write(&file, "__global__ void pr_k() { /* edit */ }").unwrap();
    fails.lock().unwrap().push_back(true);
    assert!(rank.submit(0..2).is_err());
    assert_eq!((ctx.stats_snapshot().2, ctx.stats_snapshot().3), (c0, cap0));
    rank.submit(0..2).unwrap();
    assert_eq!(
        (ctx.stats_snapshot().2, ctx.stats_snapshot().3),
        (c0 + 1, cap0 + 1)
    );
    assert_eq!(*log.lock().unwrap(), (2, 1));
    // Remove file, fail pause once, retry without touching disk.
    std::fs::remove_file(&file).unwrap();
    fails.lock().unwrap().push_back(true);
    assert!(rank.submit(0..2).is_err());
    assert_eq!(ctx.stats_snapshot().3, cap0 + 1);
    rank.submit(0..2).unwrap();
    assert_eq!(ctx.stats_snapshot().3, cap0 + 2);
    assert_eq!(*log.lock().unwrap(), (3, 2));
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
    const { assert!(!Check::<sizer::SizerGuard>::IS_SEND) };
    const { assert!(Check::<sizer::Scope>::IS_SEND) };
    let sized = |bytes| {
        make_plan(
            vec![bytes],
            vec![make_step(plain_kernel("s", "s_k", None), 0, 16)],
        )
    };
    // Out-of-order drop: dropping outer never clobbers or resurrects.
    let outer = sizer::sizing();
    let inner = sizer::sizing();
    let _r1 = tessera_run::open(sized(2048), &host()).unwrap();
    assert!(inner.sized().scratch >= 2048);
    assert_eq!(outer.sized().scratch, 0);
    drop(outer);
    assert!(sizer::counting());
    let _r2 = tessera_run::open(sized(4096), &host()).unwrap();
    assert!(inner.sized().scratch >= 4096);
    drop(inner);
    assert!(!sizer::counting());
    // Normal LIFO drop restores outer until it drops.
    let outer = sizer::sizing();
    let inner = sizer::sizing();
    drop(inner);
    assert!(sizer::counting());
    let _r3 = tessera_run::open(sized(1024), &host()).unwrap();
    assert!(outer.sized().scratch >= 1024);
    drop(outer);
    assert!(!sizer::counting());
}
