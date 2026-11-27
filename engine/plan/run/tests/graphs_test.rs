//! Multi-plan graph retention and scratch buffer stability on device runners.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};
use tessera_engine_comm::{Error as CommError, Transport};
use tessera_engine_settings::{Backend, Settings};
use tessera_gpukit::device::{Caches, Device, KernelAttrs, KernelStats};
use tessera_gpukit::launch::{Bound, KernelArgs, Launch};
use tessera_gpukit::{Error as GpuError, Limits, Platform};
use tessera_graph::DimName::Context;
use tessera_lower::kernels::{AllGather, AllReduce, Map};
use tessera_lower::{KernelRef, Launching, Plan, Slot, Step};
use tessera_mem::{DType, Tensor};
use tessera_run::testing::{device, host};
use tessera_run::{Error as RunError, Rank, Run, Sum};

#[derive(Clone, Copy)]
struct HandOp {
    job: u64,
    from: u64,
    to: u64,
    len: usize,
    gather: bool,
}

#[derive(Clone, Copy)]
struct StampOp {
    starts: u64,
    accum: u64,
    slot: usize,
    mode: u32,
}

#[derive(Default)]
struct State {
    next_ptr: u64,
    live: HashMap<u64, Vec<u8>>,
    allocs: usize,
    frees: usize,
    compiles: usize,
    captures: usize,
    replays: usize,
    d2d_copies: usize,
    clock: u64,
    recording: Option<(Vec<u64>, Vec<HandOp>, Vec<StampOp>)>,
    last_replayed: Vec<u64>,
    hand_job: u64,
    grid_running: bool,
}

#[derive(Clone)]
struct MockDevice(Arc<Mutex<State>>);

struct MockGraph {
    state: Arc<Mutex<State>>,
    ptrs: Vec<u64>,
    hands: Vec<HandOp>,
    stamps: Vec<StampOp>,
}

impl Drop for MockGraph {
    fn drop(&mut self) {
        let s = self.state.lock().unwrap();
        assert!(!s.grid_running, "dropped graph while persistent grid ran");
    }
}

fn buf_slice(live: &HashMap<u64, Vec<u8>>, ptr: u64, len: usize) -> &[u8] {
    let (&base, buf) = live
        .iter()
        .find(|(b, v)| ptr >= **b && ptr + len as u64 <= **b + v.len() as u64)
        .unwrap_or_else(|| panic!("invalid device read {ptr:#x} ({len} bytes)"));
    let off = (ptr - base) as usize;
    &buf[off..off + len]
}

fn buf_slice_mut(live: &mut HashMap<u64, Vec<u8>>, ptr: u64, len: usize) -> &mut [u8] {
    let (&base, buf) = live
        .iter_mut()
        .find(|(b, v)| ptr >= **b && ptr + len as u64 <= **b + v.len() as u64)
        .unwrap_or_else(|| panic!("invalid device write {ptr:#x} ({len} bytes)"));
    let off = (ptr - base) as usize;
    &mut buf[off..off + len]
}

fn run_hand(live: &mut HashMap<u64, Vec<u8>>, op: HandOp) {
    let seq_bytes = buf_slice(live, op.job, 8);
    let seq = u64::from_le_bytes(seq_bytes.try_into().unwrap()) + 1;
    buf_slice_mut(live, op.job, 8).copy_from_slice(&seq.to_le_bytes());
    if op.len == 0 {
        return;
    }
    let src = buf_slice(live, op.from, op.len * 4).to_vec();
    if op.gather {
        let mut out = Vec::with_capacity(src.len() * 2);
        out.extend_from_slice(&src);
        out.extend_from_slice(&src);
        buf_slice_mut(live, op.to, out.len()).copy_from_slice(&out);
    } else {
        let mut out = Vec::with_capacity(src.len());
        for &chunk in src.as_chunks::<4>().0 {
            let v = f32::from_le_bytes(chunk) * 2.0;
            out.extend_from_slice(&v.to_le_bytes());
        }
        buf_slice_mut(live, op.to, out.len()).copy_from_slice(&out);
    }
}

fn run_stamp(live: &mut HashMap<u64, Vec<u8>>, clock: &mut u64, op: StampOp) {
    *clock += 1_000;
    let now = *clock;
    let rd = |l: &HashMap<u64, Vec<u8>>, p: u64, i: usize| {
        u64::from_ne_bytes(buf_slice(l, p + (i * 8) as u64, 8).try_into().unwrap())
    };
    let wr = |l: &mut HashMap<u64, Vec<u8>>, p: u64, i: usize, v: u64| {
        buf_slice_mut(l, p + (i * 8) as u64, 8).copy_from_slice(&v.to_ne_bytes());
    };
    let add = |l: &mut HashMap<u64, Vec<u8>>, i: usize, from: usize| {
        let v = rd(l, op.accum, i) + now.wrapping_sub(rd(l, op.starts, from));
        wr(l, op.accum, i, v);
    };
    match op.mode {
        0 => {
            wr(live, op.starts, 0, now);
            wr(live, op.starts, op.slot, now);
        }
        1 => {
            add(live, op.slot, 0);
            wr(live, op.starts, 0, now);
        }
        2 | 4 => add(live, op.slot, op.slot),
        3 => wr(live, op.starts, op.slot, now),
        5 => {
            add(live, op.slot, 0);
            add(live, op.slot + 1, op.slot + 1);
        }
        _ => {}
    }
}

impl MockDevice {
    fn reset(&self) {
        let mut s = self.0.lock().unwrap();
        s.allocs = 0;
        s.frees = 0;
        s.compiles = 0;
        s.captures = 0;
        s.replays = 0;
        s.d2d_copies = 0;
        s.last_replayed.clear();
        s.hand_job = 0;
        s.grid_running = false;
    }

    fn counts(&self) -> (usize, usize, usize, usize, usize) {
        let s = self.0.lock().unwrap();
        (s.allocs, s.frees, s.compiles, s.captures, s.replays)
    }

    fn d2d_copies(&self) -> usize {
        self.0.lock().unwrap().d2d_copies
    }

    fn replayed(&self) -> Vec<u64> {
        self.0.lock().unwrap().last_replayed.clone()
    }

    fn read_u64(&self, ptr: u64) -> u64 {
        let s = self.0.lock().unwrap();
        u64::from_le_bytes(buf_slice(&s.live, ptr, 8).try_into().unwrap())
    }
}

impl Launch for MockDevice {
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
        let mut ptrs = Vec::new();
        let mut vals = Vec::new();
        for b in args.bound() {
            match b {
                Bound::Pointer(p) => ptrs.push(p),
                Bound::Bytes(bs) if bs.len() == 4 => {
                    vals.push(u32::from_le_bytes(bs.try_into().unwrap()));
                }
                Bound::Bytes(_) => {}
            }
        }
        let hand =
            (s.hand_job != 0 && ptrs.first() == Some(&s.hand_job) && ptrs.len() == 3).then(|| {
                HandOp {
                    job: ptrs[0],
                    from: ptrs[1],
                    to: ptrs[2],
                    len: vals[0] as usize,
                    gather: vals[1] != 0,
                }
            });
        let stamp =
            (ptrs.len() == 2 && vals.len() == 2 && ptrs[0] != s.hand_job).then(|| StampOp {
                starts: ptrs[0],
                accum: ptrs[1],
                slot: vals[0] as usize,
                mode: vals[1],
            });
        if let Some((rec_ptrs, rec_hands, rec_stamps)) = &mut s.recording {
            rec_ptrs.extend_from_slice(&ptrs);
            if let Some(op) = hand {
                rec_hands.push(op);
            }
            if let Some(op) = stamp {
                rec_stamps.push(op);
            }
        } else if let Some(op) = hand {
            run_hand(&mut s.live, op);
        } else if let Some(op) = stamp {
            let mut clock = s.clock;
            run_stamp(&mut s.live, &mut clock, op);
            s.clock = clock;
        }
        Ok(())
    }
}

impl Device for MockDevice {
    type Kernel = ();
    type Graph = MockGraph;

    const BACKEND: Backend = Backend::Cuda;

    fn open() -> Result<Self, GpuError> {
        Ok(Self(Arc::new(Mutex::new(State {
            next_ptr: 0x1_0000,
            ..State::default()
        }))))
    }

    fn error(msg: String) -> GpuError {
        GpuError::Cuda(msg)
    }

    fn caches() -> &'static Caches<Self> {
        static CACHES: Caches<MockDevice> = Caches::new();
        &CACHES
    }

    fn name(&self) -> String {
        "mock".into()
    }

    fn platform(&self) -> Platform {
        Platform::assumed(Limits::assumed())
    }

    fn alloc(&self, size: usize) -> Result<u64, GpuError> {
        let mut s = self.0.lock().unwrap();
        assert!(!s.grid_running, "alloc while persistent grid ran");
        let ptr = s.next_ptr;
        s.next_ptr += (size.max(4096) as u64).next_multiple_of(4096) + 4096;
        s.live.insert(ptr, vec![0u8; size]);
        s.allocs += 1;
        Ok(ptr)
    }

    fn free(&self, ptr: u64) {
        let mut s = self.0.lock().unwrap();
        assert!(!s.grid_running, "free while persistent grid ran");
        assert!(s.live.remove(&ptr).is_some(), "double free {ptr:#x}");
        s.frees += 1;
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), GpuError> {
        let mut s = self.0.lock().unwrap();
        buf_slice_mut(&mut s.live, ptr, bytes.len()).copy_from_slice(bytes);
        Ok(())
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), GpuError> {
        let s = self.0.lock().unwrap();
        bytes.copy_from_slice(buf_slice(&s.live, ptr, bytes.len()));
        Ok(())
    }

    fn copy_d2d(&self, dst: u64, src: u64, size: usize) -> Result<(), GpuError> {
        if size == 0 || dst == src {
            return Ok(());
        }
        let mut s = self.0.lock().unwrap();
        let data = buf_slice(&s.live, src, size).to_vec();
        buf_slice_mut(&mut s.live, dst, size).copy_from_slice(&data);
        s.d2d_copies += 1;
        Ok(())
    }

    fn zero(&self, ptr: u64, size: usize) -> Result<(), GpuError> {
        let mut s = self.0.lock().unwrap();
        assert!(!s.grid_running, "zero while persistent grid ran");
        buf_slice_mut(&mut s.live, ptr, size).fill(0);
        Ok(())
    }

    fn compile(&self, _: &str, _: &str) -> Result<(), GpuError> {
        let mut s = self.0.lock().unwrap();
        assert!(!s.grid_running, "compile while persistent grid ran");
        s.compiles += 1;
        Ok(())
    }

    fn finish(&self) -> Result<Option<f64>, GpuError> {
        Ok(None)
    }

    fn resident(&self, _: &(), _: u32, _: u32) -> Result<u32, GpuError> {
        Ok(4)
    }

    fn stats(&self, _: &()) -> Option<KernelStats> {
        let (regs, shared_bytes, max_threads) = (40, 2048, 512);
        Some(KernelStats {
            regs,
            shared_bytes,
            local_bytes: 0,
            max_threads,
        })
    }

    fn graphs(&self) -> bool {
        true
    }

    fn capturing(&self) -> bool {
        self.0.lock().unwrap().recording.is_some()
    }

    fn capture<E: From<GpuError>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<MockGraph, E> {
        {
            let mut s = self.0.lock().unwrap();
            assert!(!s.grid_running, "capture while persistent grid ran");
            s.recording = Some((Vec::new(), Vec::new(), Vec::new()));
        }
        let done = record();
        let (ptrs, hands, stamps) = self.0.lock().unwrap().recording.take().unwrap_or_default();
        done?;
        self.0.lock().unwrap().captures += 1;
        Ok(MockGraph {
            state: self.0.clone(),
            ptrs,
            hands,
            stamps,
        })
    }

    fn replay(&self, graph: &MockGraph) -> Result<(), GpuError> {
        let mut s = self.0.lock().unwrap();
        for &ptr in &graph.ptrs {
            assert!(
                s.live
                    .iter()
                    .any(|(&base, buf)| ptr >= base && ptr < base + buf.len() as u64),
                "replayed stale pointer {ptr:#x}"
            );
        }
        for &op in &graph.hands {
            run_hand(&mut s.live, op);
        }
        let mut clock = s.clock;
        for &op in &graph.stamps {
            run_stamp(&mut s.live, &mut clock, op);
        }
        s.clock = clock;
        s.replays += 1;
        s.last_replayed = graph.ptrs.clone();
        Ok(())
    }
}

fn copy_step(buffer: u32, len: usize) -> Step {
    let slot = Slot::floats(buffer, len);
    Step {
        kernel: KernelRef::new(Map::plain(Vec::new())),
        rule: "map",
        site: None,
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

fn scratch_plan(sizes: &[usize]) -> Plan {
    let steps: Vec<Step> = sizes
        .iter()
        .enumerate()
        .map(|(i, &bytes)| copy_step(i as u32, (bytes / 4).max(1)))
        .collect();
    Plan {
        id: Plan::fresh(),
        buffers: sizes.to_vec(),
        zones: vec![0; sizes.len()],
        steps,
        ..Plan::default()
    }
}

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn open_mock(
    plan: Plan,
) -> (
    MutexGuard<'static, ()>,
    Box<dyn tessera_run::Run>,
    &'static MockDevice,
) {
    let guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = tessera_gpukit::device::cached_context::<MockDevice>().unwrap();
    ctx.reset();
    let settings = Settings {
        memory: tessera_engine_settings::Memory::Off,
        ..host()
    };
    let mut runner = device::<MockDevice>(plan, &settings).unwrap();
    runner.launching(Launching {
        graphs: true,
        pdl: false,
    });
    (guard, runner, ctx)
}

#[test]
fn plan_graphs_stay() {
    let plan_a = scratch_plan(&[1024, 512]);
    let plan_b = scratch_plan(&[256]);
    let (_guard, mut runner, ctx) = open_mock(plan_a.clone());
    let addr_a0 = runner.address(plan_a.steps[0].write).unwrap();
    let addr_a1 = runner.address(plan_a.steps[1].write).unwrap();

    // Capture both ranges of plan A, then switch to B.
    runner.bind(Context, 16);
    runner.submit(0..1).unwrap();
    runner.submit(1..2).unwrap();
    runner.replan(&plan_b).unwrap();
    assert_eq!(runner.address(plan_b.steps[0].write).unwrap(), addr_a0);
    runner.submit(0..1).unwrap();
    let (allocs, frees, compiles, captures, replays) = ctx.counts();
    assert_eq!((allocs, frees, captures, replays), (2, 0, 3, 3));

    // Switching back retains both buffers and cached graphs.
    runner.replan(&plan_a).unwrap();
    assert_eq!(runner.address(plan_a.steps[0].write).unwrap(), addr_a0);
    assert_eq!(runner.address(plan_a.steps[1].write).unwrap(), addr_a1);
    runner.submit(0..1).unwrap();
    runner.submit(1..2).unwrap();
    runner.replan(&plan_b).unwrap();
    runner.submit(0..1).unwrap();
    assert_eq!(ctx.counts(), (2, 0, compiles, 3, 6));

    // Unreferenced runtime bindings keep reusing the cached graph.
    runner.bind(Context, 32);
    runner.submit(0..1).unwrap();
    runner.bind(Context, 16);
    runner.submit(0..1).unwrap();
    assert_eq!(ctx.counts(), (2, 0, compiles, 3, 8));
}

#[test]
fn growth_clears_graphs() {
    let small = scratch_plan(&[256]);
    let large = scratch_plan(&[65_536]);
    let (_guard, mut runner, ctx) = open_mock(small.clone());
    let old_ptr = runner.address(small.steps[0].write).unwrap();
    runner.submit(0..1).unwrap();
    assert_eq!(ctx.replayed(), [old_ptr, old_ptr]);

    // Growing scratch frees the old buffer and clears graphs.
    runner.replan(&large).unwrap();
    let new_ptr = runner.address(large.steps[0].write).unwrap();
    assert_ne!(new_ptr, old_ptr);
    runner.submit(0..1).unwrap();
    assert_eq!(ctx.replayed(), [new_ptr, new_ptr]);
    let (allocs, frees, _, captures, replays) = ctx.counts();
    assert_eq!((allocs, frees, captures, replays), (2, 1, 2, 2));

    // Returning to small reuses the larger buffer, capturing once.
    runner.replan(&small).unwrap();
    assert_eq!(runner.address(small.steps[0].write).unwrap(), new_ptr);
    runner.submit(0..1).unwrap();
    assert_eq!(ctx.replayed(), [new_ptr, new_ptr]);
    assert_eq!(ctx.counts().0, 2);
    assert_eq!(ctx.counts().3, 3);

    // Later switches hit cached graphs with zero allocations.
    runner.replan(&large).unwrap();
    runner.submit(0..1).unwrap();
    runner.replan(&small).unwrap();
    runner.submit(0..1).unwrap();
    let (allocs, frees, _, captures, replays) = ctx.counts();
    assert_eq!((allocs, frees, captures, replays), (2, 1, 3, 5));
}

struct MockTransport(Arc<Mutex<usize>>);

impl Transport for MockTransport {
    fn rank(&self) -> u32 {
        0
    }

    fn ranks(&self) -> u32 {
        2
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, CommError> {
        *self.0.lock().unwrap() += 1;
        Ok(vec![mine.to_vec(), mine.to_vec()])
    }
}

#[derive(Default)]
struct SumLog {
    starts: usize,
    stops: usize,
    checks: usize,
    timed: bool,
}

struct MockSum {
    ctx: &'static MockDevice,
    job: u64,
    capturable: bool,
    allow_gather: bool,
    running: bool,
    timed: bool,
    log: Arc<Mutex<SumLog>>,
}

impl MockSum {
    fn new(
        ctx: &'static MockDevice,
        capturable: bool,
        allow_gather: bool,
    ) -> (Self, Arc<Mutex<SumLog>>, u64) {
        let job = ctx.alloc(64).unwrap();
        ctx.0.lock().unwrap().hand_job = job;
        let log = Arc::new(Mutex::new(SumLog::default()));
        let sum = Self {
            ctx,
            job,
            capturable,
            allow_gather,
            running: false,
            timed: false,
            log: log.clone(),
        };
        (sum, log, job)
    }

    fn start(&mut self) {
        if !self.running {
            self.running = true;
            self.ctx.0.lock().unwrap().grid_running = true;
            self.log.lock().unwrap().starts += 1;
        }
    }

    fn stop(&mut self) {
        if std::mem::take(&mut self.running) {
            let _ = self.hand(self.job, self.job, 0, false);
            self.ctx.0.lock().unwrap().grid_running = false;
            self.log.lock().unwrap().stops += 1;
        }
    }

    fn hand(&self, from: u64, to: u64, len: usize, gather: bool) -> Result<(), RunError> {
        let mut args = KernelArgs::new();
        args.device_ptr(self.job)
            .device_ptr(from)
            .device_ptr(to)
            .u32(len as u32)
            .u32(u32::from(gather));
        self.ctx.launch(&(), (1, 1, 1), (1, 1, 1), 0, &mut args)?;
        Ok(())
    }
}

impl Drop for MockSum {
    fn drop(&mut self) {
        self.stop();
        self.ctx.free(self.job);
    }
}

impl Sum for MockSum {
    fn sum(&mut self, from: u64, to: u64, len: usize, _: DType) -> Result<(), RunError> {
        if len == 0 {
            self.start();
            return Ok(());
        }
        if !self.ctx.capturing() {
            self.start();
        }
        self.hand(from, to, len, false)
    }

    fn gather(
        &mut self,
        from: u64,
        to: u64,
        part: usize,
        _: usize,
        _: usize,
    ) -> Result<bool, RunError> {
        if !self.allow_gather {
            return Ok(false);
        }
        if !self.ctx.capturing() {
            self.start();
        }
        self.hand(from, to, part, true)?;
        Ok(true)
    }

    fn capturable(&self) -> bool {
        self.capturable
    }

    fn pause(&mut self) -> Result<(), RunError> {
        self.stop();
        Ok(())
    }

    fn check(&mut self) -> Result<(), RunError> {
        self.log.lock().unwrap().checks += 1;
        Ok(())
    }

    fn timing(&mut self, on: bool) {
        self.timed = on;
        self.log.lock().unwrap().timed = on;
    }

    fn spent(&mut self) -> Result<Option<[f64; 2]>, RunError> {
        Ok(self.timed.then_some([0.001, 0.002]))
    }
}

fn ranked_plan(scratch_bytes: usize) -> Plan {
    let a = Slot::floats(0, 4);
    let b = Slot::floats(1, 4);
    let c = Slot::floats(2, 8);
    let step = |kernel: KernelRef, rule, reads, write| {
        let lane = u8::from(tessera_lower::kernels::crosses(kernel.name()));
        Step {
            kernel,
            rule,
            reads,
            write,
            lane,
            ..copy_step(0, 1)
        }
    };
    Plan {
        id: Plan::fresh(),
        rank: 0,
        ranks: 2,
        buffers: vec![scratch_bytes, scratch_bytes, scratch_bytes.max(32)],
        zones: vec![0, 0, 0],
        steps: vec![
            copy_step(0, 4),
            step(
                KernelRef::new(AllReduce {
                    len: 4,
                    wire: DType::F32,
                }),
                "all_reduce",
                vec![a],
                b,
            ),
            copy_step(1, 4),
            step(
                KernelRef::new(AllReduce {
                    len: 4,
                    wire: DType::F32,
                }),
                "all_reduce",
                vec![b],
                a,
            ),
            step(
                KernelRef::new(AllGather { len: 8, rows: 1 }),
                "all_gather",
                vec![a],
                c,
            ),
            copy_step(2, 8),
        ],
        ..Plan::default()
    }
}

type RankFixture = (
    MutexGuard<'static, ()>,
    Rank<MockTransport>,
    &'static MockDevice,
    Arc<Mutex<usize>>,
    Option<(Arc<Mutex<SumLog>>, u64)>,
);

fn open_rank(plan: Plan, grid: Option<(bool, bool)>) -> RankFixture {
    let guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = tessera_gpukit::device::cached_context::<MockDevice>().unwrap();
    ctx.reset();
    let (sum, meta) = match grid {
        Some((capturable, allow_gather)) => {
            let (s, log, job) = MockSum::new(ctx, capturable, allow_gather);
            (Some(Box::new(s) as Box<dyn Sum>), Some((log, job)))
        }
        None => (None, None),
    };
    let exchanges = Arc::new(Mutex::new(0));
    let settings = Settings {
        memory: tessera_engine_settings::Memory::Off,
        ..host()
    };
    let mut rank =
        Rank::open_device::<MockDevice>(plan, &settings, MockTransport(exchanges.clone()))
            .unwrap()
            .with_grid(sum)
            .unwrap();
    rank.launching(Launching {
        graphs: true,
        pdl: false,
    });
    (guard, rank, ctx, exchanges, meta)
}

fn sum_counts(log: &Arc<Mutex<SumLog>>) -> (usize, usize, usize) {
    let l = log.lock().unwrap();
    (l.starts, l.stops, l.checks)
}

#[test]
fn sums_capture_once() {
    let plan = ranked_plan(256);
    let (_guard, mut rank, ctx, _, meta) = open_rank(plan.clone(), Some((true, true)));
    let (log, job) = meta.unwrap();

    let input = Tensor::f32(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    rank.write(plan.steps[0].write, input.view()).unwrap();
    rank.submit(0..plan.steps.len()).unwrap();
    // All 6 steps capture into 1 graph.
    assert_eq!((ctx.counts().3, ctx.counts().4), (1, 1));
    assert_eq!(ctx.read_u64(job), 3);
    assert_eq!(
        rank.read(plan.steps[5].write).unwrap().floats().unwrap(),
        [4.0, 8.0, 12.0, 16.0, 4.0, 8.0, 12.0, 16.0]
    );

    // Replays reuse the graph and keep the grid up.
    for pass in 2..=3 {
        rank.write(plan.steps[0].write, input.view()).unwrap();
        rank.submit(0..plan.steps.len()).unwrap();
        assert_eq!((ctx.counts().3, ctx.counts().4), (1, pass));
        assert_eq!(ctx.read_u64(job), pass as u64 * 3);
    }
    assert_eq!(sum_counts(&log), (1, 0, 3));

    // Scratch growth pauses first, invalidates, then re-captures cleanly.
    let large = ranked_plan(65_536);
    rank.replan(&large).unwrap();
    assert_eq!(sum_counts(&log), (1, 1, 3));
    rank.write(large.steps[0].write, input.view()).unwrap();
    rank.submit(0..large.steps.len()).unwrap();
    assert_eq!((ctx.counts().3, ctx.counts().4), (2, 4));
    assert_eq!(ctx.read_u64(job), 13);
    assert_eq!(sum_counts(&log), (2, 1, 4));

    // Warm replan reuses buffers and graphs without pausing.
    let small = ranked_plan(256);
    rank.replan(&small).unwrap();
    rank.submit(0..small.steps.len()).unwrap();
    assert_eq!(sum_counts(&log), (3, 2, 5));
    for _ in 0..3 {
        rank.replan(&large).unwrap();
        rank.submit(0..large.steps.len()).unwrap();
        rank.replan(&small).unwrap();
        rank.submit(0..small.steps.len()).unwrap();
    }
    // Zero extra pauses or restarts across warm A <-> B switches.
    assert_eq!(sum_counts(&log), (3, 2, 11));
    assert_eq!((ctx.counts().3, ctx.counts().4), (3, 11));

    // Dropping pauses the persistent grid before freeing buffers.
    drop(rank);
    assert_eq!(sum_counts(&log).1, 3);
}

#[test]
fn sums_split_fallbacks() {
    let plan = ranked_plan(256);
    let input = Tensor::f32(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

    // Non-capturable Sum splits and pauses on cold capture.
    {
        let (_guard, mut rank, ctx, _, meta) = open_rank(plan.clone(), Some((false, true)));
        let (log, job) = meta.unwrap();
        rank.submit(0..plan.steps.len()).unwrap();
        assert_eq!((ctx.counts().3, ctx.counts().4), (3, 3));
        assert_eq!(ctx.read_u64(job), 5);
        assert_eq!(sum_counts(&log), (2, 2, 1));
        // Second pass hits all 3 graphs with zero pauses.
        rank.submit(0..plan.steps.len()).unwrap();
        assert_eq!((ctx.counts().3, ctx.counts().4), (3, 6));
        assert_eq!(ctx.read_u64(job), 8);
        assert_eq!(sum_counts(&log), (3, 2, 2));
    }

    // Profiling captures a stamped graph and keeps the grid warm.
    {
        let (_guard, mut rank, ctx, _, meta) = open_rank(plan.clone(), Some((true, true)));
        let (log, _) = meta.unwrap();
        let range = 0..plan.steps.len();
        rank.submit(range.clone()).unwrap();
        assert_eq!((ctx.counts().3, ctx.counts().4), (1, 1));
        assert_eq!(sum_counts(&log), (1, 0, 1));
        rank.profiling(true);
        for _ in 0..3 {
            rank.submit(range.clone()).unwrap();
        }
        assert_eq!((ctx.counts().3, ctx.counts().4), (2, 4));
        assert!(log.lock().unwrap().timed);
        assert_eq!(sum_counts(&log), (2, 1, 4));
        let prof = rank.take_profile().unwrap();
        assert_eq!(prof.spent["gpu: graph"].count, 3);
        assert!(prof.spent["gpu: graph"].seconds > 0.0);
        assert_eq!(prof.spent["all_reduce: wait"].count, 3);
        assert_eq!(prof.spent["all_reduce: transfer"].count, 3);
        let ar = tessera_run::Profile::key(&plan.steps[1]);
        assert!(ar.starts_with("all_reduce ["), "{ar}");
        assert_eq!(prof.spent[&ar].count, 6);
        assert_eq!(prof.spent["gpu: graph"].attrs, None);
        assert_eq!(
            prof.spent[&ar].attrs,
            Some(KernelAttrs {
                regs: 40,
                local_bytes: 0,
                shared_bytes: 2048,
                ctas_per_sm: 4
            })
        );
        rank.profiling(false);
        rank.submit(range).unwrap();
        assert_eq!((ctx.counts().3, ctx.counts().4), (2, 5));
        assert_eq!(sum_counts(&log), (2, 1, 5));
    }

    // Host-fallback AllGather splits while AllReduces capture together.
    {
        let (_guard, mut rank, ctx, exchanges, meta) = open_rank(plan.clone(), Some((true, false)));
        let (log, job) = meta.unwrap();
        rank.write(plan.steps[0].write, input.view()).unwrap();
        rank.submit(0..plan.steps.len()).unwrap();
        assert_eq!((ctx.counts().3, ctx.counts().4), (2, 2));
        assert_eq!(*exchanges.lock().unwrap(), 1);
        assert_eq!(ctx.read_u64(job), 3);
        assert_eq!(
            rank.read(plan.steps[5].write).unwrap().floats().unwrap(),
            [4.0, 8.0, 12.0, 16.0, 4.0, 8.0, 12.0, 16.0]
        );
        // Second pass reuses the cached split with zero pauses.
        rank.submit(0..plan.steps.len()).unwrap();
        assert_eq!((ctx.counts().3, ctx.counts().4), (2, 4));
        assert_eq!(sum_counts(&log), (2, 1, 2));
    }

    // Absent Sum grid falls back to host transport.
    {
        let (_guard, mut rank, ctx, exchanges, _) = open_rank(plan.clone(), None);
        rank.write(plan.steps[0].write, input.view()).unwrap();
        rank.submit(0..plan.steps.len()).unwrap();
        assert_eq!((ctx.counts().3, ctx.counts().4), (3, 3));
        assert_eq!(*exchanges.lock().unwrap(), 3);
    }

    // Shared grid pauses on cold peer replan, stays hot warm.
    {
        struct Share(Rc<RefCell<MockSum>>);
        impl Sum for Share {
            fn sum(&mut self, f: u64, t: u64, n: usize, w: DType) -> Result<(), RunError> {
                self.0.borrow_mut().sum(f, t, n, w)
            }
            fn gather(
                &mut self,
                f: u64,
                t: u64,
                p: usize,
                k: usize,
                r: usize,
            ) -> Result<bool, RunError> {
                self.0.borrow_mut().gather(f, t, p, k, r)
            }
            fn capturable(&self) -> bool {
                self.0.borrow().capturable()
            }
            fn pause(&mut self) -> Result<(), RunError> {
                self.0.borrow_mut().pause()
            }
            fn check(&mut self) -> Result<(), RunError> {
                self.0.borrow_mut().check()
            }
        }
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = tessera_gpukit::device::cached_context::<MockDevice>().unwrap();
        ctx.reset();
        let (mock_sum, log, job) = MockSum::new(ctx, true, true);
        let shared = Rc::new(RefCell::new(mock_sum));
        let settings = Settings {
            memory: tessera_engine_settings::Memory::Off,
            ..host()
        };
        let make = |p: Plan| {
            let t = MockTransport(Arc::new(Mutex::new(0)));
            let mut r = Rank::open_device::<MockDevice>(p, &settings, t).unwrap();
            r = r.with_grid(Some(Box::new(Share(shared.clone())))).unwrap();
            r.launching(Launching {
                graphs: true,
                pdl: false,
            });
            r
        };
        let (p1, p2, p2_grow) = (ranked_plan(256), ranked_plan(256), ranked_plan(65_536));
        let (mut r1, mut r2) = (make(p1.clone()), make(p2.clone()));
        r1.submit(0..p1.steps.len()).unwrap();
        // Peer cold replan + capture pauses the live grid first.
        r2.replan(&p2_grow).unwrap();
        r2.submit(0..p2_grow.steps.len()).unwrap();
        let cold = sum_counts(&log);
        // Alternating warm submits keep the shared grid running.
        for _ in 0..3 {
            r1.submit(0..p1.steps.len()).unwrap();
            r2.submit(0..p2_grow.steps.len()).unwrap();
        }
        assert_eq!((sum_counts(&log).0, sum_counts(&log).1), (cold.0, cold.1));
        assert!(ctx.read_u64(job) > 20);
        // Dropping r1 while r2 lives pauses before freeing r1's buffers.
        drop(r1);
        drop(r2);
    }
}

#[test]
fn d2d_copy_keeps_graphs() {
    let (src_plan, dst_plan) = (scratch_plan(&[64]), scratch_plan(&[64]));
    let (_guard, mut dst, ctx) = open_mock(dst_plan.clone());
    let settings = Settings {
        memory: tessera_engine_settings::Memory::Off,
        ..host()
    };
    let mut src = device::<MockDevice>(src_plan.clone(), &settings).unwrap();
    let (src_slot, dst_slot) = (src_plan.steps[0].write, dst_plan.steps[0].write);
    let vals: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
    src.write(src_slot, Tensor::f32(vec![16], vals.clone()).view())
        .unwrap();
    dst.submit(0..1).unwrap();
    assert_eq!(ctx.counts().3, 1);
    let src_addr = src.address(src_slot).unwrap();
    dst.copy_from(dst_slot, 0, src_addr, 32).unwrap();
    dst.copy_from(dst_slot, 32, src_addr + 32, 32).unwrap();
    let dst_addr = dst.address(dst_slot).unwrap();
    dst.copy_from(dst_slot, 4, dst_addr, 16).unwrap();
    dst.submit(0..1).unwrap();
    assert_eq!(
        (ctx.counts().3, ctx.counts().4, ctx.d2d_copies()),
        (1, 2, 3)
    );
    let got = dst.read(dst_slot).unwrap();
    let floats = got.floats().unwrap();
    assert_eq!(
        (&floats[0..5], &floats[8..16]),
        (&[1.0, 1.0, 2.0, 3.0, 4.0][..], &vals[8..16])
    );
}
