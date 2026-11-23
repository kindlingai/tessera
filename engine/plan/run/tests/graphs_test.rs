//! Multi-plan graph retention and scratch buffer stability on device runners.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use tessera_engine_settings::{Backend, Settings};
use tessera_gpukit::device::{Caches, Device};
use tessera_gpukit::launch::{Bound, KernelArgs, Launch};
use tessera_gpukit::{Error as GpuError, Limits, Platform};
use tessera_graph::DimName::Context;
use tessera_lower::kernels::Map;
use tessera_lower::{KernelRef, Launching, Plan, Slot, Step};
use tessera_run::testing::{device, host};

#[derive(Default)]
struct State {
    next_ptr: u64,
    live: HashMap<u64, usize>,
    allocs: usize,
    frees: usize,
    compiles: usize,
    captures: usize,
    replays: usize,
    recording: Option<Vec<u64>>,
    last_replayed: Vec<u64>,
}

#[derive(Clone)]
struct MockDevice(Arc<Mutex<State>>);

struct MockGraph {
    ptrs: Vec<u64>,
}

impl MockDevice {
    fn reset(&self) {
        let mut s = self.0.lock().unwrap();
        s.allocs = 0;
        s.frees = 0;
        s.compiles = 0;
        s.captures = 0;
        s.replays = 0;
        s.last_replayed.clear();
    }

    fn counts(&self) -> (usize, usize, usize, usize, usize) {
        let s = self.0.lock().unwrap();
        (s.allocs, s.frees, s.compiles, s.captures, s.replays)
    }

    fn replayed(&self) -> Vec<u64> {
        self.0.lock().unwrap().last_replayed.clone()
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
        let ptrs: Vec<u64> = args
            .bound()
            .filter_map(|b| match b {
                Bound::Pointer(p) => Some(p),
                Bound::Bytes(_) => None,
            })
            .collect();
        if let Some(rec) = &mut s.recording {
            rec.extend_from_slice(&ptrs);
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
        let ptr = s.next_ptr;
        s.next_ptr += (size.max(4096) as u64).next_multiple_of(4096) + 4096;
        s.live.insert(ptr, size);
        s.allocs += 1;
        Ok(ptr)
    }

    fn free(&self, ptr: u64) {
        let mut s = self.0.lock().unwrap();
        assert!(s.live.remove(&ptr).is_some(), "double free {ptr:#x}");
        s.frees += 1;
    }

    fn upload(&self, _: u64, _: &[u8]) -> Result<(), GpuError> {
        Ok(())
    }

    fn download(&self, _: u64, _: &mut [u8]) -> Result<(), GpuError> {
        Ok(())
    }

    fn zero(&self, _: u64, _: usize) -> Result<(), GpuError> {
        Ok(())
    }

    fn compile(&self, _: &str, _: &str) -> Result<(), GpuError> {
        self.0.lock().unwrap().compiles += 1;
        Ok(())
    }

    fn finish(&self) -> Result<Option<f64>, GpuError> {
        Ok(None)
    }

    fn graphs(&self) -> bool {
        true
    }

    fn capture<E: From<GpuError>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<MockGraph, E> {
        self.0.lock().unwrap().recording = Some(Vec::new());
        let done = record();
        let ptrs = self.0.lock().unwrap().recording.take().unwrap_or_default();
        done?;
        self.0.lock().unwrap().captures += 1;
        Ok(MockGraph { ptrs })
    }

    fn replay(&self, graph: &MockGraph) -> Result<(), GpuError> {
        let mut s = self.0.lock().unwrap();
        for &ptr in &graph.ptrs {
            assert!(
                s.live
                    .iter()
                    .any(|(&base, &len)| ptr >= base && ptr < base + len as u64),
                "replayed stale pointer {ptr:#x}"
            );
        }
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
        reads: vec![slot],
        write: slot,
        covers: Vec::new(),
        zone: 0,
        table: None,
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

fn open_mock(
    plan: Plan,
) -> (
    MutexGuard<'static, ()>,
    Box<dyn tessera_run::Run>,
    &'static MockDevice,
) {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

    // A new runtime binding keeps the previous binding cached.
    runner.bind(Context, 32);
    runner.submit(0..1).unwrap();
    runner.bind(Context, 16);
    runner.submit(0..1).unwrap();
    assert_eq!(ctx.counts(), (2, 0, compiles, 4, 8));
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
