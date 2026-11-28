use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tessera_graph::DimName::Rows;
use tessera_graph::{
    Act, DType, Dim, Entry, EntryKind, Graph, Node, NodeId, Op, Output, Placement, Port, Role, Row,
    Shape, Stage, TensorType, Weight,
};
use tessera_lower::kernels::{Map, MapStage};
use tessera_lower::sites::Chain;
use tessera_lower::{
    ByBackend, Device, Kernel, KernelRef, Launch, Plan, Slot, Source, Step, Tree, Wants,
    kernel_identity,
};

struct CountingAlloc;

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && TRACK.with(Cell::get) {
            LIVE.with(|n| n.set(n.get() + layout.size() as isize));
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACK.with(Cell::get) {
            LIVE.with(|n| n.set(n.get() - layout.size() as isize));
        }
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(ptr, layout, new_size) };
        if !next.is_null() && TRACK.with(Cell::get) {
            LIVE.with(|n| n.set(n.get() + new_size as isize - layout.size() as isize));
        }
        next
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn f32_ty(dims: Vec<tessera_graph::DimExpr>) -> TensorType {
    TensorType {
        dtype: DType::F32,
        shape: Shape(dims),
        placement: Placement::Replicated,
    }
}

#[test]
fn cut_small_dense() {
    use tessera_graph::{CalibTap, Composite, DimName::Hidden, Region, verify};
    for n in [4usize, 32] {
        let mut g = Graph::new(vec![Dim::bounded(Rows, 64), Dim::fixed(Hidden, 16)]);
        let ty = TensorType::new([Rows, Hidden], DType::F32);
        g.entries.push(Entry {
            kind: EntryKind::Forward,
            inputs: vec![Port {
                role: Role::Tokens,
                ty: ty.clone(),
            }],
            outputs: vec![],
        });
        g.nodes.push(Node {
            op: Op::Input { entry: 0, port: 0 },
            args: vec![],
            ty: ty.clone(),
            numerics: None,
        });
        for i in 1..=n {
            g.nodes.push(Node {
                op: Op::Row(Row(vec![Stage::Add])),
                args: vec![NodeId((i - 1) as u32), NodeId(i.saturating_sub(3) as u32)],
                ty: ty.clone(),
                numerics: None,
            });
        }
        let last = NodeId(n as u32);
        g.entries[0].outputs.push(Output {
            role: Role::Logits,
            node: last,
        });
        g.entries[0].outputs.push(Output {
            role: Role::Calib(CalibTap::AttnIn),
            node: NodeId((n / 2).max(1) as u32),
        });
        let r = Region {
            kind: Composite::Layer,
            start: NodeId(1),
            end: NodeId((n + 1) as u32),
            inputs: vec![NodeId(0)],
            outputs: vec![last],
        };
        g.regions.push(r.clone());
        let ids: Vec<NodeId> = (1..=n).map(|i| NodeId(i as u32)).collect();
        let plain = tessera_lower::cut(&g, &ids);
        verify(&plain).unwrap();
        assert_eq!(
            (plain.nodes.len(), plain.entries[0].outputs.len()),
            (n + 1, 1)
        );
        let reg = tessera_lower::cut_region(&g, &r);
        verify(&reg).unwrap();
        assert_eq!(
            (
                reg.nodes.len(),
                reg.regions.len(),
                reg.entries[0].outputs.len()
            ),
            (n + 1, 1, 2)
        );
    }
}

#[test]
fn chain_stops_turn() {
    let mut g = Graph::new(vec![Dim::bounded(Rows, 8)]);
    let x_ty = f32_ty(vec![Rows.into(), 128u64.into()]);
    let w_ty = f32_ty(vec![128u64.into(), 128u64.into()]);
    let v_ty = f32_ty(vec![128u64.into()]);
    g.weights.push(Weight {
        path: vec![],
        ty: w_ty.clone(),
    });
    g.weights.push(Weight {
        path: vec![],
        ty: v_ty.clone(),
    });
    g.weights.push(Weight {
        path: vec![],
        ty: v_ty.clone(),
    });
    let mut push = |op: Op, args: Vec<NodeId>, ty: TensorType| -> NodeId {
        let id = NodeId(g.nodes.len() as u32);
        g.nodes.push(Node {
            op,
            args,
            ty,
            numerics: None,
        });
        id
    };
    let x = push(Op::Input { entry: 0, port: 0 }, vec![], x_ty.clone());
    let up = push(Op::Input { entry: 0, port: 1 }, vec![], x_ty.clone());
    let w = push(Op::Weight(0), vec![], w_ty);
    let suh = push(Op::Weight(1), vec![], v_ty.clone());
    let svh = push(Op::Weight(2), vec![], v_ty);
    let s1 = push(Op::Row(Row(vec![Stage::Mul])), vec![x, suh], x_ty.clone());
    let h1 = push(
        Op::Row(Row(vec![Stage::Hadamard { block: 128 }])),
        vec![s1],
        x_ty.clone(),
    );
    let mm = push(Op::Matmul, vec![h1, w], x_ty.clone());
    let h2 = push(
        Op::Row(Row(vec![Stage::Hadamard { block: 128 }])),
        vec![mm],
        x_ty.clone(),
    );
    let gate = push(Op::Row(Row(vec![Stage::Mul])), vec![h2, svh], x_ty.clone());
    let act = push(
        Op::Row(Row(vec![Stage::Act(Act::Silu)])),
        vec![gate],
        x_ty.clone(),
    );
    let out = push(Op::Row(Row(vec![Stage::Mul])), vec![act, up], x_ty.clone());
    g.entries.push(Entry {
        kind: EntryKind::Step,
        inputs: vec![
            Port {
                role: Role::Tokens,
                ty: x_ty.clone(),
            },
            Port {
                role: Role::Tokens,
                ty: x_ty,
            },
        ],
        outputs: vec![Output {
            role: Role::Logits,
            node: out,
        }],
    });
    let dev = Device::default();
    let parts = dev.division();
    let tree = Tree::upto(
        &g,
        0,
        Wants::ALL,
        &[(Rows, 1)],
        &[],
        &dev.pins,
        dev.kept,
        &parts,
        0,
    )
    .expect("tree builds");
    let root = tree.outputs[0].0;
    let chain: Chain = tree.site(root).expect("chain matches");
    // Chain stops at gate rather than svh.
    assert_eq!(chain.stages.len(), 2);
    assert_eq!(chain.reads[0], tree.of(gate).unwrap());
}

struct SameNameKernel {
    entry: &'static str,
    src: &'static str,
    guess: f64,
}

impl std::fmt::Debug for SameNameKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("same_debug")
    }
}

impl Kernel for SameNameKernel {
    fn name(&self) -> &'static str {
        "fast_mm_mxfp8"
    }
    fn sizes(&self) -> Vec<usize> {
        vec![8, 640, 2560]
    }
    fn guess(&self) -> f64 {
        self.guess
    }
    fn source(&self) -> Source {
        Source::Fixed {
            text: ByBackend::from_fn(|_| self.src),
            entry: self.entry,
        }
    }
    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }
}

#[test]
fn memo_distinguishes_entry_source() {
    let dev = Device::default();
    let a = SameNameKernel {
        entry: "entry_a",
        src: "kernel_v1",
        guess: 1e-6,
    };
    let b = SameNameKernel {
        entry: "entry_b",
        src: "kernel_v1",
        guess: 9e-6,
    };
    let c = SameNameKernel {
        entry: "entry_a",
        src: "kernel_v2",
        guess: 5e-6,
    };
    let (sa, sb, sc) = (
        dev.seconds(&a, 16),
        dev.seconds(&b, 16),
        dev.seconds(&c, 16),
    );
    assert!(
        (sa - sb).abs() > 1e-5,
        "distinct entries collided: {sa} vs {sb}"
    );
    assert!(
        (sa - sc).abs() > 1e-5,
        "distinct sources collided: {sa} vs {sc}"
    );
    assert!((sb - sc).abs() > 1e-5, "variants collided: {sb} vs {sc}");
}

struct SameMadeKernel {
    entry: &'static str,
    stage: MapStage,
    guess: f64,
}

impl std::fmt::Debug for SameMadeKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("same_made")
    }
}

impl Kernel for SameMadeKernel {
    fn name(&self) -> &'static str {
        "made_op"
    }
    fn sizes(&self) -> Vec<usize> {
        vec![8, 640]
    }
    fn guess(&self) -> f64 {
        self.guess
    }
    fn source(&self) -> Source {
        let Source::Made(mut body) = Map::plain(vec![self.stage]).source() else {
            unreachable!();
        };
        body.entry = self.entry.into();
        Source::Made(body)
    }
    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }
}

fn step_of<K: Kernel + 'static>(k: K, len: usize) -> Step {
    Step {
        kernel: KernelRef::new(k),
        rule: "r",
        site: None,
        reads: Vec::new(),
        write: Slot::floats(0, len),
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane: 0,
        table: None,
        index_table: None,
    }
}

#[test]
fn made_source_identity() {
    let dev = Device::default();
    let a = SameMadeKernel {
        entry: "made_a",
        stage: MapStage::Add,
        guess: 1e-6,
    };
    let b = SameMadeKernel {
        entry: "made_b",
        stage: MapStage::Add,
        guess: 9e-6,
    };
    let c = SameMadeKernel {
        entry: "made_a",
        stage: MapStage::Mul,
        guess: 5e-6,
    };
    let (ia, ib, ic) = (
        kernel_identity(&a),
        kernel_identity(&b),
        kernel_identity(&c),
    );
    assert_eq!(ia.0.as_ref(), "made_a");
    assert_eq!(ib.0.as_ref(), "made_b");
    assert_ne!(ia, ib);
    assert_ne!(ia, ic);
    let (sa, sb, sc) = (
        dev.seconds(&a, 16),
        dev.seconds(&b, 16),
        dev.seconds(&c, 16),
    );
    assert!((sa - sb).abs() > 1e-5, "{sa} vs {sb}");
    assert!((sa - sc).abs() > 1e-5, "{sa} vs {sc}");
    assert!((sb - sc).abs() > 1e-5, "{sb} vs {sc}");
}

#[test]
fn timings_prefer_source() {
    let mut dev = Device::default();
    let a = SameMadeKernel {
        entry: "made_a",
        stage: MapStage::Add,
        guess: 1e-6,
    };
    let b = SameMadeKernel {
        entry: "made_b",
        stage: MapStage::Mul,
        guess: 1e-6,
    };
    let c = SameMadeKernel {
        entry: "made_c",
        stage: MapStage::Act(Act::Silu),
        guess: 1e-6,
    };
    let sa = step_of(
        SameMadeKernel {
            entry: "made_a",
            stage: MapStage::Add,
            guess: 1e-6,
        },
        16,
    );
    let sb = step_of(
        SameMadeKernel {
            entry: "made_b",
            stage: MapStage::Mul,
            guess: 1e-6,
        },
        16,
    );
    dev.fit([(&sa, 1e-3), (&sb, 10e-3)]);
    assert_eq!(dev.samples(), 2);
    assert!((dev.seconds(&a, 16) - 1e-3).abs() < 1e-9);
    assert!((dev.seconds(&b, 16) - 10e-3).abs() < 1e-9);
    let fallback = dev.seconds(&c, 16);
    assert!((fallback - 5.5e-3).abs() < 1e-6, "fallback {fallback}");
    dev.fit_cold([(&sa, 2e-3), (&sb, 40e-3)]);
    assert!((dev.seconds_in(0, &a, 16, true) - 2e-3).abs() < 1e-9);
    assert!((dev.seconds_in(0, &b, 16, true) - 40e-3).abs() < 1e-9);
}

struct CountingMade {
    calls: Arc<AtomicUsize>,
    loads: Option<Arc<AtomicUsize>>,
    key: Option<u64>,
    scale: bool,
}

impl std::fmt::Debug for CountingMade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("counting_made")
    }
}

impl Kernel for CountingMade {
    fn name(&self) -> &'static str {
        "counting_made"
    }
    fn source(&self) -> Source {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let stage = if self.scale {
            MapStage::Scale(1.5)
        } else {
            MapStage::Add
        };
        Map::plain(vec![stage]).source()
    }
    fn load(&self, len: usize) -> Option<tessera_lower::Load> {
        let loads = self.loads.as_ref()?;
        loads.fetch_add(1, Ordering::Relaxed);
        let stage = if self.scale {
            MapStage::Scale(1.5)
        } else {
            MapStage::Add
        };
        Map::plain(vec![stage]).load(len)
    }
    fn load_key(&self) -> Option<u64> {
        self.key
    }
    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }
}

#[test]
fn made_identity_no_leak() {
    let dev = Device::default();
    let map = Map::plain(vec![MapStage::Add]);
    let warm = dev.seconds(&map, 16);
    assert!(warm > 0.0);

    // Repeated warmed Map hits retain zero bytes.
    for _ in 0..2 {
        LIVE.with(|n| n.set(0));
        TRACK.with(|t| t.set(true));
        for _ in 0..10_000 {
            assert_eq!(dev.seconds(&map, 16), warm);
        }
        TRACK.with(|t| t.set(false));
        assert_eq!(LIVE.with(Cell::get), 0, "made identity leaked bytes");
    }

    // KernelRef caches entry and identity without re-rendering source.
    let calls = Arc::new(AtomicUsize::new(0));
    let step = step_of(
        CountingMade {
            calls: Arc::clone(&calls),
            loads: None,
            key: None,
            scale: false,
        },
        16,
    );
    let plan = Plan::default();
    let id = step.kernel.identity();
    for _ in 0..100 {
        assert_eq!(step.kernel.identity(), id);
        assert_eq!(step.kernel.entry(), id.0);
        assert!(dev.step_seconds(&plan, &step).unwrap() > 0.0);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    // Entry-only calls cache without computing identity hash.
    let entry_calls = Arc::new(AtomicUsize::new(0));
    let k = KernelRef::new(CountingMade {
        calls: Arc::clone(&entry_calls),
        loads: None,
        key: None,
        scale: false,
    });
    let first = k.entry();
    for _ in 0..100 {
        assert_eq!(k.entry(), first);
    }
    assert_eq!(entry_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn knobs_and_prefixes() {
    use tessera_lower::{Knob, Knobs, define_prefix, find_knob, prefixed, register_knob};

    assert_eq!(Knobs::split_flag("k:n=42").unwrap(), ("k", "n", 42));
    assert!(Knobs::split_flag("bad_spec").is_err());

    register_knob(Knob {
        kernel: "guest_test_k",
        name: "tile",
        least: 8,
        most: 64,
    });
    assert!(find_knob("guest_test_k", "tile").is_some());
    let mut k = Knobs::default();
    k.parse("guest_test_k:tile=16").unwrap();
    assert_eq!(k.get("guest_test_k", "tile"), Some(16));
    assert!(k.parse("guest_test_k:tile=128").is_err());

    let def = define_prefix([("BLK_N", "16"), ("USE_X", "1")]).unwrap();
    assert!(def.contains("#define BLK_N 16\n#define USE_X 1\n#line 1\n"));
    assert!(define_prefix([("1bad", "2")]).is_err());
    let base: &'static str = "void f() {}";
    assert!(std::ptr::eq(prefixed("", base), base));
    assert_ne!(prefixed(&def, base), base);
}

#[test]
fn unmeasured_skips_source() {
    let mut dev = Device::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let k = CountingMade {
        calls: Arc::clone(&calls),
        loads: None,
        key: None,
        scale: false,
    };
    let step = step_of(
        CountingMade {
            calls: Arc::clone(&calls),
            loads: None,
            key: None,
            scale: false,
        },
        16,
    );
    let plan = Plan::default();
    assert!(dev.seconds(&k, 16) > 0.0);
    assert!(dev.step_seconds(&plan, &step).unwrap() > 0.0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    dev.fit([(&step, 2e-3)]);
    assert!((dev.seconds(&k, 16) - 2e-3).abs() < 1e-9);
    assert!(calls.load(Ordering::Relaxed) > 0);
}

fn gb10_device() -> Device {
    Device::default().on(tessera_lower::Platform {
        name: "NVIDIA GB10".into(),
        backend: "cuda".into(),
        units: 48,
        bandwidth: 273e9,
        limits: tessera_lower::Limits::assumed(),
        ..tessera_lower::Platform::default()
    })
}

#[test]
fn unmeasured_caches_chip_model() {
    let dev = gb10_device();
    let calls = Arc::new(AtomicUsize::new(0));
    let loads = Arc::new(AtomicUsize::new(0));
    let k = CountingMade {
        calls: Arc::clone(&calls),
        loads: Some(Arc::clone(&loads)),
        key: Some(1),
        scale: false,
    };
    let first = dev.seconds(&k, 256);
    assert!(first > 0.0);
    for _ in 0..100 {
        assert_eq!(dev.seconds(&k, 256), first);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(loads.load(Ordering::Relaxed), 1);

    // Distinct load keys never alias on the same device.
    let other_loads = Arc::new(AtomicUsize::new(0));
    let other = CountingMade {
        calls: Arc::clone(&calls),
        loads: Some(Arc::clone(&other_loads)),
        key: Some(2),
        scale: true,
    };
    assert_eq!(
        dev.seconds(&other, 16_777_216),
        gb10_device().seconds(&other, 16_777_216)
    );
    assert_eq!(other_loads.load(Ordering::Relaxed), 2);
}

#[test]
fn unmeasured_map_order_independent() {
    let len = 16_777_216;
    let add = Map::plain(vec![MapStage::Add]);
    let scale = Map::plain(vec![MapStage::Scale(1.5)]);
    assert_eq!(add.sizes(), scale.sizes());
    assert_eq!(add.work(len), scale.work(len));

    let fresh_add = gb10_device().seconds(&add, len);
    let fresh_scale = gb10_device().seconds(&scale, len);
    assert_ne!(fresh_add, fresh_scale);

    let ab = gb10_device();
    assert_eq!(ab.seconds(&add, len), fresh_add);
    assert_eq!(ab.seconds(&scale, len), fresh_scale);

    let ba = gb10_device();
    assert_eq!(ba.seconds(&scale, len), fresh_scale);
    assert_eq!(ba.seconds(&add, len), fresh_add);

    // Unkeyed kernels skip model_memo and never alias across orders.
    let mk = |scale| CountingMade {
        calls: Arc::new(AtomicUsize::new(0)),
        loads: Some(Arc::new(AtomicUsize::new(0))),
        key: None,
        scale,
    };
    let (u_add, u_scale) = (mk(false), mk(true));
    let fresh_u_add = gb10_device().seconds(&u_add, len);
    let fresh_u_scale = gb10_device().seconds(&u_scale, len);
    assert_ne!(fresh_u_add, fresh_u_scale);

    let u_ab = gb10_device();
    assert_eq!(u_ab.seconds(&u_add, len), fresh_u_add);
    assert_eq!(u_ab.seconds(&u_scale, len), fresh_u_scale);

    let u_ba = gb10_device();
    assert_eq!(u_ba.seconds(&u_scale, len), fresh_u_scale);
    assert_eq!(u_ba.seconds(&u_add, len), fresh_u_add);
}

thread_local! {
    static DERIVED_LOADS: Cell<usize> = const { Cell::new(0) };
}

#[derive(Debug, Kernel)]
#[launch(
    name = "derived_keyed",
    sizes = [16],
    source = Map::plain(vec![MapStage::Add]).source(),
    launches = Vec::new(),
    load = derived_keyed_load,
    load_key = tessera_lower::load_key(self.scale),
)]
struct DerivedKeyed {
    scale: bool,
}

fn derived_keyed_load(k: &DerivedKeyed, len: usize) -> Option<tessera_lower::Load> {
    DERIVED_LOADS.with(|n| n.set(n.get() + 1));
    let stage = if k.scale {
        MapStage::Scale(1.5)
    } else {
        MapStage::Add
    };
    Map::plain(vec![stage]).load(len)
}

struct LoadPolicy {
    binary: bool,
}

impl std::fmt::Debug for LoadPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LoadPolicy")
    }
}

#[derive(Debug, Kernel)]
#[launch(
    name = "derived_policy",
    sizes = [16],
    source = self.map().source(),
    launches = Vec::new(),
    load = self.map().load(len),
)]
struct PolicyKernel {
    policy: LoadPolicy,
}

impl PolicyKernel {
    fn map(&self) -> Map {
        let stage = if self.policy.binary {
            MapStage::Add
        } else {
            MapStage::Scale(1.5)
        };
        Map::plain(vec![stage])
    }
}

#[test]
fn derived_load_key_explicit() {
    let len = 16_777_216;
    let a = DerivedKeyed { scale: false };
    let b = DerivedKeyed { scale: true };
    assert!(a.load_key().is_some());
    assert!(b.load_key().is_some());
    assert_ne!(a.load_key(), b.load_key());

    let dev = gb10_device();
    DERIVED_LOADS.with(|n| n.set(0));
    let sa = dev.seconds(&a, len);
    for _ in 0..100 {
        assert_eq!(dev.seconds(&a, len), sa);
    }
    assert_eq!(DERIVED_LOADS.with(Cell::get), 1);

    let sb = dev.seconds(&b, len);
    assert_eq!(DERIVED_LOADS.with(Cell::get), 2);
    assert_ne!(sa, sb);
    assert_eq!(sa, gb10_device().seconds(&a, len));
    assert_eq!(sb, gb10_device().seconds(&b, len));

    // Order-independent across reversed evaluation order.
    let rev = gb10_device();
    assert_eq!(rev.seconds(&b, len), sb);
    assert_eq!(rev.seconds(&a, len), sa);

    // Missing load_key stays uncached even with equal Debug text.
    let pa = PolicyKernel {
        policy: LoadPolicy { binary: true },
    };
    let pb = PolicyKernel {
        policy: LoadPolicy { binary: false },
    };
    assert_eq!(format!("{pa:?}"), format!("{pb:?}"));
    assert_eq!(pa.load_key(), None);
    assert_eq!(pb.load_key(), None);

    let fa = gb10_device().seconds(&pa, len);
    let fb = gb10_device().seconds(&pb, len);
    assert_ne!(fa, fb);

    let ab = gb10_device();
    assert_eq!(ab.seconds(&pa, len), fa);
    assert_eq!(ab.seconds(&pb, len), fb);

    let ba = gb10_device();
    assert_eq!(ba.seconds(&pb, len), fb);
    assert_eq!(ba.seconds(&pa, len), fa);
}
