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
        Map::plain(vec![MapStage::Add]).source()
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
    };
    let step = step_of(
        CountingMade {
            calls: Arc::clone(&calls),
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
