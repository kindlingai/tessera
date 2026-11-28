//! The per-row causal filter against the interpreter.

use tessera_attention as _;
use tessera_kernels as _;

use tessera_graph::DimName::{self, Hidden, Rows, Seqs};
use tessera_graph::{DType, Dim, DimExpr, Graph, TensorType};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::{Device, Limits, Platform, lower};
use tessera_run::testing::host;
use tessera_run::{Run, open};
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{EntryKind, Model, Port, Role};

const RULE: &str = "conv_span";

const WIDTH: usize = 6;

/// Ragged rows through `taps` taps, each row's own if `dynamic`.
fn graph(taps: u64, dilation: u32, dynamic: bool) -> Graph {
    let dims = vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 128),
        Dim::fixed(Hidden, WIDTH as u64),
    ];
    let m = Model::new(dims);
    let port = |role, shape: Vec<DimExpr>, dtype| Port {
        role,
        ty: TensorType::new(shape, dtype),
    };
    let rows: DimExpr = Rows.into();
    let mut w_shape = vec![Hidden.into(), taps.into()];
    if dynamic {
        w_shape.insert(0, rows.clone());
    }
    let ports = [
        port(Role::Tokens, vec![rows.clone(), Hidden.into()], DType::F32),
        port(Role::Tokens, w_shape, DType::F32),
        port(Role::Slots, vec![rows.clone()], DType::U32),
        port(Role::Positions, vec![rows], DType::U32),
    ];
    let [x, w, slots, positions] = m.entry(EntryKind::Step, ports);
    let hist = (taps - 1) * dilation as u64;
    let state = TensorType::new(vec![Seqs.into(), hist.into(), Hidden.into()], DType::F32);
    let out = x.conv_rows_dynamic(&w, dilation, &m.kept(state), [&slots, &positions]);
    m.finish([(Role::Logits, &out)])
}

fn noise(shape: Vec<usize>, draws: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| draws.draw()).collect();
    Tensor::f32(shape, data)
}

/// Feeds for rows of (slot, position).
fn feeds(at: &[(u32, u32)], taps: usize, dynamic: bool, draws: &mut Noise) -> Vec<Tensor> {
    let n = at.len();
    let w_shape = match dynamic {
        true => vec![n, WIDTH, taps],
        false => vec![WIDTH, taps],
    };
    vec![
        noise(vec![n, WIDTH], draws),
        noise(w_shape, draws),
        Tensor::u32(vec![n], at.iter().map(|r| r.0).collect()),
        Tensor::u32(vec![n], at.iter().map(|r| r.1).collect()),
    ]
}

/// `count` rows of `slot` from `first`.
fn run_of(slot: u32, first: u32, count: u32) -> Vec<(u32, u32)> {
    (first..first + count).map(|p| (slot, p)).collect()
}

/// Launches with mixed slots, fillers and restarts.
fn launches() -> Vec<Vec<(u32, u32)>> {
    let mut mixed = run_of(1, 0, 3);
    for p in 40..75 {
        mixed.push((0, p));
        mixed.push((2, p - 40));
    }
    // Fillers repeat a position. Position 0 starts afresh.
    let mut filled = run_of(0, 75, 20);
    filled.extend([(0, 94), (0, 94), (1, 3), (1, 3)]);
    filled.extend(run_of(0, 95, 12));
    filled.extend(run_of(2, 0, 2));
    filled.extend(run_of(1, 4, 9));
    filled.extend([(2, 0), (2, 0), (2, 1), (3, 0)]);
    vec![
        run_of(0, 0, 40),
        mixed,
        filled,
        [run_of(3, 1, 33), run_of(1, 0, 1), run_of(1, 0, 40)].concat(),
    ]
}

fn sizes(rows: usize) -> [(DimName, u64); 2] {
    [(Seqs, 4), (Rows, rows as u64)]
}

/// Taps, dilation, and whether each row has its own.
const CASES: [(u64, u32, bool); 4] = [(3, 2, true), (4, 1, false), (2, 5, false), (5, 3, true)];

#[test]
fn span_matches_interp() {
    let mut draws = Noise(23);
    let settings = host();
    for (taps, dilation, dynamic) in CASES {
        let g = graph(taps, dilation, dynamic);
        let (mut store, mut runner): (Store, Option<Box<dyn Run>>) = (Store::default(), None);
        for at in launches() {
            let inputs = feeds(&at, taps as usize, dynamic, &mut draws);
            let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
            let device = Device::guess(&settings).favoring(RULE);
            let plan = lower(&g, 0, &sizes(at.len()), &device).unwrap();
            assert_eq!(plan.steps[0].kernel.name(), RULE);
            let r = match runner.as_mut() {
                Some(r) => {
                    r.replan(&plan).unwrap();
                    r
                }
                None => runner.insert(open(plan, &settings).unwrap()),
            };
            let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
            let got = r.run(&views).unwrap().remove(0);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(off < 1e-5, "taps {taps} by {dilation}: off {off:e}");
        }
    }
}

/// Metal spans long launches. Others scan each slot.
#[test]
fn metal_takes_span() {
    let on = |backend: &str, arch: &str, rows| {
        let platform = Platform {
            backend: backend.into(),
            arch: arch.into(),
            name: arch.into(),
            ..Platform::assumed(Limits::assumed())
        };
        let device = Device::guess(&host()).on(platform);
        let plan = lower(&graph(4, 1, false), 0, &sizes(rows), &device).unwrap();
        plan.steps[0].kernel.name()
    };
    assert_eq!(on("metal", "apple9", 64), RULE);
    assert_eq!(on("metal", "apple9", 8), "conv_rows");
    assert_eq!(on("cuda", "sm_121", 64), "conv_rows");
    assert_eq!(on("rocm", "gfx11_5", 64), "conv_rows");
}
