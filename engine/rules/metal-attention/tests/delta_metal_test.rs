//! Verifies native Metal Gated DeltaNet and causal conv1d kernels.

use tessera_attention as _;
use tessera_kernels as _;
use tessera_metal_attention as _;

use tessera_graph::DimName::{HeadDim, Heads, Hidden, KvHeads, Rows, Seqs};
use tessera_graph::{DType, Dim, DimExpr, Graph, PathSeg, Role};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::form::{BF16_FORM, Load};
use tessera_lower::{Backend, Device, Pick, Pins, View, lower};
use tessera_run::testing::runs;
use tessera_run::{Run, open};
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{EntryKind, Model, Port, Taps, TensorType, Weights};

fn metal_settings() -> Option<tessera_lower::Settings> {
    runs()
        .into_iter()
        .find(|s| matches!(s.backend, Pick::Gpu(Backend::Metal)))
}

fn unit_keys(shape: Vec<usize>, width: usize, draws: &mut Noise) -> Tensor {
    let len: usize = shape.iter().product();
    let mut vals: Vec<f32> = (0..len).map(|_| draws.draw()).collect();
    for row in vals.chunks_mut(width) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        for v in row {
            *v /= norm;
        }
    }
    Tensor::f32(shape, vals)
}

fn port(role: Role, ty: TensorType) -> Port {
    Port { role, ty }
}

fn ragged_delta_graph(
    seqs: u64,
    heads: u64,
    kv_heads: u64,
    dk: u64,
    dv: u64,
    max_rows: u64,
) -> Graph {
    let dims = vec![
        Dim::fixed(Seqs, seqs),
        Dim::fixed(Heads, heads),
        Dim::fixed(KvHeads, kv_heads),
        Dim::fixed(HeadDim, dk),
        Dim::fixed(Hidden, dv),
        Dim::bounded(Rows, max_rows),
    ];
    let m = Model::new(dims);
    let q_ty = TensorType::new([Rows, KvHeads, HeadDim], DType::F32);
    let v_ty = TensorType::new([Rows, Heads, Hidden], DType::F32);
    let g_ty = TensorType::new([Rows, Heads], DType::F32);
    let id_ty = TensorType::new([Rows], DType::U32);
    let [q, k, v, decay, rate, slots, pos] = m.entry(
        EntryKind::Step,
        [
            port(Role::Tokens, q_ty.clone()),
            port(Role::Tokens, q_ty),
            port(Role::Tokens, v_ty),
            port(Role::Tokens, g_ty.clone()),
            port(Role::Tokens, g_ty),
            port(Role::Slots, id_ty.clone()),
            port(Role::Positions, id_ty),
        ],
    );
    let state = m.kept(TensorType::new([Seqs, Heads, HeadDim, Hidden], DType::F32));
    let out = m.delta_rows(
        1.0 / (dk as f32).sqrt(),
        &state,
        [&q, &k, &v],
        [&decay, &rate],
        [&slots, &pos],
    );
    m.finish([(Role::Logits, &out)])
}

fn ragged_conv_graph(seqs: u64, width: u64, taps: u64, dilation: u64, max_rows: u64) -> Graph {
    let dims = vec![
        Dim::fixed(Seqs, seqs),
        Dim::fixed(Hidden, width),
        Dim::bounded(Rows, max_rows),
    ];
    let m = Model::new(dims);
    let x_ty = TensorType::new([Rows, Hidden], DType::F32);
    let id_ty = TensorType::new([Rows], DType::U32);
    let [x, slots, pos] = m.entry(
        EntryKind::Step,
        [
            port(Role::Tokens, x_ty),
            port(Role::Slots, id_ty.clone()),
            port(Role::Positions, id_ty),
        ],
    );
    let mut path = vec![PathSeg::Field("taps".into())];
    let w = Taps::declare(&m, &mut path, &[width.into(), taps.into()]);
    let hist = (taps - 1) * dilation;
    let state = m.kept(TensorType::new(
        vec![
            DimExpr::from(Seqs),
            DimExpr::Const(hist),
            DimExpr::from(Hidden),
        ],
        DType::F32,
    ));
    let out = x.conv_rows(&w, dilation as u32, &state, [&slots, &pos]);
    m.finish([(Role::Logits, &out)])
}

#[test]
fn delta_metal_matches() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let mut draws = Noise(41);
    for &(dk, dv) in &[(64u64, 64u64), (128, 128)] {
        let (seqs, heads, kv) = (4u64, 4u64, 2u64);
        let g = ragged_delta_graph(seqs, heads, kv, dk, dv, 32);
        let mut store = Store::default();
        let mut runner: Option<Box<dyn Run>> = None;
        // Prefill multi-slot, then coalesced decode with filler and reset.
        let batches: [(Vec<u32>, Vec<u32>); 3] = [
            (
                (0..12).map(|_| 0).chain((0..10).map(|_| 2)).collect(),
                (0..12).chain(0..10).collect(),
            ),
            (vec![0, 0, 2], vec![12, 12, 10]),
            (vec![0, 2, 2], vec![0, 11, 12]),
        ];
        for (slots, pos) in batches {
            let n = slots.len();
            let q = unit_keys(vec![n, kv as usize, dk as usize], dk as usize, &mut draws);
            let k = unit_keys(vec![n, kv as usize, dk as usize], dk as usize, &mut draws);
            let v = Tensor::f32(
                vec![n, heads as usize, dv as usize],
                (0..n * heads as usize * dv as usize)
                    .map(|_| draws.draw())
                    .collect(),
            );
            let decay = Tensor::f32(
                vec![n, heads as usize],
                (0..n * heads as usize)
                    .map(|_| -draws.draw().abs())
                    .collect(),
            );
            let rate = Tensor::f32(
                vec![n, heads as usize],
                (0..n * heads as usize)
                    .map(|_| (draws.draw() + 1.0) * 0.5)
                    .collect(),
            );
            let inputs = [
                q,
                k,
                v,
                decay,
                rate,
                Tensor::u32(vec![n], slots),
                Tensor::u32(vec![n], pos),
            ];
            let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
            let dev = Device::guess(&settings).favoring("delta_metal");
            let plan = lower(&g, 0, &[(Seqs, seqs), (Rows, n as u64)], &dev).unwrap();
            assert!(
                plan.steps
                    .iter()
                    .any(|s| s.kernel.name() == "delta_metal_rows")
            );
            let r = match runner.as_mut() {
                Some(r) => {
                    r.replan(&plan).unwrap();
                    r
                }
                None => runner.insert(open(plan, &settings).unwrap()),
            };
            let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
            let got = r.run(&views).unwrap().remove(0);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(off < 2e-4, "dk={dk} n={n}: off {off:e}");
        }
    }
}

#[test]
fn conv_metal_matches() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let mut draws = Noise(57);
    let forms: [(Option<&'static dyn Load>, f32); 2] = [(None, 1e-4), (Some(BF16_FORM), 2e-2)];
    for &dilation in &[1u64, 2u64] {
        let (seqs, width, taps) = (4u64, 32u64, 4u64);
        let g = ragged_conv_graph(seqs, width, taps, dilation, 32);
        let w = Tensor::f32(
            vec![width as usize, taps as usize],
            (0..(width * taps) as usize).map(|_| draws.draw()).collect(),
        );
        for (w_form, tol) in forms {
            let mut pins = Pins::default();
            if let Some(f) = w_form {
                pins.weights = f;
            }
            let mut store = Store::default();
            let mut runner: Option<Box<dyn Run>> = None;
            let batches: [(Vec<u32>, Vec<u32>); 3] = [
                (
                    (0..12).map(|_| 0).chain((0..10).map(|_| 2)).collect(),
                    (0..12).chain(0..10).collect(),
                ),
                (vec![0, 0, 2], vec![12, 12, 10]),
                (vec![0, 2, 2], vec![0, 11, 12]),
            ];
            for (slots, pos) in batches {
                let n = slots.len();
                let x = Tensor::f32(
                    vec![n, width as usize],
                    (0..n * width as usize).map(|_| draws.draw()).collect(),
                );
                let inputs = [x, Tensor::u32(vec![n], slots), Tensor::u32(vec![n], pos)];
                let want = run_kept(&g, std::slice::from_ref(&w), &mut store, 0, &inputs).unwrap();
                let dev = Device::guess(&settings)
                    .favoring("conv_metal")
                    .pinned(pins.clone());
                let plan = lower(&g, 0, &[(Seqs, seqs), (Rows, n as u64)], &dev).unwrap();
                assert!(
                    plan.steps
                        .iter()
                        .any(|s| s.kernel.name() == "conv_metal_rows")
                );
                let r = match runner.as_mut() {
                    Some(r) => {
                        r.replan(&plan).unwrap();
                        r
                    }
                    None => {
                        let r = runner.insert(open(plan, &settings).unwrap());
                        r.load(0, w.view()).unwrap();
                        r
                    }
                };
                let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
                let got = r.run(&views).unwrap().remove(0);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                assert!(off < tol, "dil={dilation} n={n}: off {off:e}");
            }
        }
    }
}
