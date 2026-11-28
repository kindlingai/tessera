//! Parity tests for native Metal simdgroup_matrix Lightning Indexer.

use tessera_attention as _;
use tessera_kernels as _;
use tessera_metal_attention as _;

use tessera_graph::DimName::{Context, HeadDim, Heads, Hidden, Rows, Seqs};
use tessera_graph::{DType, Dim, Graph, TensorType};
use tessera_interp::{Store, Tensor, run, run_kept};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load as Stored};
use tessera_lower::kept::{self, Layout};
use tessera_lower::{Backend, Device, Pins, Settings, View, lower};
use tessera_run::testing::runs;
use tessera_run::{Run, open};
use tessera_testkit::Noise;
use tessera_trace::{Act, EntryKind, Mask, Model, Port, Role};

fn metal_settings() -> Option<Settings> {
    runs().into_iter().find(|s| s.gpu() == Some(Backend::Metal))
}

fn draw(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

fn token_port<const N: usize>(dims: [tessera_graph::DimName; N]) -> Port {
    Port {
        role: Role::Tokens,
        ty: TensorType::new(dims, DType::F32),
    }
}

fn build_step_graph(heads: u64, dim: u64, max_ctx: u64, causal: bool) -> Graph {
    let model = Model::new(vec![
        Dim::fixed(Hidden, 1),
        Dim::fixed(HeadDim, dim),
        Dim::fixed(Heads, heads),
        Dim::bounded(Context, max_ctx),
        Dim::bounded(Rows, 64),
        Dim::bounded(Seqs, 4),
    ]);
    let u32_row = TensorType::new([Rows], DType::U32);
    let [q, k, w, slots, positions] = model.entry(
        EntryKind::Step,
        [
            token_port([Rows, Heads, HeadDim]),
            token_port([Rows, HeadDim]),
            token_port([Rows, Hidden, Heads]),
            Port {
                role: Role::Slots,
                ty: u32_row.clone(),
            },
            Port {
                role: Role::Positions,
                ty: u32_row,
            },
        ],
    );
    let cache = model.kept(TensorType::new([Seqs, Context, HeadDim], DType::F32));
    let keys = k.append(&cache, [&slots, &positions]).gather(&slots);
    let scores = q
        .matmul_rows(&keys)
        .act(Act::Relu)
        .permute(&[0, 2, 1])
        .matmul_rows(&w);
    let out = if causal {
        scores.reshape([Rows, Context]).causal(&positions)
    } else {
        scores
    };
    model.finish([(Role::Logits, &out)])
}

fn assert_close_scores(got: &[f32], want: &[f32], tol: f32, tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: size mismatch");
    let mut worst = 0.0f32;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        if b.is_infinite() {
            assert_eq!(a, b, "{tag}[{i}]: masked mismatch {a} vs {b}");
            continue;
        }
        worst = worst.max((a - b).abs());
    }
    assert!(worst < tol, "{tag}: max diff {worst:e} >= {tol:e}");
}

fn run_indexed(
    runner: &mut Option<Box<dyn Run>>,
    plan: tessera_lower::Plan,
    settings: &Settings,
    inputs: &[Tensor],
) -> Tensor {
    let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
    if let Some(live) = runner.as_mut() {
        live.replan(&plan).unwrap();
        return live.run(&views).unwrap().remove(0);
    }
    let table_slot = plan.table;
    let mut live = open(plan, settings).unwrap();
    if let Some(slot) = table_slot {
        let mut raw = Vec::with_capacity(slot.len * 4);
        for p in (0..slot.len as u32).rev() {
            raw.extend_from_slice(&p.to_le_bytes());
        }
        live.held_mut().upload(slot, &raw).unwrap();
    }
    let out = live.run(&views).unwrap().remove(0);
    *runner = Some(live);
    out
}

#[test]
fn indexer_metal_ragged_matches() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let mut draws = Noise(131);
    type RaggedCase = (
        u64,
        u64,
        bool,
        &'static dyn Stored,
        &'static dyn Stored,
        f32,
    );
    let cases: &[RaggedCase] = &[
        (32, 128, true, F32_FORM, F32_FORM, 1e-3),
        (32, 128, true, F32_FORM, BF16_FORM, 0.3),
        (32, 128, true, BF16_FORM, BF16_FORM, 0.5),
        (32, 128, true, F32_FORM, FP8_FORM, 2.0),
        (64, 128, true, F32_FORM, BF16_FORM, 0.6),
        (20, 64, false, F32_FORM, F32_FORM, 1e-3),
        (16, 192, true, F32_FORM, BF16_FORM, 0.5),
    ];
    let layouts: [&'static dyn Layout; 2] = [kept::DENSE, &kept::PAGED16];
    for &(heads, dim, causal, act_form, kv_form, tol) in cases {
        let g = build_step_graph(heads, dim, 256, causal);
        let mut pins = Pins::default();
        if act_form.name() != "f32" {
            pins = pins.activations(act_form);
        }
        if kv_form.name() != "f32" {
            pins = pins.kept(kv_form);
        }
        for &layout in &layouts {
            let (mut store, mut runner) = (Store::default(), None);
            let steps: [&[(u32, u32, u32)]; 2] = [
                &[(0, 0, 19), (1, 0, 7)],
                &[(0, 19, 1), (1, 7, 1), (2, 0, 5)],
            ];
            for spec in steps {
                let slots: Vec<u32> = spec
                    .iter()
                    .flat_map(|(s, _, n)| std::iter::repeat_n(*s, *n as usize))
                    .collect();
                let pos: Vec<u32> = spec
                    .iter()
                    .flat_map(|(_, p, n)| (0..*n).map(move |i| p + i))
                    .collect();
                let n = slots.len();
                let furthest = *pos.iter().max().unwrap() as u64 + 1;
                let inputs = [
                    draw(vec![n, heads as usize, dim as usize], &mut draws),
                    draw(vec![n, dim as usize], &mut draws),
                    draw(vec![n, 1, heads as usize], &mut draws),
                    Tensor::u32(vec![n], slots),
                    Tensor::u32(vec![n], pos),
                ];
                let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
                let sizes = [(Seqs, 4), (Rows, n as u64), (Context, furthest)];
                let mut device = Device::guess(&settings)
                    .favoring("indexer_metal")
                    .pinned(pins.clone());
                device.kept = layout;
                let plan = lower(&g, 0, &sizes, &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                assert!(names.contains(&"indexer_metal"), "{names:?}");
                let got = run_indexed(&mut runner, plan, &settings, &inputs);
                let tag = format!("h={heads} d={dim} {} {}", kv_form.name(), layout.name());
                assert_close_scores(got.floats().unwrap(), want[0].floats().unwrap(), tol, &tag);
            }
        }
    }
}

#[test]
fn indexer_metal_whole_matches() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let mut draws = Noise(137);
    for (heads, dim, mask) in [
        (32u64, 128u64, Some(Mask::Causal)),
        (48, 128, Some(Mask::Back(5))),
        (16, 64, None),
    ] {
        let model = Model::new(vec![
            Dim::bounded(Seqs, 4),
            Dim::bounded(Context, 64),
            Dim::fixed(Heads, heads),
            Dim::fixed(HeadDim, dim),
            Dim::fixed(Hidden, 1),
        ]);
        let port = |ty| Port {
            role: Role::Tokens,
            ty,
        };
        let [q, k, w] = model.entry(
            EntryKind::Forward,
            [
                port(TensorType::new([Seqs, Context, Heads, HeadDim], DType::F32)),
                port(TensorType::new([Seqs, Context, HeadDim], DType::F32)),
                port(TensorType::new([Seqs, Context, Hidden, Heads], DType::F32)),
            ],
        );
        let scored = q
            .matmul_rows(&k)
            .act(Act::Relu)
            .permute(&[0, 1, 3, 2])
            .matmul_rows(&w);
        let out = if let Some(kind) = mask {
            scored.reshape([Seqs, Context, Context]).mask(kind)
        } else {
            scored
        };
        let g = model.finish([(Role::Logits, &out)]);
        let (seqs, keys) = (2usize, 21usize);
        let inputs = [
            draw(vec![seqs, keys, heads as usize, dim as usize], &mut draws),
            draw(vec![seqs, keys, dim as usize], &mut draws),
            draw(vec![seqs, keys, 1, heads as usize], &mut draws),
        ];
        let want = run(&g, &[], 0, &inputs).unwrap();
        let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
        let sizes = [(Seqs, seqs as u64), (Context, keys as u64)];
        let device = Device::guess(&settings).favoring("indexer_metal");
        let plan = lower(&g, 0, &sizes, &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert_eq!(names, ["indexer_metal"]);
        let mut runner = open(plan, &settings).unwrap();
        let got = runner.run(&views).unwrap().remove(0);
        let tag = format!("whole h={heads} d={dim} {mask:?}");
        assert_close_scores(got.floats().unwrap(), want[0].floats().unwrap(), 1e-3, &tag);
    }
}
