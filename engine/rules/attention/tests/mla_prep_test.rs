//! Parity tests for fused MLA Q and KV prep.

use tessera_attention as _;
use tessera_kernels as _;

use tessera_graph::DimName::{self, Context, Heads, Hidden, Rows, Seqs};
use tessera_graph::{DType, Dim, DimExpr, Graph, TensorType};
use tessera_interp::{Store, Tensor, run, run_kept};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM};
use tessera_lower::kept::{self, Layout};
use tessera_lower::{Device, Pins, Settings, View, lower};
use tessera_run::testing::runs;
use tessera_run::{Run, open};
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{EntryKind, Mask, Model, Norm, Pairs, Port, Role, Weights, declare};

const LATENT: DimName = DimName::named("Latent");
const ROPE: DimName = DimName::named("Rope");

#[derive(Weights)]
struct KvNorm {
    #[shape(LATENT)]
    norm: Norm,
}

fn draw(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

/// Rows for `(slot, first_pos, count)` runs.
fn make_rows(runs: &[(u32, u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let slots = runs.iter().flat_map(|(s, _, n)| (0..*n).map(move |_| *s));
    let pos = runs
        .iter()
        .flat_map(|(_, p, n)| (0..*n).map(move |i| p + i));
    (slots.collect(), pos.collect())
}

/// Single MLA KV-prep + scatter followed by ragged attention readback.
fn kv_graph(latent: u64, rope: u64, pairs: Pairs) -> Graph {
    let dims = vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 64),
        Dim::bounded(Context, 64),
        Dim::fixed(Heads, 2),
        Dim::fixed(LATENT, latent),
        Dim::fixed(ROPE, rope),
    ];
    let m = Model::new(dims);
    let w: KvNorm = declare(&m);
    let port = |role, ty| Port { role, ty };
    let ids = TensorType::new([Rows], DType::U32);
    let [q, kv, slots, positions] = m.entry(
        EntryKind::Step,
        [
            port(
                Role::Tokens,
                TensorType::new(vec![Rows.into(), Heads.into(), LATENT + ROPE], DType::F32),
            ),
            port(
                Role::Tokens,
                TensorType::new(vec![Rows.into(), LATENT + ROPE], DType::F32),
            ),
            port(Role::Slots, ids.clone()),
            port(Role::Positions, ids),
        ],
    );
    let kept = m.kept(TensorType::new(
        vec![
            Seqs.into(),
            Context.into(),
            DimExpr::Const(1),
            LATENT + ROPE,
        ],
        DType::F32,
    ));
    let lat = kv.slice(1, 0u64, latent).rms_norm(&w.norm, 1e-5);
    let shared = kv.slice(1, latent, rope).rope(&positions, 1e4, pairs);
    let row = lat
        .concat(&shared)
        .reshape(vec![Rows.into(), DimExpr::Const(1), LATENT + ROPE]);
    let k = row.append(&kept, [&slots, &positions]);
    let out = m.attend_rows(Mask::Causal, 0.25, [&q, &k, &k], [&slots, &positions], None);
    m.finish([(Role::Logits, &out)])
}

/// Runs one step, initializing a reversed page table when paged.
fn run_step(
    runner: &mut Option<Box<dyn Run>>,
    plan: tessera_lower::Plan,
    settings: &Settings,
    norm_w: &Tensor,
    inputs: &[Tensor],
) -> Tensor {
    let r = match runner {
        Some(r) => {
            r.replan(&plan).unwrap();
            r
        }
        None => {
            let table = plan.table;
            let r = runner.insert(open(plan, settings).unwrap());
            r.load(0, norm_w.view()).unwrap();
            if let Some(table) = table {
                let pages: Vec<u8> = (0..table.len as u32)
                    .rev()
                    .flat_map(u32::to_le_bytes)
                    .collect();
                r.held_mut().upload(table, &pages).unwrap();
            }
            r
        }
    };
    let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
    r.run(&views).unwrap().remove(0)
}

#[test]
fn kv_scatter_matches() {
    let mut draws = Noise(71);
    let layouts: [&'static dyn Layout; 4] = [
        kept::DENSE,
        kept::INTERLEAVED,
        kept::named("paged").unwrap(),
        &kept::PAGED16,
    ];
    let launches: [&[(u32, u32, u32)]; 3] = [
        &[(0, 0, 1)],
        &[(0, 1, 8), (1, 0, 5), (2, 0, 7)],
        &[(0, 9, 1), (1, 5, 1), (3, 0, 3)],
    ];
    let cases = [
        (32u64, 16u64, Pairs::Adjacent, F32_FORM, 1e-4f32, false),
        (32, 16, Pairs::Halves, F32_FORM, 1e-4, true),
        (16, 6, Pairs::Adjacent, F32_FORM, 1e-4, false),
        (32, 16, Pairs::Adjacent, BF16_FORM, 2e-2, false),
        (32, 16, Pairs::Adjacent, FP8_FORM, 0.25, false),
        (512, 64, Pairs::Adjacent, F32_FORM, 1e-4, false),
        (1152, 16, Pairs::Adjacent, F32_FORM, 1e-4, false),
    ];
    for (lat, rope, pairs, form, tol, force_plain) in cases {
        let g = kv_graph(lat, rope, pairs);
        let w = draw(vec![lat as usize], &mut draws);
        let pins = match form.name() {
            "f32" => Pins::default(),
            _ => Pins::default().activations(BF16_FORM).kept(form),
        };
        for &layout in &layouts {
            // Odd-rope and wide-latent cases run on dense layouts.
            if (rope == 6 && layout.name() != "dense" && layout.name() != "paged16")
                || (lat > 32 && layout.name() != "dense")
            {
                continue;
            }
            for settings in runs() {
                let (mut store, mut runner): (Store, Option<Box<dyn Run>>) =
                    (Store::default(), None);
                for spec in launches {
                    let (slots, positions) = make_rows(spec);
                    let n = slots.len();
                    let width = (lat + rope) as usize;
                    let furthest = *positions.iter().max().unwrap() as u64 + 1;
                    let inputs = [
                        draw(vec![n, 2, width], &mut draws),
                        draw(vec![n, width], &mut draws),
                        Tensor::u32(vec![n], slots),
                        Tensor::u32(vec![n], positions),
                    ];
                    let want =
                        run_kept(&g, std::slice::from_ref(&w), &mut store, 0, &inputs).unwrap();
                    let sizes = [(Seqs, 4), (Rows, n as u64), (Context, furthest)];
                    let mut device = Device::guess(&settings).pinned(pins.clone());
                    device.kept = layout;
                    if force_plain {
                        device = device.without("mla_kv_scatter");
                    }
                    let plan = lower(&g, 0, &sizes, &device).unwrap();
                    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                    let expect = if force_plain || rope == 6 {
                        "mla_kv_scatter_plain"
                    } else {
                        "mla_kv_scatter"
                    };
                    assert!(names.contains(&expect), "{names:?}");
                    assert!(!names.contains(&"scatter"), "{names:?}");
                    assert!(!names.contains(&"join"), "{names:?}");
                    assert!(!names.contains(&"strided"), "{names:?}");
                    assert!(plan.kept.iter().all(|k| k.1.form == form));
                    let got = run_step(&mut runner, plan, &settings, &w, &inputs);
                    let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                    assert!(
                        off < tol,
                        "{expect} {} {layout:?} {pairs:?}: off by {off:e}",
                        form.name()
                    );
                }
            }
        }
    }
}

/// Absorbed MLA Q-prep: permuted latent joined with sliced rotary part.
fn absorbed_q(heads: u64, latent: u64, nope: u64, rope: u64, pairs: Pairs) -> Graph {
    let dims = vec![
        Dim::bounded(Rows, 32),
        Dim::fixed(Heads, heads),
        Dim::fixed(LATENT, latent),
        Dim::fixed(Hidden, nope),
        Dim::fixed(ROPE, rope),
    ];
    let m = Model::new(dims);
    let port = |ty| Port {
        role: Role::Tokens,
        ty,
    };
    let [q_lat, q_raw, pos] = m.entry(
        EntryKind::Forward,
        [
            port(TensorType::new([Heads, Rows, LATENT], DType::F32)),
            port(TensorType::new(
                vec![Rows.into(), Heads.into(), Hidden + ROPE],
                DType::F32,
            )),
            Port {
                role: Role::Positions,
                ty: TensorType::new([Rows], DType::U32),
            },
        ],
    );
    let q_p = q_lat.permute(&[1, 0, 2]);
    let q_r = q_raw.slice(2, nope, rope).rope(&pos, 1e4, pairs);
    m.finish([(Role::Logits, &q_p.concat(&q_r))])
}

/// Whole-sequence partial RoPE, either trailing or leading rotary slice.
fn partial_q(heads: u64, nope: u64, rope: u64, rope_first: bool, pairs: Pairs) -> Graph {
    let dims = vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Context, 32),
        Dim::fixed(Heads, heads),
        Dim::fixed(Hidden, nope),
        Dim::fixed(ROPE, rope),
    ];
    let m = Model::new(dims);
    let [x, pos] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role: Role::Tokens,
                ty: TensorType::new(
                    vec![Seqs.into(), Context.into(), Heads.into(), Hidden + ROPE],
                    DType::F32,
                ),
            },
            Port {
                role: Role::Positions,
                ty: TensorType::new([Seqs, Context], DType::U32),
            },
        ],
    );
    let out = if rope_first {
        let front = x.slice(3, 0u64, rope).rope(&pos, 1e6, pairs);
        let back = x.slice(3, rope, nope);
        front.concat(&back)
    } else {
        let front = x.slice(3, 0u64, nope);
        let back = x.slice(3, nope, rope).rope(&pos, 1e4, pairs);
        front.concat(&back)
    };
    m.finish([(Role::Logits, &out)])
}

#[test]
fn join_rope_matches() {
    let mut draws = Noise(83);
    for (rows, pairs, form, tol) in [
        (1usize, Pairs::Adjacent, F32_FORM, 1e-4f32),
        (8, Pairs::Adjacent, F32_FORM, 1e-4),
        (8, Pairs::Halves, BF16_FORM, 2e-2),
    ] {
        let g = absorbed_q(4, 32, 16, 16, pairs);
        let inputs = [
            draw(vec![4, rows, 32], &mut draws),
            draw(vec![rows, 4, 32], &mut draws),
            Tensor::u32(vec![rows], (0..rows as u32).map(|i| i + 3).collect()),
        ];
        let want = run(&g, &[], 0, &inputs).unwrap();
        let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
        let pins = match form.name() {
            "f32" => Pins::default(),
            _ => Pins::default().activations(form),
        };
        for settings in runs() {
            let device = Device::guess(&settings).pinned(pins.clone());
            let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert_eq!(names, ["join_rope"], "rows={rows} {pairs:?}");
            let mut runner = open(plan, &settings).unwrap();
            let got = runner.run(&views).unwrap().remove(0);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(off < tol, "absorbed rows={rows}: off by {off:e}");
        }
    }
    for (rope_first, pairs, rope) in [
        (false, Pairs::Adjacent, 16u64),
        (true, Pairs::Halves, 16),
        (true, Pairs::Adjacent, 6),
    ] {
        let g = partial_q(4, 24, rope, rope_first, pairs);
        let (seqs, ctx) = (2usize, 5usize);
        let inputs = [
            draw(vec![seqs, ctx, 4, (24 + rope) as usize], &mut draws),
            Tensor::u32(
                vec![seqs, ctx],
                (0..(seqs * ctx) as u32).map(|i| i % ctx as u32).collect(),
            ),
        ];
        let want = run(&g, &[], 0, &inputs).unwrap();
        let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
        for settings in runs() {
            let device = Device::guess(&settings);
            let plan = lower(
                &g,
                0,
                &[(Seqs, seqs as u64), (Context, ctx as u64)],
                &device,
            )
            .unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert_eq!(names, ["join_rope"], "rope_first={rope_first} rope={rope}");
            let mut runner = open(plan, &settings).unwrap();
            let got = runner.run(&views).unwrap().remove(0);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(
                off < 1e-4,
                "partial rope_first={rope_first}: off by {off:e}"
            );
        }
    }
}
