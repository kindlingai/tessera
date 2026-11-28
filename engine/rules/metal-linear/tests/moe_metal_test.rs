//! Native Metal coalesced picked MoE matmul against the CPU interpreter.

use tessera_kernels as _;
use tessera_linear as _;
use tessera_metal_linear as _;

use tessera_graph::DimName::{Experts, Ffn, Hidden, Rows, Seqs, TopK};
use tessera_graph::{DType, Dim, EntryKind, Graph, Role, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::named;
use tessera_lower::{Device, Pins, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, cosine_f32, fill, graphs, max_abs_diff, weights};
use tessera_trace::{Linears, Model, Port, Weights};

const FORMS: [&str; 8] = [
    "f32", "bf16", "f16", "fp8", "mxfp8", "q8_0", "int4_g32", "q4_0",
];

#[derive(Weights)]
struct Bank {
    #[shape(Experts, Hidden, Ffn)]
    w: Linears,
}

/// Bare picked matmul, optionally slicing `offset..offset + slice_n`.
fn bare_picked(experts: u64, k: u64, stride: u64, slice: Option<(u64, u64)>, top_k: u64) -> Graph {
    let mut dims = vec![
        Dim::bounded(Rows, 64),
        Dim::fixed(Experts, experts),
        Dim::fixed(Hidden, k),
        Dim::fixed(Ffn, stride),
    ];
    if top_k > 1 {
        dims.push(Dim::fixed(TopK, top_k));
    }
    let m = Model::new(dims);
    let bank: Bank = tessera_trace::declare(&m);
    let x_ty = TensorType::new([Rows, Hidden], DType::F32);
    let id_ty = if top_k > 1 {
        TensorType::new([Rows, TopK], DType::U32)
    } else {
        TensorType::new([Rows], DType::U32)
    };
    let [x, ids] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role: Role::Logits,
                ty: x_ty,
            },
            Port {
                role: Role::Logits,
                ty: id_ty,
            },
        ],
    );
    let picked = match slice {
        Some((off, len)) => m.pick(&bank.w.rows(off, len), &ids),
        None => m.pick(&bank.w, &ids),
    };
    let out = x.matmul_rows(&picked);
    m.finish([(Role::Logits, &out)])
}

#[test]
fn routed_matches_cpu() {
    let sizes = [(Seqs, 2), (Rows, 7)];
    let bounded = vec![Dim::bounded(Seqs, 4), Dim::bounded(Rows, 16)];
    let g = graphs::routed(&bounded, [4, 64, 96]);
    let mut rng = Noise(41);
    let drawn = weights(&g, &mut rng);
    let input = fill(&g, &g.entries[0].inputs[0].ty, &sizes, &mut rng).unwrap();
    for settings in runs() {
        let device = Device::guess(&settings);
        if device.platform.backend != "metal" {
            continue;
        }
        for name in FORMS {
            let form = named(name).unwrap();
            let dev = device.clone().pinned(Pins::all(form));
            let kept: Vec<Tensor> = drawn
                .iter()
                .map(|t| Tensor::f32(t.shape.clone(), form.roundtrip(t.floats().unwrap())))
                .collect();
            let want = run(&g, &kept, 0, std::slice::from_ref(&input)).unwrap();
            let plan = lower(&g, 0, &sizes, &dev).unwrap();
            let moe_steps = plan
                .steps
                .iter()
                .filter(|s| s.kernel.name().starts_with("moe_metal"))
                .count();
            assert_eq!(moe_steps, 2, "{name}: expected 2 moe_metal steps");
            let mut runner = open(plan, &settings).unwrap();
            for (at, w) in kept.iter().enumerate() {
                runner.load(at as u32, w.view()).unwrap();
            }
            let got = runner.run(&[input.view()]).unwrap().remove(0);
            let (got, want) = (got.floats().unwrap(), want[0].floats().unwrap());
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
            let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
            assert!(off < 1e-3 && cos > 0.9999, "{name}: off {off:e} cos {cos}");
        }
    }
}

#[test]
fn edges_match_cpu() {
    let mut rng = Noise(73);
    // Odd `k`, sliced columns, repeat, and out-of-range ids.
    let cases = [
        ("f32", 37u64, 50u64, Some((7u64, 19u64)), 3u64),
        ("bf16", 65, 48, Some((5, 31)), 2),
        ("f16", 3, 20, None, 1),
        ("fp8", 131, 35, Some((3, 17)), 2),
        ("mxfp8", 64, 48, Some((8, 25)), 2),
        ("q8_0", 96, 48, Some((4, 33)), 3),
        ("int4_g32", 64, 40, Some((6, 19)), 2),
        ("q4_0", 128, 36, None, 2),
    ];
    for settings in runs() {
        let device = Device::guess(&settings);
        if device.platform.backend != "metal" {
            continue;
        }
        for (name, k, stride, slice, top_k) in cases {
            let experts = 4u64;
            let rows = 5u64;
            let g = bare_picked(experts, k, stride, slice, top_k);
            let form = named(name).unwrap();
            let raw_w = weights(&g, &mut rng);
            let kept: Vec<Tensor> = raw_w
                .iter()
                .map(|t| Tensor::f32(t.shape.clone(), form.roundtrip(t.floats().unwrap())))
                .collect();
            let x = fill(&g, &g.entries[0].inputs[0].ty, &[(Rows, rows)], &mut rng).unwrap();
            let id_count = (rows * top_k) as usize;
            let id_vals: Vec<u32> = (0..id_count)
                .map(|i| match i % 5 {
                    4 => (experts + 2) as u32,
                    v => v as u32,
                })
                .collect();
            let id_shape = if top_k > 1 {
                vec![rows as usize, top_k as usize]
            } else {
                vec![rows as usize]
            };
            let ids = Tensor::u32(id_shape, id_vals);
            let want = run(&g, &kept, 0, &[x.clone(), ids.clone()]).unwrap();
            let dev = device.clone().pinned(Pins::all(form));
            let plan = lower(&g, 0, &[(Rows, rows)], &dev).unwrap();
            assert!(
                plan.steps
                    .iter()
                    .any(|s| s.kernel.name().starts_with("moe_metal")),
                "{name}: expected moe_metal"
            );
            let mut runner = open(plan, &settings).unwrap();
            runner.load(0, kept[0].view()).unwrap();
            let got = runner.run(&[x.view(), ids.view()]).unwrap().remove(0);
            let (got, want) = (got.floats().unwrap(), want[0].floats().unwrap());
            let off = max_abs_diff(got, want);
            assert!(off < 1e-3, "{name} k={k}: off {off:e}");
        }
    }
}
