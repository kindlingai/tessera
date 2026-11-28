//! Which launch takes picked trellis matmuls on each backend.

// The other matmul rules come from here.
use tessera_kernels as _;
use tessera_linear as _;
use tessera_metal_linear as _;

use tessera_codegen::trellis::{INPUT_SIDE, alloc_mixed, packed};
use tessera_gpukit::platform::{Asked, Platform};
use tessera_graph::Act;
use tessera_graph::DimName::{Experts, Ffn, Hidden, Rows, TopK};
use tessera_interp::{Tensor, run};
use tessera_lower::form::named;
use tessera_lower::{Device, Pins, Plan, Source, Weight, lower};
use tessera_run::open;
use tessera_run::testing::{host, runs};
use tessera_testkit::{Noise, cosine_f32};
use tessera_trace::{
    DType, Dim, DimExpr, Embed, EntryKind, Graph, Linears, Model, Port, Role, TensorType, Vector,
    Weights,
};

const RULE: &str = "picked_trellis_metal";
const MIXED: &str = "tm_pickmix_pmcgx_t";
const GENERAL: &str = "tm_picked_gen";
const INP: usize = 256;
const OUT: usize = 128;
const COUNT: usize = 4;

#[derive(Weights)]
struct Set {
    #[shape(Experts, Hidden, Ffn)]
    codes: Linears,
    #[shape(Experts, Hidden)]
    su: Embed,
    #[shape(Experts, Ffn)]
    sv: Embed,
    #[shape(Ffn)]
    signs: Vector,
}

/// Rows on picked experts. `post`: 0, 1 signs, 2 gate.
fn picked_graph(rotated: bool, post: u32, repeat: usize) -> Graph {
    let mut dims = vec![
        Dim::bounded(Rows, 256),
        Dim::fixed(Hidden, INP as u64),
        Dim::fixed(Ffn, OUT as u64),
        Dim::fixed(Experts, COUNT as u64),
    ];
    if repeat > 1 {
        dims.push(Dim::fixed(TopK, repeat as u64));
    }
    let m = Model::new(dims);
    let set: Set = tessera_trace::declare(&m);
    let role = Role::Logits;
    let x_ty = TensorType::new([Rows, Hidden], DType::F32);
    let id_ty = if repeat > 1 {
        TensorType::new([Rows, TopK], DType::U32)
    } else {
        TensorType::new([Rows], DType::U32)
    };
    let gate_ty = if repeat > 1 {
        TensorType::new([Rows, TopK, Ffn], DType::F32)
    } else {
        TensorType::new([Rows, Ffn], DType::F32)
    };
    let (x, ids, extra) = if post == 2 {
        let [x, ids, g] = m.entry(
            EntryKind::Forward,
            [
                Port { role, ty: x_ty },
                Port { role, ty: id_ty },
                Port { role, ty: gate_ty },
            ],
        );
        (x, ids, Some(g))
    } else {
        let [x, ids] = m.entry(
            EntryKind::Forward,
            [Port { role, ty: x_ty }, Port { role, ty: id_ty }],
        );
        (x, ids, None)
    };
    let x_in = if repeat > 1 {
        x.reshape(vec![Rows.into(), DimExpr::Const(1), Hidden.into()])
            .expand(1, TopK)
    } else {
        x
    };
    let codes = m.pick(&set.codes, &ids);
    let mut out = match rotated {
        true => {
            let turned = x_in.mul(&m.embed(&ids, &set.su)).hadamard(128);
            let product = turned.matmul_rows(&codes).hadamard(128);
            product.mul(&m.embed(&ids, &set.sv))
        }
        false => x_in.matmul_rows(&codes),
    };
    match post {
        1 => out = out.mul_vector(&set.signs),
        2 => out = out.act(Act::Silu).mul(extra.as_ref().unwrap()),
        _ => {}
    }
    m.finish([(Role::Logits, &out)])
}

/// Rows on picked experts. `post` scales rotated rows by signs.
fn picked(rotated: bool, post: bool) -> Graph {
    picked_graph(rotated, u32::from(post), 1)
}

fn platform(backend: &'static str, arch: &str) -> Platform {
    Platform::new(Asked {
        backend,
        arch: arch.into(),
        name: arch.into(),
        driver: String::new(),
        limits: tessera_gpukit::Limits::assumed(),
        units: 0,
        bandwidth: 0.0,
        integrated: true,
        memory: 0,
    })
}

fn plan_rows(g: &Graph, form: &str, rows: u64, on: Platform) -> Plan {
    let pins = Pins::default().pin("codes", named(form).unwrap());
    let device = Device::guess(&host()).on(on).pinned(pins);
    lower(g, 0, &[(Rows, rows)], &device).unwrap()
}

fn plan(g: &Graph, form: &str, on: Platform) -> Plan {
    plan_rows(g, form, 8, on)
}

/// The entry of each launch the Metal rule took.
fn entries(plan: &Plan) -> Vec<&'static str> {
    let taken = plan.steps.iter().filter(|s| s.rule == RULE);
    let entry = |s: &tessera_lower::Step| match s.kernel.source() {
        Source::Fixed { entry, .. } => entry,
        Source::Made(_) => panic!("a fixed kernel"),
    };
    taken.map(entry).collect()
}

fn apple() -> Platform {
    platform("metal", "apple9")
}

#[test]
fn mixed_takes_fast_entry() {
    for post in [false, true] {
        let plan = plan(&picked(true, post), "trellis_mix_pmcgx", apple());
        assert_eq!(entries(&plan), [MIXED], "post {post}");
        // The signs ride in the same launch.
        assert_eq!(plan.steps.len(), 1, "post {post}");
        let args = plan.steps[0].kernel.launches(&plan.steps[0]).remove(0).args;
        assert_eq!(args[13].u32(), Some(0));
        assert_eq!(args[16].u32(), Some(u32::from(post)));
    }
}

#[test]
fn uniform_takes_general_entry() {
    for rotated in [false, true] {
        let plan = plan(&picked(rotated, false), "trellis_k3_pmcgx", apple());
        assert_eq!(entries(&plan), [GENERAL], "rotated {rotated}");
        let args = plan.steps[0].kernel.launches(&plan.steps[0]).remove(0).args;
        assert_eq!(args[13].u32(), Some(3));
        assert_eq!(args[15].u32(), Some(u32::from(rotated)));
    }
    let plain = plan(&picked(false, false), "trellis_mix_pmcgx", apple());
    assert_eq!(entries(&plain), [GENERAL]);
}

#[test]
fn prefill_takes_mma() {
    let pf16 = plan_rows(&picked(true, true), "trellis_mix_pmcgx", 64, apple());
    assert_eq!(entries(&pf16), ["tm_pick_group", "tm_pickmix_pf16_pmcgx_t"]);
    let pf32 = plan_rows(&picked(true, false), "trellis_mix_pmcgx", 80, apple());
    assert_eq!(entries(&pf32), ["tm_pick_group", "tm_pickmix_pf32_pmcgx_t"]);
    let wide = plan_rows(&picked(true, false), "trellis_mix_pmcgx", 256, apple());
    assert_eq!(entries(&wide), ["tm_pick_group", "tm_pickmix_pf32_pmcgx_t"]);
    let needs = wide.steps[1].kernel.needs();
    assert_eq!((needs.group_threads, needs.group_bytes), (128, 12416));
    let plain = plan_rows(&picked(false, false), "trellis_k3_pmcgx", 80, apple());
    assert_eq!(entries(&plain), ["tm_pick_group", "tm_pick_pf32_gen"]);
}

#[test]
fn other_backends_skip() {
    for (backend, arch) in [("cuda", "sm_121"), ("rocm", "gfx1151")] {
        for form in ["trellis_mix_pmcgx", "trellis_k3_pmcgx"] {
            let plan = plan(&picked(true, true), form, platform(backend, arch));
            assert!(entries(&plan).is_empty(), "{backend} {form}");
        }
    }
}

fn fill_f32(shape: Vec<usize>, draws: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| draws.draw()).collect();
    Tensor::f32(shape, data)
}

/// Packs `COUNT` expert matrices, mixed or uniform.
fn pack_experts(side: u32, mixed: bool, draws: &mut Noise) -> (&'static str, Vec<u8>) {
    if mixed {
        let widths = [2u32, 3, 4, 2];
        let (mut bytes, at) = alloc_mixed(INP, OUT, side, &widths);
        for (k, pos) in widths.into_iter().zip(at) {
            let len = INP * OUT * k as usize / 8;
            for b in &mut bytes[pos..pos + len] {
                *b = (draws.draw() * 256.0) as u8;
            }
        }
        return ("trellis_mix_pmcgx", bytes);
    }
    let len = COUNT * OUT * INP * 3 / 8;
    let codes: Vec<u8> = (0..len).map(|_| (draws.draw() * 256.0) as u8).collect();
    ("trellis_k3_pmcgx", packed(INP, OUT, side, &codes))
}

#[test]
fn metal_picked_matches_cpu() {
    let Some(settings) = runs().into_iter().find(|s| s.gpu().is_some()) else {
        return;
    };
    let mut draws = Noise(53);
    let su = fill_f32(vec![COUNT, INP], &mut draws);
    let sv = fill_f32(vec![COUNT, OUT], &mut draws);
    let signs = fill_f32(vec![OUT], &mut draws);
    for side in [1u32, INPUT_SIDE] {
        for (mixed, rotated, post, repeat, rows, skew, want_entries) in [
            (true, true, 0, 1, 5usize, false, vec![MIXED]),
            (true, true, 1, 1, 5, false, vec![MIXED]),
            (true, true, 2, 2, 5, false, vec![MIXED]),
            (true, false, 0, 1, 5, false, vec![GENERAL]),
            (false, true, 1, 2, 5, false, vec![GENERAL]),
            (false, false, 0, 1, 5, false, vec![GENERAL]),
            (
                true,
                true,
                0,
                1,
                64,
                false,
                vec!["tm_pick_group", "tm_pickmix_pf16_pmcgx_t"],
            ),
            // Skewed counts force short tail tiles.
            (
                true,
                true,
                0,
                1,
                64,
                true,
                vec!["tm_pick_group", "tm_pickmix_pf16_pmcgx_t"],
            ),
            (
                true,
                true,
                1,
                2,
                40,
                false,
                vec!["tm_pick_group", "tm_pickmix_pf32_pmcgx_t"],
            ),
            (
                true,
                true,
                2,
                2,
                40,
                false,
                vec!["tm_pick_group", "tm_pickmix_pf32_pmcgx_t"],
            ),
            (
                true,
                true,
                1,
                2,
                128,
                false,
                vec!["tm_pick_group", "tm_pickmix_pf32_pmcgx_t"],
            ),
            (
                false,
                false,
                0,
                1,
                80,
                false,
                vec!["tm_pick_group", "tm_pick_pf32_gen"],
            ),
        ] {
            let (form_name, bytes) = pack_experts(side, mixed, &mut draws);
            let form = named(form_name).unwrap();
            let g = picked_graph(rotated, post, repeat);
            let x = fill_f32(vec![rows, INP], &mut draws);
            let id_vals: Vec<u32> = (0..rows * repeat)
                .map(|i| {
                    if skew {
                        match i {
                            0..37 => 0,
                            37..56 => 1,
                            56..61 => 2,
                            _ => 3,
                        }
                    } else {
                        ((i * 3 + 1) % COUNT) as u32
                    }
                })
                .collect();
            let id_shape = if repeat > 1 {
                vec![rows, repeat]
            } else {
                vec![rows]
            };
            let ids = Tensor::u32(id_shape, id_vals);
            let extra = fill_f32(
                if repeat > 1 {
                    vec![rows, repeat, OUT]
                } else {
                    vec![rows, OUT]
                },
                &mut draws,
            );
            let mut ins = vec![x.clone(), ids.clone()];
            if post == 2 {
                ins.push(extra.clone());
            }
            let unpacked = Tensor::f32(vec![COUNT, OUT, INP], form.unpack_all(&bytes));
            let weights = [unpacked, su.clone(), sv.clone(), signs.clone()];
            let want = run(&g, &weights, 0, &ins).unwrap();
            let pins = Pins::default().pin("codes", form);
            let device = Device::guess(&settings).pinned(pins);
            let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
            if device.platform.backend == "metal" {
                assert_eq!(
                    entries(&plan),
                    want_entries,
                    "side={side} mixed={mixed} rot={rotated} post={post} rep={repeat} rows={rows}"
                );
            }
            let mut runner = open(plan, &settings).unwrap();
            runner
                .load_weight(0, &Weight::Packed(form, bytes.clone()))
                .unwrap();
            runner.load_weight(1, &Weight::Floats(su.clone())).unwrap();
            runner.load_weight(2, &Weight::Floats(sv.clone())).unwrap();
            runner
                .load_weight(3, &Weight::Floats(signs.clone()))
                .unwrap();
            let views: Vec<_> = ins.iter().map(Tensor::view).collect();
            let got = runner.run(&views).unwrap().remove(0);
            let cos = cosine_f32(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(
                cos > 0.99999,
                "side={side} mixed={mixed} rot={rotated} post={post} rep={repeat} rows={rows}: cos {cos}"
            );
        }
    }
}
