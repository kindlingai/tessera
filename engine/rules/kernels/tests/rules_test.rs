//! Each rule's plans, and its launches against the CPU interpreter.

use tessera_attention as _;
use tessera_kernels as _;
use tessera_linear as _;

use tessera_graph::DimName::{Context, HeadDim, Heads, Rows, Seqs};
use tessera_graph::{DType, Dim, DimName, Graph, NodeId, verify};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM};
use tessera_lower::{Device, Limits, Pins, Plan, Platform, Settings, View, cut, lower, rules};
use tessera_run::open;
use tessera_run::testing::{host, lockstep, runs};
use tessera_testkit::graphs::{self, floats, ids};
use tessera_testkit::{Noise, fill, max_abs_diff};
use tessera_trace::{Act, EntryKind, Model, Pairs, Role};

const SIZES: [(DimName, u64); 3] = [(Seqs, 2), (Rows, 3), (Context, 5)];

/// The bounds every shared graph here is built with.
fn dims() -> Vec<Dim> {
    vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::bounded(Context, 16),
    ]
}

/// Entry 0 against the interpreter. Returns the largest difference.
fn gap(g: &Graph, inputs: &[Tensor], device: &Device, settings: &Settings) -> f32 {
    let want = run(g, &[], 0, inputs).unwrap();
    let plan = || lower(g, 0, &SIZES, device).unwrap();
    let mut runners = vec![open(plan(), settings).unwrap()];
    // The host also runs each tree in lockstep.
    if settings.gpu().is_none() {
        runners.push(lockstep(plan(), settings).unwrap());
    }
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let mut worst = 0.0f32;
    for runner in &mut runners {
        let got = runner.run(&feeds).unwrap().remove(0);
        worst = worst.max(match want[0].dtype() {
            // Indices must match exactly.
            DType::U32 if got.indices().unwrap() == want[0].indices().unwrap() => 0.0,
            DType::U32 => f32::INFINITY,
            _ => max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap()),
        });
    }
    worst
}

/// The rules this crate registers.
fn ours() -> Vec<&'static tessera_lower::Rule> {
    let ours = rules().into_iter();
    ours.filter(|r| r.module.starts_with("tessera_kernels::"))
        .collect()
}

/// Residual add, center, RMS norm and bias.
fn layer_norm_graph(d: &[Dim], width: u64) -> Graph {
    use tessera_graph::DimName::Hidden;
    use tessera_trace::{Norm, Vector, Weights, declare};
    #[derive(Weights)]
    struct LnW {
        #[shape(Hidden)]
        w: Norm,
        #[shape(Hidden)]
        b: Vector,
    }
    let mut dims = d.to_vec();
    dims.push(Dim::fixed(Hidden, width));
    let m = Model::new(dims);
    let lw: LnW = declare(&m);
    let [a, b] = m.entry(
        EntryKind::Forward,
        [floats([Seqs, Rows, Hidden]), floats([Seqs, Rows, Hidden])],
    );
    let sum = a.add(&b);
    let ln = sum.center().rms_norm(&lw.w, 1e-6).add_vector(&lw.b);
    m.finish([(Role::Logits, &ln.add(&sum))])
}

/// Every rule's launches under `settings`. Returns the rules used.
fn rules_match(settings: &Settings) -> Vec<&'static str> {
    let d = dims();
    // Rows of 600: more entries than threads for the argmax.
    let graphs = [
        graphs::rows(&d, 32),
        graphs::rows(&d, 600),
        graphs::attention(&d, [2, 2, 16]),
        graphs::rotation(&d),
        graphs::partial_rope(&d, 32),
        graphs::table(&d, 32, 16),
        graphs::neighbours(&d, [6, 9, 16]),
        graphs::picks(&d, 16),
        graphs::indexed(&d, [2, 16, 2]),
        graphs::indexed_rows(&d, [2, 16, 2]),
        graphs::residual_norm(&d, 32),
        graphs::residual_norm(&d, 600),
        clipped(),
        layer_norm_graph(&d, 32),
        layer_norm_graph(&d, 1024),
    ];
    let ours = ours();
    let mut used: Vec<&str> = Vec::new();
    for (g, rule) in graphs.iter().flat_map(|g| ours.iter().map(move |r| (g, r))) {
        let mut noise = Noise(7);
        let device = Device::guess(settings).favoring(rule.name);
        let plan = lower(g, 0, &SIZES, &device).unwrap();
        let mut seen: Vec<&[NodeId]> = Vec::new();
        for step in &plan.steps {
            used.extend((step.rule == rule.name).then_some(rule.name));
            if seen.contains(&&step.covers[..]) {
                continue;
            }
            seen.push(&step.covers);
            // The nodes one rule covered, as their own graph.
            let sub = cut(g, &step.covers);
            verify(&sub).unwrap();
            let ports = &sub.entries[0].inputs;
            let inputs: Vec<Tensor> = ports
                .iter()
                .map(|p| fill(&sub, &p.ty, &SIZES, &mut noise).unwrap())
                .collect();
            let off = gap(&sub, &inputs, &device, settings);
            assert!(off < 1e-4, "{:?} is off by {off:e}", step.kernel);
        }
    }
    used
}

#[test]
fn rules_match_cpu() {
    for settings in runs() {
        let used = rules_match(&settings);
        // A new rule needs a graph in `testkit::graphs`.
        for rule in ours() {
            let name = rule.name;
            assert!(used.contains(&name), "{name} is never exercised");
        }
    }
}

/// Rows clipped to length 1, then scaled by position.
fn clipped() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(HeadDim, 16),
    ]);
    let ports = [floats([Seqs, Rows, HeadDim]), ids([Seqs, Rows])];
    let [x, at] = m.entry(EntryKind::Forward, ports);
    let out = x.l2_clip(1.0, 1e-3).log_scale(&at, 4.0, 0.5);
    m.finish([(Role::Logits, &out)])
}

#[test]
fn clip_scale_match() {
    // Row lengths 0 to 2.5. Positions straddle 4.
    let x: Vec<f32> = (0..96)
        .map(|i| (i / 16) as f32 * if i % 2 == 0 { 0.125 } else { -0.125 })
        .collect();
    let inputs = [
        Tensor::f32(vec![2, 3, 16], x),
        Tensor::u32(vec![2, 3], vec![0, 1, 3, 4, 9, 100]),
    ];
    let g = clipped();
    let want = run(&g, &[], 0, &inputs).unwrap();
    let tau = 1.0 + 0.5 * (101f32 / 4.0).ln();
    let last = want[0].floats().unwrap()[95];
    assert!((last + tau / (2.5 + 1e-3) * 0.625).abs() < 1e-6, "{last}");
    for settings in runs() {
        let off = gap(&g, &inputs, &Device::guess(&settings), &settings);
        assert!(off < 1e-6, "off by {off:e}");
    }
}

/// Two rotations by one set of positions, then a map.
fn rotations() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Heads, 2),
        Dim::fixed(HeadDim, 16),
    ]);
    let port = || floats([Seqs, Rows, Heads, HeadDim]);
    let [x, y, at] = m.entry(EntryKind::Forward, [port(), port(), ids([Seqs, Rows])]);
    let turn = |t: &tessera_trace::Tensor| t.rope(&at, 1e4, Pairs::Halves);
    m.finish([(Role::Logits, &turn(&x).add(&turn(&y)).act(Act::Silu))])
}

fn count(plan: &Plan, kernel: &str) -> usize {
    let named = |s: &&tessera_lower::Step| s.kernel.name() == kernel;
    plan.steps.iter().filter(named).count()
}

fn names(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.kernel.name()).collect()
}

#[test]
fn rotations_share_angles() {
    let plan = lower(&rotations(), 0, &SIZES, &Device::guess(&host())).unwrap();
    assert_eq!((count(&plan, "angles"), count(&plan, "rope")), (1, 2));
}

#[test]
fn favoring_stays_local() {
    // Rope has no competing rule, so favoring it changes nothing.
    let g = rotations();
    let plain = lower(&g, 0, &SIZES, &Device::guess(&host())).unwrap();
    let favored = Device::guess(&host()).favoring("rope");
    assert!(
        plain == lower(&g, 0, &SIZES, &favored).unwrap(),
        "a favored rule skewed its readers' choice"
    );
}

#[test]
fn trips_are_counted() {
    // Walking a row opens a line every sixteen floats.
    let norm = tessera_kernels::RmsNorm {
        rows: 4,
        width: 64,
        eps: 1e-5,
        weighted: true,
        w_form: tessera_lower::form::F32_FORM,
    };
    let trip = tessera_lower::Kernel::load(&norm, 64).unwrap().trip;
    assert_eq!((trip.loads, trip.stores), (3.0, 1.0));
    assert!((trip.fresh - 1.0 / 16.0).abs() < 1e-9, "{trip:?}");
    assert_eq!(trip.special, 0.0, "the root lies outside the loops");
}

#[test]
fn limits_bar_kernels() {
    let g = graphs::rows(&dims(), 32);
    let guess = Device::guess(&host()).favoring("lane_norm");
    assert!(names(&lower(&g, 0, &SIZES, &guess).unwrap()).contains(&"lane_norm"));
    // A 64-lane device runs none of the 32-lane kernels.
    let wide = Limits {
        lanes: 64,
        ..Limits::assumed()
    };
    let narrow = guess.on(Platform::assumed(wide));
    let picked = names(&lower(&g, 0, &SIZES, &narrow).unwrap());
    assert!(picked.contains(&"rms_norm") && !picked.contains(&"lane_norm"));
    // An uncalibrated GPU picks lane_norm on wide rows.
    let gpu = Platform {
        backend: "cuda".into(),
        name: "Test GPU".into(),
        units: 128,
        bandwidth: 1e12,
        ..Platform::assumed(Limits::assumed())
    };
    let wide_rows = graphs::rows(&dims(), 600);
    let plan = lower(&wide_rows, 0, &SIZES, &Device::default().on(gpu)).unwrap();
    assert!(names(&plan).contains(&"lane_norm"));
}

#[test]
fn cast_wide_narrow_match() {
    use tessera_graph::DimName::Hidden;
    use tessera_graph::TensorType;
    use tessera_lower::form::{FP8_FORM, Load};
    use tessera_trace::{Act, Port};
    let forms: [(&'static dyn Load, f32); 3] =
        [(F32_FORM, 1e-5), (BF16_FORM, 2.0), (FP8_FORM, 4.0)];
    for width in [8u64, 5u64] {
        let m = Model::new(vec![Dim::bounded(Rows, 4), Dim::fixed(Hidden, width)]);
        let [x] = m.entry(
            EntryKind::Forward,
            [Port {
                role: Role::Logits,
                ty: TensorType::new([Rows, Hidden], DType::F32),
            }],
        );
        let g_u32 = m.finish([(Role::Tokens, &x.act(Act::Relu).cast(DType::U32))]);
        let vals: Vec<f32> = (0..3 * width as usize)
            .map(|i| match i % 4 {
                0 => 0.0,
                _ => (i as f32) * 1.5,
            })
            .collect();
        let inp_f = Tensor::f32(vec![3, width as usize], vals);
        let want_u = run(&g_u32, &[], 0, std::slice::from_ref(&inp_f)).unwrap();
        let m2 = Model::new(vec![Dim::bounded(Rows, 4), Dim::fixed(Hidden, width)]);
        let [u] = m2.entry(
            EntryKind::Forward,
            [Port {
                role: Role::Tokens,
                ty: TensorType::new([Rows, Hidden], DType::U32),
            }],
        );
        let g_f32 = m2.finish([(Role::Logits, &u.cast(DType::F32))]);
        let inp_u = Tensor::u32(
            vec![3, width as usize],
            (0..3 * width as u32).map(|i| i * 3).collect(),
        );
        let want_f = run(&g_f32, &[], 0, std::slice::from_ref(&inp_u)).unwrap();
        for (form, tol) in forms {
            let pins = Pins::default().activations(form).outputs(form);
            for settings in runs() {
                let dev = Device::guess(&settings).pinned(pins.clone());
                let p_u = lower(&g_u32, 0, &[(Rows, 3)], &dev).unwrap();
                let mut r_u = open(p_u, &settings).unwrap();
                let got_u = r_u.run(&[inp_f.view()]).unwrap();
                if form.name() == "f32" {
                    assert_eq!(got_u[0].indices().unwrap(), want_u[0].indices().unwrap());
                }
                let p_f = lower(&g_f32, 0, &[(Rows, 3)], &dev).unwrap();
                assert_eq!(p_f.steps.len(), 1, "direct narrow cast_f32");
                let tessera_lower::Source::Made(body) = p_f.steps[0].kernel.source() else {
                    panic!("expected Made source");
                };
                assert_eq!(body.entry.contains("_x4"), width.is_multiple_of(4));
                let mut r_f = open(p_f, &settings).unwrap();
                let got_f = r_f.run(&[inp_u.view()]).unwrap();
                let diff = max_abs_diff(got_f[0].floats().unwrap(), want_f[0].floats().unwrap());
                assert!(diff <= tol, "{} w={width}: {diff:e}", body.entry);
            }
        }
    }
}

#[test]
fn tables_keep_forms() {
    let g = graphs::table(&dims(), 32, 16);
    let guess = Device::guess(&host());
    let narrow = guess.clone().pinned(Pins::all(BF16_FORM));
    assert!(names(&lower(&g, 0, &SIZES, &narrow).unwrap()).contains(&"gather_bf16"));
    // A table pinned f32 stays f32.
    let pins = Pins::all(BF16_FORM).pin("embed", F32_FORM);
    let plan = lower(&g, 0, &SIZES, &guess.pinned(pins)).unwrap();
    assert_eq!(names(&plan), ["gather"]);
}

#[test]
fn wide_kept_join_matches() {
    use tessera_graph::DimName::{Context, HeadDim};
    use tessera_graph::{Size, TensorType};
    use tessera_interp::{Store, run_kept};
    use tessera_lower::form::{FP8_FORM, Load};
    use tessera_lower::kept::{self, Layout};
    use tessera_trace::Port;
    let right_dim = tessera_graph::DimName::named("Right");
    let build = |left: u64, right: u64, both: bool| {
        let m = Model::new(vec![
            Dim::bounded(Seqs, 2),
            Dim::bounded(Rows, 64),
            Dim::bounded(Context, 32),
            Dim::fixed(HeadDim, left),
            Dim {
                name: right_dim,
                size: Size::Fixed(right),
            },
        ]);
        let b_ty = if both {
            TensorType::new([Rows, right_dim], DType::F32)
        } else {
            TensorType::new([Seqs, Context, right_dim], DType::F32)
        };
        let [a_in, b_in, slots, pos] = m.entry(
            EntryKind::Step,
            [
                Port {
                    role: Role::Tokens,
                    ty: TensorType::new([Rows, HeadDim], DType::F32),
                },
                Port {
                    role: Role::Tokens,
                    ty: b_ty,
                },
                Port {
                    role: Role::Slots,
                    ty: TensorType::new([Rows], DType::U32),
                },
                Port {
                    role: Role::Positions,
                    ty: TensorType::new([Rows], DType::U32),
                },
            ],
        );
        let ka = m.kept(TensorType::new([Seqs, Context, HeadDim], DType::F32));
        let a = a_in.append(&ka, [&slots, &pos]);
        let b = if both {
            let kb = m.kept(TensorType::new([Seqs, Context, right_dim], DType::F32));
            b_in.append(&kb, [&slots, &pos])
        } else {
            b_in
        };
        m.finish([(Role::Logits, &a.concat(&b))])
    };
    let mut noise = Noise(42);
    for (left, right, both, expect_x4) in [
        (16u64, 8u64, true, true),
        (12, 8, false, true),
        (7, 5, true, false),
    ] {
        let g = build(left, right, both);
        let a = Tensor::f32(
            vec![64, left as usize],
            (0..64 * left as usize).map(|_| noise.draw()).collect(),
        );
        let b_shape = if both {
            vec![64, right as usize]
        } else {
            vec![2, 32, right as usize]
        };
        let b = Tensor::f32(
            b_shape,
            (0..64 * right as usize).map(|_| noise.draw()).collect(),
        );
        let slots: Vec<u32> = (0..2u32).flat_map(|s| vec![s; 32]).collect();
        let pos: Vec<u32> = (0..2).flat_map(|_| 0..32u32).collect();
        let inputs = [
            a,
            b,
            Tensor::u32(vec![64], slots),
            Tensor::u32(vec![64], pos),
        ];
        let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
        let want = run_kept(&g, &[], &mut Store::default(), 0, &inputs).unwrap();
        let sizes = [(Seqs, 2), (Rows, 64), (Context, 32)];
        let cases: [(&'static dyn Load, f32); 3] =
            [(F32_FORM, 1e-5), (BF16_FORM, 2e-2), (FP8_FORM, 2e-1)];
        for (form, tol) in cases {
            for layout in [&kept::Dense as &'static dyn Layout, &kept::PAGED16] {
                for settings in runs() {
                    let pins = Pins::default().kept(form).activations(form).outputs(form);
                    let mut dev = Device::guess(&settings).pinned(pins);
                    dev.kept = layout;
                    let plan = lower(&g, 0, &sizes, &dev).unwrap();
                    let kj = plan
                        .steps
                        .iter()
                        .find(|s| s.kernel.name() == "kept_join")
                        .unwrap();
                    let tessera_lower::Source::Made(body) = kj.kernel.source() else {
                        panic!("expected Made source");
                    };
                    assert_eq!(body.entry.contains("_x4_"), expect_x4, "{}", body.entry);
                    let mut runner = open(plan.clone(), &settings).unwrap();
                    if let Some(tbl) = plan.table {
                        let rev: Vec<u8> = (0..tbl.len as u32)
                            .rev()
                            .flat_map(u32::to_le_bytes)
                            .collect();
                        runner.held_mut().upload(tbl, &rev).unwrap();
                    }
                    let got = runner.run(&views).unwrap();
                    let diff = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
                    assert!(diff < tol, "{left}+{right} {}: {diff:e}", body.entry);
                }
            }
        }
    }
}

#[test]
fn stage_rules() {
    let rules = rules();
    for s in tessera_graph::STAGES {
        if let tessera_graph::Lower::Kernel(name) = s.facts().lower {
            assert!(rules.iter().any(|r| r.name == name), "no rule {name}");
        }
    }
}

#[test]
fn wide_strided_narrow_forms() {
    use tessera_graph::DimName::Hidden;
    use tessera_graph::TensorType;
    use tessera_lower::Pins;
    use tessera_lower::form::{BF16_FORM, F32_FORM};
    use tessera_trace::{Act, Port};
    let m = Model::new(vec![Dim::bounded(Rows, 8), Dim::fixed(Hidden, 16)]);
    let port = Port {
        role: Role::Logits,
        ty: TensorType::new([Rows, Hidden], DType::F32),
    };
    let [x] = m.entry(EntryKind::Forward, [port]);
    let sliced = x.act(Act::Silu).slice(1, 4, 8);
    let g = m.finish([(Role::Logits, &sliced)]);
    let dev = Device::default().pinned(
        Pins::all(BF16_FORM)
            .activations(BF16_FORM)
            .outputs(F32_FORM),
    );
    let plan = lower(&g, 0, &[(Rows, 4)], &dev).unwrap();
    let strided = plan.steps.iter().find(|s| s.rule == "view").unwrap();
    let tessera_lower::Source::Made(body) = strided.kernel.source() else {
        panic!("expected Made source")
    };
    assert_eq!(body.entry, "plan_strided_x4_bf16_f32");
    // Same-form narrow bf16->bf16 also uses vectorized x4.
    let dev_same = Device::default().pinned(
        Pins::all(BF16_FORM)
            .activations(BF16_FORM)
            .outputs(BF16_FORM),
    );
    let plan_same = lower(&g, 0, &[(Rows, 4)], &dev_same).unwrap();
    let strided_same = plan_same.steps.iter().find(|s| s.rule == "view").unwrap();
    let tessera_lower::Source::Made(body_same) = strided_same.kernel.source() else {
        panic!("expected Made source")
    };
    assert_eq!(body_same.entry, "plan_strided_x4_bf16_bf16");
    let mut noise = Noise(7);
    let inp = fill(&g, &g.entries[0].inputs[0].ty, &[(Rows, 4)], &mut noise).unwrap();
    let want = run(&g, &[], 0, std::slice::from_ref(&inp)).unwrap();
    for settings in runs() {
        let mut runner = open(plan.clone(), &settings).unwrap();
        let got = runner.run(&[inp.view()]).unwrap();
        let diff = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
        assert!(diff < 2e-2, "diff={diff:e}");
    }
}

#[test]
fn kernel_launch_knobs() {
    let mut knobs = tessera_lower::Knobs::default();
    knobs.parse("lane_norm:warps=4").unwrap();
    knobs.parse("lane_norm:block_rows=64").unwrap();
    knobs.parse("lane_norm:ahead=2").unwrap();
    knobs.parse("softmax:threads=64").unwrap();
    let gpu = Platform {
        backend: "cuda".into(),
        arch: "sm_90".into(),
        name: "GPU".into(),
        units: 128,
        bandwidth: 1e12,
        ..Platform::assumed(Limits::assumed())
    };
    let wide = graphs::rows(&dims(), 600);
    let base = lower(&wide, 0, &SIZES, &Device::default().on(gpu.clone())).unwrap();
    let mut tuned_dev = Device::default().on(gpu);
    tuned_dev.knobs = knobs;
    let tuned = lower(&wide, 0, &SIZES, &tuned_dev).unwrap();
    let norm = |p: &Plan| {
        p.steps
            .iter()
            .find(|s| s.kernel.name().contains("norm"))
            .unwrap()
            .clone()
    };
    assert_ne!(
        norm(&base).kernel.identity(),
        norm(&tuned).kernel.identity()
    );
    let n = norm(&tuned).launches()[0].shape;
    assert!(matches!(n, tessera_lower::Shape::Grid { threads: 128, .. }));

    let step = tuned
        .steps
        .iter()
        .find(|s| s.kernel.name() == "softmax")
        .unwrap();
    assert!(matches!(
        step.launches()[0].shape,
        tessera_lower::Shape::Grid { threads: 64, .. }
    ));
}

/// Exact GELU and 2D/3D M-RoPE match the interpreter.
#[test]
fn gelu_and_mrope() {
    use tessera_graph::{Shape, TensorType};
    use tessera_trace::Port;
    let mut noise = Noise(19);
    let cases = [
        (72u64, 2u64, 10_000.0f32, [18u32, 18, 0], false),
        (128u64, 3u64, 5_000_000.0f32, [24u32, 20, 20], true),
    ];
    for (width, axes, theta, sections, interleaved) in cases {
        use tessera_trace::{Norm, Weights, declare};
        #[derive(Weights)]
        struct QkW {
            #[shape(HeadDim)]
            w: Norm,
        }
        let m = Model::new(vec![
            Dim::bounded(Rows, 8),
            Dim::fixed(Heads, 2),
            Dim::fixed(HeadDim, width),
        ]);
        let qw: QkW = declare(&m);
        let xport = Port {
            role: Role::Tokens,
            ty: TensorType::new([Rows, Heads, HeadDim], DType::F32),
        };
        let cport = Port {
            role: Role::Positions,
            ty: TensorType::new(Shape(vec![Rows.into(), axes.into()]), DType::U32),
        };
        let [x, coords] = m.entry(EntryKind::Forward, [xport, cport]);
        let out =
            x.act(Act::Gelu)
                .rms_norm(&qw.w, 1e-6)
                .mrope(&coords, theta, sections, interleaved);
        let g = m.finish([(Role::Logits, &out)]);
        let rows = 5usize;
        let x_val = Tensor::f32(
            vec![rows, 2, width as usize],
            (0..rows * 2 * width as usize)
                .map(|_| noise.draw())
                .collect(),
        );
        let c_val = Tensor::u32(
            vec![rows, axes as usize],
            (0..rows * axes as usize)
                .map(|i| (i * 3 % 17) as u32)
                .collect(),
        );
        let w_val = Tensor::f32(
            vec![width as usize],
            (0..width as usize).map(|_| 0.5 + noise.draw()).collect(),
        );
        let weights = [w_val];
        let want = run(&g, &weights, 0, &[x_val.clone(), c_val.clone()]).unwrap();
        let views = [x_val.view(), c_val.view()];
        for settings in runs() {
            let device = Device::guess(&settings);
            let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
            let want_norm = if width == 128 {
                "norm_rope4"
            } else {
                "norm_rope"
            };
            assert!(
                names(&plan).contains(&want_norm),
                "expected fused {want_norm} in {:?}",
                names(&plan)
            );
            let mut runner = open(plan, &settings).unwrap();
            runner.load(0, weights[0].view()).unwrap();
            let got = runner.run(&views).unwrap();
            let off = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
            assert!(
                off < 1e-4,
                "w={width} interleaved={interleaved} {:?}: {off:e}",
                settings.backend
            );
        }
    }
}

/// Fused layer norm and wide-row block norm match the interpreter.
#[test]
fn fused_layer_norm() {
    use tessera_graph::DimName::Hidden;
    use tessera_trace::{Norm, Vector, Weights, declare};
    #[derive(Weights)]
    struct LnW {
        #[shape(Hidden)]
        w: Norm,
        #[shape(Hidden)]
        b: Vector,
    }
    let m = Model::new(vec![Dim::bounded(Rows, 4096), Dim::fixed(Hidden, 1152)]);
    let lw: LnW = declare(&m);
    let [a, b] = m.entry(
        EntryKind::Forward,
        [floats([Rows, Hidden]), floats([Rows, Hidden])],
    );
    let sum = a.add(&b);
    let ln = sum.center().rms_norm(&lw.w, 1e-6).add_vector(&lw.b);
    let g = m.finish([(Role::Logits, &ln), (Role::Logits, &sum)]);
    let mut noise = Noise(29);
    let rows = 3usize;
    let width = 1152usize;
    let a_val = Tensor::f32(
        vec![rows, width],
        (0..rows * width).map(|_| noise.draw()).collect(),
    );
    let b_val = Tensor::f32(
        vec![rows, width],
        (0..rows * width).map(|_| noise.draw()).collect(),
    );
    let w_val = Tensor::f32(
        vec![width],
        (0..width).map(|_| 0.5 + noise.draw()).collect(),
    );
    let bias_val = Tensor::f32(vec![width], (0..width).map(|_| noise.draw()).collect());
    let weights = [w_val, bias_val];
    let want = run(&g, &weights, 0, &[a_val.clone(), b_val.clone()]).unwrap();
    let views = [a_val.view(), b_val.view()];
    for settings in runs() {
        let device = Device::guess(&settings);
        let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
        assert_eq!(names(&plan), ["block_add_ln_bsave"]);
        let mut runner = open(plan, &settings).unwrap();
        runner.load(0, weights[0].view()).unwrap();
        runner.load(1, weights[1].view()).unwrap();
        let got = runner.run(&views).unwrap();
        for idx in 0..2 {
            let off = max_abs_diff(got[idx].floats().unwrap(), want[idx].floats().unwrap());
            assert!(off < 1e-4, "out {idx} {:?}: {off:e}", settings.backend);
        }
    }
    // Wide rows above 2048 still stay in single-pass block mode.
    let gpu = Platform {
        backend: "cuda".into(),
        name: "Test GPU".into(),
        units: 128,
        bandwidth: 1e12,
        ..Platform::assumed(Limits::assumed())
    };
    let big = lower(&g, 0, &[(Rows, 4096)], &Device::default().on(gpu)).unwrap();
    assert_eq!(names(&big), ["block_add_ln_bsave"]);
}

/// Fused slice-rope and biased residual layer norm match the interpreter.
#[test]
fn fused_slice_ln() {
    use tessera_graph::DimName::Hidden;
    use tessera_trace::{DType, DimExpr, Norm, Port, Shape, TensorType, Vector, Weights, declare};
    #[derive(Weights)]
    struct W {
        #[shape(Hidden)]
        ib: Vector,
        #[shape(Hidden)]
        nw: Norm,
        #[shape(Hidden)]
        nb: Vector,
    }
    let qkv_dim = DimName::named("Qkv");
    let m = Model::new(vec![
        Dim::bounded(Rows, 16),
        Dim::fixed(Hidden, 144),
        Dim::fixed(Heads, 2),
        Dim::fixed(HeadDim, 72),
        Dim::fixed(qkv_dim, 432),
    ]);
    let w: W = declare(&m);
    let [qkv, rot, res, raw] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role: Role::Tokens,
                ty: TensorType::new([Rows, qkv_dim], DType::F32),
            },
            Port {
                role: Role::Positions,
                ty: TensorType::new(Shape(vec![Rows.into(), 2u64.into()]), DType::U32),
            },
            floats([Rows, Hidden]),
            floats([Rows, Hidden]),
        ],
    );
    let q = qkv
        .slice(1, 144u64, DimExpr::Dim(Hidden))
        .reshape(vec![Rows.into(), Heads.into(), HeadDim.into()])
        .mrope(&rot, 10_000.0, [18, 18, 0], false)
        .reshape(vec![Rows.into(), Hidden.into()]);
    let sum = res.add(&raw.add_vector(&w.ib));
    let ln = sum.center().rms_norm(&w.nw, 1e-6).add_vector(&w.nb);
    let g = m.finish([
        (Role::Logits, &q),
        (Role::Logits, &ln),
        (Role::Logits, &sum),
    ]);
    let mut noise = Noise(41);
    let rows = 5usize;
    let qkv_v = Tensor::f32(
        vec![rows, 432],
        (0..rows * 432).map(|_| noise.draw()).collect(),
    );
    let rot_v = Tensor::u32(
        vec![rows, 2],
        (0..rows * 2).map(|i| (i % 7) as u32).collect(),
    );
    let res_v = Tensor::f32(
        vec![rows, 144],
        (0..rows * 144).map(|_| noise.draw()).collect(),
    );
    let raw_v = Tensor::f32(
        vec![rows, 144],
        (0..rows * 144).map(|_| noise.draw()).collect(),
    );
    let ws = [
        Tensor::f32(vec![144], (0..144).map(|_| noise.draw()).collect()),
        Tensor::f32(vec![144], (0..144).map(|_| 0.5 + noise.draw()).collect()),
        Tensor::f32(vec![144], (0..144).map(|_| noise.draw()).collect()),
    ];
    let want = run(
        &g,
        &ws,
        0,
        &[qkv_v.clone(), rot_v.clone(), res_v.clone(), raw_v.clone()],
    )
    .unwrap();
    let views = [qkv_v.view(), rot_v.view(), res_v.view(), raw_v.view()];
    for settings in runs() {
        let device = Device::guess(&settings);
        let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
        let ns = names(&plan);
        assert!(
            ns.contains(&"slice_rope_x4"),
            "expected slice_rope_x4 in {ns:?}"
        );
        assert!(
            ns.contains(&"lane_add_ln_bsave"),
            "expected fused biased ln in {ns:?}"
        );
        assert!(
            !ns.contains(&"strided_x4"),
            "strided_x4 should be fused in {ns:?}"
        );
        let mut runner = open(plan, &settings).unwrap();
        for (i, t) in ws.iter().enumerate() {
            runner.load(i as u32, t.view()).unwrap();
        }
        let got = runner.run(&views).unwrap();
        for idx in 0..3 {
            let off = max_abs_diff(got[idx].floats().unwrap(), want[idx].floats().unwrap());
            assert!(off < 1e-4, "out {idx} {:?}: {off:e}", settings.backend);
        }
    }
}

/// Rows of `width` floats, clipped to length 1.
fn clip_rows(width: u64) -> Graph {
    let m = Model::new(vec![Dim::bounded(Rows, 1024), Dim::fixed(HeadDim, width)]);
    let [x] = m.entry(EntryKind::Forward, [floats([Rows, HeadDim])]);
    m.finish([(Role::Logits, &x.l2_clip(1.0, 1e-3))])
}

#[test]
fn lane_clip_is_picked() {
    // The chip table prices both kernels.
    let metal = Platform {
        backend: "metal".into(),
        arch: "apple9".into(),
        name: "Apple M3 Pro".into(),
        ..Platform::assumed(Limits::assumed())
    };
    let device = Device::guess(&host()).on(metal);
    let shapes = [
        (8, 128),
        (24, 128),
        (24, 16),
        (4, 256),
        (256, 128),
        (768, 128),
    ];
    for (rows, width) in shapes {
        let plan = lower(&clip_rows(width), 0, &[(Rows, rows)], &device).unwrap();
        assert_eq!(plan.steps[0].rule, "lane_l2_clip", "{rows}x{width}");
    }
    // A row off the four-float grid stays scalar.
    let plan = lower(&clip_rows(18), 0, &[(Rows, 8)], &device).unwrap();
    assert_eq!(plan.steps[0].rule, "l2_clip");
}

#[test]
fn lane_clip_matches() {
    // Rows cycle: zero, shorter than 1, longer than 1.
    const SCALE: [f32; 3] = [0.0, 0.01, 1.0];
    for width in [16usize, 128, 144, 256] {
        let x: Vec<f32> = (0..5 * width)
            .map(|i| SCALE[i / width % 3] * ((i * 37 % 23) as f32 - 11.0) / 11.0)
            .collect();
        let inputs = [Tensor::f32(vec![5, width], x)];
        let g = clip_rows(width as u64);
        let want = run(&g, &[], 0, &inputs).unwrap();
        let want = want[0].floats().unwrap();
        assert!(want[..width].iter().all(|v| *v == 0.0));
        for settings in runs() {
            let device = Device::guess(&settings).favoring("lane_l2_clip");
            let plan = lower(&g, 0, &[(Rows, 5)], &device).unwrap();
            assert_eq!(plan.steps[0].rule, "lane_l2_clip");
            let mut runner = open(plan, &settings).unwrap();
            let got = runner.run(&[inputs[0].view()]).unwrap();
            let off = max_abs_diff(got[0].floats().unwrap(), want);
            assert!(off < 1e-6, "width {width} is off by {off:e}");
        }
    }
}

#[test]
fn join_bounds() {
    use tessera_graph::DimName::Hidden;
    use tessera_graph::TensorType;
    use tessera_lower::Shape;
    use tessera_trace::Port;
    let metal = Platform {
        backend: "metal".into(),
        arch: "apple9".into(),
        name: "Apple M4 Pro".into(),
        ..Platform::assumed(Limits::assumed())
    };
    let dev = Device::guess(&host()).on(metal);
    let build = |rows: u64, width: u64| {
        let m = Model::new(vec![Dim::bounded(Rows, rows), Dim::fixed(Hidden, width)]);
        let port = || Port {
            role: Role::Logits,
            ty: TensorType::new([Rows, Hidden], DType::F32),
        };
        let [x, y] = m.entry(EntryKind::Forward, [port(), port()]);
        m.finish([(Role::Logits, &x.concat(&y))])
    };
    for (rows, width, want_threads) in [
        (2u64, 0u64, 0usize),
        (1u64, 2_147_483_648u64, 1_073_741_824usize),
        (32_769, 65_536, 1_073_774_592),
    ] {
        let g = build(rows, width);
        let plan = lower(&g, 0, &[(Rows, rows)], &dev).unwrap();
        let step = plan
            .steps
            .iter()
            .find(|s| s.kernel.name() == "join")
            .unwrap();
        let launch = &step.launches()[0];
        assert_eq!(launch.shape, Shape::Line(want_threads));
        assert!(want_threads <= u32::MAX as usize);
        assert_eq!(launch.args[3].u32(), Some(width as u32));
        assert_eq!(launch.args[4].u32(), Some(width as u32));
        let want_rows = if width == 0 { 0 } else { rows as u32 };
        assert_eq!(launch.args[5].u32(), Some(want_rows));
        assert_eq!(want_threads * 4, step.write.len);
        if width == 0 {
            let empty = Tensor::f32(vec![rows as usize, 0], vec![]);
            let host_plan = lower(&g, 0, &[(Rows, rows)], &Device::guess(&host())).unwrap();
            let mut runner = open(host_plan, &host()).unwrap();
            let got = runner.run(&[empty.view(), empty.view()]).unwrap();
            assert_eq!(got[0].shape, [rows as usize, 0]);
        }
    }
    for (rows, width) in [(1u64, 4_294_967_296u64), (32_769, 65_535)] {
        assert!(
            lower(&build(rows, width), 0, &[(Rows, rows)], &dev).is_err(),
            "expected join to reject {rows}x({width}+{width})"
        );
    }
}
