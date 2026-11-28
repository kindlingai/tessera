use crate::device_test::{push_node, with_stage};
use crate::rocm_mm::rocm_linear_source;
use crate::rocm_trellis::rocm_trellis_source;
use tessera_gpukit::backend::{Backend, picked};
use tessera_gpukit::platform::{Asked, Platform};
use tessera_graph::DimName::Rows;
use tessera_graph::{Act, DType, Dim, F16, Graph, Op, PathSeg, Row, Stage, TensorType};
use tessera_interp::{Tensor, run};
use tessera_kernels as _;
use tessera_linear as _;
use tessera_lower::form::{BF16_FORM, named};
use tessera_lower::{Device, Limits, Pins, Settings, Weight, explain, lower};
use tessera_mem::trellis;
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::graphs::{linear, turned};
use tessera_testkit::{Noise, cosine_f32, max_abs_diff};

fn rdna3_platform() -> Platform {
    Platform::new(Asked {
        backend: "rocm",
        arch: "gfx1151".into(),
        name: "AMD Strix Halo".into(),
        driver: String::new(),
        limits: Limits {
            group_bytes: 64 * 1024,
            ..Limits::assumed()
        },
        units: 40,
        bandwidth: 256e9,
        integrated: true,
        memory: 112 << 30,
    })
}

fn with_bias_weight(mut g: Graph, name: &str) -> Graph {
    let mat = g.entries[0].outputs[0].node;
    let ty = g.nodes[mat.0 as usize].ty.clone();
    let b_ty = TensorType::new(vec![ty.shape.0[1].clone()], DType::F32);
    let w_id = g.weights.len() as u32;
    g.weights.push(tessera_graph::Weight {
        path: vec![PathSeg::Field(name.into())],
        ty: b_ty.clone(),
    });
    let b_node = push_node(&mut g, Op::Weight(w_id), Vec::new(), b_ty);
    let out = push_node(
        &mut g,
        Op::Row(Row(vec![Stage::Add])),
        vec![mat, b_node],
        ty,
    );
    g.entries[0].outputs[0].node = out;
    g
}

fn epi_graphs(base: &Graph, with_bias: bool, with_swiglu: bool) -> Vec<(Graph, bool)> {
    let out_ty = base.nodes[base.entries[0].outputs[0].node.0 as usize]
        .ty
        .clone();
    let mut out = Vec::new();
    if with_bias {
        out.push((with_bias_weight(base.clone(), "b"), false));
    }
    out.push((
        with_stage(
            with_bias_weight(base.clone(), "b"),
            Stage::Act(Act::GeluTanh),
            None,
        ),
        false,
    ));
    out.push((
        with_stage(
            with_bias_weight(base.clone(), "b"),
            Stage::Act(Act::Gelu),
            None,
        ),
        false,
    ));
    out.push((
        with_stage(
            with_bias_weight(base.clone(), "b"),
            Stage::Add,
            Some(out_ty.clone()),
        ),
        true,
    ));
    if with_swiglu {
        out.push((
            with_stage(
                with_stage(base.clone(), Stage::Act(Act::Silu), None),
                Stage::Mul,
                Some(out_ty),
            ),
            true,
        ));
    }
    out
}

#[test]
fn rocm_sources_list_entries() {
    let tr = rocm_trellis_source();
    for entry in [
        "rocm_dtr_k4_t1",
        "rocm_dtr_k4_t1s",
        "rocm_dtr_k4_p1",
        "rocm_dtr_k4_t2",
        "rocm_dtr_gen1",
        "rocm_dtr_gen2",
        "rocm_dtr_reduce",
        "rocm_dtr_reduce_epi",
        "rocm_dtr_prep",
        "rocm_dtr_pre_k4_t4",
        "rocm_dtr_pre_k4_t4s",
        "rocm_dtr_pre_k4_t8",
        "rocm_dtr_pre_k4_t8s",
        "rocm_dtr_pre_k4_t16",
        "rocm_dtr_pre_gen4",
        "rocm_dtr_pre_gen8",
        "rocm_dtr_pre_gen16",
    ] {
        assert!(tr.contains(entry), "missing {entry}");
    }
    let lin = rocm_linear_source();
    for entry in [
        "rocm_gemv_bf16",
        "rocm_gemv_bf16_narrow",
        "rocm_gemv_int4",
        "rocm_gemv_mxfp8",
        "rocm_mm_bf16_r16",
        "rocm_mm_bf16_r64",
        "rocm_mm_bf16_r128",
        "rocm_mm_int4_r16",
        "rocm_mm_int4_r64",
        "rocm_mm_int4_r128",
        "rocm_mm_mxfp8_r16",
        "rocm_mm_mxfp8_r64",
        "rocm_mm_mxfp8_r128",
        "rocm_mm_reduce",
    ] {
        assert!(lin.contains(entry), "missing {entry}");
    }
}

#[test]
fn rocm_rdna3_selection() {
    let rows = [Dim::bounded(Rows, 2048)];
    let rdna = rdna3_platform();

    // Trellis K4 rotated: small-M and prefill.
    let tr_form = named("trellis_k4").unwrap();
    let dev_tr = Device::default()
        .on(rdna.clone())
        .pinned(Pins::default().pin("w", tr_form));
    let g_tr = turned(&rows);
    for (m, want) in [
        (1u64, "rocm_trellis_turned"),
        (16, "rocm_trellis_turned"),
        (64, "rocm_trellis_prep"),
        (256, "rocm_trellis_prep"),
    ] {
        let plan = lower(&g_tr, 0, &[(Rows, m)], &dev_tr).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(
            names.iter().any(|n| n.contains(want)),
            "m={m} got {names:?}"
        );
        let r = explain(&g_tr, 0, &[(Rows, m)], &dev_tr).unwrap();
        assert!(!r.has_fallbacks(), "m={m}: {}", r.format(10));
    }

    // BF16 including 8B ViT MLP shapes.
    let dev_bf = Device::default()
        .on(rdna.clone())
        .pinned(Pins::all(BF16_FORM));
    for (k, n) in [(1152u64, 4304u64), (4304, 1152), (2048, 2048)] {
        let g = linear(&rows, k, n);
        let p1 = lower(&g, 0, &[(Rows, 1)], &dev_bf).unwrap();
        assert_eq!(p1.steps[0].kernel.name(), "rocm_gemv_bf16");
        for m in [16u64, 64, 256] {
            let pm = lower(&g, 0, &[(Rows, m)], &dev_bf).unwrap();
            assert!(
                pm.steps[0].kernel.name().starts_with("rocm_mm_bf16"),
                "{k}x{n} m={m} got {:?}",
                pm.steps[0].kernel.name()
            );
            let r = explain(&g, 0, &[(Rows, m)], &dev_bf).unwrap();
            assert!(!r.has_fallbacks(), "{k}x{n} m={m}: {}", r.format(10));
        }
    }

    // Fused epilogues on BF16 (f32/bf16 bias, gelu, residual, SwiGLU).
    let base = linear(&rows, 1152, 4304);
    let out_ty = base.nodes[base.entries[0].outputs[0].node.0 as usize]
        .ty
        .clone();
    let bias_ty = TensorType::new(vec![out_ty.shape.0[1].clone()], DType::F32);
    let dev_bf_all = Device::default()
        .on(rdna.clone())
        .pinned(Pins::all(BF16_FORM));
    let mut bf_sel = vec![
        with_stage(base.clone(), Stage::Add, Some(bias_ty)),
        with_stage(base.clone(), Stage::Add, Some(out_ty)),
    ];
    bf_sel.extend(epi_graphs(&base, true, true).into_iter().map(|(g, _)| g));
    for g in bf_sel {
        let pm = lower(&g, 0, &[(Rows, 64)], &dev_bf_all).unwrap();
        assert!(pm.steps[0].kernel.name().starts_with("rocm_mm_bf16"));
        assert!(!pm.steps.iter().any(|s| s.kernel.name() == "map"));
    }

    // Fused epilogues on Trellis K4 prefill (ks=1 and split-K).
    let dev_tr_bf = Device::default()
        .on(rdna.clone())
        .pinned(Pins::default().pin("w", tr_form).pin("b", BF16_FORM));
    for (g, _) in epi_graphs(&g_tr, true, true) {
        for m in [64u64, 256] {
            let pm = lower(&g, 0, &[(Rows, m)], &dev_tr_bf).unwrap();
            assert!(
                pm.steps
                    .iter()
                    .any(|s| s.kernel.name().contains("rocm_trellis"))
            );
            assert!(
                !pm.steps.iter().any(|s| s.kernel.name() == "map"),
                "m={m} unfused map in {:?}",
                pm.steps.iter().map(|s| s.kernel.name()).collect::<Vec<_>>()
            );
        }
    }

    // INT4_G32 and MXFP8 selection for GEMV and GEMM.
    for (fname, gemv_name, mm_prefix, k, n) in [
        (
            "int4_g32",
            "rocm_gemv_int4_g32",
            "rocm_mm_int4_g32",
            6144,
            2048,
        ),
        (
            "int4_g32",
            "rocm_gemv_int4_g32",
            "rocm_mm_int4_g32",
            4096,
            12288,
        ),
        ("mxfp8", "rocm_gemv_mxfp8", "rocm_mm_mxfp8", 2048, 2048),
    ] {
        let dev = Device::default()
            .on(rdna.clone())
            .pinned(Pins::all(named(fname).unwrap()));
        let g = linear(&rows, k, n);
        let p1 = lower(&g, 0, &[(Rows, 1)], &dev).unwrap();
        assert_eq!(p1.steps[0].kernel.name(), gemv_name);
        let p64 = lower(&g, 0, &[(Rows, 64)], &dev).unwrap();
        assert!(p64.steps[0].kernel.name().starts_with(mm_prefix));
    }

    // CDNA wave64 skips wave32 ROCm rules cleanly.
    let cdna = Platform::new(Asked {
        backend: "rocm",
        arch: "gfx942".into(),
        name: "AMD MI300X".into(),
        driver: String::new(),
        limits: Limits {
            lanes: 64,
            ..Limits::assumed()
        },
        units: 304,
        bandwidth: 5300e9,
        integrated: false,
        memory: 192 << 30,
    });
    let dev_cdna = Device::default().on(cdna).pinned(Pins::all(BF16_FORM));
    let p_cdna = lower(&linear(&rows, 1152, 4304), 0, &[(Rows, 1)], &dev_cdna).unwrap();
    assert!(!p_cdna.steps[0].kernel.name().starts_with("rocm_"));
}

struct GpuEnv<'a> {
    settings: &'a Settings,
    platform: &'a Platform,
    draws: Noise,
}

fn check_gpu_case(
    g: &Graph,
    form_name: &str,
    m: u64,
    x_shape: Vec<usize>,
    min_cos: f32,
    env: &mut GpuEnv<'_>,
) {
    let form = named(form_name).unwrap();
    let pins = if g.weights.len() > 1 {
        Pins::default().pin("w", form)
    } else {
        Pins::all(form)
    };
    let dev = Device::default().on(env.platform.clone()).pinned(pins);
    let plan = lower(g, 0, &[(Rows, m)], &dev).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    if !names.iter().any(|n| n.contains("rocm_")) {
        let sel = tessera_lower::select(g, 0, &[(Rows, m)], &dev).unwrap();
        println!("picked={names:?} ways={:?}", sel.ways);
    }
    assert!(
        names.iter().any(|n| n.contains("rocm_")),
        "{form_name} m={m} did not pick rocm kernel: {names:?}"
    );
    let x_len: usize = x_shape.iter().product();
    let x = Tensor::f32(x_shape, (0..x_len).map(|_| env.draws.draw()).collect());
    let mut weights = Vec::new();
    for w in &g.weights {
        let sh: Vec<usize> =
            w.ty.shape
                .0
                .iter()
                .map(|d| d.value(&g.dims, &|_| None).unwrap() as usize)
                .collect();
        let n_el: usize = sh.iter().product();
        let raw: Vec<f32> = (0..n_el).map(|_| env.draws.draw() * 0.5).collect();
        let data = if sh.len() == 2 {
            form.roundtrip(&raw)
        } else {
            raw
        };
        weights.push(Tensor::f32(sh, data));
    }
    let want = run(g, &weights, 0, std::slice::from_ref(&x)).unwrap();
    let mut runner = open(plan, env.settings).unwrap();
    for (i, w) in weights.iter().enumerate() {
        runner.load(i as u32, w.view()).unwrap();
    }
    let got = runner.run(&[x.view()]).unwrap().remove(0);
    let cos = cosine_f32(got.floats().unwrap(), want[0].floats().unwrap());
    let diff = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
    println!("{form_name:14} m={m:4} {names:?} cos={cos:.6} max_abs={diff:.3e}");
    assert!(
        cos >= min_cos,
        "{form_name} m={m}: cos={cos} < {min_cos} (max_abs={diff})"
    );
}

fn check_trellis_case(
    spec: (u32, &str, bool, u32),
    [inp, out]: [usize; 2],
    m: u64,
    env: &mut GpuEnv<'_>,
) {
    let (k_bits, book, rot, side) = spec;
    let rows = [Dim::bounded(Rows, 2048)];
    let g = linear(&rows, inp as u64, out as u64);
    let base_name = format!("trellis_k{k_bits}{book}");
    let form_name = if rot {
        format!("{base_name}_rot")
    } else {
        base_name
    };
    let form = named(&form_name).unwrap();
    let entries: Vec<u8> = (0..out * inp * k_bits as usize / 8)
        .map(|_| (env.draws.word() & 0xFF) as u8)
        .collect();
    let mut bytes = trellis::packed(inp, out, side, &entries);
    if rot {
        let scales =
            (0..inp + out).flat_map(|i| F16::from(1.0 + (i % 7) as f32 / 8.0).0.to_le_bytes());
        bytes.extend(scales);
    }
    let w = Tensor::f32(vec![out, inp], form.unpack_all(&bytes));
    let x = Tensor::f32(
        vec![m as usize, inp],
        (0..m as usize * inp).map(|_| env.draws.draw()).collect(),
    );
    let want = run(&g, &[w], 0, std::slice::from_ref(&x)).unwrap();
    let dev = Device::default()
        .on(env.platform.clone())
        .pinned(Pins::all(form));
    let plan = lower(&g, 0, &[(Rows, m)], &dev).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert!(
        names.iter().any(|n| n.contains("rocm_trellis")),
        "{form_name} side={side} m={m} picked {names:?}"
    );
    let mut runner = open(plan, env.settings).unwrap();
    runner.load_weight(0, &Weight::Packed(form, bytes)).unwrap();
    let got = runner.run(&[x.view()]).unwrap().remove(0);
    let cos = cosine_f32(got.floats().unwrap(), want[0].floats().unwrap());
    let diff = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
    println!("{form_name:18} s={side} m={m:4} {names:?} cos={cos:.6} max_abs={diff:.3e}");
    assert!(
        cos >= 0.9999,
        "{form_name} side={side} m={m}: cos={cos} (max_abs={diff})"
    );
}

fn check_epi_gpu(
    g: &Graph,
    pins: Pins,
    m: u64,
    in_shapes: &[Vec<usize>],
    min_cos: f32,
    env: &mut GpuEnv<'_>,
) {
    let dev = Device::default().on(env.platform.clone()).pinned(pins);
    let plan = lower(g, 0, &[(Rows, m)], &dev).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert!(
        names.iter().any(|n| n.contains("rocm_")) && !names.contains(&"map"),
        "epi m={m} picked {names:?}"
    );
    let inputs: Vec<Tensor> = in_shapes
        .iter()
        .map(|sh| {
            let len: usize = sh.iter().product();
            Tensor::f32(
                sh.clone(),
                (0..len).map(|_| env.draws.draw() * 0.5).collect(),
            )
        })
        .collect();
    let mut weights = Vec::new();
    let mut packed: Vec<(u32, Weight)> = Vec::new();
    for (i, w) in g.weights.iter().enumerate() {
        let sh: Vec<usize> =
            w.ty.shape
                .0
                .iter()
                .map(|d| d.value(&g.dims, &|_| None).unwrap() as usize)
                .collect();
        let n_el: usize = sh.iter().product();
        let form = dev.pins.form(&w.path);
        if form.name().starts_with("trellis_") {
            let (out, inp) = (sh[0], sh[1]);
            let entries: Vec<u8> = (0..out * inp / 2)
                .map(|_| (env.draws.word() & 0xFF) as u8)
                .collect();
            let mut bytes = trellis::packed(inp, out, 1, &entries);
            let scales =
                (0..inp + out).flat_map(|j| F16::from(1.0 + (j % 7) as f32 / 8.0).0.to_le_bytes());
            bytes.extend(scales);
            weights.push(Tensor::f32(sh, form.unpack_all(&bytes)));
            packed.push((i as u32, Weight::Packed(form, bytes)));
            continue;
        }
        let raw: Vec<f32> = (0..n_el).map(|_| env.draws.draw() * 0.5).collect();
        let data = form.roundtrip(&raw);
        weights.push(Tensor::f32(sh, data));
    }
    let want = run(g, &weights, 0, &inputs).unwrap();
    let mut runner = open(plan, env.settings).unwrap();
    for (i, w) in weights.iter().enumerate() {
        if let Some((_, pw)) = packed.iter().find(|(idx, _)| *idx == i as u32) {
            runner.load_weight(i as u32, pw).unwrap();
        } else {
            runner.load(i as u32, w.view()).unwrap();
        }
    }
    let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
    let got = runner.run(&views).unwrap().remove(0);
    let cos = cosine_f32(got.floats().unwrap(), want[0].floats().unwrap());
    let diff = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
    println!("epi m={m:4} {names:?} cos={cos:.6} max_abs={diff:.3e}");
    assert!(cos >= min_cos, "epi m={m} {names:?}: cos={cos} < {min_cos}");
}

#[test]
fn rocm_gpu_parity() {
    if picked() != Some(Backend::Rocm) {
        return;
    }
    let Some(settings) = runs().into_iter().find(|s| s.gpu() == Some(Backend::Rocm)) else {
        return;
    };
    let rows = [Dim::bounded(Rows, 2048)];
    let probe = lower(
        &linear(&rows, 64, 64),
        0,
        &[(Rows, 1)],
        &Device::guess(&settings),
    )
    .unwrap();
    let platform = open(probe, &settings).unwrap().platform();
    let mut env = GpuEnv {
        settings: &settings,
        platform: &platform,
        draws: Noise(42),
    };

    // BF16 ViT MLP shapes (4304x1152 and 1152x4304).
    for (k, n) in [(1152u64, 4304u64), (4304, 1152)] {
        let g = linear(&rows, k, n);
        for m in [1u64, 16, 64, 256] {
            check_gpu_case(
                &g,
                "bf16",
                m,
                vec![m as usize, k as usize],
                0.99995,
                &mut env,
            );
        }
    }

    // INT4_G32 and MXFP8 (GEMV, WMMA r16/r64/r128, split-K).
    for (fname, k, n) in [
        ("int4_g32", 1024u64, 512u64),
        ("int4_g32", 512, 2048),
        ("mxfp8", 1024, 512),
    ] {
        let g = linear(&rows, k, n);
        for m in [1u64, 16, 64, 256] {
            check_gpu_case(&g, fname, m, vec![m as usize, k as usize], 0.9995, &mut env);
        }
    }

    // Trellis K4 rotated (side=OUT and side=IN), small-M and prefill.
    for side in [1u32, 2] {
        for m in [1u64, 16, 24, 64] {
            check_trellis_case((4, "", true, side), [512, 512], m, &mut env);
        }
    }
    // Prefill split-K at M=64 and M=65.
    check_trellis_case((4, "", true, 1), [1024, 512], 64, &mut env);
    check_trellis_case((4, "", true, 2), [1024, 512], 64, &mut env);
    check_trellis_case((4, "", true, 1), [1024, 512], 65, &mut env);
    // Prefill NT=8 and NT=16 (128x128 and 256x128 tiles).
    check_trellis_case((4, "", true, 1), [256, 1280], 130, &mut env);
    check_trellis_case((4, "", true, 1), [256, 2560], 512, &mut env);
    // Plain Trellis K4, PMCGX, and MUL1 codebooks.
    for (k_bits, book, rot, m) in [
        (4u32, "", false, 16u64),
        (4, "", false, 64),
        (4, "_pmcgx", true, 16),
        (4, "_pmcgx", true, 64),
        (3, "_mul1", true, 16),
        (3, "_mul1", true, 64),
    ] {
        check_trellis_case((k_bits, book, rot, 1), [512, 512], m, &mut env);
    }

    // Fused ViT epilogues on BF16 (direct and split-K reduce).
    let bf_base = linear(&rows, 1024, 512);
    for (g, has_res) in epi_graphs(&bf_base, true, false) {
        for m in [16u64, 64] {
            let mut sh = vec![vec![m as usize, 1024]];
            if has_res {
                sh.push(vec![m as usize, 512]);
            }
            check_epi_gpu(&g, Pins::all(BF16_FORM), m, &sh, 0.9999, &mut env);
        }
    }

    // Fused ViT and SwiGLU epilogues on Trellis K4.
    let tr_form = named("trellis_k4_rot").unwrap();
    let tr_pins = || Pins::default().pin("w", tr_form).pin("b", BF16_FORM);
    for (k, n, m) in [(1024u64, 512u64, 65u64), (256, 2560, 256)] {
        let base = linear(&rows, k, n);
        for (g, has_res) in epi_graphs(&base, false, true) {
            let mut sh = vec![vec![m as usize, k as usize]];
            if has_res {
                sh.push(vec![m as usize, n as usize]);
            }
            check_epi_gpu(&g, tr_pins(), m, &sh, 0.9999, &mut env);
        }
    }
}
