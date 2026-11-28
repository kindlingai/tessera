use tessera_attention as _;
use tessera_device_rules::{packed_cuda_source, packed_rocm_source};
use tessera_gpukit::backend::{Backend, picked};
use tessera_gpukit::platform::{Asked, Platform};
use tessera_graph::DimName::{HeadDim, Heads, KvHeads, Rows};
use tessera_graph::{DType, Dim, EntryKind, Graph, Port, Role, Shape, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::{Device, Limits, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, cosine_f32, max_abs_diff};
use tessera_trace::Model;

fn packed_graph(heads: u64, kv_heads: u64, width: u64, causal: bool) -> Graph {
    let dims = vec![
        Dim::bounded(Rows, 2048),
        Dim::fixed(Heads, heads),
        Dim::fixed(KvHeads, kv_heads),
        Dim::fixed(HeadDim, width),
    ];
    let m = Model::new(dims);
    let port = |h| Port {
        role: Role::Tokens,
        ty: TensorType::new([Rows, h, HeadDim], DType::F32),
    };
    let bport = Port {
        role: Role::Positions,
        ty: TensorType::new(Shape(vec![Rows.into(), 2u64.into()]), DType::U32),
    };
    let [q, k, v, b] = m.entry(
        EntryKind::Forward,
        [port(Heads), port(KvHeads), port(KvHeads), bport],
    );
    let scale = (width as f32).powf(-0.5);
    let out = m.attend_packed(causal, scale, [&q, &k, &v], &b);
    m.finish([(Role::Logits, &out)])
}

fn mock_gpu(backend: &'static str, arch: &str) -> Platform {
    Platform::new(Asked {
        backend,
        arch: arch.into(),
        name: arch.into(),
        driver: String::new(),
        limits: Limits {
            group_bytes: 64 * 1024,
            ..Limits::assumed()
        },
        units: 40,
        bandwidth: 256e9,
        integrated: true,
        memory: 64 << 30,
    })
}

fn draw_tensor(shape: Vec<usize>, rng: &mut Noise) -> Tensor {
    let n: usize = shape.iter().product();
    let data = (0..n).map(|_| rng.draw() * 0.5).collect();
    Tensor::f32(shape, data)
}

#[test]
fn packed_gpu_entries() {
    for src in [packed_cuda_source(), packed_rocm_source()] {
        for entry in [
            "fa2_packed_d16",
            "fa2_packed_d64",
            "fa2_packed_d80",
            "fa2_packed_d128",
        ] {
            assert!(src.contains(entry), "missing {entry}");
        }
    }
}

#[test]
fn packed_gpu_selection() {
    let cases = [
        (16u64, 8u64, 128u64, true),
        (32, 8, 128, true),
        (16, 16, 64, false),
        (16, 16, 72, false),
        (4, 2, 16, true),
    ];
    for (backend, arch, want) in [
        ("cuda", "sm_120", "flash_packed_cuda"),
        ("rocm", "gfx1151", "flash_packed_rocm"),
    ] {
        let dev = Device::default().on(mock_gpu(backend, arch));
        for &(h, hk, d, causal) in &cases {
            let g = packed_graph(h, hk, d, causal);
            let plan = lower(&g, 0, &[(Rows, 128)], &dev).unwrap();
            assert_eq!(plan.steps.len(), 1, "{backend} h={h} d={d}");
            assert_eq!(plan.steps[0].kernel.name(), want);
        }
    }
}

#[test]
fn packed_gpu_parity() {
    let Some(backend) = picked() else {
        return;
    };
    if !matches!(backend, Backend::Cuda | Backend::Rocm) {
        return;
    }
    let Some(settings) = runs().into_iter().find(|s| s.gpu() == Some(backend)) else {
        return;
    };
    let want_name = match backend {
        Backend::Cuda => "flash_packed_cuda",
        Backend::Rocm => "flash_packed_rocm",
        _ => return,
    };
    let probe = lower(
        &packed_graph(4, 2, 16, true),
        0,
        &[(Rows, 16)],
        &Device::guess(&settings),
    )
    .unwrap();
    let platform = open(probe, &settings).unwrap().platform();
    let dev = Device::guess(&settings).on(platform);
    let mut rng = Noise(12345);
    let cases: &[([usize; 3], &[usize], bool)] = &[
        ([16, 8, 128], &[64, 19, 130, 5], true),
        ([32, 8, 128], &[37, 91], true),
        ([16, 16, 64], &[256, 73], false),
        ([16, 16, 72], &[196, 68], false),
        ([4, 2, 16], &[13, 21], true),
    ];
    for &([h, hk, d], lens, causal) in cases {
        let m: usize = lens.iter().sum();
        let g = packed_graph(h as u64, hk as u64, d as u64, causal);
        let plan = lower(&g, 0, &[(Rows, m as u64)], &dev).unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].kernel.name(), want_name);
        let mut bounds = Vec::with_capacity(m * 2);
        let mut pos = 0u32;
        for &len in lens {
            let end = pos + len as u32;
            for _ in 0..len {
                bounds.push(pos);
                bounds.push(end);
            }
            pos = end;
        }
        let inputs = [
            draw_tensor(vec![m, h, d], &mut rng),
            draw_tensor(vec![m, hk, d], &mut rng),
            draw_tensor(vec![m, hk, d], &mut rng),
            Tensor::u32(vec![m, 2], bounds),
        ];
        let want = run(&g, &[], 0, &inputs).unwrap();
        let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
        let mut runner = open(plan, &settings).unwrap();
        let got = runner.run(&views).unwrap();
        let got_f32 = got[0].floats().unwrap();
        let want_f32 = want[0].floats().unwrap();
        let cos = cosine_f32(got_f32, want_f32);
        let diff = max_abs_diff(got_f32, want_f32);
        println!("{want_name} h={h}/{hk} d={d:3} m={m:3} c={causal} cos={cos:.6} max={diff:.3e}");
        assert!(
            cos >= 0.99999,
            "{want_name} h={h} d={d} cos={cos} max={diff}"
        );
    }
}

/// Packed causal and bidirectional attention across portable rules.
#[test]
fn packed_matches_cpu() {
    let mut draws = Noise(83);
    let cases = [
        (false, 4u64, 4u64, 72u64, [19usize, 18]),
        (true, 4u64, 2u64, 128u64, [13, 28]),
    ];
    for (causal, heads, kv_heads, width, lens) in cases {
        let rows: usize = lens.iter().sum();
        let g = packed_graph(heads, kv_heads, width, causal);
        let mut bounds = Vec::with_capacity(rows * 2);
        let mut pos = 0u32;
        for len in lens {
            let end = pos + len as u32;
            for _ in 0..len {
                bounds.extend_from_slice(&[pos, end]);
            }
            pos = end;
        }
        let inputs = [
            draw_tensor(vec![rows, heads as usize, width as usize], &mut draws),
            draw_tensor(vec![rows, kv_heads as usize, width as usize], &mut draws),
            draw_tensor(vec![rows, kv_heads as usize, width as usize], &mut draws),
            Tensor::u32(vec![rows, 2], bounds),
        ];
        let want = run(&g, &[], 0, &inputs).unwrap();
        let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
        for settings in runs() {
            for rule in ["flash_packed", "attend", "cpu_attend"] {
                if rule == "cpu_attend" && settings.gpu().is_some() {
                    continue;
                }
                let device = Device::guess(&settings).favoring(rule);
                let Ok(plan) = lower(&g, 0, &[(Rows, rows as u64)], &device) else {
                    continue;
                };
                let mut runner = open(plan, &settings).unwrap();
                let got = runner.run(&views).unwrap();
                let off = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
                assert!(
                    off < 1e-4,
                    "causal={causal} w={width} {rule} {:?}: {off:e}",
                    settings.backend
                );
            }
        }
    }
}
