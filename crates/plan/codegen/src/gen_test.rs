use crate::{Dialect, Gen, LANES, Text, float, hash};
use graph::DType;

/// A dot product over `k` weights by 32 lanes.
fn dot(dtype: DType) -> (Text, sim::Ops) {
    let mut g = Gen::new("test_dot");
    g.floats("x", false);
    g.buffer("w", "u16", false);
    g.floats("out", true);
    g.scalar("k", "u32");
    let lane = g.lane();
    g.guard("gid / 32 >= 1");
    g.line("float acc = 0.0f;");
    g.repeat("i", &format!("{lane} * 4"), "k", "128", 2, |g| {
        let c = g.four(dtype, "w + i");
        let x = g.four(DType::F32, "x + i");
        for (x, c) in x.iter().zip(&c) {
            g.fma("acc", x, c);
        }
    });
    g.sum32("acc");
    let inv = g.rsqrt("acc");
    g.line(format!("if (lane == 0) out[0] = {inv};"));
    let trip = g.trip();
    (g.finish(), trip)
}

#[test]
fn bodies_render() {
    let (text, trip) = dot(DType::Bf16);
    let metal = text.render(Dialect::Metal);
    let cuda = text.render(Dialect::Cuda);
    assert!(metal.contains("simd_shuffle_xor") && cuda.contains("__shfl_xor_sync"));
    assert!(
        cuda.contains("__shfl_xor(v, m)"),
        "the HIP spelling is there too"
    );
    assert!(metal.contains("u32 lid [[thread_index_in_threadgroup]]"));
    assert!(cuda.contains("u32 lid = threadIdx.x;"));
    for body in [&metal, &cuda] {
        assert!(body.contains(&format!("u32 lane = gid % {LANES};")));
        assert!(body.contains("#pragma unroll 2"));
        assert!(body.contains("uint2 v1 = f_u32x2(w + i);"));
        assert!(body.contains("acc += v2.x * f_bits(v1.x << 16);"));
        assert!(body.contains("acc += f_shfl_xor(acc, 16);"));
        assert!(body.contains("acc += f_shfl_xor(acc, 1);"));
        assert!(body.contains("f_rsqrt(acc)"));
    }
    // One trip: two wide loads and four multiply-adds.
    assert_eq!((trip.loads, trip.float, trip.width), (2.0, 4.0, 16.0));
    assert_eq!(trip.int, 2.0, "the loop's own index work");
    assert_eq!(trip.special, 0.0, "the root lies outside the loop");
}

#[test]
fn widths_follow() {
    let (_, bf16) = dot(DType::Bf16);
    let (text, f32) = dot(DType::F32);
    assert_eq!(f32.loads, bf16.loads);
    assert!(text.body.contains("float4 v1 = f_f32x4(w + i);"));
    assert_eq!(hash(&text.body), hash(&dot(DType::F32).0.body));
    assert_ne!(hash(&text.body), hash(&dot(DType::Bf16).0.body));
    assert_eq!(float("x", true), ("x".to_string(), "float", true));
}
