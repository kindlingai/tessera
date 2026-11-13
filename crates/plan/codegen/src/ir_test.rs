use crate as codegen;
use crate::Dialect;
use crate::prelude::*;

/// A dot product over `k` weights by 32 lanes.
#[kernel]
fn dot<const AHEAD: u32, F: Form>(x: Buf<F32>, w: Buf<F>, mut out: BufMut<F32>, k: u32) {
    if gid() / LANES as u32 >= 1 {
        return;
    }
    let mut acc = 0.0f32;
    for i in (lane() * 4..k).by(128).unroll(AHEAD) {
        let c = w.load4(i as u64);
        let v = x.load4(i as u64);
        acc += v.x * c.x;
        acc += v.y * c.y;
        acc += v.z * c.z;
        acc += v.w * c.w;
    }
    acc = sum32(acc);
    let inv = rsqrt(acc);
    if lane() == 0 {
        out[0] = inv;
    }
}

#[test]
fn trees_render() {
    let body = dot_ir::<2, Bf16>();
    let text = body.text();
    let metal = text.render(Dialect::Metal);
    let cuda = text.render(Dialect::Cuda);
    assert!(metal.contains("simd_shuffle_xor") && cuda.contains("__shfl_xor_sync"));
    for rendered in [&metal, &cuda] {
        assert!(rendered.contains(&format!("u32 lane = gid % {LANES};")));
        assert!(rendered.contains("#pragma unroll 2"));
        assert!(rendered.contains("for (u32 i = (lane * 4); i < k; i += 128) {"));
        assert!(rendered.contains("uint2 c = f_u32x2(w + (u64)i);"));
        assert!(rendered.contains("float4 v = f_f32x4(x + (u64)i);"));
        assert!(rendered.contains("acc += (v.x * f_bits(c.x << 16));"));
        assert!(rendered.contains("acc += (v.w * f_bits(c.y & 0xffff0000u));"));
        assert!(rendered.contains("acc += f_shfl_xor(acc, 16);"));
        assert!(rendered.contains("float inv = f_rsqrt(acc);"));
        assert!(rendered.contains("out[0] = inv;"));
        assert!(rendered.contains("const u16 *w") || rendered.contains("device const u16 *w"));
    }
    // One trip: two wide loads and four multiply-adds.
    let trip = body.trip();
    assert_eq!((trip.loads, trip.float, trip.width), (2.0, 4.0, 12.0));
    assert_eq!(trip.fresh, 1.0, "each trip moves 128 elements");
    assert_eq!(trip.int, 2.0, "the loop's own index work");
    assert_eq!(trip.special, 0.0, "the root lies outside the loop");
}

#[test]
fn constants_fold() {
    let bf16 = dot_ir::<4, Bf16>();
    let f32 = dot_ir::<4, F32>();
    assert_ne!(bf16, f32, "the form is part of the tree");
    assert!(f32.text().body.contains("float4 c = f_f32x4(w + (u64)i);"));
    assert!(f32.text().body.contains("#pragma unroll 4"));
    assert_eq!(f32.trip().width, 16.0);
}
