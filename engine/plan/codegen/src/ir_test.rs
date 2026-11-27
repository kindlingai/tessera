use crate as tessera_codegen;
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

/// Shadowing in a nested block and in the same block.
#[kernel]
fn shadows(mut out: BufMut<F32>, base: u32, second: bool) {
    let col = base + lid() * 16;
    if second {
        let col = col + 64;
        out[col as u64] = 1.0;
    }
    let col = col + 1;
    let col = col * 2;
    out[col as u64] = 2.0;
}

#[test]
fn shadows_read_outer() {
    let text = shadows_ir().text().body;
    for line in [
        "u32 col = (base + (lid * 16));",
        "u32 col__1 = (col + 64);",
        "out[(u64)col__1] = 1.0f;",
        "u32 col__1 = (col + 1);",
        "u32 col__2 = (col__1 * 2);",
        "out[(u64)col__2] = 2.0f;",
    ] {
        assert!(text.contains(line), "`{line}` in\n{text}");
    }
}

/// `N` bits of `x` from `at`, as a float.
#[helper("field{N}")]
fn field<const N: u32>(x: u32, at: u32) -> f32 {
    let mask = !((1u32 << N) - 1);
    let v = (x >> at) & !mask;
    if N >= 8 {
        v as f32
    } else if at == 0 {
        -(v as f32)
    } else {
        0.0
    }
}

/// The two bytes of `w[i]` as fields, summed.
#[helper]
fn pair(w: Buf<U8>, i: u64) -> f32 {
    let b = w + i * 2;
    let x = b[0] | (b[1] << 8);
    field::<4>(x, 0) + field::<8>(x, 4)
}

#[test]
fn helpers_render() {
    let helper = pair_ir();
    assert_eq!(helper.name, "pair");
    let text = helper.text();
    assert!(
        text.starts_with("f_fn float pair(f_const(u8) w, u64 i) {\n"),
        "{text}"
    );
    assert!(text.contains("f_const(u8) b = (w + (i * 2));"));
    assert!(text.contains("return (field4(x, 0) + field8(x, 4));"));
    // Called helpers come along, each once.
    let names: Vec<&str> = helper
        .body
        .helpers
        .iter()
        .map(|h| h.name.as_str())
        .collect();
    assert_eq!(names, ["field4", "field8"]);
    let field = &helper.body.helpers[0];
    let text = field.text();
    assert!(text.contains("u32 mask = (~15);"), "{text}");
    // A build-time choice leaves one branch.
    assert!(
        text.contains("return ((at == 0) ? (-(float)v) : 0.0f);"),
        "{text}"
    );
    assert!(helper.body.helpers[1].text().contains("return (float)v;"));
}

/// A packed four that is changed or stored whole.
#[kernel]
fn bump<F: Form>(x: Buf<F>, mut out: BufMut<F>) {
    let mut s = x.load4(0);
    s.x += 1.0;
    out.store4(0, s);
    let c = x.load4(4);
    out.store4(4, c);
}

/// Packed fours decode before a change or a whole use.
#[test]
fn packed_fours_decode() {
    let text = bump_ir::<Bf16>().text();
    for rendered in [text.render(Dialect::Metal), text.render(Dialect::Cuda)] {
        assert!(rendered.contains("float4 s;"), "{rendered}");
        assert!(
            rendered.contains("f_store4_bf16(out + 4, f_make4("),
            "{rendered}"
        );
    }
}
