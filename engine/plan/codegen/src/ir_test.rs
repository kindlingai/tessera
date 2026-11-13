use crate as codegen;
use crate::exec::{self, ArgVal, Grid};
use crate::prelude::*;
use crate::{Dialect, Precision};

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

/// A row's sum of squares: `n` from `at`, in order.
#[kernel]
fn squares(x: Buf<F32>, at: u64, n: u32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += x[at + i as u64] * x[at + i as u64];
    }
    sum
}

/// Scales `n` of `x` from `at` by `by`, into `out`.
#[kernel]
fn scale<const AHEAD: u32>(x: Buf<F32>, mut out: BufMut<F32>, at: u64, n: u32, by: f32) {
    for i in (0..n).unroll(AHEAD) {
        out[at + i as u64] = x[at + i as u64] * by;
    }
}

/// An RMS norm from the two kernels above.
#[kernel]
fn calls(x: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let inv = rsqrt(squares(x, at, width) / width as f32);
    scale::<2>(x, out, at, width, inv);
}

#[test]
fn kernels_call() {
    let body = calls_ir();
    let text = body.text().render(Dialect::Cuda);
    // Callee locals get the call's number. Passed names stay.
    assert!(!text.contains("(u64)at;") && !text.contains("(u32)width;"));
    assert!(text.contains("for (u32 i_1 = 0; i_1 < width; i_1 += 1) {"));
    assert!(text.contains("sum_1 += (x[(at + (u64)i_1)] * x[(at + (u64)i_1)]);"));
    assert!(text.contains("float inv = f_rsqrt((sum_1 / (float)width));"));
    assert!(text.contains("#pragma unroll 2"));
    assert!(text.contains("out[(at + (u64)i_2)] = (x[(at + (u64)i_2)] * inv);"));

    let rows = [[1.0f32, 2.0, 2.0, 4.0], [3.0, 0.0, 0.0, 4.0]];
    let mut x: Vec<u8> = rows
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut out = vec![0u8; x.len()];
    let buf = |at| ArgVal::Buf {
        at,
        skip: 0,
        form: crate::form::F32_FORM,
    };
    let args = [buf(0), buf(1), ArgVal::U32(2), ArgVal::U32(4)];
    let grid = Grid {
        groups: 1,
        threads: 2,
    };
    exec::run(
        &body,
        &args,
        &mut [&mut x, &mut out],
        grid,
        LANES,
        Precision::EXACT,
    );
    let got: Vec<f32> = out
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    // Each row's mean square is 6.25: scale 0.4.
    let want = [0.4f32, 0.8, 0.8, 1.6, 1.2, 0.0, 0.0, 1.6];
    assert_eq!(got, want);
}

/// Sums `red` into its first slot, span halving each level.
#[kernel]
fn tree_sum<const N: u32, const LEVELS: u32>(red: &mut Shared<f32>) -> f32 {
    for level in 0..LEVELS {
        let span = N >> (level + 1);
        if lid() < span {
            red[lid()] += red[lid() + span];
        }
        barrier();
    }
    red[0]
}

/// Scales each row of `acc` by its entry in `alpha`.
#[kernel]
fn scale_rows(acc: &mut Tiles, alpha: &Shared<f32>) {
    acc.scale(|r| alpha[r]);
}

/// Adds `by` to both of a thread's running sums.
#[kernel]
fn bump(sums: &mut [f32; 2], by: f32) {
    for i in 0..2 {
        sums[i as usize] += by;
    }
}

/// Eight threads sum their elements through group memory.
#[kernel]
fn group_sum(x: Buf<F32>, mut out: BufMut<F32>) {
    let mut red = shared::<f32>(8);
    red[lid()] = x[gid() as u64];
    barrier();
    let total = tree_sum::<8, 3>(&mut red);
    let mut sums = [0.0f32; 2];
    bump(&mut sums, total);
    if lid() == 0 {
        out[(gid() / 8) as u64] = sums[1];
    }
    let mut o = tiles(16, 16);
    let mut alpha = shared::<f32>(16);
    alpha[lid()] = total;
    scale_rows(&mut o, &alpha);
}

#[test]
fn callers_lend_memory() {
    let body = group_sum_ir();
    let text = body.text().render(Dialect::Metal);
    // The callee's loop is renamed. The caller's memory is not.
    assert!(text.contains("for (u32 level_1 = 0; level_1 < 3; level_1 += 1) {"));
    assert!(text.contains("u32 span_1 = (8 >> (level_1 + 1));"));
    assert!(text.contains("red[lid] += red[(lid + span_1)];"));
    assert!(text.contains("float total = red[0];"));
    assert!(text.contains("sums[(u32)i_2] += total;"));
    assert!(text.contains("o[i][j][e] *= alpha[r];"));

    let mut x: Vec<u8> = (1..=8).flat_map(|v| (v as f32).to_le_bytes()).collect();
    let mut out = vec![0u8; 4];
    let buf = |at| ArgVal::Buf {
        at,
        skip: 0,
        form: crate::form::F32_FORM,
    };
    let grid = Grid {
        groups: 1,
        threads: 8,
    };
    let args = [buf(0), buf(1)];
    exec::run(
        &body,
        &args,
        &mut [&mut x, &mut out],
        grid,
        LANES,
        Precision::EXACT,
    );
    assert_eq!(f32::from_le_bytes(out[..4].try_into().unwrap()), 36.0);
}
