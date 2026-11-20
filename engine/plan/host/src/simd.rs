//! Inner loops per instruction set, picked at run time.

use std::sync::OnceLock;

/// The widest instruction set this CPU runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    /// The target's own baseline.
    Base,
    /// x86 AVX2 with fused multiply-add.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// x86 AVX-512.
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Level {
    /// Every level this CPU runs, the best last.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn all() -> Vec<Level> {
        let fma = is_x86_feature_detected!("fma");
        let mut out = vec![Level::Base];
        if fma && is_x86_feature_detected!("avx2") {
            out.push(Level::Avx2);
        }
        if fma && is_x86_feature_detected!("avx512f") {
            out.push(Level::Avx512);
        }
        out
    }

    /// Every level this CPU runs: its baseline.
    #[cfg(not(target_arch = "x86_64"))]
    pub(crate) fn all() -> Vec<Level> {
        vec![Level::Base]
    }

    /// The best level, found once.
    pub(crate) fn best() -> Level {
        static BEST: OnceLock<Level> = OnceLock::new();
        *BEST.get_or_init(|| *Level::all().last().expect("the baseline"))
    }
}

/// Fused only where the instruction set fuses in one step.
const FUSED: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// `a + x * y` lane by lane.
#[inline(always)]
fn madd<const L: usize, const F: bool>(a: [f32; L], x: [f32; L], y: [f32; L]) -> [f32; L] {
    std::array::from_fn(|l| match F {
        true => x[l].mul_add(y[l], a[l]),
        false => a[l] + x[l] * y[l],
    })
}

/// The sum of `a` times `b`, in `N` running sums.
#[inline(always)]
fn dot_in<const N: usize, const F: bool>(a: &[f32], b: &[f32]) -> f32 {
    let whole = a.len() / N * N;
    let (xs, _) = a[..whole].as_chunks::<N>();
    let (ys, _) = b[..whole].as_chunks::<N>();
    let mut acc = [0.0f32; N];
    for (x, y) in xs.iter().zip(ys) {
        acc = madd::<N, F>(acc, *x, *y);
    }
    let mut sum = acc.iter().sum::<f32>();
    for i in whole..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// `out * keep + p * b`. A zero `keep` drops `out`.
#[inline(always)]
fn mix_in<const F: bool>(out: &mut [f32], b: &[f32], keep: f32, p: f32) {
    if keep == 0.0 {
        out.iter_mut().zip(b).for_each(|(o, x)| *o = p * x);
        return;
    }
    for (o, x) in out.iter_mut().zip(b) {
        *o = match F {
            true => x.mul_add(p, *o * keep),
            false => *o * keep + p * x,
        };
    }
}

/// Rows of `x` by rows of `w`, in `L` lanes.
#[inline(always)]
fn tile_in<const R: usize, const C: usize, const L: usize, const F: bool>(
    x: [&[f32]; R],
    w: [&[f32]; C],
) -> [[f32; C]; R] {
    let len = w[0].len();
    let steps = len / L;
    let xs = x.map(|r| &r[..steps * L].as_chunks::<L>().0[..steps]);
    let ws = w.map(|r| &r[..steps * L].as_chunks::<L>().0[..steps]);
    // Each load feeds `C` or `R` products.
    let mut acc = [[[0.0f32; L]; C]; R];
    for i in 0..steps {
        let wv: [[f32; L]; C] = std::array::from_fn(|c| ws[c][i]);
        for r in 0..R {
            let xv = xs[r][i];
            for c in 0..C {
                acc[r][c] = madd::<L, F>(acc[r][c], xv, wv[c]);
            }
        }
    }
    let mut out = [[0.0f32; C]; R];
    for r in 0..R {
        for c in 0..C {
            out[r][c] = acc[r][c].iter().sum::<f32>();
            for i in steps * L..len {
                out[r][c] += x[r][i] * w[c][i];
            }
        }
    }
    out
}

/// Four columns as two passes of two, for 16 registers.
#[inline(always)]
fn halves<const R: usize, const L: usize, const F: bool>(
    x: [&[f32]; R],
    w: [&[f32]; 4],
) -> [[f32; 4]; R] {
    let a = tile_in::<R, 2, L, F>(x, [w[0], w[1]]);
    let b = tile_in::<R, 2, L, F>(x, [w[2], w[3]]);
    std::array::from_fn(|r| [a[r][0], a[r][1], b[r][0], b[r][1]])
}

/// One x86 vector width's loops, in its own intrinsics.
#[cfg(target_arch = "x86_64")]
macro_rules! level {
    (
        $name:ident, $feature:literal, $vec:ident, $lanes:literal, $group:literal,
        $zero:ident, $load:ident, $fma:ident, $store:ident, $set1:ident
    ) => {
        mod $name {
            use std::arch::x86_64::*;

            const L: usize = $lanes;

            #[target_feature(enable = $feature)]
            unsafe fn sum(v: $vec) -> f32 {
                let mut lanes = [0.0f32; L];
                // SAFETY: `lanes` holds one vector.
                unsafe { $store(lanes.as_mut_ptr(), v) };
                lanes.iter().sum()
            }

            /// Two running vectors, then the tail one at a time.
            #[target_feature(enable = $feature)]
            pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
                let n = a.len().min(b.len());
                let steps = n / (2 * L);
                let (mut s0, mut s1) = ($zero(), $zero());
                for i in 0..steps {
                    let o = i * 2 * L;
                    // SAFETY: `o + 2 L` is at most `n`.
                    unsafe {
                        let (pa, pb) = (a.as_ptr().add(o), b.as_ptr().add(o));
                        s0 = $fma($load(pa), $load(pb), s0);
                        s1 = $fma($load(pa.add(L)), $load(pb.add(L)), s1);
                    }
                }
                // SAFETY: the features are this function's own.
                let mut total = unsafe { sum(s0) + sum(s1) };
                for i in steps * 2 * L..n {
                    total += a[i] * b[i];
                }
                total
            }

            #[target_feature(enable = $feature)]
            pub unsafe fn mix(out: &mut [f32], b: &[f32], keep: f32, p: f32) {
                super::mix_in::<true>(out, b, keep, p)
            }

            /// Vectors per row in a pass, and their columns.
            const V: usize = 2;
            const PASS: usize = V * L;

            /// Rows of `x` times packed `w` from `col`.
            #[target_feature(enable = $feature)]
            unsafe fn rows<const R: usize>(
                x: [&[f32]; R],
                packed: &[f32],
                (col, acc): (usize, &mut [[f32; super::WIDE]]),
            ) {
                let k = packed.len() / super::WIDE;
                let mut held = [[$zero(); V]; R];
                let mut wv = [$zero(); V];
                // SAFETY: `panel` checks every length used here.
                unsafe {
                    for r in 0..R {
                        for v in 0..V {
                            held[r][v] = $load(acc[r].as_ptr().add(col + v * L));
                        }
                    }
                    for kk in 0..k {
                        let p = packed.as_ptr().add(kk * super::WIDE + col);
                        for v in 0..V {
                            wv[v] = $load(p.add(v * L));
                        }
                        for r in 0..R {
                            let b = $set1(*x[r].get_unchecked(kk));
                            for v in 0..V {
                                held[r][v] = $fma(b, wv[v], held[r][v]);
                            }
                        }
                    }
                    for r in 0..R {
                        for v in 0..V {
                            $store(acc[r].as_mut_ptr().add(col + v * L), held[r][v]);
                        }
                    }
                }
            }

            /// Every row of `x` times packed `w`, in groups.
            #[target_feature(enable = $feature)]
            pub unsafe fn panel(x: &[&[f32]], packed: &[f32], acc: &mut [[f32; super::WIDE]]) {
                let k = packed.len() / super::WIDE;
                assert!(x.iter().all(|r| r.len() == k), "rows of one length");
                assert_eq!(acc.len(), x.len(), "a sum row per row");
                for (col, first) in (0..super::WIDE)
                    .step_by(PASS)
                    .flat_map(|c| (0..x.len()).step_by($group).map(move |f| (c, f)))
                {
                    let row = |r: usize| x[first + r];
                    let acc = (col, &mut acc[first..]);
                    // SAFETY: the features are this function's own.
                    unsafe {
                        match (x.len() - first).min($group) {
                            1 => rows::<1>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                            2 => rows::<2>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                            3 => rows::<3>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                            4 => rows::<4>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                            5 => rows::<5>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                            6 => rows::<6>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                            7 => rows::<7>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                            8 => rows::<8>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                            9 => rows::<9>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                            10 => {
                                rows::<10>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1))
                            }
                            11 => {
                                rows::<11>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1))
                            }
                            _ => rows::<12>(std::array::from_fn(row), packed, (acc.0, &mut *acc.1)),
                        }
                    }
                }
            }
        }
    };
}

/// All four columns at once, for 32 registers.
#[inline(always)]
fn whole<const R: usize, const L: usize, const F: bool>(
    x: [&[f32]; R],
    w: [&[f32]; 4],
) -> [[f32; 4]; R] {
    tile_in::<R, 4, L, F>(x, w)
}

// AVX2 holds 6 rows of two vectors, AVX-512 holds 12.
#[cfg(target_arch = "x86_64")]
level!(
    avx2,
    "avx2,fma",
    __m256,
    8,
    6,
    _mm256_setzero_ps,
    _mm256_loadu_ps,
    _mm256_fmadd_ps,
    _mm256_storeu_ps,
    _mm256_set1_ps
);
#[cfg(target_arch = "x86_64")]
level!(
    avx512,
    "avx512f,fma",
    __m512,
    16,
    12,
    _mm512_setzero_ps,
    _mm512_loadu_ps,
    _mm512_fmadd_ps,
    _mm512_storeu_ps,
    _mm512_set1_ps
);

/// The sum of `a` times `b` at `level`.
pub(crate) fn dot_at(level: Level, a: &[f32], b: &[f32]) -> f32 {
    match level {
        // SAFETY: `Level::all` found the features.
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => unsafe { avx2::dot(a, b) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => unsafe { avx512::dot(a, b) },
        _ => dot_in::<16, FUSED>(a, b),
    }
}

/// `out * keep + p * b` into `out` at `level`.
pub(crate) fn mix_at(level: Level, out: &mut [f32], b: &[f32], keep: f32, p: f32) {
    match level {
        // SAFETY: `Level::all` found the features.
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => unsafe { avx2::mix(out, b, keep, p) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => unsafe { avx512::mix(out, b, keep, p) },
        _ => mix_in::<FUSED>(out, b, keep, p),
    }
}

/// Rows of `x` by four of `w`, into lane 0.
fn quad_base<const R: usize>(x: [&[f32]; R], w: [&[f32]; 4], acc: &mut Lanes) {
    // NEON holds four columns. The x86 baseline holds two.
    let sums = match cfg!(target_arch = "aarch64") {
        true => whole::<R, 4, FUSED>(x, w),
        false => halves::<R, 4, FUSED>(x, w),
    };
    for (r, line) in sums.iter().enumerate() {
        for (c, s) in line.iter().enumerate() {
            acc[r][c][0] += s;
        }
    }
}

/// Rows of `x` by 16 rows of `w`, one span.
pub(crate) fn block_at(x: &[&[f32]], rows: usize, w: &[&[f32]], sums: &mut Block) {
    for first in (0..rows).step_by(4) {
        let row = |r: usize| x[first + r];
        for col in (0..w.len()).step_by(4) {
            let (acc, w) = (
                &mut sums[first / 4][col / 4],
                std::array::from_fn(|c| w[col + c]),
            );
            match rows - first {
                1 => quad_base::<1>(std::array::from_fn(row), w, acc),
                2 => quad_base::<2>(std::array::from_fn(row), w, acc),
                3 => quad_base::<3>(std::array::from_fn(row), w, acc),
                _ => quad_base::<4>(std::array::from_fn(row), w, acc),
            }
        }
    }
}

/// A 4 by 4 block's partial sums, in lanes.
pub(crate) type Lanes = [[[f32; 16]; 4]; 4];

/// A 16 by 16 block's quads.
pub(crate) type Block = [[Lanes; 4]; 4];

/// Each sum of `acc`, its lanes added.
pub(crate) fn reduce(acc: &Lanes) -> [[f32; 4]; 4] {
    acc.map(|line| line.map(|lanes| lanes.iter().sum()))
}

/// Columns in one packed block.
pub(crate) const WIDE: usize = 32;

/// Depth one packed panel holds, so it stays in cache.
#[cfg(target_arch = "x86_64")]
const PANEL: usize = 128;

/// Rows of `w` as `[depth][WIDE]`, eight at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack(w: &[&[f32]], packed: &mut [f32]) {
    use std::arch::x86_64::*;
    let len = w[0].len();
    assert!(w.len() == WIDE && w.iter().all(|r| r.len() == len) && packed.len() >= len * WIDE);
    let whole = len / 8 * 8;
    for group in 0..WIDE / 8 {
        let rows = &w[group * 8..group * 8 + 8];
        let at = |kk: usize| kk * WIDE + group * 8;
        for kk in (0..whole).step_by(8) {
            // SAFETY: the assert covers every index here.
            unsafe {
                let mut r = [_mm256_setzero_ps(); 8];
                for i in 0..8 {
                    r[i] = _mm256_loadu_ps(rows[i].as_ptr().add(kk));
                }
                // An 8 by 8 transpose: pairs, then quads, then halves.
                let mut t = [_mm256_setzero_ps(); 8];
                for i in (0..8).step_by(2) {
                    t[i] = _mm256_unpacklo_ps(r[i], r[i + 1]);
                    t[i + 1] = _mm256_unpackhi_ps(r[i], r[i + 1]);
                }
                let mut q = [_mm256_setzero_ps(); 8];
                for i in [0, 4] {
                    q[i] = _mm256_shuffle_ps::<0x44>(t[i], t[i + 2]);
                    q[i + 1] = _mm256_shuffle_ps::<0xEE>(t[i], t[i + 2]);
                    q[i + 2] = _mm256_shuffle_ps::<0x44>(t[i + 1], t[i + 3]);
                    q[i + 3] = _mm256_shuffle_ps::<0xEE>(t[i + 1], t[i + 3]);
                }
                for j in 0..4 {
                    let lo = _mm256_permute2f128_ps::<0x20>(q[j], q[j + 4]);
                    let hi = _mm256_permute2f128_ps::<0x31>(q[j], q[j + 4]);
                    _mm256_storeu_ps(packed.as_mut_ptr().add(at(kk + j)), lo);
                    _mm256_storeu_ps(packed.as_mut_ptr().add(at(kk + j + 4)), hi);
                }
            }
        }
        for kk in whole..len {
            for (i, row) in rows.iter().enumerate() {
                packed[at(kk) + i] = row[kk];
            }
        }
    }
}

/// Rows of `x` by those of `w`, if `level` packs.
pub(crate) fn panel_at(level: Level, x: &[&[f32]], w: &[&[f32]], acc: &mut [[f32; WIDE]]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if level != Level::Base {
        let k = w[0].len();
        let mut packed = [0.0f32; PANEL * WIDE];
        for start in (0..k).step_by(PANEL) {
            let end = (start + PANEL).min(k);
            let (ws, xs): (Vec<&[f32]>, Vec<&[f32]>) = (
                w.iter().map(|r| &r[start..end]).collect(),
                x.iter().map(|r| &r[start..end]).collect(),
            );
            let packed = &mut packed[..(end - start) * WIDE];
            // SAFETY: `Level::all` found the features.
            unsafe {
                pack(&ws, packed);
                match level {
                    Level::Avx512 => avx512::panel(&xs, packed, acc),
                    _ => avx2::panel(&xs, packed, acc),
                }
            }
        }
        return true;
    }
    let _ = (level, x, w, acc);
    false
}
