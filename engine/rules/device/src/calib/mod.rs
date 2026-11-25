//! Host references for grouped SYRK sums and the expert sort.

use tessera_graph::Bf16;

/// Tile side length for grouped SYRK and BF16 GEMM.
pub const TILE: usize = 128;

/// Deterministic stable counting sort matching `calib_sort_*` kernels.
pub fn sort_cpu(ids: &[u32], experts: usize, first: u32) -> (Vec<u32>, Vec<u32>, Vec<i64>) {
    let mut counts = vec![0u32; experts];
    for &raw in ids {
        if raw >= first {
            let e = (raw - first) as usize;
            if e < experts {
                counts[e] += 1;
            }
        }
    }
    let mut offsets = vec![0u32; experts + 1];
    for e in 0..experts {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let total = offsets[experts] as usize;
    let mut perm = vec![0u32; total];
    let mut cur = offsets[..experts].to_vec();
    for (i, &raw) in ids.iter().enumerate() {
        if raw >= first {
            let e = (raw - first) as usize;
            if e < experts {
                let at = cur[e] as usize;
                perm[at] = i as u32;
                cur[e] += 1;
            }
        }
    }
    let i64_counts: Vec<i64> = counts.into_iter().map(i64::from).collect();
    (offsets, perm, i64_counts)
}

/// Rounds `v` to BF16 and widens back to `f32`.
pub(crate) fn round_bf16(v: f32) -> f32 {
    f32::from(Bf16::from(v))
}

/// Accumulates packed upper-triangle tiles `X_e^T X_e` in fixed order.
pub fn syrk_tiles_cpu(
    x: &[f32],
    perm: &[u32],
    offsets: &[u32],
    dim: usize,
    tile: usize,
    repeat: usize,
    acc: &mut [f64],
) {
    assert!(tile > 0 && dim.is_multiple_of(tile));
    let experts = offsets.len() - 1;
    let blocks = dim / tile;
    let total_tiles = blocks * (blocks + 1) / 2;
    assert_eq!(acc.len(), experts * total_tiles * tile * tile);
    for e in 0..experts {
        let start = offsets[e] as usize;
        let end = offsets[e + 1] as usize;
        if start == end {
            continue;
        }
        let mut tidx = 0usize;
        for br in 0..blocks {
            for bc in br..blocks {
                let dst = &mut acc[(e * total_tiles + tidx) * tile * tile..][..tile * tile];
                for &pick in &perm[start..end] {
                    let row = (pick as usize) / repeat;
                    let xr = &x[row * dim + br * tile..][..tile];
                    let xc = &x[row * dim + bc * tile..][..tile];
                    for lr in 0..tile {
                        let a = f64::from(round_bf16(xr[lr]));
                        for lc in 0..tile {
                            dst[lr * tile + lc] += a * f64::from(round_bf16(xc[lc]));
                        }
                    }
                }
                tidx += 1;
            }
        }
    }
}

/// Accumulates full symmetric `[dim, dim]` dense Gram `X^T X`.
pub fn syrk_dense_cpu(x: &[f32], rows: usize, dim: usize, acc: &mut [f64]) {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(acc.len(), dim * dim);
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        for i in 0..dim {
            let a = f64::from(round_bf16(xr[i]));
            for j in i..dim {
                let prod = a * f64::from(round_bf16(xr[j]));
                acc[i * dim + j] += prod;
                if i != j {
                    acc[j * dim + i] += prod;
                }
            }
        }
    }
}
