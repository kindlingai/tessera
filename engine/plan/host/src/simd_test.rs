use crate::simd::{Block, Level, WIDE, block_at, dot_at, mix_at, panel_at, reduce};

/// Every level this CPU runs, against plain sums.
#[test]
fn levels_match_sums() {
    let len = 300;
    let row = |seed: usize| -> Vec<f32> {
        let value = |i: usize| ((i * seed + 3) % 13) as f32 * 0.5 - 3.0;
        (0..len).map(value).collect()
    };
    let rows: Vec<Vec<f32>> = (1..=3 + WIDE).map(row).collect();
    let sum = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let (x, w): (Vec<&[f32]>, Vec<&[f32]>) = (
        rows[..3].iter().map(Vec::as_slice).collect(),
        rows[3..].iter().map(Vec::as_slice).collect(),
    );
    for level in Level::all() {
        assert_eq!(dot_at(level, x[0], x[1]), sum(x[0], x[1]));
        let mut out = x[2].to_vec();
        mix_at(level, &mut out, w[0], 0.5, 2.0);
        let want: Vec<f32> = (0..len).map(|i| x[2][i] * 0.5 + 2.0 * w[0][i]).collect();
        assert_eq!(out, want, "{level:?} mix");
        let mut acc = vec![[0.0f32; WIDE]; 3];
        // Two spans, added in the order a long depth uses.
        if !panel_at(level, &x, &w, &mut acc) {
            for col in (0..WIDE).step_by(16) {
                let mut sums: Block = [[[[[0.0; 16]; 4]; 4]; 4]; 4];
                for span in [0..96, 96..len] {
                    let xs: Vec<&[f32]> = x.iter().map(|r| &r[span.clone()]).collect();
                    let ws: Vec<&[f32]> =
                        w[col..col + 16].iter().map(|r| &r[span.clone()]).collect();
                    block_at(&xs, 3, &ws, &mut sums);
                }
                for (r, line) in acc.iter_mut().enumerate() {
                    for (cq, quad) in sums[0].iter().enumerate() {
                        line[col + 4 * cq..col + 4 * cq + 4].copy_from_slice(&reduce(quad)[r]);
                    }
                }
            }
        }
        for (r, line) in acc.iter().enumerate() {
            for (c, v) in line.iter().enumerate() {
                assert_eq!(*v, sum(x[r], w[c]), "{level:?} block {r} {c}");
            }
        }
    }
}

/// Every NEON/AVX row-group remainder matches plain sums.
#[test]
fn panel_row_counts_match() {
    let len = 301;
    let row = |seed: usize| -> Vec<f32> {
        (0..len)
            .map(|i| ((i * seed + 5) % 11) as f32 * 0.5 - 2.5)
            .collect()
    };
    let w_rows: Vec<Vec<f32>> = (1..=WIDE).map(row).collect();
    let w: Vec<&[f32]> = w_rows.iter().map(Vec::as_slice).collect();
    for m in [5usize, 6, 7, 8, 9, 10, 13] {
        let x_rows: Vec<Vec<f32>> = (100..100 + m).map(row).collect();
        let x: Vec<&[f32]> = x_rows.iter().map(Vec::as_slice).collect();
        let mut acc = vec![[0.0f32; WIDE]; m];
        if panel_at(Level::best(), &x, &w, &mut acc) {
            for r in 0..m {
                for c in 0..WIDE {
                    let want: f32 = x[r].iter().zip(w[c]).map(|(a, b)| a * b).sum();
                    assert_eq!(acc[r][c], want, "m={m} r={r} c={c}");
                }
            }
        }
    }
}
