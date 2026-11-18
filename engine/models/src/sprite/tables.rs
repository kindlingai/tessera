//! Tables the graph reads that depend on the config alone.

use super::{Config, HEAD_DIM, PREFIX, TAPS};
use std::f32::consts::PI;

/// Fourier terms per color channel, and for the mask ratio.
pub(crate) const COLOR_FREQS: u64 = 6;
pub(crate) const RATIO_FREQS: u64 = 8;

/// Color features: channels, their Fourier terms, then alpha.
pub(crate) const COLOR_FEATS: u64 = 3 + 3 * 2 * COLOR_FREQS + 1;

/// One table, by the path the graph declares it at.
pub struct Constant {
    pub path: String,
    pub data: Vec<f32>,
}

/// Sines then cosines of `v * 2^k * pi`.
fn fourier(v: f32, n: u64, out: &mut Vec<f32>) {
    let angles: Vec<f32> = (0..n).map(|k| v * ((1u32 << k) as f32 * PI)).collect();
    out.extend(angles.iter().map(|a| a.sin()));
    out.extend(angles.iter().map(|a| a.cos()));
}

/// RoPE `cos` and `sin` tables, prefix rows unrotated.
fn rope(grid: usize, stride: usize, base: f64) -> [Vec<f32>; 2] {
    let n = HEAD_DIM as usize / 4;
    let freqs: Vec<f64> = (0..n)
        .map(|j| 1.0 / base.powf(j as f64 / n as f64))
        .collect();
    let off = (stride as f64 - 1.0) / 2.0;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for row in 0..PREFIX as usize + grid * grid {
        let mut angles = vec![0.0f64; 2 * n];
        if let Some(p) = row.checked_sub(PREFIX as usize) {
            for (j, f) in freqs.iter().enumerate() {
                angles[j] = ((p / grid * stride) as f64 + off) * f;
                angles[n + j] = ((p % grid * stride) as f64 + off) * f;
            }
        }
        let c: Vec<f32> = angles.iter().map(|a| a.cos() as f32).collect();
        let s: Vec<f32> = angles.iter().map(|a| a.sin() as f32).collect();
        cos.extend(&c);
        cos.extend(&c);
        sin.extend(s.iter().map(|v| -v));
        sin.extend(&s);
    }
    [cos, sin]
}

/// Each row's 3x3 neighbours. Padding reads one past the end.
fn taps(grid: usize) -> [Vec<f32>; 2] {
    let (prefix, taps) = (PREFIX as usize, TAPS as usize);
    let rows = prefix + grid * grid;
    let mut ids = vec![rows as f32; rows * taps];
    for r in 0..grid {
        for c in 0..grid {
            for t in 0..taps {
                let (rr, cc) = ((r + t / 3) as isize - 1, (c + t % 3) as isize - 1);
                if (0..grid as isize).contains(&rr) && (0..grid as isize).contains(&cc) {
                    let at = (prefix + r * grid + c) * taps + t;
                    ids[at] = (prefix + rr as usize * grid + cc as usize) as f32;
                }
            }
        }
    }
    let image = (0..rows).map(|r| (r < prefix) as u8 as f32).collect();
    [ids, image]
}

/// Color tables. Row 0, transparent, is all zero.
fn color(palette: usize) -> [Vec<f32>; 4] {
    let (mut scale, mut shift, mut freqs, mut alpha) = (vec![], vec![], vec![], vec![]);
    for row in 0..palette {
        let on = (row > 0) as u8 as f32;
        scale.extend([2.0 * on, 2.5 * on, 2.5 * on]);
        shift.extend([-on, 0.0, 0.0]);
        for _ in 0..3 {
            freqs.extend((0..COLOR_FREQS).map(|k| (1u32 << k) as f32 * PI));
        }
        alpha.push(on);
    }
    [scale, shift, freqs, alpha]
}

/// Every table the graph reads, for `c`.
pub fn constants(c: &Config) -> Vec<Constant> {
    let (grid, palette, hidden) = (c.grid as usize, c.palette as usize, c.hidden as usize);
    let mut out = Vec::new();
    let mut add = |path: &str, data: Vec<f32>| {
        let path = format!("const.{path}");
        out.push(Constant { path, data });
    };
    for (level, side, stride) in [("pixel", grid, 1), ("patch", grid / 2, 2)] {
        let [cos, sin] = rope(side, stride, c.rope);
        let [ids, image] = taps(side);
        add(&format!("{level}.cos"), cos);
        add(&format!("{level}.sin"), sin);
        add(&format!("{level}.taps"), ids);
        add(&format!("{level}.image"), image);
    }
    let [scale, shift, freqs, alpha] = color(palette);
    add("color.scale", scale);
    add("color.shift", shift);
    add("color.freqs", freqs);
    add("color.alpha", alpha);
    // Palette rows, then a zero row for the mask token.
    let mut pad = vec![0.0; (palette + 1) * palette];
    (0..palette).for_each(|i| pad[i * palette + i] = 1.0);
    add("embed_pad", pad);
    let steps = grid * grid + 1;
    let mut ratio = Vec::new();
    for i in 0..steps {
        let r = i as f32 / (steps - 1) as f32 * 2.0 - 1.0;
        fourier(r, RATIO_FREQS, &mut ratio);
    }
    add("ratio", ratio);
    // Shift, scale, gate twice: both scales start at one.
    let mut six = vec![0.0; 6 * hidden];
    for row in [1, 4] {
        six[row * hidden..][..hidden].fill(1.0);
    }
    add("modulation", six);
    let mut two = vec![0.0; 2 * hidden];
    two[hidden..].fill(1.0);
    add("final_mod", two);
    out
}
