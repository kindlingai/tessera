//! tessera-small's upstream weights, offered under the denoiser's names.

use sensai_checkpoint::{Error, Offer, Slice, Stored};
use sensai_engine::sprite::{self, Config, REGISTERS, TEXT};
use std::collections::BTreeMap;
use std::f32::consts::PI;

const HEAD_DIM: usize = 64;
const PREFIX: usize = (REGISTERS + TEXT) as usize;
const COLOR_FREQS: usize = 6;
const RATIO_FREQS: usize = 8;
const RATIO_ROWS: usize = 257;
const TAPS: usize = sprite::TAPS as usize;
const PATCH: usize = 2;
const ROPE_BASE: f64 = 100.0;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `x @ w.T + b` for `x` with rows of `w`'s width.
fn linear(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let (outs, ins) = (b.len(), w.len() / b.len());
    let mut y = Vec::with_capacity(x.len() / ins * outs);
    for row in x.chunks(ins) {
        for (o, bias) in b.iter().enumerate() {
            let w = &w[o * ins..][..ins];
            let dot: f32 = row.iter().zip(w).map(|(a, b)| a * b).sum();
            y.push(dot + bias);
        }
    }
    y
}

/// Sines then cosines of `x * 2^k * pi`, per value.
fn fourier(x: &[f32], n: usize, out: &mut Vec<f32>) {
    for &v in x {
        let angles: Vec<f32> = (0..n).map(|k| v * ((1u32 << k) as f32 * PI)).collect();
        out.extend(angles.iter().map(|a| a.sin()));
        out.extend(angles.iter().map(|a| a.cos()));
    }
}

/// Color features per Lab palette row. Row 0 is transparent.
fn color_feats(lab: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for (i, c) in lab.chunks(3).enumerate() {
        let alpha = if i == 0 { 0.0 } else { 1.0 };
        let x = [
            (c[0] * 2.0 - 1.0) * alpha,
            c[1] / 0.4 * alpha,
            c[2] / 0.4 * alpha,
        ];
        out.extend(x);
        fourier(&x, COLOR_FREQS, &mut out);
        out.push(alpha);
    }
    out
}

/// RoPE `cos` and `sin` tables, prefix rows unrotated.
fn rope(grid: usize, stride: usize) -> (Vec<f32>, Vec<f32>) {
    let n = HEAD_DIM / 4;
    let freqs: Vec<f64> = (0..n)
        .map(|j| 1.0 / ROPE_BASE.powf(j as f64 / n as f64))
        .collect();
    let off = (stride as f64 - 1.0) / 2.0;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for row in 0..PREFIX + grid * grid {
        let mut angles = vec![0.0f64; 2 * n];
        if let Some(p) = row.checked_sub(PREFIX) {
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
    (cos, sin)
}

/// Each row's 3x3 neighbours. Padding is one past the end.
fn taps(grid: usize) -> (Vec<f32>, Vec<f32>) {
    let rows = PREFIX + grid * grid;
    let mut ids = vec![rows as f32; rows * TAPS];
    for r in 0..grid {
        for c in 0..grid {
            for t in 0..TAPS {
                let (rr, cc) = ((r + t / 3) as isize - 1, (c + t % 3) as isize - 1);
                if (0..grid as isize).contains(&rr) && (0..grid as isize).contains(&cc) {
                    let at = (PREFIX + r * grid + c) * TAPS + t;
                    ids[at] = (PREFIX + rr as usize * grid + cc as usize) as f32;
                }
            }
        }
    }
    let image = (0..rows).map(|r| (r < PREFIX) as u8 as f32).collect();
    (ids, image)
}

/// Two linears with SiLU between, `name.0` then `name.2`.
fn mlp(inner: &dyn Offer, name: &str, x: &[f32]) -> Result<Vec<f32>, Error> {
    let w = |i: usize, part: &str| inner.floats(&format!("ema.{name}.{i}.{part}"));
    let mut hidden = linear(x, &w(0, "weight")?, &w(0, "bias")?);
    hidden.iter_mut().for_each(|h| *h = silu(*h));
    Ok(linear(&hidden, &w(2, "weight")?, &w(2, "bias")?))
}

/// Tensors taken as they are: new name, `ema.` name.
const COPIES: [(&str, &str); 24] = [
    ("pos", "pos_emb"),
    ("draft", "sc_proj.weight"),
    ("text_norm", "text_norm.weight"),
    ("text", "text_proj.weight"),
    ("text_bias", "text_proj.bias"),
    ("null_text", "null_text"),
    ("null_cond", "null_cond"),
    ("pool_norm", "pool_mlp.0.weight"),
    ("pool_in", "pool_mlp.1.weight"),
    ("pool_in_bias", "pool_mlp.1.bias"),
    ("pool_out", "pool_mlp.3.weight"),
    ("pool_out_bias", "pool_mlp.3.bias"),
    ("modulate", "t_block.1.weight"),
    ("modulate_bias", "t_block.1.bias"),
    ("registers", "registers"),
    ("merge_norm", "down.0.weight"),
    ("merge", "down.1.weight"),
    ("merge_bias", "down.1.bias"),
    ("split", "up.weight"),
    ("split_bias", "up.bias"),
    ("final_mod", "final_mod.1.weight"),
    ("head", "head.weight"),
    ("head_bias", "head.bias"),
    ("palette", "palette_rgb"),
];

/// Block parts taken as they are.
const BLOCK_COPIES: [(&str, &str); 5] = [
    ("o", "attn.proj.weight"),
    ("q_norm", "attn.q_norm.weight"),
    ("k_norm", "attn.k_norm.weight"),
    ("down", "mlp.w3.weight"),
    ("conv_bias", "mlp.dw.bias"),
];

/// Block parts cut from a fused weight: piece of pieces.
const BLOCK_CUTS: [(&str, &str, usize, usize); 5] = [
    ("q", "attn.qkv.weight", 0, 3),
    ("k", "attn.qkv.weight", 1, 3),
    ("v", "attn.qkv.weight", 2, 3),
    ("up", "mlp.w12.weight", 0, 2),
    ("gate", "mlp.w12.weight", 1, 2),
];

const STAGES: [&str; 3] = ["in", "core", "out"];

/// Upstream `ema.*` tensors, folded when read. `palette` holds RGB.
pub(crate) struct Upstream<O> {
    inner: O,
    grid: u64,
    depth: [u64; 3],
}

impl<O: Offer> Upstream<O> {
    pub fn new(inner: O, config: &Config) -> Self {
        let (grid, depth) = (config.grid, config.depth);
        Self { inner, grid, depth }
    }

    fn ema(&self, name: &str) -> Result<Vec<f32>, Error> {
        self.inner.floats(&format!("ema.{name}"))
    }

    /// The grid and stride of a level's tables.
    fn level(&self, name: &str) -> Option<(usize, usize)> {
        let grid = self.grid as usize;
        match name {
            "pixel" => Some((grid, 1)),
            "patch" => Some((grid / PATCH, PATCH)),
            _ => None,
        }
    }

    fn table(&self, level: (usize, usize), table: &str) -> Option<Vec<f32>> {
        let (grid, stride) = level;
        Some(match table {
            "cos" => rope(grid, stride).0,
            "sin" => rope(grid, stride).1,
            "taps" => taps(grid).0,
            "image" => taps(grid).1,
            _ => return None,
        })
    }

    /// Part `part` of block `src`.
    fn block(&self, src: &str, part: &str) -> Result<Vec<f32>, Error> {
        let get = |name: &str| self.ema(&format!("{src}.{name}"));
        if let Some((_, from)) = BLOCK_COPIES.iter().find(|(to, _)| *to == part) {
            return get(from);
        }
        if let Some(&(_, from, piece, pieces)) = BLOCK_CUTS.iter().find(|c| c.0 == part) {
            let fused = get(from)?;
            let len = fused.len() / pieces;
            return Ok(fused[piece * len..][..len].to_vec());
        }
        match part {
            "modulation" => {
                let mut m = get("mod_bias")?;
                let width = m.len() / 6;
                for row in [1, 4] {
                    m[row * width..][..width].iter_mut().for_each(|v| *v += 1.0);
                }
                Ok(m)
            }
            "taps" => {
                let dw = get("mlp.dw.weight")?;
                let channels = dw.len() / TAPS;
                let mut taps = Vec::with_capacity(dw.len());
                for tap in 0..TAPS {
                    taps.extend((0..channels).map(|c| dw[c * TAPS + tap] * TAPS as f32));
                }
                Ok(taps)
            }
            _ => Err(Error::Missing(format!("{src}.{part}"))),
        }
    }

    fn folded(&self, name: &str) -> Result<Vec<f32>, Error> {
        match name {
            "embed" => {
                let mut embed = self.ema("tok_emb.weight")?;
                let feats = color_feats(&self.ema("palette_lab")?);
                let colors = mlp(&self.inner, "color_mlp", &feats)?;
                embed.iter_mut().zip(&colors).for_each(|(e, c)| *e += c);
                Ok(embed)
            }
            "ratio" => {
                let steps: Vec<f32> = (0..RATIO_ROWS)
                    .map(|i| i as f32 / (RATIO_ROWS - 1) as f32 * 2.0 - 1.0)
                    .collect();
                let mut feats = Vec::new();
                fourier(&steps, RATIO_FREQS, &mut feats);
                mlp(&self.inner, "ratio_mlp", &feats)
            }
            "final_mod_bias" => {
                let mut bias = self.ema("final_mod.1.bias")?;
                let half = bias.len() / 2;
                bias[half..].iter_mut().for_each(|v| *v += 1.0);
                Ok(bias)
            }
            _ => Err(Error::Missing(name.into())),
        }
    }

    /// Every name on offer, with its shape.
    fn names(&self) -> Vec<(String, Vec<usize>)> {
        let shapes: BTreeMap<String, Vec<usize>> = (self.inner.slices().into_iter())
            .map(|s| (s.name, s.shape))
            .collect();
        let shape = |n: &str| shapes.get(&format!("ema.{n}")).cloned().unwrap_or_default();
        let mut out = Vec::new();
        for (to, from) in COPIES {
            let mut s = shape(from);
            if s.len() == 3 && s[0] == 1 {
                s.remove(0);
            }
            out.push((to.to_string(), s));
        }
        let ratio = shape("ratio_mlp.2.bias");
        out.push(("embed".into(), shape("tok_emb.weight")));
        out.push(("ratio".into(), vec![RATIO_ROWS, ratio.iter().product()]));
        out.push(("final_mod_bias".into(), shape("final_mod.1.bias")));
        for level in ["pixel", "patch"] {
            let (grid, _) = self.level(level).expect("a level");
            let rows = PREFIX + grid * grid;
            for (t, s) in [("cos", vec![rows, HEAD_DIM]), ("sin", vec![rows, HEAD_DIM])] {
                out.push((format!("{level}.{t}"), s));
            }
            out.push((format!("{level}.taps"), vec![rows, TAPS]));
            out.push((format!("{level}.image"), vec![rows]));
        }
        for (stage, depth) in STAGES.iter().zip(self.depth) {
            for i in 0..depth {
                let src = |n: &str| shape(&format!("blocks_{stage}.{i}.{n}"));
                let dst = |n: &str| format!("{stage}.{i}.{n}");
                for (to, from) in BLOCK_COPIES {
                    let mut s = src(from);
                    if to == "conv_bias" {
                        s.insert(0, 1);
                    }
                    out.push((dst(to), s));
                }
                for (to, from, _, pieces) in BLOCK_CUTS {
                    let mut s = src(from);
                    s[0] /= pieces;
                    out.push((dst(to), s));
                }
                let modulation = src("mod_bias").iter().product();
                out.push((dst("modulation"), vec![modulation]));
                let channels = src("mlp.dw.bias").iter().product();
                out.push((dst("taps"), vec![TAPS, channels]));
            }
        }
        out
    }
}

impl<O: Offer> Offer for Upstream<O> {
    fn slices(&self) -> Vec<Slice> {
        (self.names().into_iter())
            .map(|(name, shape)| Slice {
                name,
                shape,
                stored: Stored::F32,
            })
            .collect()
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        if let Some((_, from)) = COPIES.iter().find(|(to, _)| *to == name) {
            return self.ema(from);
        }
        if let Some((level, table)) = name.split_once('.')
            && let Some(level) = self.level(level)
        {
            return self
                .table(level, table)
                .ok_or_else(|| Error::Missing(name.into()));
        }
        let mut parts = name.splitn(3, '.');
        match (parts.next(), parts.next(), parts.next()) {
            (Some(stage), Some(i), Some(part)) if STAGES.contains(&stage) => {
                self.block(&format!("blocks_{stage}.{i}"), part)
            }
            _ => self.folded(name),
        }
    }

    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        Err(Error::DType(format!("{name} holds floats, not a form")))
    }
}
