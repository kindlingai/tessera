//! Checkpoint tensors, and the graph work that makes layers.

use super::tables::{COLOR_FEATS, COLOR_FREQS, RATIO_FREQS};
use super::{Block, Config, Level, PREFIX, REGISTERS, Sprite, Stages, TAPS, TEXT, Table};
use sensai_trace::DimName::*;
use sensai_trace::{
    Act, DType, DimExpr, DimName, Linear, Model, Norm, PathSeg, Shape, Tensor, TensorType, Vector,
};

/// A checkpoint tensor, or a `const.` table, by dotted name.
fn at(m: &Model, name: &str, shape: Vec<DimExpr>) -> Tensor {
    let seg = |p: &str| match p.parse::<u32>() {
        Ok(i) => PathSeg::Index(i),
        Err(_) => PathSeg::Field(p.into()),
    };
    let path: Vec<PathSeg> = name.split('.').map(seg).collect();
    m.weight(&path, TensorType::new(Shape(shape), DType::F32))
}

/// A torch `nn.Linear` weight, `[out, in]`.
fn linear(m: &Model, name: &str, out: impl Into<DimExpr>, ins: impl Into<DimExpr>) -> Linear {
    Linear::of(at(
        m,
        &format!("{name}.weight"),
        vec![out.into(), ins.into()],
    ))
}

fn bias(m: &Model, name: &str, n: impl Into<DimExpr>) -> Vector {
    Vector::of(at(m, &format!("{name}.bias"), vec![n.into()]))
}

/// `x` through `name.0`, SiLU and `name.2`, as torch's `Sequential`.
fn mlp(m: &Model, name: &str, x: &Tensor, ins: u64) -> Tensor {
    let first = format!("{name}.0");
    let second = format!("{name}.2");
    let hidden = x.matmul(&linear(m, &first, Hidden, ins));
    let hidden = hidden.add_vector(&bias(m, &first, Hidden)).act(Act::Silu);
    let out = hidden.matmul(&linear(m, &second, Hidden, Hidden));
    out.add_vector(&bias(m, &second, Hidden))
}

/// Rows `from..from + len` of a fused matrix, as one matrix.
fn rows(fused: &Tensor, from: impl Into<DimExpr>, len: DimName) -> Linear {
    Linear::of(fused.slice(0, from, len))
}

fn block(m: &Model, stage: &str, i: u64, offset: &Tensor) -> Block {
    let w = |part: &str, shape: Vec<DimExpr>| at(m, &format!("blocks_{stage}.{i}.{part}"), shape);
    let qkv = w("attn.qkv.weight", vec![Hidden * 3, Hidden.into()]);
    let w12 = w("mlp.w12.weight", vec![Ffn * 2, Hidden.into()]);
    let mods = w("mod_bias", vec![6u64.into(), Hidden.into()]);
    let dw = w(
        "mlp.dw.weight",
        vec![Ffn.into(), 1u64.into(), 3u64.into(), 3u64.into()],
    );
    let taps = dw.reshape(vec![Ffn.into(), TAPS.into()]).permute(&[1, 0]);
    let conv_bias = w("mlp.dw.bias", vec![Ffn.into()]);
    Block {
        modulation: Vector::of(mods.reshape(vec![Hidden * 6]).add(offset)),
        q: rows(&qkv, 0u64, Hidden),
        k: rows(&qkv, Hidden, Hidden),
        v: rows(&qkv, Hidden * 2, Hidden),
        o: Linear::of(w("attn.proj.weight", vec![Hidden.into(), Hidden.into()])),
        q_norm: Norm::of(w("attn.q_norm.weight", vec![HeadDim.into()])),
        k_norm: Norm::of(w("attn.k_norm.weight", vec![HeadDim.into()])),
        up: rows(&w12, 0u64, Ffn),
        gate: rows(&w12, Ffn, Ffn),
        down: Linear::of(w("mlp.w3.weight", vec![Hidden.into(), Ffn.into()])),
        // The mean reads a sum: each tap counts nine times.
        taps: Table(taps.scale(TAPS as f32)),
        conv_bias: Table(conv_bias.reshape(vec![1u64.into(), Ffn.into()])),
    }
}

/// The palette's color embeddings, from its Lab values.
fn colors(m: &Model) -> Tensor {
    let k: DimExpr = Vocab.into();
    let table = |name: &str, shape: Vec<DimExpr>| at(m, &format!("const.color.{name}"), shape);
    let lab = at(m, "palette_lab", vec![k.clone(), 3u64.into()]);
    let three = vec![k.clone(), 3u64.into()];
    let x = lab
        .mul(&table("scale", three.clone()))
        .add(&table("shift", three));
    let each = vec![k.clone(), 3u64.into(), COLOR_FREQS.into()];
    let angles = x.reshape(vec![k.clone(), 3u64.into(), 1u64.into()]);
    let angles = angles.expand(2, COLOR_FREQS).mul(&table("freqs", each));
    let terms = angles.sin().concat(&angles.cos());
    let terms = terms.reshape(vec![k.clone(), (6 * COLOR_FREQS).into()]);
    let alpha = table("alpha", vec![k, 1u64.into()]);
    mlp(
        m,
        "color_mlp",
        &x.concat(&terms).concat(&alpha),
        COLOR_FEATS,
    )
}

/// A level's tables, `rows` long.
fn level(m: &Model, name: &str, rows: u64) -> Level {
    let t =
        |table: &str, shape: Vec<DimExpr>| Table(at(m, &format!("const.{name}.{table}"), shape));
    Level {
        cos: t("cos", vec![rows.into(), HeadDim.into()]),
        sin: t("sin", vec![rows.into(), HeadDim.into()]),
        taps: t("taps", vec![rows.into(), TAPS.into()]),
        image: t("image", vec![rows.into()]),
    }
}

/// Declares every weight and the work that makes each layer's.
pub fn declare(m: &Model, c: &Config) -> (Sprite, Stages) {
    let pixels = c.pixels();
    let one: DimExpr = 1u64.into();
    let flat = |name: &str, shape: Vec<DimExpr>| {
        let lead = [one.clone()].into_iter().chain(shape.iter().cloned());
        Table(at(m, name, lead.collect()).reshape(shape))
    };
    let pad = at(m, "const.embed_pad", vec![Vocab + 1, Vocab.into()]);
    let colors = Linear::of(colors(m).permute(&[1, 0]));
    let embed = at(m, "tok_emb.weight", vec![Vocab + 1, Hidden.into()]);
    let ratio = at(
        m,
        "const.ratio",
        vec![(pixels + 1).into(), (2 * RATIO_FREQS).into()],
    );
    let final_bias = at(m, "final_mod.1.bias", vec![Hidden * 2]);
    let final_offset = at(m, "const.final_mod", vec![Hidden * 2]);
    let w = Sprite {
        embed: Table(embed.add(&pad.matmul(&colors))),
        pos: flat("pos_emb", vec![pixels.into(), Hidden.into()]),
        draft: linear(m, "sc_proj", Hidden, Vocab),
        text_norm: Norm::of(at(m, "text_norm.weight", vec![Embedding.into()])),
        text: linear(m, "text_proj", Hidden, Embedding),
        text_bias: bias(m, "text_proj", Hidden),
        null_text: flat("null_text", vec![TEXT.into(), Hidden.into()]),
        null_cond: flat("null_cond", vec![Hidden.into()]),
        pool_norm: Norm::of(at(m, "pool_mlp.0.weight", vec![Embedding.into()])),
        pool_in: linear(m, "pool_mlp.1", Hidden, Embedding),
        pool_in_bias: bias(m, "pool_mlp.1", Hidden),
        pool_out: linear(m, "pool_mlp.3", Hidden, Hidden),
        pool_out_bias: bias(m, "pool_mlp.3", Hidden),
        ratio: Table(mlp(m, "ratio_mlp", &ratio, 2 * RATIO_FREQS)),
        modulate: linear(m, "t_block.1", Hidden * 6, Hidden),
        modulate_bias: bias(m, "t_block.1", Hidden * 6),
        registers: flat("registers", vec![REGISTERS.into(), Hidden.into()]),
        merge_norm: Norm::of(at(m, "down.0.weight", vec![Hidden * 4])),
        merge: linear(m, "down.1", Hidden, Hidden * 4),
        merge_bias: bias(m, "down.1", Hidden),
        split: linear(m, "up", Hidden * 4, Hidden),
        split_bias: bias(m, "up", Hidden * 4),
        final_mod: linear(m, "final_mod.1", Hidden * 2, Hidden),
        final_mod_bias: Vector::of(final_bias.add(&final_offset)),
        head: linear(m, "head", Vocab, Hidden),
        head_bias: bias(m, "head", Vocab),
    };
    let offset = at(m, "const.modulation", vec![Hidden * 6]);
    let stage = |name: &str, n: u64| (0..n).map(|i| block(m, name, i, &offset)).collect();
    let [a, b, d] = c.depth;
    let stages = Stages {
        blocks: [stage("in", a), stage("core", b), stage("out", d)],
        pixel: level(m, "pixel", PREFIX + pixels),
        patch: level(m, "patch", PREFIX + pixels / 4),
    };
    (w, stages)
}
