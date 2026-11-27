//! A masked-diffusion model of 16x16 palette sprites.

mod tables;
mod upstream;
mod weights;

pub use tables::constants;
pub use weights::declare;

use crate::shared::{Attend, Side, join, spread};
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, DType, Dim, DimExpr, EntryKind, Graph, Linear, Mask, Model, Norm, Port, Role, Tensor,
    TensorType, Vector,
};

/// Tokens before the image: registers, then text.
const REGISTERS: u64 = 4;
pub const TEXT: u64 = 24;
const PREFIX: u64 = REGISTERS + TEXT;
/// The 3x3 neighbours a depthwise convolution reads.
pub const TAPS: u64 = 9;
/// Rotary tables assume this head width.
const HEAD_DIM: u64 = 64;
/// Every norm's variance floor.
const EPS: f32 = 1e-6;

pub struct Config {
    pub grid: u64,
    pub palette: u64,
    pub hidden: u64,
    pub heads: u64,
    pub ffn: u64,
    pub text_dim: u64,
    /// Blocks before, inside and after the merged middle.
    pub depth: [u64; 3],
    pub max_seqs: u64,
    /// The base of the 2D rotary tables.
    pub rope: f64,
}

impl Config {
    pub fn pixels(&self) -> u64 {
        self.grid * self.grid
    }

    pub fn dims(&self) -> Vec<Dim> {
        vec![
            Dim::bounded(Seqs, self.max_seqs),
            Dim::fixed(Vocab, self.palette),
            Dim::fixed(Hidden, self.hidden),
            Dim::fixed(Heads, self.heads),
            Dim::fixed(HeadDim, self.hidden / self.heads),
            Dim::fixed(Ffn, self.ffn),
            Dim::fixed(Embedding, self.text_dim),
        ]
    }
}

/// A weight used as a plain tensor, any rank.
pub struct Table(pub Tensor);

pub struct Block {
    /// Shift, scale and gate twice, scales offset by one.
    pub modulation: Vector,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub q_norm: Norm,
    pub k_norm: Norm,
    pub up: Linear,
    pub gate: Linear,
    pub down: Linear,
    /// Depthwise weights per tap, times `TAPS` for the mean.
    pub taps: Table,
    pub conv_bias: Table,
}

/// Fixed tables for one grid size.
pub struct Level {
    /// `x * cos + swap(x) * sin`, swap exchanging halves.
    pub cos: Table,
    pub sin: Table,
    /// Each row's neighbours. Past the end reads zero.
    pub taps: Table,
    /// Zero on image rows: they read the bias.
    pub image: Table,
}

pub struct Sprite {
    pub embed: Table,
    pub pos: Table,
    /// The last step's colors, for masked pixels.
    pub draft: Linear,
    pub text_norm: Norm,
    pub text: Linear,
    pub text_bias: Vector,
    pub null_text: Table,
    pub null_cond: Table,
    pub pool_norm: Norm,
    pub pool_in: Linear,
    pub pool_in_bias: Vector,
    pub pool_out: Linear,
    pub pool_out_bias: Vector,
    /// The mask-ratio MLP, one row per masked count.
    pub ratio: Table,
    pub modulate: Linear,
    pub modulate_bias: Vector,
    pub registers: Table,
    pub merge_norm: Norm,
    pub merge: Linear,
    pub merge_bias: Vector,
    pub split: Linear,
    pub split_bias: Vector,
    pub final_mod: Linear,
    pub final_mod_bias: Vector,
    pub head: Linear,
    pub head_bias: Vector,
}

/// Blocks at each level, and each level's tables.
pub struct Stages {
    pub blocks: [Vec<Block>; 3],
    pub pixel: Level,
    pub patch: Level,
}

/// The graph's inputs, one row per canvas.
pub struct Inputs {
    /// Palette ids, `Vocab` where masked.
    pub canvas: Tensor,
    /// Masked pixels per canvas.
    pub masked: Tensor,
    pub text: Tensor,
    pub pooled: Tensor,
    /// `[1, 0]` keeps the text, `[0, 1]` takes the null.
    pub guide: Tensor,
    /// Last step's probabilities, zero on committed pixels.
    pub draft: Tensor,
}

fn port(role: Role, shape: Vec<DimExpr>, dtype: DType) -> Port {
    Port {
        role,
        ty: TensorType::new(shape, dtype),
    }
}

fn inputs(m: &Model, c: &Config) -> Inputs {
    let (s, pixels): (DimExpr, DimExpr) = (Seqs.into(), c.pixels().into());
    let ports = [
        port(Role::Tokens, vec![s.clone(), pixels.clone()], DType::U32),
        port(Role::Positions, vec![s.clone()], DType::U32),
        port(
            Role::Keys,
            vec![s.clone(), TEXT.into(), Embedding.into()],
            DType::F32,
        ),
        port(Role::Keys, vec![s.clone(), Embedding.into()], DType::F32),
        port(Role::Keys, vec![s.clone(), 2u64.into()], DType::F32),
        port(Role::Keys, vec![s, pixels, Vocab.into()], DType::F32),
    ];
    let [canvas, masked, text, pooled, guide, draft] = m.entry(EntryKind::Forward, ports);
    Inputs {
        canvas,
        masked,
        text,
        pooled,
        guide,
        draft,
    }
}

/// `t` where the guide keeps text, else `null`.
fn choose(t: &Tensor, null: &Tensor, guide: &Tensor, rest: &[DimExpr]) -> Tensor {
    let width = rest.iter().cloned().reduce(|a, b| a * b).expect("rank");
    let flat = |x: &Tensor, lead: DimExpr| x.reshape(vec![lead, width.clone()]);
    let null = flat(null, 1u64.into()).expand(0, Seqs);
    let both = flat(t, Seqs.into()).concat(&null);
    let mut shape = vec![Seqs.into(), 2u64.into()];
    shape.extend_from_slice(rest);
    both.reshape(shape).mix(1, guide)
}

/// Text tokens and the global condition.
fn condition(w: &Sprite, x: &Inputs) -> (Tensor, Tensor) {
    let t = x.text.rms_norm(&w.text_norm, EPS).matmul(&w.text);
    let t = t.add_vector(&w.text_bias);
    let c = x.pooled.rms_norm(&w.pool_norm, EPS).matmul(&w.pool_in);
    let c = c.add_vector(&w.pool_in_bias).act(Act::Silu);
    let c = c.matmul(&w.pool_out).add_vector(&w.pool_out_bias);
    let t = choose(&t, &w.null_text.0, &x.guide, &[TEXT.into(), Hidden.into()]);
    let c = choose(&c, &w.null_cond.0, &x.guide, &[Hidden.into()]);
    (t, c)
}

/// One stage's sequence length and grid side.
#[derive(Clone, Copy)]
struct Grid {
    side: u64,
}

impl Grid {
    fn rows(self) -> u64 {
        PREFIX + self.side * self.side
    }
}

/// 2D rotary angles from the level's tables.
fn rotate(t: Tensor, lv: &Level, rows: u64) -> Tensor {
    let table = |x: &Table| {
        x.0.reshape(vec![rows.into(), 1u64.into(), HeadDim.into()])
            .expand(1, Heads)
    };
    let half = HEAD_DIM / 2;
    let swapped = t.slice(3, half, half).concat(&t.slice(3, 0u64, half));
    t.mul(&table(&lv.cos)).add(&swapped.mul(&table(&lv.sin)))
}

fn attention(m: &Model, b: &Block, lv: &Level, h: &Tensor, g: Grid) -> Tensor {
    let lead = vec![Seqs.into(), g.rows().into()];
    let heads = Attend::new(&b.o, lead, [Heads.into(), Heads.into(), HeadDim.into()]);
    let turn = |t: Tensor, side: Side| match side.of([&b.q_norm, &b.k_norm]) {
        Some(norm) => rotate(t.rms_norm(norm, EPS), lv, g.rows()),
        None => t,
    };
    let scale = 1.0 / ((m.size(HeadDim) as f32).sqrt());
    heads.apply([&b.q, &b.k, &b.v], h, turn, |qkv| {
        m.attend(Mask::Bidirectional, scale, qkv, None)
    })
}

/// Depthwise 3x3 over image rows. Prefix rows get zero.
fn conv(b: &Block, lv: &Level, x: &Tensor) -> Tensor {
    let ids = lv.taps.0.cast(DType::U32);
    let rows = x.permute(&[1, 0, 2]).gather(&ids).permute(&[2, 0, 1, 3]);
    let summed = rows.mul(&b.taps.0).mean(2);
    let bias = b.conv_bias.0.gather(&lv.image.0.cast(DType::U32));
    summed.add(&bias)
}

fn mlp(b: &Block, lv: &Level, h: &Tensor) -> Tensor {
    let gate = h.matmul(&b.gate);
    let gate = gate.add(&conv(b, lv, &gate));
    let inner = h.matmul(&b.up).mul(&gate.act(Act::Silu));
    inner.matmul(&b.down)
}

fn block(m: &Model, b: &Block, lv: &Level, x: Tensor, cond: &Tensor, g: Grid) -> Tensor {
    let rows = g.rows();
    let hidden = m.size(Hidden);
    let mods = cond.add_vector(&b.modulation);
    let part = |i: u64| spread(&mods.slice(1, i * hidden, hidden), rows);
    let h = x.rms(EPS).mul(&part(1)).add(&part(0));
    let x = x.add(&attention(m, b, lv, &h, g).mul(&part(2)));
    let h = x.rms(EPS).mul(&part(4)).add(&part(3));
    x.add(&mlp(b, lv, &h).mul(&part(5)))
}

/// `[Seqs, side^2, Hidden]` to 2x2 patches, `[Seqs, side^2/4, 4 Hidden]`.
fn patches(img: &Tensor, side: u64, width: DimExpr) -> Tensor {
    let half = side / 2;
    let shape: Vec<DimExpr> = vec![
        Seqs.into(),
        half.into(),
        2u64.into(),
        half.into(),
        2u64.into(),
        width,
    ];
    let cells = img.reshape(shape).permute(&[0, 1, 3, 2, 4, 5]);
    cells.reshape(vec![
        Seqs.into(),
        (half * half).into(),
        DimExpr::from(Hidden) * 4u64,
    ])
}

/// The inverse of `patches` on `[Seqs, side^2/4, 4 Hidden]`.
fn pixels(core: &Tensor, side: u64) -> Tensor {
    let half = side / 2;
    let shape: Vec<DimExpr> = vec![
        Seqs.into(),
        half.into(),
        half.into(),
        2u64.into(),
        2u64.into(),
        Hidden.into(),
    ];
    let cells = core.reshape(shape).permute(&[0, 1, 3, 2, 4, 5]);
    cells.reshape(vec![Seqs.into(), (side * side).into(), Hidden.into()])
}

/// Logits over the palette for every pixel, `[Seqs, pixels, Vocab]`.
pub fn forward(m: &Model, c: &Config, w: &Sprite, s: &Stages, x: &Inputs) -> Tensor {
    let (fine, coarse) = (Grid { side: c.grid }, Grid { side: c.grid / 2 });
    let px = c.pixels();
    let img = w.embed.0.gather(&x.canvas).add(&w.pos.0);
    let img = img.add(&x.draft.matmul(&w.draft));
    let (text, cond) = condition(w, x);
    let cond = cond.add(&w.ratio.0.gather(&x.masked));
    let mods = cond.act(Act::Silu).matmul(&w.modulate);
    let mods = mods.add_vector(&w.modulate_bias);
    let shape = vec![1u64.into(), REGISTERS.into(), Hidden.into()];
    let regs = w.registers.0.reshape(shape).expand(0, Seqs);
    let mut xs = join(&join(&regs, &text), &img);
    for b in &s.blocks[0] {
        xs = block(m, b, &s.pixel, xs, &mods, fine);
    }
    let (pre, skip) = (xs.slice(1, 0u64, PREFIX), xs.slice(1, PREFIX, px));
    let merged = patches(&skip, c.grid, Hidden.into()).rms_norm(&w.merge_norm, EPS);
    let merged = merged.matmul(&w.merge).add_vector(&w.merge_bias);
    xs = join(&pre, &merged);
    for b in &s.blocks[1] {
        xs = block(m, b, &s.patch, xs, &mods, coarse);
    }
    let (pre, core) = (xs.slice(1, 0u64, PREFIX), xs.slice(1, PREFIX, px / 4));
    let up = core.matmul(&w.split).add_vector(&w.split_bias);
    xs = join(&pre, &skip.add(&pixels(&up, c.grid)));
    for b in &s.blocks[2] {
        xs = block(m, b, &s.pixel, xs, &mods, fine);
    }
    let img = xs.slice(1, PREFIX, px);
    let fin = cond
        .act(Act::Silu)
        .matmul(&w.final_mod)
        .add_vector(&w.final_mod_bias);
    let hidden = m.size(Hidden);
    let (shift, scale) = (fin.slice(1, 0u64, hidden), fin.slice(1, hidden, hidden));
    let img = img
        .rms(EPS)
        .mul(&spread(&scale, px))
        .add(&spread(&shift, px));
    img.matmul(&w.head).add_vector(&w.head_bias)
}

/// One forward entry: canvas and condition in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let (w, s) = declare(&m, c);
    let x = inputs(&m, c);
    let logits = forward(&m, c, &w, &s, &x);
    m.finish([(Role::Logits, &logits)])
}
