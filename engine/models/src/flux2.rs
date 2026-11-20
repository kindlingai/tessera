//! FLUX.2's flow transformer: double-stream blocks, then single-stream.

use crate::keys::Keys;
use crate::shared::{join, spread};
use serde_json::json;
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, Composite, DType, Dim, DimExpr, EntryKind, Graph, Linear, Mask, Model, Norm, Pairs,
    PathSeg, Port, Role, Tensor, TensorType, Weights,
};

/// Channels per latent token.
pub const LATENT: u64 = 128;
/// Sinusoid channels the host computes from the timestep.
pub const TIME: u64 = 256;
/// RoPE axes: time, height, width, layer.
pub const AXES: usize = 4;

pub struct Config {
    pub hidden: u64,
    pub heads: u64,
    pub ffn: u64,
    /// Width of the text encoder's rows.
    pub text_dim: u64,
    /// Text rows, first and fixed: image rows start after them.
    pub text: u64,
    pub double: u64,
    pub single: u64,
    pub theta: f32,
    /// Channels each RoPE axis turns, in order.
    pub axes: [u64; AXES],
    pub eps: f32,
    pub max_seqs: u64,
    /// The most image rows in one launch.
    pub max_rows: u64,
}

impl Config {
    /// The transformer `k` describes, `text` rows and up to `max_rows`.
    pub fn read(k: &mut Keys, max_seqs: u64, text: u64, max_rows: u64) -> Self {
        k.fixed("in_channels", json!(LATENT));
        k.fixed("patch_size", json!(1));
        k.fixed("guidance_embeds", json!(false));
        k.fixed("timestep_guidance_channels", json!(TIME));
        let heads: u64 = k.need(&["num_attention_heads"]);
        let head_dim: u64 = k.need(&["attention_head_dim"]);
        let hidden = heads * head_dim;
        let ratio: f32 = k.need(&["mlp_ratio"]);
        let axes: Vec<u64> = k.need(&["axes_dims_rope"]);
        let axes = match <[u64; AXES]>::try_from(axes) {
            Ok(a) if a.iter().sum::<u64>() == head_dim => a,
            _ => {
                let why = format!("the graph takes {AXES} axes that fill attention_head_dim");
                k.misfit("axes_dims_rope", why);
                [0; AXES]
            }
        };
        Self {
            hidden,
            heads,
            ffn: (hidden as f32 * ratio) as u64,
            text_dim: k.need(&["joint_attention_dim"]),
            text,
            double: k.need(&["num_layers"]),
            single: k.need(&["num_single_layers"]),
            theta: k.need(&["rope_theta"]),
            axes,
            eps: k.need(&["eps"]),
            max_seqs,
            max_rows,
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        vec![
            Dim::bounded(Seqs, self.max_seqs),
            Dim::bounded(Rows, self.max_rows),
            Dim::fixed(Context, self.text),
            Dim::fixed(Hidden, self.hidden),
            Dim::fixed(Heads, self.heads),
            Dim::fixed(HeadDim, self.hidden / self.heads),
            Dim::fixed(Ffn, self.ffn),
            Dim::fixed(Embedding, self.text_dim),
        ]
    }
}

/// Separate text and image weights around one joint attention.
#[derive(Weights)]
pub struct Double {
    #[shape(Hidden, Hidden)]
    pub q: Linear,
    #[shape(Hidden, Hidden)]
    pub k: Linear,
    #[shape(Hidden, Hidden)]
    pub v: Linear,
    #[shape(HeadDim)]
    pub q_norm: Norm,
    #[shape(HeadDim)]
    pub k_norm: Norm,
    #[shape(Hidden, Hidden)]
    pub out: Linear,
    #[shape(Hidden, Hidden)]
    pub text_q: Linear,
    #[shape(Hidden, Hidden)]
    pub text_k: Linear,
    #[shape(Hidden, Hidden)]
    pub text_v: Linear,
    #[shape(HeadDim)]
    pub text_q_norm: Norm,
    #[shape(HeadDim)]
    pub text_k_norm: Norm,
    #[shape(Hidden, Hidden)]
    pub text_out: Linear,
    /// Gate rows, then up rows.
    #[shape(Hidden, Ffn * 2)]
    pub ff_in: Linear,
    #[shape(Ffn, Hidden)]
    pub ff_out: Linear,
    #[shape(Hidden, Ffn * 2)]
    pub text_ff_in: Linear,
    #[shape(Ffn, Hidden)]
    pub text_ff_out: Linear,
}

/// One projection feeds attention and the MLP side by side.
#[derive(Weights)]
pub struct Single {
    /// Queries, keys, values, then the MLP's gate and up rows.
    #[shape(Hidden, Hidden * 3 + Ffn * 2)]
    pub fused: Linear,
    #[shape(HeadDim)]
    pub q_norm: Norm,
    #[shape(HeadDim)]
    pub k_norm: Norm,
    #[shape(Hidden + Ffn, Hidden)]
    pub out: Linear,
}

#[derive(Weights)]
pub struct Flux2 {
    #[shape(TIME, Hidden)]
    pub time_in: Linear,
    #[shape(Hidden, Hidden)]
    pub time_out: Linear,
    /// Shift, scale and gate for attention, then for the MLP.
    #[shape(Hidden, Hidden * 6)]
    pub image_mod: Linear,
    #[shape(Hidden, Hidden * 6)]
    pub text_mod: Linear,
    #[shape(Hidden, Hidden * 3)]
    pub single_mod: Linear,
    #[shape(LATENT, Hidden)]
    pub image_in: Linear,
    #[shape(Embedding, Hidden)]
    pub text_in: Linear,
    /// Scale, then shift.
    #[shape(Hidden, Hidden * 2)]
    pub out_mod: Linear,
    #[shape(Hidden, LATENT)]
    pub out: Linear,
}

/// Every weight. Paths name the blocks `double.i` and `single.i`.
pub fn declare(m: &Model, c: &Config) -> (Flux2, Vec<Double>, Vec<Single>) {
    fn each<W: Weights>(m: &Model, name: &str, n: u64) -> Vec<W> {
        let one = |i: u64| {
            let mut path = vec![PathSeg::Field(name.into()), PathSeg::Index(i as u32)];
            W::declare(m, &mut path, &[])
        };
        (0..n).map(one).collect()
    }
    let w: Flux2 = tessera_trace::declare(m);
    (w, each(m, "double", c.double), each(m, "single", c.single))
}

/// The graph's inputs, one row per canvas.
pub struct Inputs {
    /// Noisy latents, `[Seqs, Rows, LATENT]`.
    pub image: Tensor,
    /// Text features, `[Seqs, Context, Embedding]`.
    pub text: Tensor,
    /// The timestep's sinusoids, `[Seqs, TIME]`.
    pub time: Tensor,
    /// Positions on each RoPE axis, text rows then image rows.
    pub ids: [Tensor; AXES],
}

fn inputs(m: &Model) -> Inputs {
    let port = |role, shape: Vec<DimExpr>, dtype| Port {
        role,
        ty: TensorType::new(shape, dtype),
    };
    let joint = || vec![Seqs.into(), Context + Rows];
    let ports = [
        port(
            Role::Tokens,
            vec![Seqs.into(), Rows.into(), LATENT.into()],
            DType::F32,
        ),
        port(
            Role::Keys,
            vec![Seqs.into(), Context.into(), Embedding.into()],
            DType::F32,
        ),
        port(Role::Keys, vec![Seqs.into(), TIME.into()], DType::F32),
        port(Role::Positions, joint(), DType::U32),
        port(Role::Positions, joint(), DType::U32),
        port(Role::Positions, joint(), DType::U32),
        port(Role::Positions, joint(), DType::U32),
    ];
    let [image, text, time, t, h, w, l] = m.entry(EntryKind::Forward, ports);
    Inputs {
        image,
        text,
        time,
        ids: [t, h, w, l],
    }
}

/// One modulation set: rows of shift, scale and gate.
struct Mods<'a> {
    all: &'a Tensor,
    rows: DimExpr,
    eps: f32,
}

impl Mods<'_> {
    fn part(&self, i: u64, hidden: u64) -> Tensor {
        spread(&self.all.slice(1, i * hidden, hidden), self.rows.clone())
    }

    /// `(1 + scale) * layer_norm(x) + shift`, the set at `at`.
    fn apply(&self, x: &Tensor, at: u64, hidden: u64) -> Tensor {
        let n = x.center().rms(self.eps);
        n.mul(&self.part(at + 1, hidden))
            .add(&n)
            .add(&self.part(at, hidden))
    }
}

/// FLUX.2's SwiGLU: the first half gates the second.
fn glu(x: &Tensor, ffn: u64) -> Tensor {
    let gate = x.slice(2, 0u64, ffn).act(Act::Silu);
    gate.mul(&x.slice(2, ffn, ffn))
}

/// Rows as heads, each normed, then rows again.
fn normed(x: &Tensor, norm: &Norm, rows: DimExpr, eps: f32) -> Tensor {
    let heads = vec![Seqs.into(), rows.clone(), Heads.into(), HeadDim.into()];
    let flat = vec![Seqs.into(), rows, Hidden.into()];
    x.reshape(heads).rms_norm(norm, eps).reshape(flat)
}

/// Joint attention. Each RoPE axis turns its own channels.
fn attend(m: &Model, c: &Config, [q, k, v]: [&Tensor; 3], ids: &[Tensor; AXES]) -> Tensor {
    let rows: DimExpr = Context + Rows;
    let heads = vec![Seqs.into(), rows.clone(), Heads.into(), HeadDim.into()];
    let turn = |x: &Tensor| {
        let x = x.reshape(heads.clone());
        let axis = |a: usize| {
            let at: u64 = c.axes[..a].iter().sum();
            x.slice(3, at, c.axes[a])
                .rope(&ids[a], c.theta, Pairs::Adjacent)
        };
        (1..AXES).fold(axis(0), |all, a| all.concat(&axis(a)))
    };
    let scale = 1.0 / ((c.hidden / c.heads) as f32).sqrt();
    let out = m.attend(
        Mask::Bidirectional,
        scale,
        [&turn(q), &turn(k), &v.reshape(heads.clone())],
        None,
    );
    out.reshape(vec![Seqs.into(), rows, Hidden.into()])
}

/// One double block on the text and image streams.
fn double(
    m: &Model,
    c: &Config,
    d: &Double,
    [text, image]: [Tensor; 2],
    [tm, im]: [&Mods; 2],
    ids: &[Tensor; AXES],
) -> [Tensor; 2] {
    let h = c.hidden;
    let (nt, ni) = (tm.apply(&text, 0, h), im.apply(&image, 0, h));
    let side = |x: &Tensor, w: &Linear, norm: Option<&Norm>, rows: DimExpr| {
        let y = x.matmul(w);
        norm.map_or(y.clone(), |n| normed(&y, n, rows, c.eps))
    };
    let (ctx, img): (DimExpr, DimExpr) = (Context.into(), Rows.into());
    let q = join(
        &side(&nt, &d.text_q, Some(&d.text_q_norm), ctx.clone()),
        &side(&ni, &d.q, Some(&d.q_norm), img.clone()),
    );
    let k = join(
        &side(&nt, &d.text_k, Some(&d.text_k_norm), ctx.clone()),
        &side(&ni, &d.k, Some(&d.k_norm), img.clone()),
    );
    let v = join(&nt.matmul(&d.text_v), &ni.matmul(&d.v));
    let a = attend(m, c, [&q, &k, &v], ids);
    let at = a.slice(1, 0u64, c.text).matmul(&d.text_out);
    let ai = a.slice(1, c.text, Rows).matmul(&d.out);
    let image = image.add(&ai.mul(&im.part(2, h)));
    let ff = glu(&im.apply(&image, 3, h).matmul(&d.ff_in), c.ffn).matmul(&d.ff_out);
    let image = image.add(&ff.mul(&im.part(5, h)));
    let text = text.add(&at.mul(&tm.part(2, h)));
    let ff = glu(&tm.apply(&text, 3, h).matmul(&d.text_ff_in), c.ffn).matmul(&d.text_ff_out);
    let text = text.add(&ff.mul(&tm.part(5, h)));
    [text, image]
}

/// One single block on the joint rows.
fn single(
    m: &Model,
    c: &Config,
    s: &Single,
    x: Tensor,
    mods: &Mods,
    ids: &[Tensor; AXES],
) -> Tensor {
    let h = c.hidden;
    let rows: DimExpr = Context + Rows;
    let f = mods.apply(&x, 0, h).matmul(&s.fused);
    let q = normed(&f.slice(2, 0u64, h), &s.q_norm, rows.clone(), c.eps);
    let k = normed(&f.slice(2, h, h), &s.k_norm, rows, c.eps);
    let a = attend(m, c, [&q, &k, &f.slice(2, 2 * h, h)], ids);
    let mlp = glu(&f.slice(2, 3 * h, 2 * c.ffn), c.ffn);
    x.add(&a.concat(&mlp).matmul(&s.out).mul(&mods.part(2, h)))
}

/// The predicted velocity for each image row, `[Seqs, Rows, LATENT]`.
pub fn forward(
    m: &Model,
    c: &Config,
    (w, doubles, singles): &(Flux2, Vec<Double>, Vec<Single>),
    x: &Inputs,
) -> Tensor {
    let temb = x.time.matmul(&w.time_in).act(Act::Silu).matmul(&w.time_out);
    let t = temb.act(Act::Silu);
    let set = |lin: &Linear, rows: DimExpr| (t.matmul(lin), rows);
    let (im, ir) = set(&w.image_mod, Rows.into());
    let (tm, tr) = set(&w.text_mod, Context.into());
    let (sm, sr) = set(&w.single_mod, Context + Rows);
    let (om, or) = set(&w.out_mod, Rows.into());
    let eps = c.eps;
    let mods = |all, rows| Mods { all, rows, eps };
    let (im, tm, sm, om) = (mods(&im, ir), mods(&tm, tr), mods(&sm, sr), mods(&om, or));
    let mut streams = [x.text.matmul(&w.text_in), x.image.matmul(&w.image_in)];
    for d in doubles {
        let block = || double(m, c, d, streams, [&tm, &im], &x.ids);
        streams = m.composite(Composite::Layer, block);
    }
    let [text, image] = streams;
    let mut xs = join(&text, &image);
    for s in singles {
        [xs] = m.composite(Composite::Layer, || [single(m, c, s, xs, &sm, &x.ids)]);
    }
    let image = xs.slice(1, c.text, Rows);
    let n = image.center().rms(c.eps);
    let shifted = n
        .mul(&om.part(0, c.hidden))
        .add(&n)
        .add(&om.part(1, c.hidden));
    shifted.matmul(&w.out)
}

/// One forward entry: latents, text, time and positions in.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let weights = declare(&m, c);
    let x = inputs(&m);
    let velocity = forward(&m, c, &weights, &x);
    m.finish([(Role::Logits, &velocity)])
}
