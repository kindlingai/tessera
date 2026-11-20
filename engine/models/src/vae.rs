//! FLUX.2's VAE decoder: transformer latents to RGB.

use sensai_trace::DimName::*;
use sensai_trace::{
    Act, DType, Dim, DimExpr, EntryKind, Graph, Linear, Mask, Model, PathSeg, Port, Role, Tensor,
    TensorType, Vector, Weights,
};

/// Channels per transformer token: 32 latent channels in 2x2 patches.
pub const TOKEN: u64 = 128;
const LATENT: u64 = 32;
const GROUPS: u64 = 32;
const EPS: f32 = 1e-6;
/// Neighbours a 3x3 convolution reads.
pub const TAPS: u64 = 9;

pub struct Config {
    /// Transformer tokens per side. The latent is twice that.
    pub side: u64,
    /// Channels from the deepest block out, as the decoder runs.
    pub widths: Vec<u64>,
    /// Resnets per up block.
    pub depth: u64,
    pub max_seqs: u64,
}

impl Config {
    /// FLUX.2's VAE, `side` tokens a side.
    pub fn flux2(side: u64, max_seqs: u64) -> Self {
        Self {
            side,
            widths: vec![512, 512, 256, 128],
            depth: 3,
            max_seqs,
        }
    }

    /// Pixels a side at each level, the latent first.
    pub fn sides(&self) -> Vec<u64> {
        (0..self.widths.len() as u32)
            .map(|i| (2 * self.side) << i)
            .collect()
    }

    pub fn dims(&self) -> Vec<Dim> {
        vec![Dim::bounded(Seqs, self.max_seqs)]
    }
}

/// A matrix and its bias. 3x3 kernels read 9 pixels.
pub struct Affine {
    pub weight: Linear,
    pub bias: Vector,
}

pub struct GroupNorm {
    pub weight: Vector,
    pub bias: Vector,
}

pub struct Resnet {
    pub norm1: GroupNorm,
    pub conv1: Affine,
    pub norm2: GroupNorm,
    pub conv2: Affine,
    pub shortcut: Option<Affine>,
}

pub struct Attention {
    pub norm: GroupNorm,
    pub q: Affine,
    pub k: Affine,
    pub v: Affine,
    pub out: Affine,
}

pub struct Up {
    pub resnets: Vec<Resnet>,
    pub upsample: Option<Affine>,
}

pub struct Decoder {
    /// The latents' mean and spread per token channel.
    pub mean: Vector,
    pub std: Vector,
    pub post_quant: Affine,
    pub conv_in: Affine,
    pub mid: [Resnet; 2],
    pub attention: Attention,
    pub ups: Vec<Up>,
    pub norm_out: GroupNorm,
    pub conv_out: Affine,
}

/// Declares weights under one path, with diffusers' names.
struct At<'a> {
    m: &'a Model,
    path: Vec<PathSeg>,
}

impl At<'_> {
    fn sub(&self, parts: &[&str]) -> Self {
        let mut path = self.path.clone();
        for p in parts {
            path.push(match p.parse::<u32>() {
                Ok(i) => PathSeg::Index(i),
                Err(_) => PathSeg::Field((*p).into()),
            });
        }
        At { m: self.m, path }
    }

    fn leaf<W: Weights>(&self, name: &str, shape: &[u64]) -> W {
        let shape: Vec<DimExpr> = shape.iter().map(|&n| n.into()).collect();
        W::declare(self.m, &mut self.sub(&[name]).path, &shape)
    }

    /// A convolution `k` wide from `cin` to `cout` channels.
    fn conv(&self, name: &str, k: u64, [cin, cout]: [u64; 2]) -> Affine {
        let at = self.sub(&[name]);
        Affine {
            weight: at.leaf("weight", &[cin * k * k, cout]),
            bias: at.leaf("bias", &[cout]),
        }
    }

    fn norm(&self, name: &str, width: u64) -> GroupNorm {
        let at = self.sub(&[name]);
        GroupNorm {
            weight: at.leaf("weight", &[width]),
            bias: at.leaf("bias", &[width]),
        }
    }

    fn resnet(&self, [cin, cout]: [u64; 2]) -> Resnet {
        Resnet {
            norm1: self.norm("norm1", cin),
            conv1: self.conv("conv1", 3, [cin, cout]),
            norm2: self.norm("norm2", cout),
            conv2: self.conv("conv2", 3, [cout, cout]),
            shortcut: (cin != cout).then(|| self.conv("conv_shortcut", 1, [cin, cout])),
        }
    }
}

pub fn declare(m: &Model, c: &Config) -> Decoder {
    let root = At {
        m,
        path: Vec::new(),
    };
    let d = root.sub(&["decoder"]);
    let top = c.widths[0];
    let bn = root.sub(&["bn"]);
    let a = d.sub(&["mid_block", "attentions", "0"]);
    let dense = |name: &str| a.conv(name, 1, [top, top]);
    let mut ups = Vec::new();
    let mut cin = top;
    for (i, &cout) in c.widths.iter().enumerate() {
        let up = d.sub(&["up_blocks", &i.to_string()]);
        let resnets = (0..c.depth)
            .map(|j| {
                let channels = [if j == 0 { cin } else { cout }, cout];
                up.sub(&["resnets", &j.to_string()]).resnet(channels)
            })
            .collect();
        let last = i + 1 == c.widths.len();
        let upsample = (!last).then(|| up.sub(&["upsamplers", "0"]).conv("conv", 3, [cout, cout]));
        ups.push(Up { resnets, upsample });
        cin = cout;
    }
    let mid = |j: &str| d.sub(&["mid_block", "resnets", j]).resnet([top, top]);
    Decoder {
        mean: bn.leaf("running_mean", &[TOKEN]),
        std: bn.leaf("std", &[TOKEN]),
        post_quant: root.conv("post_quant_conv", 1, [LATENT, LATENT]),
        conv_in: d.conv("conv_in", 3, [LATENT, top]),
        mid: [mid("0"), mid("1")],
        attention: Attention {
            norm: a.norm("group_norm", top),
            q: dense("to_q"),
            k: dense("to_k"),
            v: dense("to_v"),
            out: a.sub(&["to_out"]).conv("0", 1, [top, top]),
        },
        ups,
        norm_out: d.norm("conv_norm_out", *c.widths.last().expect("widths")),
        conv_out: d.conv("conv_out", 3, [*c.widths.last().expect("widths"), 3]),
    }
}

fn shape(dims: &[u64]) -> Vec<DimExpr> {
    std::iter::once(Seqs.into())
        .chain(dims.iter().map(|&n| n.into()))
        .collect()
}

impl Affine {
    fn apply(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add_vector(&self.bias)
    }

    /// A 3x3 convolution: gather neighbours, then multiply.
    fn conv(&self, x: &Tensor, ids: &Tensor, [pixels, width]: [u64; 2]) -> Tensor {
        let near = x.permute(&[1, 0, 2]).gather(ids).permute(&[2, 0, 1, 3]);
        self.apply(&near.reshape(shape(&[pixels, TAPS * width])))
    }
}

impl GroupNorm {
    fn apply(&self, x: &Tensor, [pixels, width]: [u64; 2]) -> Tensor {
        let per = width / GROUPS;
        let split = x
            .reshape(shape(&[pixels, GROUPS, per]))
            .permute(&[0, 2, 1, 3]);
        let rows = split
            .reshape(shape(&[GROUPS, pixels * per]))
            .center()
            .rms(EPS);
        let back = rows
            .reshape(shape(&[GROUPS, pixels, per]))
            .permute(&[0, 2, 1, 3]);
        let x = back.reshape(shape(&[pixels, width]));
        x.mul_vector(&self.weight).add_vector(&self.bias)
    }
}

impl Resnet {
    fn apply(&self, x: &Tensor, ids: &Tensor, [pixels, cin, cout]: [u64; 3]) -> Tensor {
        let h = self.norm1.apply(x, [pixels, cin]).act(Act::Silu);
        let h = self.conv1.conv(&h, ids, [pixels, cin]);
        let h = self.norm2.apply(&h, [pixels, cout]).act(Act::Silu);
        let h = self.conv2.conv(&h, ids, [pixels, cout]);
        let x = self.shortcut.as_ref().map_or(x.clone(), |s| s.apply(x));
        x.add(&h)
    }
}

impl Attention {
    /// One head over every pixel.
    fn apply(&self, m: &Model, x: &Tensor, [pixels, width]: [u64; 2]) -> Tensor {
        let h = self.norm.apply(x, [pixels, width]);
        let head = |a: &Affine| a.apply(&h).reshape(shape(&[pixels, 1, width]));
        let scale = 1.0 / (width as f32).sqrt();
        let o = m.attend(
            Mask::Bidirectional,
            scale,
            [&head(&self.q), &head(&self.k), &head(&self.v)],
            None,
        );
        x.add(&self.out.apply(&o.reshape(shape(&[pixels, width]))))
    }
}

/// Each pixel twice as wide and twice as tall.
fn upsample(x: &Tensor, [side, width]: [u64; 2]) -> Tensor {
    let cells = x.reshape(shape(&[side, 1, side, 1, width]));
    let wide = cells.expand(2, 2u64).expand(4, 2u64);
    wide.reshape(shape(&[4 * side * side, width]))
}

/// Tokens to latent pixels, twice as many a side.
fn unpatchify(x: &Tensor, side: u64) -> Tensor {
    let cells = x
        .reshape(shape(&[side, side, LATENT, 2, 2]))
        .permute(&[0, 1, 2, 4, 5, 3]);
    let rows = cells.permute(&[0, 1, 3, 2, 4, 5]);
    rows.reshape(shape(&[4 * side * side, LATENT]))
}

/// RGB in `[-1, 1]` per pixel, `[Seqs, (16 side)^2, 3]`.
pub fn forward(m: &Model, c: &Config, d: &Decoder, tokens: &Tensor, ids: &[Tensor]) -> Tensor {
    let sides = c.sides();
    let x = tokens.mul_vector(&d.std).add_vector(&d.mean);
    let x = d.post_quant.apply(&unpatchify(&x, c.side));
    let (mut side, mut level) = (sides[0], 0);
    let top = c.widths[0];
    let mut x = d.conv_in.conv(&x, &ids[0], [side * side, LATENT]);
    x = d.mid[0].apply(&x, &ids[0], [side * side, top, top]);
    x = d.attention.apply(m, &x, [side * side, top]);
    x = d.mid[1].apply(&x, &ids[0], [side * side, top, top]);
    let mut cin = top;
    for (up, &cout) in d.ups.iter().zip(&c.widths) {
        for (j, r) in up.resnets.iter().enumerate() {
            let from = if j == 0 { cin } else { cout };
            x = r.apply(&x, &ids[level], [side * side, from, cout]);
        }
        if let Some(conv) = &up.upsample {
            x = upsample(&x, [side, cout]);
            (side, level) = (side * 2, level + 1);
            x = conv.conv(&x, &ids[level], [side * side, cout]);
        }
        cin = cout;
    }
    let x = d.norm_out.apply(&x, [side * side, cin]).act(Act::Silu);
    d.conv_out.conv(&x, &ids[level], [side * side, cin])
}

/// One entry: tokens and neighbour lists in, RGB out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let d = declare(&m, c);
    let port = |role, shape: Vec<DimExpr>, dtype| Port {
        role,
        ty: TensorType::new(shape, dtype),
    };
    let mut ports = vec![port(
        Role::Tokens,
        shape(&[c.side * c.side, TOKEN]),
        DType::F32,
    )];
    for s in c.sides() {
        ports.push(port(
            Role::Keys,
            vec![(s * s).into(), TAPS.into()],
            DType::U32,
        ));
    }
    let entry: [Tensor; 5] = m.entry(EntryKind::Forward, ports.try_into().expect("four levels"));
    let rgb = forward(&m, c, &d, &entry[0], &entry[1..]);
    m.finish([(Role::Logits, &rgb)])
}
