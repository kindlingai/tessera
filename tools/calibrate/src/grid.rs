//! Small graphs that reach every rule, at many sizes.

use crate::measure;
use graph::DimName::{self, *};
use lower::{Device, Error, Settings, Step, Tensor, View, lower};
use run::{Run, open};
use testkit::Noise;
use trace::{
    Act, DType, Dim, Embed, EntryKind, Graph, Linear, Mask, Model, Norm, Pairs, Port, Role,
    TensorType, Weights, declare,
};

/// A timed plan holds at most this many elements.
const ROOM: usize = 750_000_000;

type Sizes = Vec<(DimName, u64)>;

/// The grid times every rule at these sizes.
pub struct Grid {
    /// Row counts, for everything that works on rows.
    pub rows: Vec<u64>,
    /// Hidden and inner widths.
    pub widths: Vec<u64>,
    pub vocabs: Vec<u64>,
    /// Heads, key heads and head width.
    pub heads: Vec<[u64; 3]>,
    /// Queries and keys of one attention.
    pub spans: Vec<[u64; 2]>,
}

/// A graph, with each size to time it at.
struct Case {
    g: Graph,
    sizes: Vec<Sizes>,
}

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

#[derive(Weights)]
struct Scale {
    #[shape(Hidden)]
    norm: Norm,
}

#[derive(Weights)]
struct Table {
    #[shape(Vocab, Hidden)]
    embed: Embed,
}

/// Every name bounded at `most`, then the fixed ones.
fn model(bounded: &[DimName], fixed: &[(DimName, u64)]) -> Model {
    let bounded = bounded.iter().map(|name| Dim::bounded(*name, 1 << 20));
    let fixed = fixed.iter().map(|(name, n)| Dim::fixed(*name, *n));
    Model::new(bounded.chain(fixed).collect())
}

fn floats<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::F32);
    let role = Role::Logits;
    Port { role, ty }
}

fn ids<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::U32);
    let role = Role::Tokens;
    Port { role, ty }
}

/// `k` inputs to `n` outputs: every matmul.
fn linear(k: u64, n: u64) -> Graph {
    let m = model(&[Rows], &[(Hidden, k), (Ffn, n)]);
    let w: Layer = declare(&m);
    let [x] = m.entry(EntryKind::Forward, [floats([Rows, Hidden])]);
    m.finish([(Role::Logits, &x.matmul(&w.w))])
}

/// Norms, maps, a softmax, an argmax and a mean.
fn rows(width: u64) -> Graph {
    let m = model(&[Rows], &[(Hidden, width)]);
    let w: Scale = declare(&m);
    let port = || floats([Rows, Hidden]);
    let [x, y] = m.entry(EntryKind::Forward, [port(), port()]);
    let gated = x.rms_norm(&w.norm, 1e-6).act(Act::Silu).mul(&y.rms(1e-6));
    let sum = y.add(&gated);
    let ids = sum.softmax().top_k(1u64);
    m.finish([(Role::Tokens, &ids), (Role::Logits, &sum.mean(0))])
}

/// A rotation, then causal attention.
fn attention([heads, kv_heads, width]: [u64; 3]) -> Graph {
    let fixed = [(Heads, heads), (KvHeads, kv_heads), (HeadDim, width)];
    let m = model(&[Seqs, Rows, Context], &fixed);
    let keys = || floats([Seqs, Context, KvHeads, HeadDim]);
    let ports = [
        floats([Seqs, Rows, Heads, HeadDim]),
        keys(),
        keys(),
        ids([Seqs, Rows]),
    ];
    let [q, k, v, at] = m.entry(EntryKind::Forward, ports);
    let q = q.rope(&at, 1e4, Pairs::Halves);
    let out = m.attend(Mask::Causal, 0.1, [&q, &k, &v], None);
    m.finish([(Role::Logits, &out)])
}

/// Rows of 256, rotated in two blocks of 128.
fn rotation() -> Graph {
    let m = model(&[Rows], &[(Hidden, 256)]);
    let [x] = m.entry(EntryKind::Forward, [floats([Rows, Hidden])]);
    m.finish([(Role::Logits, &x.hadamard(128))])
}

/// Rows read from a table.
fn table(vocab: u64, width: u64) -> Graph {
    let m = model(&[Rows], &[(Vocab, vocab), (Hidden, width)]);
    let w: Table = declare(&m);
    let [tokens] = m.entry(EntryKind::Forward, [ids([Rows])]);
    m.finish([(Role::Logits, &m.embed(&tokens, &w.embed))])
}

/// Noise for `ty` as one row. Ids stay below eight.
fn fill(g: &Graph, ty: &TensorType, sizes: &Sizes, noise: &mut Noise) -> Result<Tensor, Error> {
    let unsized_shape = || Error::Size(ty.shape.to_string());
    testkit::fill(g, ty, sizes, noise).ok_or_else(unsized_shape)
}

impl Case {
    /// Times this case at each of its sizes that fits.
    fn run(
        &self,
        device: &mut Device,
        noise: &mut Noise,
        settings: &Settings,
    ) -> Result<Vec<(Step, f64)>, Error> {
        let mut runner: Option<Box<dyn Run>> = None;
        let mut added = Vec::new();
        for sizes in &self.sizes {
            let plan = lower(&self.g, 0, sizes, &Device::guess(settings))?;
            if plan.buffers.iter().sum::<usize>() > ROOM {
                continue;
            }
            let runner = match &mut runner {
                Some(runner) => runner,
                None => runner.insert(self.open(plan, noise, settings)?),
            };
            let ports = self.g.entries[0].inputs.iter();
            let fed: Result<Vec<Tensor>, Error> =
                ports.map(|p| fill(&self.g, &p.ty, sizes, noise)).collect();
            let fed = fed?;
            let feeds: Vec<View> = fed.iter().map(Tensor::view).collect();
            added.extend(measure(
                &mut **runner,
                (&self.g, 0),
                sizes,
                &feeds,
                device,
                settings,
            )?);
        }
        Ok(added)
    }

    /// A runner with noise for weights.
    fn open(
        &self,
        plan: lower::Plan,
        noise: &mut Noise,
        settings: &Settings,
    ) -> Result<Box<dyn Run>, Error> {
        let mut runner = open(plan, settings)?;
        for (at, w) in self.g.weights.iter().enumerate() {
            let data = fill(&self.g, &w.ty, &Vec::new(), noise)?;
            runner.load(at as u32, data.view())?;
        }
        Ok(runner)
    }
}

impl Grid {
    /// Sizes from one decoded row to a long prompt.
    pub fn full() -> Self {
        Self {
            rows: vec![1, 4, 16, 64, 256, 1024],
            widths: vec![256, 1024, 4096],
            vocabs: vec![32_768, 131_072],
            heads: vec![[16, 8, 128], [8, 8, 64]],
            spans: vec![
                [1, 16],
                [1, 128],
                [1, 1024],
                [16, 16],
                [128, 128],
                [512, 512],
            ],
        }
    }

    /// Enough to reach every rule, in well under a second.
    pub fn tiny() -> Self {
        Self {
            rows: vec![1, 3],
            widths: vec![16],
            vocabs: vec![300],
            heads: vec![[4, 2, 16]],
            spans: vec![[1, 5], [4, 4]],
        }
    }

    fn cases(&self) -> Vec<Case> {
        let by_rows = |g: Graph| {
            let sizes = self.rows.iter().map(|r| vec![(Rows, *r)]).collect();
            Case { g, sizes }
        };
        let mut cases = Vec::new();
        for &k in &self.widths {
            let outs = self.widths.iter().chain(&self.vocabs);
            cases.extend(outs.map(|n| by_rows(linear(k, *n))));
        }
        cases.push(by_rows(rotation()));
        for &width in self.widths.iter().chain(&self.vocabs) {
            cases.push(by_rows(rows(width)));
            cases.extend(self.vocabs.iter().map(|v| by_rows(table(*v, width))));
        }
        for &heads in &self.heads {
            let span =
                |[queries, keys]: &[u64; 2]| vec![(Seqs, 1), (Rows, *queries), (Context, *keys)];
            let sizes = self.spans.iter().map(span).collect();
            let g = attention(heads);
            cases.push(Case { g, sizes });
        }
        cases
    }

    /// Times every case on this GPU. Returns the launches timed.
    pub fn run(&self, device: &mut Device, settings: &Settings) -> Result<Vec<(Step, f64)>, Error> {
        let mut noise = Noise(17);
        let mut added = Vec::new();
        for case in self.cases() {
            added.extend(case.run(device, &mut noise, settings)?);
        }
        Ok(added)
    }
}
