//! Whole programs: dimensions, mesh, precision, entries and ports. Sketch.

use crate::core::DimName::*;
use crate::core::*;
use crate::dgemma::{self, DGemma};
use crate::ds41::{self, Ds41};
use crate::glm53::{self, Glm53};
use crate::lsa;
use crate::spec::{self, DraftLimits, DraftPorts, Drafters, Rule};

macro_rules! shape {
    ($($d:expr),* $(,)?) => {
        Shape(vec![$(DimExpr::from($d)),*])
    };
}

fn port(io: Io, kind: PortKind, shape: Shape, dtype: DType) -> PortDecl {
    let placement = Placement::Replicated;
    let ty = TensorType {
        shape,
        dtype,
        placement,
    };
    PortDecl { io, kind, ty }
}

/// gr8's numerics: FP32 residual, FP16 activations, BF16 partials.
fn gr8_precision() -> Precision {
    Precision {
        residual: DType::F32,
        activations: DType::F16,
        partials: DType::Bf16,
        accumulate: DType::F32,
    }
}

/// Inputs every paged-KV entry takes before each launch.
fn step_in() -> Vec<PortDecl> {
    vec![
        port(Io::Tokens, PortKind::PerLaunch, shape![Rows], DType::I32),
        port(Io::Positions, PortKind::PerLaunch, shape![Rows], DType::I32),
        port(
            Io::PageTable,
            PortKind::PerLaunch,
            shape![Seqs, Pages],
            DType::I32,
        ),
    ]
}

/// GLM-5.3 at TP4 with speculation, as gr8 serves it.
pub fn glm53_serve(c: &glm53::Config, w: &Glm53, d: &Drafters, lim: &DraftLimits) -> Program {
    let prefill = Entry {
        kind: EntryKind::Prefill,
        dims: vec![(Rows, Size::Buckets(vec![512, 2048, 8192]))],
        ports: step_in(),
        body: Box::new(move |m, io| {
            glm53::forward(m, c, w, &io.tensor(Io::Tokens));
        }),
    };
    let mut ports = step_in();
    ports.extend([
        port(Io::Left, PortKind::PerLaunch, shape![Seqs], DType::I32),
        port(
            Io::DraftChoice,
            PortKind::Mailbox,
            shape![Seqs, 2],
            DType::I32,
        ),
        port(
            Io::Emitted,
            PortKind::Stream,
            shape![Seqs, Drafts],
            DType::I32,
        ),
        port(Io::Stats, PortKind::Stream, shape![Seqs, 3], DType::F32),
    ]);
    let decode = Entry {
        kind: EntryKind::Decode,
        // Each sequence verifies one row plus its drafts.
        dims: vec![(Rows, Size::Buckets(vec![8, 16, 32, 64, 128]))],
        ports,
        body: Box::new(move |m, io| {
            let dp = DraftPorts {
                choice: io.port(Io::DraftChoice),
                stats: io.port(Io::Stats),
            };
            let left = io.tensor(Io::Left);
            spec::decode(m, w, d, lim, &dp, Rule::Greedy, left, &io.port(Io::Emitted));
        }),
    };
    Program {
        dims: vec![
            (Seqs, Size::Bounded(8)),
            (Context, Size::Bounded(425_984)),
            (Pages, Size::Bounded(6656)),
            (PageRows, Size::Fixed(64)),
            (Drafts, Size::Bounded(17)),
            (Hidden, Size::Config),
            (Heads, Size::Config),
            (Latent, Size::Config),
            (Rope, Size::Config),
            (Vocab, Size::Config),
            (Experts, Size::Config),
            (TopK, Size::Config),
            (Layers, Size::Config),
        ],
        mesh: Mesh { tp: 4, ep: 1 },
        precision: gr8_precision(),
        entries: vec![prefill, decode],
    }
}

/// DiffusionGemma on one device: generation and reads share an entry.
pub fn dgemma_serve(c: &dgemma::Config, w: &DGemma) -> Program {
    let mut ports = step_in();
    ports.extend([
        port(
            Io::Canvas,
            PortKind::PerLaunch,
            shape![Seqs, Block],
            DType::I32,
        ),
        // Cap, read-only flag and allowed vocabulary per row.
        port(
            Io::RowState,
            PortKind::PerLaunch,
            shape![Seqs, 3],
            DType::I32,
        ),
        port(
            Io::Emitted,
            PortKind::Stream,
            shape![Seqs, Block],
            DType::I32,
        ),
        port(
            Io::Logprobs,
            PortKind::Stream,
            shape![Seqs, Block],
            DType::F32,
        ),
    ]);
    let denoise = Entry {
        kind: EntryKind::Denoise,
        dims: vec![(Rows, Size::Buckets(vec![32, 64, 128, 256, 512]))],
        ports,
        body: Box::new(move |m, io| {
            let rs = io.row_state(Io::RowState);
            let canvas = dgemma::generate(m, c, w, &rs, io.tensor(Io::Canvas));
            m.emit(&io.port(Io::Emitted), &canvas);
        }),
    };
    Program {
        dims: vec![
            (Seqs, Size::Bounded(32)),
            (Block, Size::Config),
            (Context, Size::Bounded(131_072)),
            (Pages, Size::Bounded(4096)),
            (PageRows, Size::Fixed(64)),
            (Hidden, Size::Config),
            (Heads, Size::Config),
            (KvHeads, Size::Config),
            (HeadDim, Size::Config),
            (Vocab, Size::Config),
            (Layers, Size::Config),
        ],
        mesh: Mesh { tp: 1, ep: 1 },
        precision: Precision {
            residual: DType::F32,
            activations: DType::Bf16,
            partials: DType::Bf16,
            accumulate: DType::F32,
        },
        entries: vec![denoise],
    }
}

/// DeepSeek V4.1 Flash at TP4, plus LSA upkeep and prefetch.
pub fn ds41_serve(c: &ds41::Config, w: &Ds41) -> Program {
    let prefill = Entry {
        kind: EntryKind::Prefill,
        dims: vec![(Rows, Size::Buckets(vec![512, 2048, 8192]))],
        ports: step_in(),
        body: Box::new(move |m, io| {
            ds41::forward(m, c, w, &io.tensor(Io::Tokens), None);
        }),
    };
    let mut ports = step_in();
    ports.push(port(
        Io::Emitted,
        PortKind::Stream,
        shape![Seqs],
        DType::I32,
    ));
    let decode = Entry {
        kind: EntryKind::Decode,
        dims: vec![(Rows, Size::Buckets(vec![1, 2, 4, 8]))],
        ports,
        body: Box::new(move |m, io| {
            let logits = ds41::forward(m, c, w, &io.tensor(Io::Tokens), None);
            let next = m.sample(&logits, Sampler::Argmax);
            ds41::prefetch_next(m, w, c, &logits, &next);
            m.emit(&io.port(Io::Emitted), &next);
        }),
    };
    // Off the step path: clusters blocks before their deadline.
    let maintain = Entry {
        kind: EntryKind::Maintain,
        dims: vec![],
        ports: vec![port(
            Io::ReadyBlocks,
            PortKind::Mailbox,
            shape![Seqs, Layers],
            DType::I32,
        )],
        body: Box::new(move |m, _io| {
            for index in m.search_indexes() {
                lsa::maintain(m, &index);
            }
        }),
    };
    // At admission, before the prompt's prefill is scheduled.
    let prefetch = Entry {
        kind: EntryKind::Prefetch,
        dims: vec![(Rows, Size::Bounded(131_072))],
        ports: vec![port(
            Io::Prompt,
            PortKind::PerLaunch,
            shape![Rows],
            DType::I32,
        )],
        body: Box::new(move |m, io| {
            ds41::prefetch_prompt(m, w, c, &io.tensor(Io::Prompt));
        }),
    };
    Program {
        dims: vec![
            (Seqs, Size::Bounded(8)),
            (Context, Size::Bounded(1_048_576)),
            (Pages, Size::Bounded(16_384)),
            (PageRows, Size::Fixed(64)),
            (Streams, Size::Config),
            (Hidden, Size::Config),
            (Heads, Size::Config),
            (Latent, Size::Config),
            (Rope, Size::Config),
            (Vocab, Size::Config),
            (Experts, Size::Config),
            (TopK, Size::Config),
            (Layers, Size::Config),
        ],
        mesh: Mesh { tp: 4, ep: 1 },
        precision: gr8_precision(),
        entries: vec![prefill, decode, maintain, prefetch],
    }
}
