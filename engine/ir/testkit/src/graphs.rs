//! Small graphs that reach every rule, for tests and calibrate.

use tessera_graph::DimName::*;
use tessera_graph::{
    Composite, DType, Dim, DimExpr, DimName, Graph, PathSeg, Port, Role, TensorType,
};
use tessera_trace::{
    Act, Embed, EntryKind, Linear, Linears, Mask, Model, Norm, Pairs, Taps, Vector, Weights,
    declare, each,
};

const NOPE: DimName = DimName::named("Nope");
const ROPE: DimName = DimName::named("Rope");
const LATENT: DimName = DimName::named("Latent");
const INDEX_KEEP: DimName = DimName::named("IndexKeep");

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

/// The way back in a chain: from `Ffn` to `Hidden`.
#[derive(Weights)]
struct Back {
    #[shape(Ffn, Hidden)]
    w: Linear,
}

/// A matrix between two scaled rotations.
#[derive(Weights)]
struct Turned {
    #[shape(Hidden)]
    suh: Vector,
    #[shape(Hidden, Ffn)]
    w: Linear,
    #[shape(Ffn)]
    svh: Vector,
}

/// Per head: key rows, then value rows.
#[derive(Weights)]
struct Blocks {
    #[shape(LATENT, Heads * (NOPE + HeadDim))]
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

/// A 3x3 kernel over gathered neighbours.
#[derive(Weights)]
struct Conv {
    #[shape(TopK * Hidden, Hidden)]
    w: Linear,
}

#[derive(Weights)]
struct Experts {
    #[shape(Hidden, Experts)]
    router: Linear,
    /// Two matrices per expert, stacked by output row.
    #[shape(Experts, Hidden, Ffn * 2)]
    set: Linears,
}

/// An f32 input port.
pub fn floats<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::F32);
    let role = Role::Logits;
    Port { role, ty }
}

/// A u32 input port.
pub fn ids<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::U32);
    let role = Role::Tokens;
    Port { role, ty }
}

/// `bounded`, then these fixed sizes.
fn model(bounded: &[Dim], fixed: &[(DimName, u64)]) -> Model {
    let fixed = fixed.iter().map(|(name, n)| Dim::fixed(*name, *n));
    Model::new(bounded.iter().cloned().chain(fixed).collect())
}

/// Rows along every bounded axis, `width` wide.
fn rows_of(bounded: &[Dim], width: DimName, dtype: DType) -> Port {
    let lead = bounded.iter().map(|d| DimExpr::from(d.name));
    let shape: Vec<DimExpr> = lead.chain([width.into()]).collect();
    let role = Role::Logits;
    Port {
        role,
        ty: TensorType::new(shape, dtype),
    }
}

/// `k` inputs to `n` outputs: every matmul.
pub fn linear(bounded: &[Dim], k: u64, n: u64) -> Graph {
    let m = model(bounded, &[(Hidden, k), (Ffn, n)]);
    let w: Layer = declare(&m);
    let [x] = m.entry(EntryKind::Forward, [rows_of(bounded, Hidden, DType::F32)]);
    m.finish([(Role::Logits, &x.matmul(&w.w))])
}

/// A product through `act`, then times and plus `y`.
pub fn finished(bounded: &[Dim], [k, n]: [u64; 2], act: Act) -> Graph {
    let m = model(bounded, &[(Hidden, k), (Ffn, n)]);
    let w: Layer = declare(&m);
    let ports = [
        rows_of(bounded, Hidden, DType::F32),
        rows_of(bounded, Ffn, DType::F32),
    ];
    let [x, y] = m.entry(EntryKind::Forward, ports);
    let out = x.matmul(&w.w).scale(0.5).act(act).mul(&y).add(&y);
    m.finish([(Role::Logits, &out)])
}

/// `pairs` links of `k` to `n` and back, unshared weights.
pub fn chain(bounded: &[Dim], k: u64, n: u64, pairs: u64) -> Graph {
    let m = model(bounded, &[(Hidden, k), (Ffn, n), (Layers, pairs)]);
    let out: Vec<Layer> = each(&m, "out", Layers, |_| Vec::new());
    let back: Vec<Back> = each(&m, "back", Layers, |_| Vec::new());
    let [mut x] = m.entry(EntryKind::Forward, [rows_of(bounded, Hidden, DType::F32)]);
    for (o, b) in out.iter().zip(&back) {
        x = x.matmul(&o.w).matmul(&b.w);
    }
    m.finish([(Role::Logits, &x)])
}

/// Per-head blocks of one matrix, one side swapped.
pub fn blocks(bounded: &[Dim], [heads, key, value]: [u64; 3]) -> Graph {
    let fixed = [
        (Heads, heads),
        (NOPE, key),
        (HeadDim, value),
        (LATENT, value),
    ];
    let m = model(bounded, &fixed);
    let w: Blocks = declare(&m);
    let ports = [floats([Heads, Rows, NOPE]), floats([Heads, Rows, LATENT])];
    let [q, o] = m.entry(EntryKind::Forward, ports);
    let shape = || vec![Heads.into(), NOPE + HeadDim, LATENT.into()];
    let set = || w.w.stored().reshape(shape());
    let keys = q.matmul_rows(&set().slice(1, 0u64, NOPE).permute(&[0, 2, 1]));
    let values = o.matmul_rows(&set().slice(1, NOPE, HeadDim));
    m.finish([(Role::Logits, &keys), (Role::Logits, &values)])
}

/// Norms, activations, centering, unit sum, softmax, mask, edits,
/// argmax, mean.
pub fn rows(bounded: &[Dim], width: u64) -> Graph {
    let fixed = [
        (Hidden, width),
        (VocabWords, width.div_ceil(32)),
        (EditWords, 6),
    ];
    let m = model(bounded, &fixed);
    let w: Scale = declare(&m);
    let port = || rows_of(bounded, Hidden, DType::F32);
    let mut bits = rows_of(bounded, VocabWords, DType::U32);
    let words = bits.ty.shape.0.pop().expect("a row axis");
    let edits = DimExpr::Add(Box::new(words), Box::new(EditWords.into()));
    bits.ty.shape.0.push(edits);
    let [x, y, bits] = m.entry(EntryKind::Forward, [port(), port(), bits]);
    let gated = x.rms_norm(&w.norm, 1e-6).act(Act::Silu).mul(&y.rms(1e-6));
    let tail = gated.act(Act::Tanh);
    let sum = y.act(Act::Softplus).act(Act::Exp).add(&tail).center();
    let probs = sum.act(Act::Sigmoid).unit_sum().softmax();
    let ids = probs.allow(&bits).top_k(1u64);
    m.finish([(Role::Tokens, &ids), (Role::Logits, &sum.mean(0))])
}

/// Residual add + RMS norm: 2-output (residual kept) and 3-addend 1-output.
pub fn residual_norm(bounded: &[Dim], width: u64) -> Graph {
    let m = model(bounded, &[(Hidden, width)]);
    let w: Scale = declare(&m);
    let port = || rows_of(bounded, Hidden, DType::F32);
    let [x, a, b] = m.entry(EntryKind::Forward, [port(), port(), port()]);
    let h = x.add(&a);
    let n1 = h.rms_norm(&w.norm, 1e-6);
    let h2 = h.add(&a.add(&b));
    let n2 = h2.rms(1e-6);
    m.finish([(Role::Logits, &n1.add(&n2))])
}

/// Row L2 clip followed by position-dependent log scaling.
pub fn clipped(bounded: &[Dim], width: u64) -> Graph {
    let m = model(bounded, &[(Hidden, width)]);
    let pos = Port {
        role: Role::Positions,
        ty: TensorType::new([Rows], DType::U32),
    };
    let [x, at] = m.entry(
        EntryKind::Forward,
        [rows_of(bounded, Hidden, DType::F32), pos],
    );
    let out = x.l2_clip(4.0, 1e-6).log_scale(&at, 4.0, 0.5);
    m.finish([(Role::Logits, &out)])
}

/// A rotation, then causal attention over `Seqs`, `Rows` and `Context`.
pub fn attention(bounded: &[Dim], [heads, kv_heads, width]: [u64; 3]) -> Graph {
    let fixed = [(Heads, heads), (KvHeads, kv_heads), (HeadDim, width)];
    let m = model(bounded, &fixed);
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

/// Filtered values, then the delta rule over key heads.
pub fn delta(bounded: &[Dim], [heads, kv_heads, width]: [u64; 3]) -> Graph {
    let fixed = [
        (Heads, heads),
        (KvHeads, kv_heads),
        (HeadDim, width),
        (Hidden, heads * width),
    ];
    let m = model(bounded, &fixed);
    let keys = || floats([Seqs, Context, KvHeads, HeadDim]);
    let gates = || floats([Seqs, Context, Heads]);
    let values = floats([Seqs, Context, Hidden]);
    let ports = [keys(), keys(), values, gates(), gates()];
    let [q, k, x, decay, rate] = m.entry(EntryKind::Forward, ports);
    let mut path = vec![PathSeg::Field("taps".into())];
    let taps = Taps::declare(&m, &mut path, &[Hidden.into(), 4u64.into()]);
    let v = x.conv(&taps).reshape([Seqs, Context, Heads, HeadDim]);
    let out = m.delta(0.25, [&q, &k, &v], [&decay, &rate]);
    m.finish([(Role::Logits, &out)])
}

/// An attention region appends K and V, then sums both.
pub fn appended(bounded: &[Dim], width: u64) -> Graph {
    let m = model(bounded, &[(Hidden, width)]);
    let slots = || Port {
        role: Role::Slots,
        ty: TensorType::new([Rows], DType::U32),
    };
    let ports = [
        floats([Rows, Hidden]),
        floats([Rows, Hidden]),
        slots(),
        slots(),
    ];
    let [k, v, slots, positions] = m.entry(EntryKind::Step, ports);
    let ty = TensorType::new([Seqs, Context, Hidden], DType::F32);
    let (keys, values) = (m.kept(ty.clone()), m.kept(ty));
    let by = [&slots, &positions];
    let [out] = m.composite(Composite::Attention, || {
        [k.append(&keys, by).add(&v.append(&values, by))]
    });
    m.finish([(Role::Logits, &out)])
}

/// Heads of `width` with a quarter of their pairs turning.
pub fn partial_rope(bounded: &[Dim], width: u64) -> Graph {
    let m = model(bounded, &[(Heads, 2), (HeadDim, width)]);
    let ports = [floats([Seqs, Rows, Heads, HeadDim]), ids([Seqs, Rows])];
    let [x, at] = m.entry(EntryKind::Forward, ports);
    let turned = (width / 8) as u32;
    m.finish([(Role::Logits, &x.rope_part(&at, 1e6, Pairs::Halves, turned))])
}

/// Rows of 256, rotated in two blocks of 128.
pub fn rotation(bounded: &[Dim]) -> Graph {
    let m = model(bounded, &[(Hidden, 256)]);
    let [x] = m.entry(EntryKind::Forward, [rows_of(bounded, Hidden, DType::F32)]);
    m.finish([(Role::Logits, &x.hadamard(128))])
}

/// Rows of 512 through a rotated matrix to 128.
pub fn turned(bounded: &[Dim]) -> Graph {
    let m = model(bounded, &[(Hidden, 512), (Ffn, 128)]);
    let w: Turned = declare(&m);
    let [x] = m.entry(EntryKind::Forward, [rows_of(bounded, Hidden, DType::F32)]);
    let product = x.mul_vector(&w.suh).hadamard(128).matmul(&w.w);
    m.finish([(Role::Logits, &product.hadamard(128).mul_vector(&w.svh))])
}

/// Each row's top two experts, picked, applied and mixed.
pub fn routed(bounded: &[Dim], sizes: [u64; 3]) -> Graph {
    experts(bounded, sizes, true)
}

/// As `routed`, each expert one whole matrix with no gate.
pub fn picked(bounded: &[Dim], sizes: [u64; 3]) -> Graph {
    experts(bounded, sizes, false)
}

fn experts(bounded: &[Dim], [experts, width, out]: [u64; 3], gated: bool) -> Graph {
    let fixed = [(Experts, experts), (TopK, 2), (Hidden, width), (Ffn, out)];
    let m = model(bounded, &fixed);
    let w: Experts = declare(&m);
    let [x] = m.entry(EntryKind::Forward, [rows_of(bounded, Hidden, DType::F32)]);
    let scores = x.matmul(&w.router);
    let ids = scores.top_k(TopK);
    let gates = scores.take(&ids).softmax();
    let on = |set: &Linears| x.matmul_rows(&m.pick(set, &ids));
    let picks = match gated {
        true => on(&w.set.rows(0u64, Ffn))
            .act(Act::Silu)
            .mul(&on(&w.set.rows(Ffn, Ffn))),
        false => on(&w.set),
    };
    let axis = bounded.len() as u32;
    m.finish([(Role::Logits, &picks.mix(axis, &gates))])
}

/// Each row's top two entries weight the second input.
pub fn picks(bounded: &[Dim], width: u64) -> Graph {
    let m = model(bounded, &[(TopK, 2), (Hidden, width)]);
    let lead = bounded.iter().map(|d| DimExpr::from(d.name));
    let shape: Vec<DimExpr> = lead.chain([TopK.into(), Hidden.into()]).collect();
    let stacked = Port {
        role: Role::Logits,
        ty: TensorType::new(shape, DType::F32),
    };
    let [x, y] = m.entry(
        EntryKind::Forward,
        [rows_of(bounded, Hidden, DType::F32), stacked],
    );
    let gates = x.take(&x.top_k(TopK)).softmax();
    m.finish([(Role::Logits, &y.mix(bounded.len() as u32, &gates))])
}

/// Rows read from a table.
pub fn table(bounded: &[Dim], vocab: u64, width: u64) -> Graph {
    let m = model(bounded, &[(Vocab, vocab), (Hidden, width)]);
    let w: Table = declare(&m);
    let shape: Vec<DimExpr> = bounded.iter().map(|d| d.name.into()).collect();
    let tokens = Port {
        role: Role::Tokens,
        ty: TensorType::new(shape, DType::U32),
    };
    let [tokens] = m.entry(EntryKind::Forward, [tokens]);
    m.finish([(Role::Logits, &m.embed(&tokens, &w.embed))])
}

/// Neighbour rows of a computed table, at weight ids.
pub fn neighbours(bounded: &[Dim], [rows, taps, width]: [u64; 3]) -> Graph {
    let m = model(bounded, &[(Vocab, rows), (TopK, taps), (Hidden, width)]);
    let table = TensorType::new([Vocab, TopK], DType::F32);
    let ids = m
        .weight(&[PathSeg::Field("ids".into())], table)
        .cast(DType::U32);
    let x = Port {
        role: Role::Logits,
        ty: TensorType::new([bounded[0].name, Vocab, Hidden], DType::F32),
    };
    let [x, y] = m.entry(EntryKind::Forward, [x.clone(), x]);
    let near = x.permute(&[1, 0, 2]).gather(&ids).permute(&[2, 1, 0, 3]);
    // A second gather, shaped as a convolution's input.
    let w: Conv = declare(&m);
    let flat = vec![bounded[0].name.into(), Vocab.into(), TopK * Hidden];
    let gathered = y.permute(&[1, 0, 2]).gather(&ids).permute(&[2, 0, 1, 3]);
    let conv = gathered.reshape(flat).matmul(&w.w);
    m.finish([(Role::Logits, &near.mean(1)), (Role::Logits, &conv)])
}

/// Scores pick each query's best keys for attention.
pub fn indexed(bounded: &[Dim], [heads, width, keep]: [u64; 3]) -> Graph {
    let fixed = [(Heads, heads), (HeadDim, width), (INDEX_KEEP, keep)];
    let m = model(bounded, &fixed);
    let heads = || floats([Seqs, Context, Heads, HeadDim]);
    let rows = floats([Seqs, Context, HeadDim]);
    let [q, v, x] = m.entry(EntryKind::Forward, [heads(), heads(), rows]);
    // One key row per position, shared by every head.
    let one = vec![
        Seqs.into(),
        Context.into(),
        DimExpr::Const(1),
        HeadDim.into(),
    ];
    let shared = x.reshape(one).expand(2, Heads);
    let half = width / 2;
    let k = shared.slice(3, 0u64, half).concat(&v.slice(3, half, half));
    let scores = x.matmul_rows(&x);
    let scores = scores.add(&scores.permute(&[0, 2, 1]));
    let ids = scores.mask(Mask::Causal).top_k(INDEX_KEEP);
    let out = m.attend(Mask::Causal, 0.25, [&q, &k, &v], Some(&ids));
    m.finish([(Role::Logits, &out)])
}

/// Weighted ReLU scores over keys: the Lightning Indexer.
pub fn indexer(bounded: &[Dim], [heads, width]: [u64; 2]) -> Graph {
    let fixed = [(Heads, heads), (HeadDim, width), (Hidden, 1)];
    let m = model(bounded, &fixed);
    let ports = [
        floats([Seqs, Context, Heads, HeadDim]),
        floats([Seqs, Context, HeadDim]),
        floats([Seqs, Context, Hidden, Heads]),
    ];
    let [q, k, w] = m.entry(EntryKind::Forward, ports);
    let hits = q.matmul_rows(&k).act(Act::Relu);
    let scores = hits.permute(&[0, 1, 3, 2]).matmul_rows(&w);
    let out = scores.reshape([Seqs, Context, Context]).mask(Mask::Causal);
    m.finish([(Role::Logits, &out)])
}

/// Ragged rows pick keys up to their position, then attend.
pub fn indexed_rows(bounded: &[Dim], [heads, width, keep]: [u64; 3]) -> Graph {
    let rows: Vec<Dim> = bounded.iter().filter(|d| d.name == Rows).cloned().collect();
    // Eight slots of eight keys hold any id below eight.
    let fixed = [
        (Seqs, 8),
        (Context, 8),
        (Heads, heads),
        (HeadDim, width),
        (INDEX_KEEP, keep),
    ];
    let m = model(&rows, &fixed);
    let keys = || floats([Seqs, Context, Heads, HeadDim]);
    let ports = [
        floats([Rows, Heads, HeadDim]),
        keys(),
        keys(),
        floats([Rows, Context]),
        ids([Rows]),
        ids([Rows]),
    ];
    let [q, k, v, scores, slots, at] = m.entry(EntryKind::Forward, ports);
    let picked = scores.causal(&at).top_k(INDEX_KEEP);
    let out = m.attend_rows(
        Mask::Causal,
        0.25,
        [&q, &k, &v],
        [&slots, &at],
        Some(&picked),
    );
    m.finish([(Role::Logits, &out)])
}

/// MLA Q partial-RoPE and KV-prep scatter into kept rows.
pub fn mla_prep(bounded: &[Dim], [heads, latent, rope]: [u64; 3]) -> Graph {
    let mut dims: Vec<Dim> = bounded.iter().filter(|d| d.name == Rows).cloned().collect();
    dims.push(Dim::bounded(Context, 8));
    let fixed = [
        (Seqs, 8),
        (Heads, heads),
        (LATENT, latent),
        (ROPE, rope),
        (Hidden, latent),
    ];
    let m = model(&dims, &fixed);
    let w: Scale = declare(&m);
    let q_ty = TensorType::new(vec![Rows.into(), Heads.into(), LATENT + ROPE], DType::F32);
    let kv_ty = TensorType::new(vec![Rows.into(), LATENT + ROPE], DType::F32);
    let port = |role, ty| Port { role, ty };
    let [q, kv, slots, positions] = m.entry(
        EntryKind::Step,
        [
            port(Role::Logits, q_ty),
            port(Role::Logits, kv_ty),
            ids([Rows]),
            ids([Rows]),
        ],
    );
    let q_out = q
        .slice(2, 0u64, LATENT)
        .concat(
            &q.slice(2, LATENT, ROPE)
                .rope(&positions, 1e4, Pairs::Adjacent),
        );
    let kept_ty = TensorType::new(
        vec![
            Seqs.into(),
            Context.into(),
            DimExpr::Const(1),
            LATENT + ROPE,
        ],
        DType::F32,
    );
    let kept = m.kept(kept_ty);
    let lat = kv.slice(1, 0u64, LATENT).rms_norm(&w.norm, 1e-6);
    let shared = kv
        .slice(1, LATENT, ROPE)
        .rope(&positions, 1e4, Pairs::Adjacent);
    let kv_row = lat
        .concat(&shared)
        .reshape(vec![Rows.into(), DimExpr::Const(1), LATENT + ROPE]);
    let kv_kept = kv_row.append(&kept, [&slots, &positions]);
    let out = m.attend_rows(
        Mask::Causal,
        0.25,
        [&q_out, &kv_kept, &kv_kept],
        [&slots, &positions],
        None,
    );
    m.finish([(Role::Logits, &out)])
}

fn kept_attend_rows(
    bounded: &[Dim],
    [heads, kv_heads, width]: [u64; 3],
    context: u64,
    seqs: u64,
    scale: f32,
    shared_kv: bool,
) -> Graph {
    let mut dims: Vec<Dim> = bounded.iter().filter(|d| d.name == Rows).cloned().collect();
    dims.push(Dim::bounded(Context, context));
    let fixed = [
        (Seqs, seqs),
        (Heads, heads),
        (KvHeads, kv_heads),
        (HeadDim, width),
    ];
    let m = model(&dims, &fixed);
    let port = |role, ty| Port { role, ty };
    let u32_rows = TensorType::new([Rows], DType::U32);
    let cache = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let out = if shared_kv {
        let [q, kv, slots, positions] = m.entry(
            EntryKind::Step,
            [
                floats([Rows, Heads, HeadDim]),
                floats([Rows, KvHeads, HeadDim]),
                port(Role::Slots, u32_rows.clone()),
                port(Role::Positions, u32_rows),
            ],
        );
        let kept = m.kept(cache);
        let by = [&slots, &positions];
        let k = kv.append(&kept, by);
        m.attend_rows(Mask::Causal, scale, [&q, &k, &k], by, None)
    } else {
        let [q, k, v, slots, positions] = m.entry(
            EntryKind::Step,
            [
                floats([Rows, Heads, HeadDim]),
                floats([Rows, KvHeads, HeadDim]),
                floats([Rows, KvHeads, HeadDim]),
                port(Role::Slots, u32_rows.clone()),
                port(Role::Positions, u32_rows),
            ],
        );
        let (keys, values) = (m.kept(cache.clone()), m.kept(cache));
        let by = [&slots, &positions];
        let (k, v) = (k.append(&keys, by), v.append(&values, by));
        m.attend_rows(Mask::Causal, scale, [&q, &k, &v], by, None)
    };
    m.finish([(Role::Logits, &out)])
}

/// Ragged kept-cache MLA matching `fast_mla_attend` (`HeadDim = 576`).
pub fn mla_attend_rows(bounded: &[Dim], heads: u64, context: u64, seqs: u64) -> Graph {
    kept_attend_rows(bounded, [heads, 1, 576], context, seqs, 0.04, true)
}

/// Ragged kept-cache GQA matching `fast_gqa_attend`.
pub fn gqa_attend_rows(bounded: &[Dim], dims: [u64; 3], context: u64, seqs: u64) -> Graph {
    kept_attend_rows(bounded, dims, context, seqs, 0.125, false)
}

/// Direct-routed MoE block with explicit `Role::Picks` expert ids.
pub fn moe_routed(bounded: &[Dim], [experts, top_k, hidden, ffn]: [u64; 4], gated: bool) -> Graph {
    let fixed = [
        (Experts, experts),
        (TopK, top_k),
        (Hidden, hidden),
        (Ffn, ffn),
    ];
    let m = model(bounded, &fixed);
    let picks_port = Port {
        role: Role::Picks,
        ty: TensorType::new([Rows, TopK], DType::U32),
    };
    let [x_in, ids, gates] = m.entry(
        EntryKind::Forward,
        [
            rows_of(bounded, Hidden, DType::F32),
            picks_port,
            floats([Rows, TopK]),
        ],
    );
    // Stage op lets BF16 activation pins narrow `x` for `picked_bf16`.
    let x = x_in.scale(1.0);
    let decl = |name: &str, shape: [DimExpr; 3]| {
        let mut path = vec![PathSeg::Field(name.into())];
        Linears::declare(&m, &mut path, &shape)
    };
    let out = if gated {
        let gate_w = decl("gate", [Experts.into(), Hidden.into(), Ffn.into()]);
        let up_w = decl("up", [Experts.into(), Hidden.into(), Ffn.into()]);
        let down_w = decl("down", [Experts.into(), Ffn.into(), Hidden.into()]);
        let g = x.matmul_rows(&m.pick(&gate_w, &ids)).act(Act::Silu);
        let u = x.matmul_rows(&m.pick(&up_w, &ids));
        g.mul(&u).matmul_rows(&m.pick(&down_w, &ids))
    } else {
        let set = decl("set", [Experts.into(), Hidden.into(), Ffn.into()]);
        x.matmul_rows(&m.pick(&set, &ids))
    };
    m.finish([(Role::Logits, &out.mix(bounded.len() as u32, &gates))])
}
