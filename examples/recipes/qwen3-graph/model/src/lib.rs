//! A Qwen3 step graph, built inside the recipe's sandbox.

use tessera_guest::{Build, Table};
use tessera_trace::DimName::*;
use tessera_trace::hf::*;
use tessera_trace::{
    Act, Composite, DType, Dim, DimExpr, Embed, Graph, Linear, Mask, Model, Norm, Pairs, Tensor,
    TensorType, Weights, pick, step_ports,
};

/// The SwiGLU block after each attention block.
#[derive(Weights)]
struct Gated {
    #[shape(Hidden, Ffn)]
    gate: Linear,
    #[shape(Hidden, Ffn)]
    up: Linear,
    #[shape(Ffn, Hidden)]
    down: Linear,
}

/// One layer: attention with normed heads, then the MLP.
#[derive(Weights)]
struct Block {
    #[shape(Hidden)]
    ln1: Norm,
    #[shape(Hidden, Heads * HeadDim)]
    q: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    k: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    v: Linear,
    #[shape(HeadDim)]
    q_norm: Norm,
    #[shape(HeadDim)]
    k_norm: Norm,
    #[shape(Heads * HeadDim, Hidden)]
    o: Linear,
    #[shape(Hidden)]
    ln2: Norm,
    mlp: Gated,
}

/// The embedding is also the output head.
#[derive(Weights)]
struct Trunk {
    #[shape(Vocab, Hidden)]
    embed: Embed,
    #[shape(Layers)]
    layers: Vec<Block>,
    #[shape(Hidden)]
    norm: Norm,
}

/// `config`'s unsigned value at `key`.
fn size(config: &serde_json::Value, key: Key<u64>) -> Result<u64, String> {
    let key = key.name;
    config[key]
        .as_u64()
        .ok_or(format!("config.json needs {key}"))
}

/// `config`'s number at `key`.
fn number(config: &serde_json::Value, key: Key<f32>) -> Result<f32, String> {
    let key = key.name;
    let n = config[key]
        .as_f64()
        .ok_or(format!("config.json needs {key}"));
    n.map(|n| n as f32)
}

/// One step entry over `seqs` lanes of `context` rows.
fn program(config: &serde_json::Value, seqs: u64, context: u64) -> Result<Graph, String> {
    let fixed = [
        (Vocab, VOCAB),
        (Hidden, HIDDEN),
        (Layers, LAYERS),
        (Heads, HEADS),
        (HeadDim, HEAD_DIM),
        (Ffn, FFN),
        (KvHeads, KV_HEADS),
    ];
    let mut dims = vec![Dim::bounded(Seqs, seqs), Dim::bounded(Context, context)];
    for (name, key) in fixed {
        dims.push(Dim::fixed(name, size(config, key)?));
    }
    dims.push(Dim::bounded(Rows, context));
    let theta = number(config, ROPE_THETA)?;
    let eps = number(config, EPS)?;
    let scale = 1.0 / (size(config, HEAD_DIM)? as f32).sqrt();

    let m = Model::new(dims);
    let w: Trunk = tessera_trace::declare(&m);
    let ([tokens, positions, slots], _) = step_ports(&m, false);
    let ty = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let by = [&slots, &positions];
    let heads = |t: Tensor, n: DimExpr| t.reshape(vec![Rows.into(), n, HeadDim.into()]);
    let spin = |t: Tensor| t.rope(&positions, theta, Pairs::Halves);
    let mut kept = Vec::new();
    for _ in &w.layers {
        kept.push([m.kept(ty.clone()), m.kept(ty.clone())]);
    }
    let [logits] = m.composite(Composite::Model, || {
        let mut x = m.embed(&tokens, &w.embed);
        for (l, [keys, values]) in w.layers.iter().zip(&kept) {
            let [out] = m.composite(Composite::Layer, || {
                let h = x.rms_norm(&l.ln1, eps);
                let [attn] = m.composite(Composite::Attention, || {
                    let q = spin(heads(h.matmul(&l.q), Heads.into()).rms_norm(&l.q_norm, eps));
                    let k = spin(heads(h.matmul(&l.k), KvHeads.into()).rms_norm(&l.k_norm, eps));
                    let v = heads(h.matmul(&l.v), KvHeads.into());
                    let (k, v) = (k.append(keys, by), v.append(values, by));
                    let out = m.attend_rows(Mask::Causal, scale, [&q, &k, &v], by, None);
                    let width = DimExpr::from(Heads) * DimExpr::from(HeadDim);
                    [out.reshape(vec![Rows.into(), width]).matmul(&l.o)]
                });
                let x = attn.add(&x);
                let h = x.rms_norm(&l.ln2, eps);
                let [mlp] = m.composite(Composite::Mlp, || {
                    let g = &l.mlp;
                    let inner = h.matmul(&g.gate).act(Act::Silu).mul(&h.matmul(&g.up));
                    [inner.matmul(&g.down)]
                });
                [mlp.add(&x)]
            });
            x = out;
        }
        [x.rms_norm(&w.norm, eps).unembed(&w.embed)]
    });
    Ok(pick(&m, &logits, None, None))
}

/// The graph as JSON, or why the input is refused.
fn built(b: &Build) -> Result<String, String> {
    let context = b.context.ok_or("the build input needs a context size")?;
    Ok(program(&b.config, b.seqs, context)?.to_json())
}

/// The `tables` export: an empty list.
fn none(_: &Build) -> Result<Vec<Table>, String> {
    Ok(Vec::new())
}

tessera_guest::exports! {
    graph: built,
    tables: none,
}
