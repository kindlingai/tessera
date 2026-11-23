//! A Qwen3 step graph, built inside the recipe's sandbox.

use serde::Deserialize;
use tessera_guest::{input, output};
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, Composite, DType, Dim, DimExpr, Embed, Graph, Linear, Mask, Model, Norm, Pairs, Tensor,
    TensorType, Weights, pick, step_ports,
};

/// Inputs the host passes to `graph`.
#[derive(Deserialize)]
struct Build {
    config: serde_json::Value,
    seqs: u64,
    context: Option<u64>,
}

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
fn size(config: &serde_json::Value, key: &str) -> Result<u64, String> {
    config[key]
        .as_u64()
        .ok_or(format!("config.json needs {key}"))
}

/// `config`'s number at `key`.
fn number(config: &serde_json::Value, key: &str) -> Result<f32, String> {
    let n = config[key]
        .as_f64()
        .ok_or(format!("config.json needs {key}"));
    n.map(|n| n as f32)
}

/// One step entry over `seqs` lanes of `context` rows.
fn program(config: &serde_json::Value, seqs: u64, context: u64) -> Result<Graph, String> {
    let fixed = [
        (Vocab, "vocab_size"),
        (Hidden, "hidden_size"),
        (Layers, "num_hidden_layers"),
        (Heads, "num_attention_heads"),
        (HeadDim, "head_dim"),
        (Ffn, "intermediate_size"),
        (KvHeads, "num_key_value_heads"),
    ];
    let mut dims = vec![Dim::bounded(Seqs, seqs), Dim::bounded(Context, context)];
    for (name, key) in fixed {
        dims.push(Dim::fixed(name, size(config, key)?));
    }
    dims.push(Dim::bounded(Rows, context));
    let theta = number(config, "rope_theta")?;
    let eps = number(config, "rms_norm_eps")?;
    let scale = 1.0 / (size(config, "head_dim")? as f32).sqrt();

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
fn built(at: i32, len: i32) -> Result<String, String> {
    // SAFETY: the host passes back what `alloc` returned.
    let bytes = unsafe { input(at, len) };
    let b: Build = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let context = b.context.ok_or("the build input needs a context size")?;
    Ok(program(&b.config, b.seqs, context)?.to_json())
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: i32) -> i32 {
    tessera_guest::alloc(len)
}

#[unsafe(no_mangle)]
pub extern "C" fn manifest() -> i64 {
    let manifest = serde_json::json!({"abi": tessera_guest::ABI, "graph": true});
    output(manifest.to_string().into_bytes())
}

/// The graph as JSON, else why the config is refused.
#[unsafe(no_mangle)]
pub extern "C" fn graph(at: i32, len: i32) -> i64 {
    output(built(at, len).unwrap_or_else(|e| e).into_bytes())
}
