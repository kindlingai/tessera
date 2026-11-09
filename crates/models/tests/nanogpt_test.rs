//! The traced nanogpt, run from its JSON alone, matches crates/nanogpt.

use interp::{Cpu, Data, Gpu, Kernels, Tensor, bind, gpu_ready, run};
use models::nanogpt::{Config, program};
use nanogpt::data::{Corpus, DEFAULT_TEXT};
use nanogpt::model::{self as gpt, EPS, GptConfig, Weights};
use nanogpt::tensor::Target;
use trace::{Graph, PathSeg};

struct Case {
    cfg: GptConfig,
    w: Weights,
    tokens: Vec<u32>,
    seq: usize,
}

fn case(batch: usize, seq: usize) -> Case {
    let corpus = Corpus::from_text(DEFAULT_TEXT);
    let cfg = GptConfig::tiny(corpus.vocab_size());
    let tokens = corpus.batch(7, batch, seq);
    Case {
        w: Weights::random(&cfg, 1337),
        cfg,
        tokens,
        seq,
    }
}

fn config(cfg: &GptConfig) -> Config {
    Config {
        vocab: cfg.vocab as u64,
        block: cfg.block as u64,
        layers: cfg.n_layer as u64,
        heads: cfg.n_head as u64,
        hidden: cfg.n_embd as u64,
        eps: EPS,
        max_seqs: 8,
    }
}

/// The interpreter only ever sees the JSON text.
fn graph(cfg: &GptConfig) -> Graph {
    Graph::from_json(&program(&config(cfg)).to_json()).unwrap()
}

fn lookup(w: &Weights, path: &[PathSeg]) -> Option<Data> {
    use PathSeg::{Field, Index};
    let v = match path {
        [Field(f)] => match f.as_str() {
            "tok_emb" => &w.tok_emb,
            "pos_emb" => &w.pos_emb,
            "ln_f" => &w.ln_f,
            "lm_head" => &w.lm_head,
            _ => return None,
        },
        [Field(layers), Index(i), Field(f)] if layers == "layers" => {
            let l = w.layers.get(*i as usize)?;
            // nanogpt packs q, k and v rows into one matrix.
            let part = |i: usize| {
                let n = w.ln_f.len().pow(2);
                Some(Data::F32(l.wqkv[i * n..(i + 1) * n].to_vec()))
            };
            match f.as_str() {
                "wq" => return part(0),
                "wk" => return part(1),
                "wv" => return part(2),
                "ln1" => &l.ln1,
                "wproj" => &l.wproj,
                "ln2" => &l.ln2,
                "w1" => &l.w1,
                "w2" => &l.w2,
                _ => return None,
            }
        }
        _ => return None,
    };
    Some(Data::F32(v.clone()))
}

fn logits(c: &Case, k: &mut dyn Kernels) -> Vec<f32> {
    let g = graph(&c.cfg);
    let weights = bind(&g, |p| lookup(&c.w, p)).unwrap();
    let batch = c.tokens.len() / c.seq;
    let shape = vec![batch, c.seq];
    let positions = (0..batch).flat_map(|_| 0..c.seq as u32).collect();
    let inputs = [
        Tensor {
            shape: shape.clone(),
            data: Data::U32(c.tokens.clone()),
        },
        Tensor {
            shape,
            data: Data::U32(positions),
        },
    ];
    let out = run(&g, k, &weights, 0, &inputs).unwrap();
    out[0].floats().unwrap().to_vec()
}

/// Largest absolute difference and cosine similarity.
fn compare(a: &[f32], b: &[f32]) -> (f32, f64) {
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        (dot, na, nb) = (dot + x * y, na + x * x, nb + y * y);
    }
    (max, dot / (na.sqrt() * nb.sqrt()))
}

/// Prints and checks one comparison against the parity tolerance.
fn close(label: &str, got: &[f32], want: &[f32]) {
    let (max, cos) = compare(got, want);
    eprintln!("{label}: max_abs {max:.3e} cos {cos:.9}");
    assert!(
        max < 1e-5 && cos > 0.99999,
        "{label}: max_abs {max:.3e} cos {cos}"
    );
}

#[test]
fn cpu_matches_reference() {
    for (batch, seq) in [(2, 64), (3, 17)] {
        let c = case(batch, seq);
        let want = gpt::forward(Target::Cpu, &c.w, &c.cfg, &c.tokens, c.seq).unwrap();
        close(&format!("cpu {batch}x{seq}"), &logits(&c, &mut Cpu), &want);
    }
}

#[test]
fn gpu_matches_nanogpt() {
    if !gpu_ready() {
        return eprintln!("no GPU here; skipped");
    }
    let (c, mut k) = (case(2, 64), Gpu::default());
    let got = logits(&c, &mut k);
    assert_eq!(k.fallbacks, 0, "every op should have a GPU kernel");
    let ours = gpt::forward(Target::Gpu, &c.w, &c.cfg, &c.tokens, c.seq).unwrap();
    close("gpu vs nanogpt gpu", &got, &ours);
    let cpu = gpt::forward(Target::Cpu, &c.w, &c.cfg, &c.tokens, c.seq).unwrap();
    close("gpu vs cpu reference", &got, &cpu);
}

#[test]
fn trace_is_deterministic() {
    let cfg = case(1, 8).cfg;
    let g = graph(&cfg);
    graph::verify(&g).unwrap();
    assert_eq!(g.hash(), program(&config(&cfg)).hash());
    assert_eq!(g.weights.len(), 4 + 8 * cfg.n_layer);
}

#[test]
fn layers_mark_composites() {
    use trace::Composite::{Attention, Mlp};
    let g = graph(&case(1, 8).cfg);
    let kinds: Vec<_> = g.regions.iter().map(|r| r.kind).collect();
    assert_eq!(kinds, [Attention, Mlp, Attention, Mlp]);
    let mlp = &g.regions[1];
    let ops = &g.nodes[mlp.start.0 as usize..mlp.end.0 as usize];
    assert_eq!(ops.iter().filter(|n| n.numerics.is_some()).count(), 2);
}
