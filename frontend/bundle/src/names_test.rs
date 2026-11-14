use crate::names::{Unresolved, resolve, rules};
use checkpoint::{Slice, Stored};
use graph::Graph;
use models::qwen3::{Config, stepper, tensor_name};

fn graph() -> Graph {
    stepper(&Config::toy(1, 8))
}

/// Every weight as `spell` names it, sized to fit.
fn offered(g: &Graph, spell: impl Fn(&str) -> String) -> Vec<Slice> {
    let mut out = Vec::new();
    for w in &g.weights {
        let name = spell(&tensor_name(&w.path).unwrap());
        let axis = |e: &graph::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
        out.push(Slice {
            name,
            shape: w.ty.shape.0.iter().map(axis).collect(),
            stored: Stored::F32,
        });
    }
    out
}

/// GGUF's spelling of a Hugging Face name.
pub(crate) fn gguf(hf: &str) -> String {
    let swaps = [
        ("model.layers.", "blk."),
        ("self_attn.q_proj", "attn_q"),
        ("self_attn.k_proj", "attn_k"),
        ("self_attn.v_proj", "attn_v"),
        ("self_attn.o_proj", "attn_output"),
        ("self_attn.q_norm", "attn_q_norm"),
        ("self_attn.k_norm", "attn_k_norm"),
        ("mlp.gate_proj", "ffn_gate"),
        ("mlp.up_proj", "ffn_up"),
        ("mlp.down_proj", "ffn_down"),
        ("input_layernorm", "attn_norm"),
        ("post_attention_layernorm", "ffn_norm"),
        ("model.embed_tokens", "token_embd"),
        ("model.norm", "output_norm"),
    ];
    swaps
        .iter()
        .fold(hf.to_string(), |s, (a, b)| s.replace(a, b))
}

#[test]
fn both_spellings_resolve() {
    let g = graph();
    for spell in [|s: &str| s.to_string(), gguf] {
        let slices = offered(&g, spell);
        let names: Vec<String> = resolve(&g, &slices, &[])
            .unwrap()
            .into_iter()
            .map(|c| c.name)
            .collect();
        let want: Vec<String> = slices.iter().map(|s| s.name.clone()).collect();
        assert_eq!(names, want);
    }
}

#[test]
fn refuses_doubt() {
    let g = graph();
    let mut slices = offered(&g, |s| s.to_string());
    let q = slices
        .iter()
        .find(|s| s.name.contains("0.self_attn.q_proj"))
        .unwrap();
    let extra = Slice {
        name: "model.layers.0.self_attn.q.weight".into(),
        ..q.clone()
    };
    slices.push(extra);
    slices.retain(|s| s.name != "model.norm.weight");
    let down = slices
        .iter_mut()
        .find(|s| s.name.contains("0.mlp.down"))
        .unwrap();
    down.shape.reverse();
    let Err(Unresolved {
        missing, ambiguous, ..
    }) = resolve(&g, &slices, &[])
    else {
        panic!("doubt refused");
    };
    assert_eq!(missing, ["layers.0.down", "norm"]);
    assert_eq!(ambiguous.len(), 1);
    assert!(ambiguous[0].0.ends_with(".q") && ambiguous[0].1.len() == 2);
}

#[test]
fn rules_decide() {
    let g = graph();
    let mut slices = offered(&g, |s| s.to_string());
    slices
        .iter_mut()
        .for_each(|s| s.name = s.name.replace("q_proj", "query"));
    assert!(resolve(&g, &slices, &[]).is_err());
    let repo = rules(r#"{"layers.*.q": "model.layers.*.self_attn.query"}"#).unwrap();
    let names = resolve(&g, &slices, &[&repo]).unwrap();
    let at = g
        .weights
        .iter()
        .position(|w| tensor_name(&w.path).unwrap().contains("0.self_attn.q_proj"));
    assert_eq!(
        names[at.unwrap()].name,
        "model.layers.0.self_attn.query.weight"
    );
    let user = rules(r#"{"layers.*.q": "elsewhere.*"}"#).unwrap();
    let failed = resolve(&g, &slices, &[&repo, &user]).unwrap_err();
    assert_eq!(failed.missing, ["layers.0.q", "layers.1.q"]);
    let wrong = rules(r#"{"norm": "model.layers.0.self_attn.query"}"#).unwrap();
    let failed = resolve(&g, &slices, &[&repo, &wrong]).unwrap_err();
    assert!(
        failed
            .to_string()
            .contains("norm: model.layers.0.self_attn.query.weight")
    );
}

/// GGUF's name for a weight path.
pub(crate) fn gguf_name(path: &[graph::PathSeg]) -> Option<String> {
    Some(gguf(&tensor_name(path)?))
}

/// A weight path as `layers.3.q`.
pub(crate) fn text(path: &[graph::PathSeg]) -> String {
    let seg = |p: &graph::PathSeg| match p {
        graph::PathSeg::Field(f) => f.clone(),
        graph::PathSeg::Index(i) => i.to_string(),
    };
    path.iter().map(seg).collect::<Vec<_>>().join(".")
}
