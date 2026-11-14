use crate::names::{repo, resolve};
use checkpoint::{Slice, Stored};
use graph::PathSeg::{Field, Index};
use models::Rotated;
use models::eg2::{Config, program};

/// The release's name for an EmbeddingGemma 2 weight path.
fn spelled(path: &[graph::PathSeg]) -> String {
    let (role, part) = match path {
        [rest @ .., Field(p)] if ["w", "suh", "svh"].contains(&p.as_str()) => {
            (rest, if p == "w" { "codes" } else { p.as_str() })
        }
        _ => (path, "weight"),
    };
    let stem = match role {
        [Field(f)] => match f.as_str() {
            "embed" => "embed_tokens".to_string(),
            "norm" => "norm".to_string(),
            "proj" => "embedding_projection".to_string(),
            _ => "ple.per_layer_projection_norm".to_string(),
        },
        [Field(kind), Index(l), Field(f)] => {
            let local = match (kind.as_str(), f.as_str()) {
                (_, "ln") => "input_layernorm".to_string(),
                (_, "post") => "post_attention_layernorm".to_string(),
                (_, "q_norm" | "k_norm") => format!("self_attn.{f}"),
                ("attention", _) => format!("self_attn.{f}_proj"),
                (_, "pre_ff") => "pre_feedforward_layernorm".to_string(),
                (_, "post_ff") => "post_feedforward_layernorm".to_string(),
                (_, "gate" | "up" | "down") => format!("mlp.{f}_proj"),
                (_, "ple_in") => {
                    return format!("language_model.ple.per_layer_model_projection.{part}");
                }
                (_, "ple_gate") => "ple_block.per_layer_input_gate".to_string(),
                (_, "ple_proj") => "ple_block.per_layer_projection".to_string(),
                (_, "ple_norm") => "ple_block.post_per_layer_input_norm".to_string(),
                _ => return format!("language_model.layers.{l}.layer_scalar"),
            };
            format!("layers.{l}.{local}")
        }
        _ => unreachable!(),
    };
    format!("language_model.{stem}.{part}")
}

/// Every weight resolves, the shared projection by rows.
#[test]
fn release_resolves() {
    let mut c = Config::release(1, 64);
    c.stack.layers = 2;
    let g = program::<Rotated>(&c);
    let mut slices: Vec<Slice> = Vec::new();
    for w in &g.weights {
        let name = spelled(&w.path);
        if slices.iter().any(|s| s.name == name) {
            continue;
        }
        let axis = |e: &graph::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
        let mut shape: Vec<usize> = w.ty.shape.0.iter().map(axis).collect();
        if name.contains("per_layer_model_projection") && !name.ends_with(".suh") {
            shape[0] *= c.stack.layers as usize;
        }
        let stored = Stored::F32;
        slices.push(Slice {
            name,
            shape,
            stored,
        });
    }
    let rules = repo("eg2").unwrap();
    let claims = resolve(&g, &slices, &[&rules]).unwrap();
    assert_eq!(claims.len(), g.weights.len());
    let per = c.per_layer as usize;
    let at = |text: &str| {
        let i = g
            .weights
            .iter()
            .position(|w| crate::names_test::text(&w.path) == text);
        claims[i.unwrap()].clone()
    };
    assert_eq!(at("layers.1.ple_in.w").rows, Some((per, per)));
    assert_eq!(at("layers.1.ple_in.svh").rows, Some((per, per)));
    assert_eq!(at("layers.1.ple_in.suh").rows, None);
    assert!(
        at("layers.1.scalar")
            .name
            .ends_with("layers.1.layer_scalar")
    );
    assert!(resolve(&g, &slices, &[]).is_err());
}
