use crate::{self as trace, *};
use DimName::*;

#[derive(Weights)]
struct Block {
    #[shape(Hidden)]
    norm: Norm,
    #[shape(Hidden, Hidden * 3)]
    qkv: Linear,
}

/// Named like a `DimName` variant, which the derive must allow.
#[derive(Weights)]
struct Experts {
    #[shape(Hidden)]
    gate: Norm,
}

#[derive(Weights)]
struct Net {
    #[shape(Vocab, Hidden)]
    embed: Embed,
    #[shape(Layers)]
    blocks: Vec<Block>,
    #[shape(Layers)]
    experts: Vec<Experts>,
}

fn model() -> Model {
    Model::new(vec![
        Dim::bounded(Rows, 16),
        Dim::fixed(Hidden, 8),
        Dim::fixed(Vocab, 32),
        Dim::fixed(Layers, 2),
    ])
}

fn field(name: &str) -> PathSeg {
    PathSeg::Field(name.into())
}

#[test]
fn weights_follow_fields() {
    let m = model();
    let net: Net = trace::declare(&m);
    let mut x = m.embed(&traced_ids(&m), &net.embed);
    for b in &net.blocks {
        x = x.rms_norm(&b.norm, 1e-5).matmul(&b.qkv).slice(1, 0, Hidden);
    }
    x = x.rms_norm(&net.experts[1].gate, 1e-5);
    let g = m.finish([(Role::Logits, &x)]);
    let paths: Vec<_> = g.weights.iter().map(|w| w.path.clone()).collect();
    assert_eq!(paths[0], [field("embed")]);
    assert_eq!(paths[4], [field("blocks"), PathSeg::Index(1), field("qkv")]);
    let qkv = &g.weights[2].ty.shape;
    assert_eq!(qkv, &Shape(vec![Hidden * 3, Hidden.into()]));
}

#[test]
#[should_panic(expected = "matmul K differs")]
fn type_errors_panic() {
    let m = model();
    let net: Net = trace::declare(&m);
    let x = m
        .embed(&traced_ids(&m), &net.embed)
        .matmul(&net.blocks[0].qkv);
    x.matmul(&net.blocks[1].qkv);
}

fn traced_ids(m: &Model) -> Tensor {
    let [ids] = m.entry(
        EntryKind::Forward,
        [Port {
            role: Role::Tokens,
            ty: TensorType::new([Rows], DType::U32),
        }],
    );
    ids
}

#[test]
fn composites_record_regions() {
    let m = model();
    let net: Net = trace::declare(&m);
    let x = m.embed(&traced_ids(&m), &net.embed);
    let b = &net.blocks[0];
    let [y] = m.composite(Composite::Mlp, || {
        let h = x.rms_norm(&b.norm, 1e-5);
        let [inner] = m.composite(Composite::Attention, || [h.matmul(&b.qkv)]);
        [inner.slice(1, 0, Hidden)]
    });
    let g = m.finish([(Role::Logits, &y)]);
    let [inner, outer] = &g.regions[..] else {
        panic!("two regions, inner first");
    };
    assert_eq!(
        (outer.kind, inner.kind),
        (Composite::Mlp, Composite::Attention)
    );
    assert_eq!(outer.outputs, [y.id]);
    assert_eq!(outer.inputs, g.reads(outer.start, outer.end));
    assert!(outer.start < inner.start && inner.end < outer.end);
    let matmul = g.node(inner.outputs[0]);
    assert_eq!(matmul.numerics, Some(graph::Numerics::default()));
    assert_eq!(
        g.node(outer.start).numerics,
        Some(graph::Numerics::default())
    );
}

#[test]
#[should_panic(expected = "must return values it computed")]
fn composites_own_outputs() {
    let m = model();
    let net: Net = trace::declare(&m);
    let x = m.embed(&traced_ids(&m), &net.embed);
    m.composite(Composite::Mlp, || {
        [x.rms_norm(&net.blocks[0].norm, 1e-5), x.clone()]
    });
}
