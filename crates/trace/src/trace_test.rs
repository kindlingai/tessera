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
        Dim {
            name: Rows,
            size: Size::Bounded(16),
        },
        Dim {
            name: Hidden,
            size: Size::Fixed(8),
        },
        Dim {
            name: Vocab,
            size: Size::Fixed(32),
        },
        Dim {
            name: Layers,
            size: Size::Fixed(2),
        },
    ])
}

fn field(name: &str) -> PathSeg {
    PathSeg::Field(name.into())
}

#[test]
fn weights_follow_fields() {
    let m = model();
    let net: Net = trace::declare(&m);
    let [ids] = m.entry(
        EntryKind::Forward,
        [Port {
            role: Role::Tokens,
            ty: TensorType::new([Rows], DType::U32),
        }],
    );
    let mut x = m.embed(&ids, &net.embed);
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
    let [ids] = m.entry(
        EntryKind::Forward,
        [Port {
            role: Role::Tokens,
            ty: TensorType::new([Rows], DType::U32),
        }],
    );
    let x = m.embed(&ids, &net.embed).matmul(&net.blocks[0].qkv);
    x.matmul(&net.blocks[1].qkv);
}
