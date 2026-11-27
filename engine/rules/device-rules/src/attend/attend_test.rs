use super::*;
use tessera_graph::DimName::{Context, HeadDim, Heads, KvHeads, Rows, Seqs};
use tessera_graph::{
    Attend, Dim, Entry, EntryKind, Graph, Mask, Node, NodeId, Op, Output, Port, Role, TensorType,
};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM};
use tessera_lower::{Device, Pins, Tree, Wants};

fn ring_graph(heads: u64, kv_heads: u64, width: u64, mla: bool) -> Graph {
    let mut g = Graph::new(vec![
        Dim::bounded(Seqs, 2),
        Dim::bounded(Rows, 4),
        Dim::bounded(Context, 4096),
        Dim::fixed(Heads, heads),
        Dim::fixed(KvHeads, kv_heads),
        Dim::fixed(HeadDim, width),
    ]);
    let q_ty = TensorType::new([Rows, Heads, HeadDim], DType::F32);
    let kv_ty = TensorType::new([Rows, KvHeads, HeadDim], DType::F32);
    let kept_ty = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let ids_ty = TensorType::new([Rows], DType::U32);
    let push = |g: &mut Graph, op: Op, args: Vec<NodeId>, ty: TensorType| -> NodeId {
        let id = NodeId(g.nodes.len() as u32);
        g.nodes.push(Node {
            op,
            args,
            ty,
            numerics: None,
        });
        id
    };
    let inp = |port| Op::Input { entry: 0, port };
    let q = push(&mut g, inp(0), vec![], q_ty.clone());
    let k_in = push(&mut g, inp(1), vec![], kv_ty.clone());
    let (v_in, s_i, p_i, mut ports) = if mla {
        (
            k_in,
            2,
            3,
            vec![
                Port {
                    role: Role::Tokens,
                    ty: q_ty.clone(),
                },
                Port {
                    role: Role::Tokens,
                    ty: kv_ty,
                },
            ],
        )
    } else {
        let v = push(&mut g, inp(2), vec![], kv_ty.clone());
        (
            v,
            3,
            4,
            vec![
                Port {
                    role: Role::Tokens,
                    ty: q_ty.clone(),
                },
                Port {
                    role: Role::Tokens,
                    ty: kv_ty.clone(),
                },
                Port {
                    role: Role::Tokens,
                    ty: kv_ty,
                },
            ],
        )
    };
    ports.push(Port {
        role: Role::Slots,
        ty: ids_ty.clone(),
    });
    ports.push(Port {
        role: Role::Positions,
        ty: ids_ty.clone(),
    });
    let slots = push(&mut g, inp(s_i), vec![], ids_ty.clone());
    let pos = push(&mut g, inp(p_i), vec![], ids_ty);
    g.kept.push(kept_ty.clone());
    let k = push(
        &mut g,
        Op::Append(0),
        vec![k_in, slots, pos],
        kept_ty.clone(),
    );
    let v = if mla {
        k
    } else {
        g.kept.push(kept_ty.clone());
        push(&mut g, Op::Append(1), vec![v_in, slots, pos], kept_ty)
    };
    let op = Op::Attend(Attend {
        mask: Mask::Back(511),
        scale: 0.125,
        picked: false,
        ragged: true,
        sinks: false,
    });
    let out = push(&mut g, op, vec![q, k, v, slots, pos], q_ty);
    g.entries.push(Entry {
        kind: EntryKind::Step,
        inputs: ports,
        outputs: vec![Output {
            role: Role::Logits,
            node: out,
        }],
    });
    g
}

fn ring_tree<'a>(g: &'a Graph, dev: &Device, rows: u64) -> Tree<'a> {
    let mut tree = Tree::upto(
        g,
        0,
        Wants::ALL,
        &[(Rows, rows), (Seqs, 2), (Context, 4096)],
        &[],
        &dev.pins,
        dev.kept,
        &dev.division(),
        0,
    )
    .unwrap();
    tree.knobs = dev.knobs.clone();
    tree
}

fn check_ring(g: &Graph, mla: bool, want: &str) {
    let mut offers = Vec::new();
    for kv_form in [FP8_FORM, BF16_FORM, F32_FORM] {
        let dev = Device::default().pinned(Pins::default().kept(kv_form));
        let tree = ring_tree(g, &dev, 4);
        let (_, layout, ring) = tree.kept_layout(0).unwrap();
        assert_eq!(layout.name(), "ring");
        assert!(ring < 4096);
        let got = if mla {
            check_mla(&tree, tree.outputs[0].0)
        } else {
            check_gqa(&tree, tree.outputs[0].0)
        };
        let mut drafts = got.unwrap().unwrap();
        assert_eq!(drafts[0].kernel.name(), want);
        offers.push(drafts.remove(0));
    }
    let gpu = || {
        Device::default().on(tessera_lower::Platform {
            name: "NVIDIA GB10".into(),
            backend: "cuda".into(),
            units: 48,
            bandwidth: 273e9,
            limits: tessera_lower::Limits::assumed(),
            ..Default::default()
        })
    };
    let (a, b) = (&offers[0], &offers[2]);
    let (x, y) = (
        gpu().seconds(&*a.kernel, a.len),
        gpu().seconds(&*b.kernel, b.len),
    );
    let (ab, ba) = (gpu(), gpu());
    assert_ne!(x, y);
    assert_eq!(
        (
            ab.seconds(&*a.kernel, a.len),
            ab.seconds(&*b.kernel, b.len),
            ba.seconds(&*b.kernel, b.len),
            ba.seconds(&*a.kernel, a.len)
        ),
        (x, y, y, x),
    );
}

#[test]
fn ring_layout_shift() {
    assert_eq!(kept_stride(kept::DENSE, 512), Some((512, kept::DENSE)));
    assert_eq!(kept_stride(kept::RING, 512), Some((512, kept::RING)));
    let p16: &dyn Layout = &kept::PAGED16;
    assert_eq!(kept_stride(p16, 512), Some((32, p16)));
    assert_eq!(page_shift(kept::DENSE), 0);
    assert_eq!(page_shift(kept::RING), RING_SHIFT);
    assert_eq!(page_shift(p16), 4);
}

#[test]
fn ring_attend_matches() {
    let mla = ring_graph(32, 1, MLA_WIDTH as u64, true);
    check_ring(&mla, true, "fast_mla_attend");
    let gqa = ring_graph(16, 4, GQA_WIDTH as u64, false);
    check_ring(&gqa, false, "fast_gqa_attend");
}

#[test]
fn mla_attend_knobs() {
    let g = ring_graph(32, 1, MLA_WIDTH as u64, true);
    let base_dev = Device::default().pinned(Pins::default().kept(BF16_FORM));
    let mut tuned_dev = base_dev.clone();
    tuned_dev.knobs.parse("fast_mla_attend:splits=4").unwrap();
    tuned_dev
        .knobs
        .parse("fast_mla_attend:min_blocks=4")
        .unwrap();
    tuned_dev
        .knobs
        .parse("fast_mla_attend:reduce_threads=128")
        .unwrap();
    let (t_base, t_tuned) = (ring_tree(&g, &base_dev, 1), ring_tree(&g, &tuned_dev, 1));
    let d_base = check_mla(&t_base, t_base.outputs[0].0).unwrap().unwrap();
    let d_tuned = check_mla(&t_tuned, t_tuned.outputs[0].0).unwrap().unwrap();
    assert_eq!(d_base.len(), 2);
    assert_eq!(d_tuned.len(), 2);
    assert_ne!(d_base[0].kernel.identity(), d_tuned[0].kernel.identity());
    let Source::Fixed { text, .. } = d_tuned[0].kernel.source() else {
        panic!("expected fixed source");
    };
    assert!(text[tessera_lower::Backend::Cuda].contains("#define MLA_MIN_BLOCKS 4"));
    assert_eq!(d_tuned[1].kernel.needs().group_threads, 128);
}

#[test]
fn kernel_flattens() {
    let options = tessera_device::Options::plain(tessera_device::Backend::Cuda);
    let flat = tessera_device::flatten("mla.cu", KERNEL, &|_| None, &options).unwrap();
    for entry in [
        "GQA_DEF(tessera_gqa_attend_fp8,",
        "GQA_DEF(tessera_gqa_attend_bf16,",
        "GQA_DEF(tessera_gqa_attend_f32,",
        "tessera_gqa_reduce(",
        "tessera_indexer_fp8",
        "tessera_indexer_bf16",
        "tessera_indexer_f32",
    ] {
        assert!(flat.text.contains(entry), "missing {entry}");
    }
}
