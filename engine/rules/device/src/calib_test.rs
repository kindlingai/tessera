use crate::calib::{
    TILE, cuda_source, propose_picked_bf16, round_bf16, sort_cpu, syrk_dense_cpu, syrk_tiles_cpu,
};
use tessera_graph::DimName::{Experts, Ffn, Hidden, Rows, TopK};
use tessera_graph::{
    Act, DType, Dim, Entry, EntryKind, Graph, Node, NodeId, Op, Output, PathSeg, Port, Role, Row,
    Select, Stage, TensorType, Weight,
};
use tessera_lower::form::{BF16_FORM, F32_FORM};
use tessera_lower::{Device, Pins, Source, Tree, Wants};
use tessera_testkit::Noise;

fn triu_tiles(dim: usize) -> usize {
    assert!(dim > 0 && dim.is_multiple_of(TILE));
    let blocks = dim / TILE;
    blocks * (blocks + 1) / 2
}

#[test]
fn calib_cuda_entries() {
    let src = cuda_source();
    for entry in [
        "calib_sort_hist(",
        "calib_sort_prefix(",
        "calib_sort_scatter(",
        "calib_syrk_bf16(",
        "calib_syrk_f32(",
        "pbf16_gemm_bf16_bf16",
        "pbf16_gemm_f32_f32",
        "pbf16_gu_bf16_bf16",
        "pbf16_gu_f32_f32",
    ] {
        assert!(src.contains(entry), "missing {entry}");
    }
    assert_eq!(triu_tiles(6144), 1176);
    assert_eq!(triu_tiles(2048), 136);
}

#[test]
fn calib_syrk_accuracy() {
    let mut rng = Noise(42);
    let (rows, top_k, experts, dim, tile) = (12usize, 2usize, 4usize, 8usize, 4usize);
    let x: Vec<f32> = (0..rows * dim).map(|_| round_bf16(rng.draw())).collect();
    let ids: Vec<u32> = (0..rows * top_k)
        .map(|i| ((i * 3 + 1) % experts) as u32)
        .collect();
    let (offsets, perm, counts) = sort_cpu(&ids, experts, 0);
    assert_eq!(counts.iter().sum::<i64>() as usize, rows * top_k);

    let tiles = (dim / tile) * (dim / tile + 1) / 2;
    let mut acc1 = vec![0.0f64; experts * tiles * tile * tile];
    let mut acc2 = vec![0.0f64; experts * tiles * tile * tile];
    syrk_tiles_cpu(&x, &perm, &offsets, dim, tile, top_k, &mut acc1);
    syrk_tiles_cpu(&x, &perm, &offsets, dim, tile, top_k, &mut acc2);
    assert_eq!(acc1, acc2);

    // Check packed tiles against float64 reference.
    for e in 0..experts {
        let mut ref64 = vec![0.0f64; dim * dim];
        for (pick, &eid) in ids.iter().enumerate() {
            if eid as usize != e {
                continue;
            }
            let r = pick / top_k;
            for i in 0..dim {
                for j in 0..dim {
                    ref64[i * dim + j] += f64::from(x[r * dim + i]) * f64::from(x[r * dim + j]);
                }
            }
        }
        let mut tidx = 0usize;
        for br in 0..(dim / tile) {
            for bc in br..(dim / tile) {
                let base = (e * tiles + tidx) * tile * tile;
                for lr in 0..tile {
                    for lc in 0..tile {
                        let got = acc1[base + lr * tile + lc];
                        let want = ref64[(br * tile + lr) * dim + (bc * tile + lc)];
                        assert!(
                            (got - want).abs() < 1e-4,
                            "e={e} ({br},{bc}) {got} vs {want}"
                        );
                    }
                }
                tidx += 1;
            }
        }
    }

    let mut dense = vec![0.0f64; dim * dim];
    syrk_dense_cpu(&x, rows, dim, &mut dense);
    for i in 0..dim {
        for j in 0..dim {
            assert_eq!(dense[i * dim + j], dense[j * dim + i]);
        }
    }
}

fn push(g: &mut Graph, op: Op, args: Vec<NodeId>, ty: TensorType) -> NodeId {
    let id = NodeId(g.nodes.len() as u32);
    g.nodes.push(Node {
        op,
        args,
        ty,
        numerics: None,
    });
    id
}

fn moe_graph(fused: bool) -> Graph {
    let mut g = Graph::new(vec![
        Dim::bounded(Rows, 64),
        Dim::fixed(Experts, 256),
        Dim::fixed(TopK, 8),
        Dim::fixed(Hidden, 6144),
        Dim::fixed(Ffn, 2048),
    ]);
    let w_ty = TensorType::new([Experts, Ffn, Hidden], DType::F32);
    g.weights.push(Weight {
        path: vec![PathSeg::Field("gate".into())],
        ty: w_ty.clone(),
    });
    if fused {
        g.weights.push(Weight {
            path: vec![PathSeg::Field("up".into())],
            ty: w_ty.clone(),
        });
    }
    let x_ty = TensorType::new([Rows, Hidden], DType::F32);
    let ids_ty = TensorType::new([Rows, TopK], DType::U32);
    let pick_ty = TensorType::new([Rows, TopK, Ffn, Hidden], DType::F32);
    let out_ty = TensorType::new([Rows, TopK, Ffn], DType::F32);
    let x_in = push(
        &mut g,
        Op::Input { entry: 0, port: 0 },
        vec![],
        x_ty.clone(),
    );
    let x = push(
        &mut g,
        Op::Row(Row(vec![Stage::Scale(1.0)])),
        vec![x_in],
        x_ty.clone(),
    );
    let ids = push(
        &mut g,
        Op::Input { entry: 0, port: 1 },
        vec![],
        ids_ty.clone(),
    );
    let wg = push(&mut g, Op::Weight(0), vec![], w_ty.clone());
    let pg = push(
        &mut g,
        Op::Select(Select::Gather),
        vec![wg, ids],
        pick_ty.clone(),
    );
    let gate = push(&mut g, Op::Matmul, vec![x, pg], out_ty.clone());
    let out = if fused {
        let wu = push(&mut g, Op::Weight(1), vec![], w_ty);
        let pu = push(&mut g, Op::Select(Select::Gather), vec![wu, ids], pick_ty);
        let up = push(&mut g, Op::Matmul, vec![x, pu], out_ty.clone());
        let silu = push(
            &mut g,
            Op::Row(Row(vec![Stage::Act(Act::Silu)])),
            vec![gate],
            out_ty.clone(),
        );
        push(
            &mut g,
            Op::Row(Row(vec![Stage::Mul])),
            vec![silu, up],
            out_ty,
        )
    } else {
        gate
    };
    g.entries.push(Entry {
        kind: EntryKind::Forward,
        inputs: vec![
            Port {
                role: Role::Tokens,
                ty: x_ty,
            },
            Port {
                role: Role::Picks,
                ty: ids_ty,
            },
        ],
        outputs: vec![Output {
            role: Role::Logits,
            node: out,
        }],
    });
    g
}

#[test]
fn picked_bf16_selection() {
    for (fused, act_form, want_entry) in [
        (false, BF16_FORM, "pbf16_gemm_bf16_bf16"),
        (false, F32_FORM, "pbf16_gemm_f32_f32"),
        (true, BF16_FORM, "pbf16_gu_bf16_bf16"),
        (true, F32_FORM, "pbf16_gu_f32_f32"),
    ] {
        let g = moe_graph(fused);
        let pins = Pins::all(BF16_FORM).activations(act_form).outputs(act_form);
        let dev = Device::default().pinned(pins);
        let parts = dev.division();
        let tree = Tree::upto(
            &g,
            0,
            Wants::ALL,
            &[(Rows, 16)],
            &[],
            &dev.pins,
            dev.kept,
            &parts,
            0,
        )
        .expect("tree builds");
        let drafts = propose_picked_bf16(&tree, tree.outputs[0].0).expect("matches picked_bf16");
        assert_eq!(drafts.len(), 4);
        let Source::Fixed { entry, .. } = drafts[3].kernel.source() else {
            panic!("fixed cuda source");
        };
        assert_eq!(entry, want_entry);
    }
}
