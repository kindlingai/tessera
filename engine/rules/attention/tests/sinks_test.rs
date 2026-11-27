use tessera_attention as _;
use tessera_graph::{DType, Dim, EntryKind, Graph, Mask, Port, Role, Size, TensorType};
use tessera_interp::Tensor;
use tessera_lower::{Device, Settings, lower};
use tessera_run::open;
use tessera_run::testing::host;
use tessera_trace::DimName::{self, Context, HeadDim, Heads, KvHeads, Rows, Seqs};
use tessera_trace::Model;

fn fill(n: usize, mul: usize, off: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * mul + 13) % 29) as f32 - 14.0) * 0.08 + off)
        .collect()
}

fn port(dims: impl Into<tessera_graph::Shape>, dtype: DType) -> Port {
    Port {
        role: Role::Tokens,
        ty: TensorType::new(dims, dtype),
    }
}

fn fixed(name: DimName, n: usize) -> Dim {
    Dim {
        name,
        size: Size::Fixed(n as u64),
    }
}

fn check_parity(g: &Graph, inputs: &[Tensor], tag: &str) -> Vec<f32> {
    let want = tessera_interp::run(g, &[], 0, inputs).unwrap();
    let want_f = want[0].floats().unwrap().to_vec();
    let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
    let device = Device::guess(&host());
    let settings = Settings::default();
    for force_passes in [false, true] {
        let dev = if force_passes {
            device.clone().without("listed_fused")
        } else {
            device.clone()
        };
        let plan = lower(g, 0, &[], &dev).unwrap();
        let mut runner = open(plan, &settings).unwrap();
        let got = runner.run(&views).unwrap().remove(0);
        for (a, b) in got.floats().unwrap().iter().zip(&want_f) {
            assert!(
                (a - b).abs() < 1e-5,
                "{tag} passes={force_passes}: {a} vs {b}"
            );
        }
    }
    want_f
}

#[test]
fn attend_sinks_parity() {
    let (seqs, ctx, heads, kv_heads, dim) = (2, 6, 4, 2, 8);
    let scale = (dim as f32).sqrt().recip();
    let base_dims = vec![
        fixed(Seqs, seqs),
        fixed(Context, ctx),
        fixed(Heads, heads),
        fixed(KvHeads, kv_heads),
        fixed(HeadDim, dim),
    ];
    let kv_p = port([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let s_p = port([Heads], DType::F32);
    let kt = Tensor::f32(
        vec![seqs, ctx, kv_heads, dim],
        fill(seqs * ctx * kv_heads * dim, 37, 0.1),
    );
    let vt = Tensor::f32(
        vec![seqs, ctx, kv_heads, dim],
        fill(seqs * ctx * kv_heads * dim, 37, -0.05),
    );

    // Whole-sequence causal and sliding-window attention with sinks.
    let q_whole = port([Seqs, Context, Heads, HeadDim], DType::F32);
    let qt_whole = Tensor::f32(
        vec![seqs, ctx, heads, dim],
        fill(seqs * ctx * heads * dim, 37, 0.0),
    );
    let st_whole = Tensor::f32(vec![heads], vec![-2.0, 0.5, 3.0, f32::NEG_INFINITY]);
    for mask in [Mask::Causal, Mask::Back(2)] {
        let m = Model::new(base_dims.clone());
        let [q, k, v, sinks] = m.entry(
            EntryKind::Forward,
            [q_whole.clone(), kv_p.clone(), kv_p.clone(), s_p.clone()],
        );
        let out = m.attend_sinks(mask, scale, [&q, &k, &v], None, &sinks);
        let g = m.finish([(Role::Tokens, &out)]);
        let want_f = check_parity(
            &g,
            &[qt_whole.clone(), kt.clone(), vt.clone(), st_whole.clone()],
            &format!("whole {mask:?}"),
        );

        // Sink = -inf on head 3 matches no-sink attention.
        let mp = Model::new(base_dims.clone());
        let [pq, pk, pv] = mp.entry(
            EntryKind::Forward,
            [q_whole.clone(), kv_p.clone(), kv_p.clone()],
        );
        let pout = mp.attend(mask, scale, [&pq, &pk, &pv], None);
        let gp = mp.finish([(Role::Tokens, &pout)]);
        let plain =
            tessera_interp::run(&gp, &[], 0, &[qt_whole.clone(), kt.clone(), vt.clone()]).unwrap();
        let plain_f = plain[0].floats().unwrap();
        for s in 0..seqs {
            for t in 0..ctx {
                let h3 = ((s * ctx + t) * heads + 3) * dim;
                for d in 0..dim {
                    assert!((want_f[h3 + d] - plain_f[h3 + d]).abs() < 1e-6);
                }
            }
        }
    }

    // Ragged attention with sinks, both dense and picked.
    let rows = 3;
    let pick_dim = DimName::named("SinkPicks");
    let q_rag = port([Rows, Heads, HeadDim], DType::F32);
    let u_p = port([Rows], DType::U32);
    let qt_rag = Tensor::f32(vec![rows, heads, dim], fill(rows * heads * dim, 23, 0.0));
    let slt = Tensor::u32(vec![rows], vec![0, 1, 0]);
    let pot = Tensor::u32(vec![rows], vec![0, 4, 5]);
    let st_rag = Tensor::f32(vec![heads], vec![-1.0, 2.5, 0.0, f32::NEG_INFINITY]);
    for picked in [false, true] {
        let mut dims = base_dims.clone();
        dims.push(fixed(Rows, rows));
        if picked {
            dims.push(fixed(pick_dim, 3));
        }
        let m = Model::new(dims);
        let g = if picked {
            let id_p = port([Rows, pick_dim], DType::U32);
            let [q, k, v, sl, po, ids, si] = m.entry(
                EntryKind::Forward,
                [
                    q_rag.clone(),
                    kv_p.clone(),
                    kv_p.clone(),
                    u_p.clone(),
                    u_p.clone(),
                    id_p,
                    s_p.clone(),
                ],
            );
            let out = m.attend_rows_sinks(
                Mask::Back(3),
                scale,
                [&q, &k, &v],
                [&sl, &po],
                Some(&ids),
                &si,
            );
            m.finish([(Role::Tokens, &out)])
        } else {
            let [q, k, v, sl, po, si] = m.entry(
                EntryKind::Forward,
                [
                    q_rag.clone(),
                    kv_p.clone(),
                    kv_p.clone(),
                    u_p.clone(),
                    u_p.clone(),
                    s_p.clone(),
                ],
            );
            let out =
                m.attend_rows_sinks(Mask::Back(3), scale, [&q, &k, &v], [&sl, &po], None, &si);
            m.finish([(Role::Tokens, &out)])
        };
        let mut inputs = vec![
            qt_rag.clone(),
            kt.clone(),
            vt.clone(),
            slt.clone(),
            pot.clone(),
        ];
        if picked {
            inputs.push(Tensor::u32(vec![rows, 3], vec![0, 9, 9, 1, 3, 4, 2, 4, 5]));
        }
        inputs.push(st_rag.clone());
        check_parity(&g, &inputs, &format!("ragged picked={picked}"));
    }
}

/// Fills sink logits, with one `-inf` to skip the sink.
fn token_sinks(shape: Vec<usize>) -> Tensor {
    let n = shape.iter().product();
    let mut vals = fill(n, 11, 0.4);
    vals[n - 1] = f32::NEG_INFINITY;
    Tensor::f32(shape, vals)
}

#[test]
fn token_sinks_parity() {
    let (seqs, ctx, heads, kv_heads, dim, win) = (2, 6, 4, 2, 8, 3);
    let scale = (dim as f32).sqrt().recip();
    let rel_win = DimName::named("RelWin");
    let base_dims = vec![
        fixed(Seqs, seqs),
        fixed(Context, ctx),
        fixed(Heads, heads),
        fixed(KvHeads, kv_heads),
        fixed(HeadDim, dim),
        fixed(rel_win, win),
    ];
    let kv_p = port([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let kv_n = seqs * ctx * kv_heads * dim;
    let kt = Tensor::f32(vec![seqs, ctx, kv_heads, dim], fill(kv_n, 37, 0.1));
    let vt = Tensor::f32(vec![seqs, ctx, kv_heads, dim], fill(kv_n, 37, -0.05));

    // Whole sequences: one sink and bias row per query head.
    let q_p = port([Seqs, Context, Heads, HeadDim], DType::F32);
    let s_p = port([Seqs, Context, Heads], DType::F32);
    let r_p = port([Seqs, Context, Heads, rel_win], DType::F32);
    let lists = seqs * ctx * heads;
    let qt = Tensor::f32(vec![seqs, ctx, heads, dim], fill(lists * dim, 37, 0.0));
    let st = token_sinks(vec![seqs, ctx, heads]);
    let rt = Tensor::f32(vec![seqs, ctx, heads, win], fill(lists * win, 17, 0.3));
    for mask in [Mask::Causal, Mask::Back(2)] {
        for (with_sinks, with_rel) in [(true, true), (false, true), (true, false)] {
            let m = Model::new(base_dims.clone());
            let ports = [
                q_p.clone(),
                kv_p.clone(),
                kv_p.clone(),
                s_p.clone(),
                r_p.clone(),
            ];
            let [q, k, v, sinks, rel] = m.entry(EntryKind::Forward, ports);
            let extra = [with_sinks.then_some(&sinks), with_rel.then_some(&rel)];
            let out = m.attend_biased(mask, scale, [&q, &k, &v], None, extra);
            let g = m.finish([(Role::Tokens, &out)]);
            let inputs = [qt.clone(), kt.clone(), vt.clone(), st.clone(), rt.clone()];
            let tag = format!("whole {mask:?} sinks={with_sinks} rel={with_rel}");
            check_parity(&g, &inputs, &tag);
        }
    }

    // Uniform token sinks and a zero bias match per-head sinks.
    let head_sinks = [-2.0, 0.5, 3.0, f32::NEG_INFINITY];
    let m = Model::new(base_dims.clone());
    let ports = [q_p.clone(), kv_p.clone(), kv_p.clone(), s_p, r_p];
    let [q, k, v, sinks, rel] = m.entry(EntryKind::Forward, ports);
    let out = m.attend_biased(
        Mask::Causal,
        scale,
        [&q, &k, &v],
        None,
        [Some(&sinks), Some(&rel)],
    );
    let g = m.finish([(Role::Tokens, &out)]);
    let spread = Tensor::f32(vec![seqs, ctx, heads], head_sinks.repeat(seqs * ctx));
    let zeros = Tensor::f32(vec![seqs, ctx, heads, win], vec![0.0; lists * win]);
    let inputs = [qt.clone(), kt.clone(), vt.clone(), spread, zeros];
    let got = check_parity(&g, &inputs, "uniform");
    let m = Model::new(base_dims.clone());
    let h_p = port([Heads], DType::F32);
    let [q, k, v, sinks] = m.entry(EntryKind::Forward, [q_p, kv_p.clone(), kv_p.clone(), h_p]);
    let out = m.attend_sinks(Mask::Causal, scale, [&q, &k, &v], None, &sinks);
    let g = m.finish([(Role::Tokens, &out)]);
    let per_head = Tensor::f32(vec![heads], head_sinks.to_vec());
    let inputs = [qt, kt.clone(), vt.clone(), per_head];
    let want = tessera_interp::run(&g, &[], 0, &inputs).unwrap();
    for (a, b) in got.iter().zip(want[0].floats().unwrap()) {
        assert!((a - b).abs() < 1e-6, "uniform sinks: {a} vs {b}");
    }

    // Ragged rows, dense and picked.
    let rows = 3;
    let pick_dim = DimName::named("SinkPicks");
    let u_p = port([Rows], DType::U32);
    let qt = Tensor::f32(vec![rows, heads, dim], fill(rows * heads * dim, 23, 0.0));
    let slt = Tensor::u32(vec![rows], vec![0, 1, 0]);
    let pot = Tensor::u32(vec![rows], vec![0, 4, 5]);
    let st = token_sinks(vec![rows, heads]);
    let rt = Tensor::f32(vec![rows, heads, win], fill(rows * heads * win, 17, 0.3));
    let idt = Tensor::u32(vec![rows, 3], vec![0, 9, 9, 1, 3, 4, 2, 4, 5]);
    for (mask, picked) in [
        (Mask::Causal, false),
        (Mask::Back(3), false),
        (Mask::Causal, true),
        (Mask::Back(3), true),
    ] {
        let mut dims = base_dims.clone();
        dims.push(fixed(Rows, rows));
        dims.push(fixed(pick_dim, 3));
        let m = Model::new(dims);
        let ports = [
            port([Rows, Heads, HeadDim], DType::F32),
            kv_p.clone(),
            kv_p.clone(),
            u_p.clone(),
            u_p.clone(),
            port([Rows, pick_dim], DType::U32),
            port([Rows, Heads], DType::F32),
            port([Rows, Heads, rel_win], DType::F32),
        ];
        let [q, k, v, sl, po, ids, si, rel] = m.entry(EntryKind::Forward, ports);
        let out = m.attend_rows_biased(
            mask,
            scale,
            [&q, &k, &v],
            [&sl, &po],
            picked.then_some(&ids),
            [Some(&si), Some(&rel)],
        );
        let g = m.finish([(Role::Tokens, &out)]);
        let inputs = [
            qt.clone(),
            kt.clone(),
            vt.clone(),
            slt.clone(),
            pot.clone(),
            idt.clone(),
            st.clone(),
            rt.clone(),
        ];
        check_parity(&g, &inputs, &format!("ragged {mask:?} picked={picked}"));
    }
}
