//! Layer-streaming calibration loop using `cut_region` and runner accumulators.

use crate::accum::{DenseAccum, ExpertAccum, RouteLog, SavedInputs, tap_name};
use crate::args::{Config, read_npy};
use crate::state::{AsyncWriter, CalibState, HiddenReader, HiddenWriter, hidden_path};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tessera_bundle::mla::rules as mla_rules;
use tessera_checkpoint::hessian::{HessianMeta, valid_hex, weights_id};
use tessera_checkpoint::{Offer, Stored, open as open_ckpt};
use tessera_device::calib::{cuda_source, round_bf16};
use tessera_graph::{Bf16, CalibTap, Composite, DimName, Graph, NodeId, Op, Role};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM};
use tessera_lower::{Pins, Site, Weight, cut, cut_region};
use tessera_models::mla;

/// Validated metadata, token batches, and resume state for one run.
pub(crate) struct RunPlan {
    pub n_rows: usize,
    pub seq_len: usize,
    pub cal_rows: usize,
    pub tok_data: Vec<u32>,
    pub meta: HessianMeta,
    pub state_path: PathBuf,
    pub state: CalibState,
    pub todo: Vec<usize>,
    pub batches: Vec<(usize, usize)>,
    pub graph: Graph,
}

fn resolve_meta(cfg: &Config) -> Result<HessianMeta, String> {
    let wid = match &cfg.weights_id {
        Some(w) if valid_hex(w) => w.clone(),
        Some(w) => return Err(format!("invalid --weights-id `{w}`")),
        None => weights_id(&cfg.model).map_err(|e| e.to_string())?,
    };
    let meta = HessianMeta {
        hessian_id: cfg.hess_id.clone(),
        weights_id: wid,
    };
    meta.validate().map_err(|e| e.to_string())?;
    Ok(meta)
}

fn batches_split(n: usize, cal: usize, batch: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (lo, hi) in [(0, cal), (cal, n)] {
        let mut s = lo;
        while s < hi {
            let e = (s + batch).min(hi);
            out.push((s, e));
            s = e;
        }
    }
    out
}

/// Pins activations and outputs to BF16, keeping router F32.
pub fn calib_pins(base: Pins) -> Pins {
    base.activations(BF16_FORM)
        .outputs(BF16_FORM)
        .site(Site::Router, F32_FORM)
}

/// Runtime dimension bindings for one calibration batch `[batch, seq_len]`.
pub fn layer_sizes(batch: usize, seq_len: usize) -> [(DimName, u64); 4] {
    [
        (DimName::Seqs, batch as u64),
        (DimName::Context, seq_len as u64),
        (DimName::Rows, (batch * seq_len) as u64),
        (DimName::Segments, 1),
    ]
}

/// Validates config, tokens, resume state, and builds the traced graph.
pub(crate) fn prepare_run(cfg: &Config, mla_cfg: &mla::Config) -> Result<RunPlan, String> {
    let _ = cuda_source();
    let (n_rows, seq_len, tok_data) = read_npy(&cfg.tokens)?;
    if let Some(ix) = &mla_cfg.index
        && seq_len as u64 > ix.keep
    {
        return Err(format!(
            "sequence length {seq_len} exceeds index_topk {}",
            ix.keep
        ));
    }
    if cfg.heldout > n_rows {
        return Err("heldout exceeds total rows".into());
    }
    let cal_rows = n_rows - cfg.heldout;
    let meta = resolve_meta(cfg)?;
    let state_path = cfg
        .state
        .clone()
        .unwrap_or_else(|| cfg.out.join("calib-state.json"));
    let mut state = CalibState::load(&state_path, cfg.resume)?;
    state.sync_meta(&meta, &state_path)?;
    let todo: Vec<usize> = cfg
        .layers
        .iter()
        .copied()
        .filter(|l| !state.done.contains(l))
        .collect();
    for w in todo.windows(2) {
        if w[1] != w[0] + 1 {
            return Err("layers must be a contiguous range".into());
        }
    }
    fs::create_dir_all(&cfg.work).map_err(|e| e.to_string())?;
    fs::create_dir_all(&cfg.out).map_err(|e| e.to_string())?;

    let mut trace_cfg = mla_cfg.clone();
    trace_cfg.index = None;
    let graph = mla::calib_program::<mla::Moe>(&trace_cfg);
    let batches = batches_split(n_rows, cal_rows, cfg.batch);
    Ok(RunPlan {
        n_rows,
        seq_len,
        cal_rows,
        tok_data,
        meta,
        state_path,
        state,
        todo,
        batches,
        graph,
    })
}

/// Writes `h-000.bin` directly from checkpoint embedding weights.
pub(crate) fn embed_offer(
    offer: &dyn Offer,
    tok_data: &[u32],
    dims: (usize, usize, usize),
    batches: &[(usize, usize)],
    path: PathBuf,
) -> Result<(), String> {
    let (n_rows, seq_len, hidden) = dims;
    let slice = offer
        .slices()
        .into_iter()
        .find(|s| s.name == "model.embed_tokens.weight")
        .ok_or("missing model.embed_tokens.weight")?;
    let table_raw: Vec<u8> = if slice.stored == Stored::BF16 {
        offer.packed(&slice.name).map_err(|e| e.to_string())?
    } else {
        let vals = offer.floats(&slice.name).map_err(|e| e.to_string())?;
        let mut raw = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            raw.extend_from_slice(&Bf16::from(v).0.to_le_bytes());
        }
        raw
    };
    let row_bytes = hidden * 2;
    let writer = HiddenWriter::spawn(path, n_rows, seq_len, hidden)?;
    for &(s0, s1) in batches {
        let count = s1 - s0;
        let toks = &tok_data[s0 * seq_len..s1 * seq_len];
        let mut raw = Vec::with_capacity(toks.len() * row_bytes);
        for &id in toks {
            let off = id as usize * row_bytes;
            raw.extend_from_slice(&table_raw[off..off + row_bytes]);
        }
        writer.send_raw(s0, count, raw)?;
    }
    writer.finish()
}

fn sub_weights(full: &Graph, sub: &Graph, all: &[Tensor]) -> Vec<Tensor> {
    let by_path: HashMap<String, usize> = full
        .weights
        .iter()
        .enumerate()
        .map(|(i, w)| (format!("{:?}", w.path), i))
        .collect();
    sub.weights
        .iter()
        .map(|w| all[by_path[&format!("{:?}", w.path)]].clone())
        .collect()
}

fn layer_from_offer(
    lg: &Graph,
    offer: &dyn Offer,
    rules: &[tessera_bundle::names::Rule],
) -> Result<Vec<Tensor>, String> {
    let mut out = Vec::with_capacity(lg.weights.len());
    tessera_bundle::each(lg, offer, &[rules], &|_| None, |w, _| {
        let t = match w {
            Weight::Floats(t) => t,
            Weight::Packed(form, bytes) => {
                let vals = form.unpack_all(&bytes);
                Tensor::f32(vec![vals.len()], vals)
            }
            Weight::Filed(form, file) => {
                let mut bytes = vec![0u8; file.bytes()];
                file.read(&mut bytes, Default::default())?;
                let vals = form.unpack_all(&bytes);
                Tensor::f32(vec![vals.len()], vals)
            }
        };
        out.push(t);
        Ok(())
    })
    .map_err(|e| format!("layer weights: {e}"))?;
    for (t, w) in out.iter_mut().zip(&lg.weights) {
        let shape: Option<Vec<usize>> =
            w.ty.shape
                .0
                .iter()
                .map(|e| e.value(&lg.dims, &|_| None).map(|n| n as usize))
                .collect();
        if let Some(s) = shape
            && t.shape != s
        {
            *t = t.reshaped(s);
        }
    }
    Ok(out)
}

fn decode_le4<T>(t: &Tensor, f: fn([u8; 4]) -> T) -> Vec<T> {
    t.bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f)
        .collect()
}

/// Flushes captures, saves resume state, and removes old hidden file.
pub(crate) fn finish_layer(
    cfg: &Config,
    plan: &mut RunPlan,
    layer: usize,
    dims: (usize, usize),
    saved: Option<SavedInputs>,
    routes: Option<RouteLog>,
    writer: &AsyncWriter,
) -> Result<(), String> {
    let (hidden, top_k) = dims;
    if let Some(sv) = saved {
        sv.flush(
            &cfg.out,
            layer,
            cfg.heldout,
            (hidden, top_k),
            &plan.meta,
            writer,
        )?;
    }
    if let Some(rt) = routes {
        rt.flush(
            &cfg.out,
            layer,
            [plan.n_rows, plan.seq_len, top_k],
            &plan.meta,
            writer,
        )?;
    }
    plan.state.done.push(layer);
    plan.state.save(&plan.state_path)?;
    if !cfg.keep_hidden {
        let _ = fs::remove_file(hidden_path(&cfg.work, layer));
    }
    Ok(())
}

fn run_cpu_layers(
    cfg: &Config,
    mla_cfg: &mla::Config,
    mut plan: RunPlan,
    mut load_layer: impl FnMut(&Graph, &Graph) -> Result<Vec<Tensor>, String>,
) -> Result<(), String> {
    let hidden = mla_cfg.stack.hidden as usize;
    let ffn = mla_cfg.stack.ffn as usize;
    let experts = mla_cfg.experts as usize;
    let top_k = mla_cfg.top_k as usize;
    let regions: Vec<_> = plan
        .graph
        .regions
        .iter()
        .filter(|r| r.kind == Composite::Layer)
        .cloned()
        .collect();
    let writer = AsyncWriter::new(4);
    let todo = plan.todo.clone();

    for layer in todo {
        let region = regions
            .get(layer)
            .ok_or_else(|| format!("layer {layer} out of range"))?;
        let lg = cut_region(&plan.graph, region);
        let lw = load_layer(&plan.graph, &lg)?;
        let sparse = layer >= mla_cfg.dense_layers as usize;
        let do_hess = cfg.hess_layers.contains(&layer);
        let mut exp_acc = (sparse && do_hess && !cfg.no_experts)
            .then(|| ExpertAccum::new(experts, hidden, ffn, cfg.tile));
        let mut dense_acc = (do_hess && !cfg.no_dense).then(DenseAccum::default);
        let mut held_dense = (do_hess && cfg.dense_heldout).then(DenseAccum::default);
        let mut saved = (sparse && cfg.save_inputs.contains(&layer)).then(SavedInputs::default);
        let mut routes = (sparse && cfg.route_log.contains(&layer)).then(RouteLog::default);

        let reader = HiddenReader::spawn(
            hidden_path(&cfg.work, layer),
            plan.seq_len,
            hidden,
            plan.batches.clone(),
        );
        let h_out = HiddenWriter::spawn(
            hidden_path(&cfg.work, layer + 1),
            plan.n_rows,
            plan.seq_len,
            hidden,
        )?;

        for &(s0, s1) in &plan.batches {
            let b = s1 - s0;
            let is_cal = s1 <= plan.cal_rows;
            let is_held = s0 >= plan.cal_rows;
            let rows_in = reader.next_batch()?;
            let x_t = Tensor::f32(vec![b, plan.seq_len, hidden], rows_in);
            let pos_v: Vec<u32> = (0..b).flat_map(|_| 0..plan.seq_len as u32).collect();
            let pos_t = Tensor::u32(vec![b, plan.seq_len], pos_v);
            let feeds: Vec<Tensor> = lg.entries[0]
                .inputs
                .iter()
                .map(|port| match port.role {
                    Role::Positions => pos_t.clone(),
                    _ => x_t.clone(),
                })
                .collect();
            let outs = run(&lg, &lw, 0, &feeds).map_err(|e| format!("layer {layer}: {e}"))?;
            let out_vals: Vec<f32> = decode_le4(&outs[0], f32::from_le_bytes)
                .into_iter()
                .map(round_bf16)
                .collect();
            h_out.send_f32(s0, b, &out_vals)?;

            let mut mlp_in_v: Option<Vec<f32>> = None;
            let mut ids_v: Option<Vec<u32>> = None;
            let mut gates_v: Option<Vec<f32>> = None;
            let mut inner_v: Option<Vec<f32>> = None;
            let n_tok = b * plan.seq_len;

            for (o, t) in lg.entries[0].outputs[1..].iter().zip(&outs[1..]) {
                let Role::Calib(tap) = o.role else {
                    continue;
                };
                match tap {
                    CalibTap::RouterIds => ids_v = Some(decode_le4(t, u32::from_le_bytes)),
                    CalibTap::RouterGates => gates_v = Some(decode_le4(t, f32::from_le_bytes)),
                    CalibTap::RoutedInner => inner_v = Some(decode_le4(t, f32::from_le_bytes)),
                    other => {
                        let vals = decode_le4(t, f32::from_le_bytes);
                        let dim = vals.len() / n_tok;
                        if let Some(name) = tap_name(other) {
                            if is_cal && let Some(acc) = dense_acc.as_mut() {
                                acc.add(name, &vals, n_tok, dim);
                            }
                            if is_held && let Some(acc) = held_dense.as_mut() {
                                acc.add(name, &vals, n_tok, dim);
                            }
                        }
                        if other == CalibTap::MlpIn {
                            mlp_in_v = Some(vals);
                        }
                    }
                }
            }

            if is_cal
                && let (Some(acc), Some(x), Some(inner), Some(ids)) =
                    (exp_acc.as_mut(), &mlp_in_v, &inner_v, &ids_v)
            {
                acc.add(x, inner, ids, top_k);
            }
            if is_held
                && let (Some(sv), Some(x), Some(ids), Some(g)) =
                    (saved.as_mut(), &mlp_in_v, &ids_v, &gates_v)
            {
                sv.push(x, ids, g);
            }
            if let (Some(rt), Some(ids), Some(g)) = (routes.as_mut(), &ids_v, &gates_v) {
                rt.push(ids, g);
            }
        }
        reader.finish();
        h_out.finish()?;

        if let Some(acc) = exp_acc {
            acc.flush(&cfg.out, layer, cfg.experts_per_file, &plan.meta, &writer)?;
        }
        if let Some(acc) = dense_acc {
            acc.flush(&cfg.out, layer, false, &plan.meta, &writer)?;
        }
        if let Some(acc) = held_dense {
            acc.flush(&cfg.out, layer, true, &plan.meta, &writer)?;
        }
        finish_layer(
            cfg,
            &mut plan,
            layer,
            (hidden, top_k),
            saved,
            routes,
            &writer,
        )?;
    }
    writer.finish()
}

/// Runs layer-streaming MLA calibration over in-memory weights (CPU interpreter).
pub fn run_mla(cfg: &Config, mla_cfg: &mla::Config, all_weights: &[Tensor]) -> Result<(), String> {
    let plan = prepare_run(cfg, mla_cfg)?;
    if plan.todo.is_empty() {
        return Ok(());
    }
    let hidden = mla_cfg.stack.hidden as usize;
    if plan.todo[0] == 0 {
        let embed_id = plan
            .graph
            .nodes
            .iter()
            .enumerate()
            .find_map(|(i, n)| matches!(n.op, Op::Select(_)).then_some(NodeId(i as u32)))
            .ok_or("missing embed node")?;
        let emb_g = cut(&plan.graph, &[embed_id]);
        let h0 = HiddenWriter::spawn(hidden_path(&cfg.work, 0), plan.n_rows, plan.seq_len, hidden)?;
        for &(s0, s1) in &plan.batches {
            let b = s1 - s0;
            let slice = plan.tok_data[s0 * plan.seq_len..s1 * plan.seq_len].to_vec();
            let tok = Tensor::u32(vec![b, plan.seq_len], slice);
            let emb = run(&emb_g, &[], 0, &[all_weights[0].clone(), tok])
                .map_err(|e| format!("embed: {e}"))?;
            let vals: Vec<f32> = decode_le4(&emb[0], f32::from_le_bytes)
                .into_iter()
                .map(round_bf16)
                .collect();
            h0.send_f32(s0, b, &vals)?;
        }
        h0.finish()?;
    } else if !hidden_path(&cfg.work, plan.todo[0]).exists() {
        return Err(format!("missing hidden state for layer {}", plan.todo[0]));
    }
    run_cpu_layers(cfg, mla_cfg, plan, |full, sub| {
        Ok(sub_weights(full, sub, all_weights))
    })
}

/// Runs layer-streaming MLA calibration from a checkpoint directory on CPU.
pub fn run_dir(cfg: &Config, mla_cfg: &mla::Config) -> Result<(), String> {
    let plan = prepare_run(cfg, mla_cfg)?;
    if plan.todo.is_empty() {
        return Ok(());
    }
    let offer = open_ckpt(&cfg.model, None).map_err(|e| format!("checkpoint: {e}"))?;
    let rules = mla_rules(mla_cfg, false);
    let hidden = mla_cfg.stack.hidden as usize;
    if plan.todo[0] == 0 {
        embed_offer(
            offer.as_ref(),
            &plan.tok_data,
            (plan.n_rows, plan.seq_len, hidden),
            &plan.batches,
            hidden_path(&cfg.work, 0),
        )?;
    } else if !hidden_path(&cfg.work, plan.todo[0]).exists() {
        return Err(format!("missing hidden state for layer {}", plan.todo[0]));
    }
    run_cpu_layers(cfg, mla_cfg, plan, |_, sub| {
        layer_from_offer(sub, offer.as_ref(), &rules)
    })
}
