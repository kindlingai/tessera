//! Layer-streaming calibration loop over `cut_region` layer graphs.

use crate::accum::{DenseAccum, ExpertAccum, Files, RouteLog, SavedInputs, Taps};
use crate::args::{Config, Tokens, read_npy};
use crate::state::{
    AsyncWriter, CalibState, HiddenReader, HiddenWriter, Stamp, bf16_bytes, hidden_path,
};
use std::fs;
use std::path::PathBuf;
use tessera_bundle::mla::{config as mla_config, rules as mla_rules};
use tessera_checkpoint::hessian::{HessianMeta, valid_hex, weights_id};
use tessera_checkpoint::{Offer, open as open_ckpt};
use tessera_graph::{CalibTap, Composite, Graph, Op, Role};
use tessera_interp::{Tensor, run};
use tessera_lower::{Weight, cut_region};
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
    /// Hidden width, routed picks per row, and dense layer count.
    pub hidden: usize,
    pub top_k: usize,
    pub dense_layers: usize,
}

/// Which sums a batch feeds.
#[derive(Clone, Copy)]
pub(crate) struct Part {
    pub cal: bool,
    pub held: bool,
}

/// What one layer collects.
pub(crate) struct Wants {
    pub experts: bool,
    pub dense: bool,
    pub held: bool,
}

/// Host copies of a batch's MoE input, picks and gates.
pub(crate) struct Routing {
    pub x: Vec<f32>,
    pub ids: Vec<u32>,
    pub gates: Vec<f32>,
}

/// Runs one layer's batches and keeps its Gram sums.
pub(crate) trait Engine {
    /// Prepares layer graph `lg` and the sums `wants` asks for.
    fn open(&mut self, layer: usize, lg: &Graph, wants: &Wants) -> Result<(), String>;
    /// Runs one batch: BF16 output rows, and routing when `route`.
    fn batch(
        &mut self,
        x: Vec<f32>,
        seqs: usize,
        part: Part,
        route: bool,
    ) -> Result<(Vec<u8>, Option<Routing>), String>;
    /// Queues the layer's sums as files.
    fn close(&mut self, files: &Files) -> Result<(), String>;
}

/// Loads the weights of a graph cut from the model.
pub(crate) type Loader<'a> = dyn FnMut(&Graph) -> Result<Vec<Tensor>, String> + 'a;

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

/// Checks tokens, sizes and resume state, and traces the graph.
pub(crate) fn prepare_run(
    cfg: &Config,
    mla_cfg: &mla::Config,
    toks: Tokens,
) -> Result<RunPlan, String> {
    let (n_rows, seq_len) = (toks.rows, toks.seq_len);
    let vocab = mla_cfg.stack.vocab;
    if let Some(&id) = toks.ids.iter().find(|&&id| u64::from(id) >= vocab) {
        return Err(format!(
            "token id {id} is outside the vocabulary of {vocab}"
        ));
    }
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
    let hidden = mla_cfg.stack.hidden as usize;
    let ffn = mla_cfg.stack.ffn as usize;
    if !cfg.no_experts && (!hidden.is_multiple_of(cfg.tile) || !ffn.is_multiple_of(cfg.tile)) {
        return Err(format!(
            "hidden {hidden} and expert width {ffn} must be multiples of {}",
            cfg.tile
        ));
    }
    let meta = resolve_meta(cfg)?;
    let state_path = (cfg.state.clone()).unwrap_or_else(|| cfg.out.join("calib-state.json"));
    let stamp = Stamp {
        weights_id: meta.weights_id.clone(),
        hessian_id: meta.hessian_id.clone(),
        tokens: toks.hash,
        rows: n_rows,
        seq_len,
        heldout: cfg.heldout,
        batch: cfg.batch,
    };
    let state = CalibState::load(&state_path, cfg.resume, stamp)?;
    let todo: Vec<usize> = (cfg.layers.iter().copied())
        .filter(|l| !state.done.contains(l))
        .collect();
    if todo.windows(2).any(|w| w[1] != w[0] + 1) {
        return Err("layers must be a contiguous range".into());
    }
    fs::create_dir_all(&cfg.work).map_err(|e| e.to_string())?;
    fs::create_dir_all(&cfg.out).map_err(|e| e.to_string())?;

    let mut trace_cfg = mla_cfg.clone();
    trace_cfg.index = None;
    Ok(RunPlan {
        n_rows,
        seq_len,
        cal_rows: n_rows - cfg.heldout,
        tok_data: toks.ids,
        meta,
        state_path,
        state,
        todo,
        batches: batches_split(n_rows, n_rows - cfg.heldout, cfg.batch),
        graph: mla::calib_program::<mla::Moe>(&trace_cfg),
        hidden,
        top_k: mla_cfg.top_k as usize,
        dense_layers: mla_cfg.dense_layers as usize,
    })
}

/// Writes `h-000.bin` from the embedding, or checks the first input.
pub(crate) fn first_hidden(cfg: &Config, plan: &RunPlan, load: &mut Loader) -> Result<(), String> {
    let (first, row) = (plan.todo[0], plan.seq_len * plan.hidden);
    let path = hidden_path(&cfg.work, first);
    if first > 0 {
        let want = (plan.n_rows * row * 2) as u64;
        let len = fs::metadata(&path).map(|m| m.len());
        return match len {
            Ok(len) if len == want => Ok(()),
            Ok(len) => Err(format!("{}: {len} bytes, want {want}", path.display())),
            Err(_) => Err(format!("missing hidden state for layer {first}")),
        };
    }
    let g = &plan.graph;
    let embed = (g.nodes.iter().find(|n| matches!(n.op, Op::Select(_))))
        .and_then(|n| match g.nodes[n.args[0].0 as usize].op {
            Op::Weight(w) => Some(w as usize),
            _ => None,
        })
        .ok_or("missing embedding table")?;
    let mut table_g = Graph::new(g.dims.clone());
    table_g.weights.push(g.weights[embed].clone());
    let table = load(&table_g)?;
    let table = bf16_bytes(&decode_le4(&table[0], f32::from_le_bytes));
    let row_bytes = plan.hidden * 2;
    let writer = HiddenWriter::spawn(path, plan.n_rows, plan.seq_len, plan.hidden)?;
    for &(s0, s1) in &plan.batches {
        let mut raw = Vec::with_capacity((s1 - s0) * row * 2);
        for &id in &plan.tok_data[s0 * plan.seq_len..s1 * plan.seq_len] {
            let off = id as usize * row_bytes;
            raw.extend_from_slice(&table[off..off + row_bytes]);
        }
        writer.send_raw(s0, s1 - s0, raw)?;
    }
    writer.finish()
}

/// Weights of `sub` picked from `all` by path.
fn sub_weights(full: &Graph, sub: &Graph, all: &[Tensor]) -> Result<Vec<Tensor>, String> {
    let mut out = Vec::with_capacity(sub.weights.len());
    for w in &sub.weights {
        let at = (full.weights.iter().position(|f| f.path == w.path))
            .ok_or_else(|| format!("weight {:?} is not in the model", w.path))?;
        out.push(all[at].clone());
    }
    Ok(out)
}

/// Reads and unpacks the weights of `lg` from a checkpoint.
pub(crate) fn layer_from_offer(
    lg: &Graph,
    offer: &dyn Offer,
    rules: &[tessera_bundle::names::Rule],
) -> Result<Vec<Tensor>, String> {
    let mut out = Vec::with_capacity(lg.weights.len());
    tessera_bundle::each(lg, offer, &[rules], &|_| None, |w, _| {
        let vals = match w {
            Weight::Floats(t) => {
                out.push(t);
                return Ok(());
            }
            Weight::Packed(form, bytes) => form.unpack_all(&bytes),
            Weight::Filed(form, file) => {
                let mut bytes = vec![0u8; file.bytes()];
                file.read(&mut bytes, Default::default())?;
                form.unpack_all(&bytes)
            }
        };
        out.push(Tensor::f32(vec![vals.len()], vals));
        Ok(())
    })
    .map_err(|e| format!("layer weights: {e}"))?;
    for (t, w) in out.iter_mut().zip(&lg.weights) {
        let shape: Option<Vec<usize>> = (w.ty.shape.0.iter())
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

pub(crate) fn decode_le4<T>(t: &Tensor, f: fn([u8; 4]) -> T) -> Vec<T> {
    t.bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f)
        .collect()
}

/// Each calibration output of `lg` and its output index.
pub(crate) fn calib_outputs(lg: &Graph) -> impl Iterator<Item = (usize, CalibTap)> + '_ {
    let outs = lg.entries[0].outputs.iter().enumerate();
    outs.filter_map(|(i, o)| match o.role {
        Role::Calib(tap) => Some((i, tap)),
        _ => None,
    })
}

/// Positions `0..seq_len` for each of `seqs` sequences.
pub(crate) fn positions(seqs: usize, seq_len: usize) -> Vec<u32> {
    (0..seqs).flat_map(|_| 0..seq_len as u32).collect()
}

/// Marks `layer` done after its files, then drops its input.
fn complete(cfg: &Config, plan: &mut RunPlan, layer: usize, w: AsyncWriter) -> Result<(), String> {
    w.finish()
        .map_err(|e| format!("layer {layer} files: {e}"))?;
    plan.state.done.push(layer);
    plan.state.save(&plan.state_path)?;
    if !cfg.keep_hidden {
        let _ = fs::remove_file(hidden_path(&cfg.work, layer));
    }
    Ok(())
}

fn run_layer(
    cfg: &Config,
    plan: &RunPlan,
    engine: &mut dyn Engine,
    layer: usize,
) -> Result<AsyncWriter, String> {
    let region = (plan.graph.regions.iter())
        .filter(|r| r.kind == Composite::Layer)
        .nth(layer)
        .ok_or_else(|| format!("layer {layer} out of range"))?;
    let lg = cut_region(&plan.graph, region);
    let sparse = layer >= plan.dense_layers;
    let hess = cfg.hess_layers.contains(&layer);
    let wants = Wants {
        experts: sparse && hess && !cfg.no_experts,
        dense: hess && !cfg.no_dense,
        held: hess && cfg.dense_heldout,
    };
    engine.open(layer, &lg, &wants)?;
    let mut saved = (sparse && cfg.save_inputs.contains(&layer)).then(SavedInputs::default);
    let mut routes = (sparse && cfg.route_log.contains(&layer)).then(RouteLog::default);
    let (seq, hidden) = (plan.seq_len, plan.hidden);
    let reader = HiddenReader::spawn(
        hidden_path(&cfg.work, layer),
        seq,
        hidden,
        plan.batches.clone(),
    );
    let h_out = HiddenWriter::spawn(hidden_path(&cfg.work, layer + 1), plan.n_rows, seq, hidden)?;
    for &(s0, s1) in &plan.batches {
        let part = Part {
            cal: s1 <= plan.cal_rows,
            held: s0 >= plan.cal_rows,
        };
        let keep = part.held && saved.is_some();
        let (out, routing) = engine.batch(
            reader.next_batch()?,
            s1 - s0,
            part,
            keep || routes.is_some(),
        )?;
        h_out.send_raw(s0, s1 - s0, out)?;
        let Some(r) = routing else { continue };
        if let Some(sv) = saved.as_mut().filter(|_| keep) {
            sv.push(&r.x, &r.ids, &r.gates);
        }
        if let Some(rt) = routes.as_mut() {
            rt.push(&r.ids, &r.gates);
        }
    }
    reader.finish();
    h_out.finish()?;
    let writer = AsyncWriter::new(4);
    let files = Files {
        out: &cfg.out,
        layer,
        meta: &plan.meta,
        writer: &writer,
    };
    engine.close(&files)?;
    if let Some(sv) = saved {
        sv.flush(&files, cfg.heldout, (hidden, plan.top_k))?;
    }
    if let Some(rt) = routes {
        rt.flush(&files, [plan.n_rows, seq, plan.top_k])?;
    }
    Ok(writer)
}

/// Runs each pending layer, writing files behind the next.
pub(crate) fn run_layers(
    cfg: &Config,
    mut plan: RunPlan,
    engine: &mut dyn Engine,
) -> Result<(), String> {
    let mut pending: Option<(usize, AsyncWriter)> = None;
    for layer in plan.todo.clone() {
        let step = run_layer(cfg, &plan, engine, layer);
        if let Some((prev, w)) = pending.take() {
            complete(cfg, &mut plan, prev, w)?;
        }
        pending = Some((layer, step?));
    }
    match pending {
        Some((layer, w)) => complete(cfg, &mut plan, layer, w),
        None => Ok(()),
    }
}

/// Runs layer graphs on the CPU interpreter.
struct Host<'a> {
    load: Box<Loader<'a>>,
    experts: usize,
    per_file: usize,
    dims: (usize, usize),
    seq_len: usize,
    top_k: usize,
    tile: usize,
    layer: Option<HostLayer>,
}

struct HostLayer {
    lg: Graph,
    weights: Vec<Tensor>,
    experts: Option<ExpertAccum>,
    dense: Option<DenseAccum>,
    held: Option<DenseAccum>,
}

impl Engine for Host<'_> {
    fn open(&mut self, _: usize, lg: &Graph, wants: &Wants) -> Result<(), String> {
        let (h, f) = self.dims;
        self.layer = Some(HostLayer {
            lg: lg.clone(),
            weights: (self.load)(lg)?,
            experts: (wants.experts).then(|| ExpertAccum::new(self.experts, h, f, self.tile)),
            dense: wants.dense.then(DenseAccum::default),
            held: wants.held.then(DenseAccum::default),
        });
        Ok(())
    }

    fn batch(
        &mut self,
        x: Vec<f32>,
        seqs: usize,
        part: Part,
        route: bool,
    ) -> Result<(Vec<u8>, Option<Routing>), String> {
        let l = self.layer.as_mut().ok_or("no open layer")?;
        let n_tok = seqs * self.seq_len;
        let x_t = Tensor::f32(vec![seqs, self.seq_len, self.dims.0], x);
        let pos_t = Tensor::u32(vec![seqs, self.seq_len], positions(seqs, self.seq_len));
        let feeds: Vec<Tensor> = (l.lg.entries[0].inputs.iter())
            .map(|port| match port.role {
                Role::Positions => pos_t.clone(),
                _ => x_t.clone(),
            })
            .collect();
        let outs = run(&l.lg, &l.weights, 0, &feeds).map_err(|e| e.to_string())?;
        let out = bf16_bytes(&decode_le4(&outs[0], f32::from_le_bytes));
        let floats = |t: &Tensor| decode_le4(t, f32::from_le_bytes);
        let taps = Taps::sort(
            calib_outputs(&l.lg).map(|(i, tap)| (tap, &outs[i])),
            |name, t| {
                let vals = floats(t);
                let dim = vals.len() / n_tok;
                for acc in [
                    l.dense.as_mut().filter(|_| part.cal),
                    l.held.as_mut().filter(|_| part.held),
                ]
                .into_iter()
                .flatten()
                {
                    acc.add(name, &vals, n_tok, dim);
                }
                Ok(())
            },
        )?;
        let ids = taps.ids.map(|t| decode_le4(t, u32::from_le_bytes));
        let mlp = taps.mlp.map(floats);
        if part.cal
            && let (Some(acc), Some(x), Some(inner), Some(ids)) =
                (l.experts.as_mut(), &mlp, taps.inner, &ids)
        {
            acc.add(x, &floats(inner), ids, self.top_k);
        }
        let routing = match (route, mlp, ids, taps.gates) {
            (true, Some(x), Some(ids), Some(g)) => Some(Routing {
                x,
                ids,
                gates: floats(g),
            }),
            _ => None,
        };
        Ok((out, routing))
    }

    fn close(&mut self, files: &Files) -> Result<(), String> {
        let l = self.layer.take().ok_or("no open layer")?;
        if let Some(acc) = l.experts {
            acc.flush(files, self.per_file)?;
        }
        if let Some(acc) = l.dense {
            acc.flush(files, false)?;
        }
        if let Some(acc) = l.held {
            acc.flush(files, true)?;
        }
        Ok(())
    }
}

fn run_host(
    cfg: &Config,
    mla_cfg: &mla::Config,
    plan: RunPlan,
    mut load: Box<Loader>,
) -> Result<(), String> {
    first_hidden(cfg, &plan, &mut load)?;
    let mut host = Host {
        load,
        experts: mla_cfg.experts as usize,
        per_file: cfg.experts_per_file,
        dims: (plan.hidden, mla_cfg.stack.ffn as usize),
        seq_len: plan.seq_len,
        top_k: plan.top_k,
        tile: cfg.tile,
        layer: None,
    };
    run_layers(cfg, plan, &mut host)
}

/// Runs layer-streaming MLA calibration over in-memory weights.
pub fn run_mla(cfg: &Config, mla_cfg: &mla::Config, all: &[Tensor]) -> Result<(), String> {
    let plan = prepare_run(cfg, mla_cfg, read_npy(&cfg.tokens)?)?;
    if plan.todo.is_empty() {
        return Ok(());
    }
    let full = plan.graph.clone();
    run_host(
        cfg,
        mla_cfg,
        plan,
        Box::new(move |sub| sub_weights(&full, sub, all)),
    )
}

/// Reads the tokens and model config, and resolves layer sets.
pub(crate) fn load_config(cfg: &mut Config) -> Result<(mla::Config, Tokens), String> {
    let toks = read_npy(&cfg.tokens)?;
    let json = fs::read_to_string(cfg.model.join("config.json"))
        .map_err(|e| format!("config.json: {e}"))?;
    let mla_cfg = mla_config(&json, cfg.batch as u64, toks.seq_len as u64)
        .map_err(|e| format!("mla config: {e}"))?;
    cfg.resolve(mla_cfg.stack.layers as usize)?;
    Ok((mla_cfg, toks))
}

/// Runs layer-streaming MLA calibration from a checkpoint directory.
pub fn run_dir(mut cfg: Config) -> Result<(), String> {
    let (mla_cfg, toks) = load_config(&mut cfg)?;
    let plan = prepare_run(&cfg, &mla_cfg, toks)?;
    if plan.todo.is_empty() {
        return Ok(());
    }
    let offer = open_ckpt(&cfg.model, None).map_err(|e| format!("checkpoint: {e}"))?;
    let rules = mla_rules(&mla_cfg, false);
    run_host(
        &cfg,
        &mla_cfg,
        plan,
        Box::new(|sub| layer_from_offer(sub, offer.as_ref(), &rules)),
    )
}
