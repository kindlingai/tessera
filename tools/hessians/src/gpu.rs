//! CUDA layer-streaming calibration, device Gram sums, and kernel bench.

use crate::accum::{Dense, Files, Taps, TileShard, submit_dense, submit_experts, tiled};
use crate::args::{BenchShape, Config, Tokens};
use crate::driver::{
    Engine, Part, Routing, Wants, calib_outputs, first_hidden, layer_from_offer, positions,
    prepare_run, run_layers,
};
use std::collections::BTreeMap;
use std::time::Instant;
use tessera_bundle::mla::rules as mla_rules;
use tessera_checkpoint::{Offer, open as open_ckpt};
use tessera_device_rules::calib::{
    MAX_EXPERTS, SORT_BLOCKS, TILE, cuda_source, sort_cpu, syrk_dense_cpu, syrk_tiles_cpu,
};
use tessera_engine_settings::{Backend, Pick, Settings};
use tessera_gpukit::cuda::{Context, Kernel};
use tessera_gpukit::device::{Device as GpuDevice, cached_context};
use tessera_gpukit::launch::{KernelArgs, Launch};
use tessera_graph::{Bf16, DimName, Graph, Role};
use tessera_lower::form::{BF16_FORM, F32_FORM};
use tessera_lower::{Cache, Compute, Device as LowerDev, Pins, Site, Slot};
use tessera_mem::{Tensor, View};
use tessera_models::mla;
use tessera_run::{Run, open as open_runner, select as select_gpu};

/// Pins activations and outputs to BF16, keeping router F32.
fn calib_pins(base: Pins) -> Pins {
    base.activations(BF16_FORM)
        .outputs(BF16_FORM)
        .site(Site::Router, F32_FORM)
}

/// Runtime dimension bindings for one calibration batch `[batch, seq_len]`.
fn layer_sizes(batch: usize, seq_len: usize) -> [(DimName, u64); 4] {
    [
        (DimName::Seqs, batch as u64),
        (DimName::Context, seq_len as u64),
        (DimName::Rows, (batch * seq_len) as u64),
        (DimName::Segments, 1),
    ]
}

/// Pinned host staging buffer for DMA hidden-state transfers.
struct PinnedBuf {
    ctx: &'static Context,
    host: u64,
    len: usize,
}

impl PinnedBuf {
    fn new(ctx: &'static Context, len: usize) -> Result<Self, String> {
        let (host, _) = ctx.pin(len.max(1)).map_err(|e| format!("pin: {e}"))?;
        Ok(Self { ctx, host, len })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host as *mut u8, self.len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        self.ctx.unpin(self.host);
    }
}

/// RAII GPU device buffer allocated through the shared CUDA context.
struct DevBuf {
    ctx: &'static Context,
    ptr: u64,
    bytes: usize,
}

impl DevBuf {
    fn alloc(ctx: &'static Context, bytes: usize) -> Result<Self, String> {
        let sz = bytes.max(4);
        let ptr = ctx.alloc(sz).map_err(|e| format!("gpu alloc {sz}: {e}"))?;
        Ok(Self { ctx, ptr, bytes })
    }

    fn zeroed(ctx: &'static Context, bytes: usize) -> Result<Self, String> {
        let buf = Self::alloc(ctx, bytes)?;
        (buf.ctx.zero(buf.ptr, bytes.max(4))).map_err(|e| format!("gpu zero: {e}"))?;
        Ok(buf)
    }

    fn upload_le<T: Copy, const N: usize>(
        &self,
        vals: &[T],
        f: fn(T) -> [u8; N],
    ) -> Result<(), String> {
        let raw: Vec<u8> = vals.iter().copied().flat_map(f).collect();
        (self.ctx.upload(self.ptr, &raw)).map_err(|e| format!("gpu upload: {e}"))
    }

    /// Downloads `bytes` bytes starting `at` bytes in.
    fn download_at(&self, at: usize, bytes: usize) -> Result<Vec<u8>, String> {
        let mut raw = vec![0u8; bytes];
        (self.ctx.download(self.ptr + at as u64, &mut raw))
            .map_err(|e| format!("gpu download: {e}"))?;
        Ok(raw)
    }

    fn download_f32(&self) -> Result<Vec<f32>, String> {
        let raw = self.download_at(0, self.bytes)?;
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn download_i64(&self) -> Result<Vec<i64>, String> {
        let raw = self.download_at(0, self.bytes)?;
        Ok(raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect())
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        self.ctx.free(self.ptr);
    }
}

/// Compiled calibration sort, SYRK, and BF16 MoE GEMM kernels.
struct CalibKernels {
    ctx: &'static Context,
    sort_hist: Kernel,
    sort_prefix: Kernel,
    sort_scatter: Kernel,
    syrk_bf16: Kernel,
    syrk_f32: Kernel,
    pbf16_gu: Kernel,
    pbf16_gemm: Kernel,
}

impl CalibKernels {
    fn compile(ctx: &'static Context) -> Result<Self, String> {
        let src = cuda_source();
        let build =
            |name: &str| (ctx.compile(src, name)).map_err(|e| format!("compile {name}: {e}"));
        Ok(Self {
            ctx,
            sort_hist: build("calib_sort_hist")?,
            sort_prefix: build("calib_sort_prefix")?,
            sort_scatter: build("calib_sort_scatter")?,
            syrk_bf16: build("calib_syrk_bf16")?,
            syrk_f32: build("calib_syrk_f32")?,
            pbf16_gu: build("pbf16_gu_bf16_bf16")?,
            pbf16_gemm: build("pbf16_gemm_bf16_bf16")?,
        })
    }

    fn launch_syrk(
        &self,
        is_f32: bool,
        ptrs: (u64, u64, u64, u64),
        shape: (u32, u32, u32, u32, u32),
    ) -> Result<(), String> {
        let (x_ptr, perm_ptr, off_ptr, acc_ptr) = ptrs;
        let (total_rows, dim, repeat, full_out, experts) = shape;
        let blocks = dim / TILE as u32;
        let tiles = blocks * (blocks + 1) / 2;
        let mut args = KernelArgs::new();
        args.device_ptr(x_ptr)
            .device_ptr(perm_ptr)
            .device_ptr(off_ptr)
            .device_ptr(acc_ptr)
            .u32(total_rows)
            .u32(dim)
            .u32(repeat)
            .u32(full_out);
        let k = if is_f32 {
            &self.syrk_f32
        } else {
            &self.syrk_bf16
        };
        (self
            .ctx
            .launch(k, (tiles, experts, 1), (256, 1, 1), 0, &mut args))
        .map_err(|e| format!("launch syrk: {e}"))
    }
}

/// Reusable GPU scratch for deterministic 3-pass expert counting sort.
struct GpuSort {
    block_hist: DevBuf,
    block_off: DevBuf,
    offsets: DevBuf,
    perm: DevBuf,
    tile_meta: DevBuf,
    max_tiles: u32,
}

impl GpuSort {
    fn new(ctx: &'static Context, experts: usize, max_picks: usize) -> Result<Self, String> {
        let hist_bytes = SORT_BLOCKS * experts * 4;
        let max_tiles = (max_picks.div_ceil(TILE) + experts).max(1);
        Ok(Self {
            block_hist: DevBuf::alloc(ctx, hist_bytes)?,
            block_off: DevBuf::alloc(ctx, hist_bytes)?,
            offsets: DevBuf::alloc(ctx, (experts + 1) * 4)?,
            perm: DevBuf::alloc(ctx, max_picks.max(1) * 4)?,
            tile_meta: DevBuf::alloc(ctx, (1 + 2 * max_tiles) * 4)?,
            max_tiles: max_tiles as u32,
        })
    }

    fn sort(
        &self,
        k: &CalibKernels,
        ids_ptr: u64,
        total_picks: u32,
        experts: u32,
        counts_ptr: u64,
        tile_m: u32,
    ) -> Result<(), String> {
        let sb = SORT_BLOCKS as u32;
        let mut a1 = KernelArgs::new();
        a1.device_ptr(ids_ptr)
            .device_ptr(self.block_hist.ptr)
            .u32(total_picks)
            .u32(experts)
            .u32(0)
            .u32(sb);
        (k.ctx
            .launch(&k.sort_hist, (sb, 1, 1), (256, 1, 1), 0, &mut a1))
        .map_err(|e| format!("sort_hist: {e}"))?;
        let meta_ptr = if tile_m > 0 { self.tile_meta.ptr } else { 0 };
        let mut a2 = KernelArgs::new();
        a2.device_ptr(self.block_hist.ptr)
            .device_ptr(self.block_off.ptr)
            .device_ptr(self.offsets.ptr)
            .device_ptr(counts_ptr)
            .device_ptr(meta_ptr)
            .u32(experts)
            .u32(sb)
            .u32(tile_m);
        (k.ctx
            .launch(&k.sort_prefix, (1, 1, 1), (256, 1, 1), 0, &mut a2))
        .map_err(|e| format!("sort_prefix: {e}"))?;
        let mut a3 = KernelArgs::new();
        a3.device_ptr(ids_ptr)
            .device_ptr(self.block_off.ptr)
            .device_ptr(self.offsets.ptr)
            .device_ptr(self.perm.ptr)
            .u32(total_picks)
            .u32(experts)
            .u32(0)
            .u32(sb);
        (k.ctx
            .launch(&k.sort_scatter, (sb, 1, 1), (32, 1, 1), 0, &mut a3))
        .map_err(|e| format!("sort_scatter: {e}"))
    }
}

/// GPU-resident packed upper-triangle accumulators for routed MoE experts.
struct GpuExperts {
    experts: usize,
    hidden: usize,
    ffn: usize,
    gu: DevBuf,
    down: DevBuf,
    counts: DevBuf,
    sort: GpuSort,
}

impl GpuExperts {
    fn new(
        ctx: &'static Context,
        experts: usize,
        (hidden, ffn): (usize, usize),
        max_picks: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            experts,
            hidden,
            ffn,
            gu: DevBuf::zeroed(ctx, experts * tiled(hidden, TILE) * 4)?,
            down: DevBuf::zeroed(ctx, experts * tiled(ffn, TILE) * 4)?,
            counts: DevBuf::zeroed(ctx, experts * 8)?,
            sort: GpuSort::new(ctx, experts, max_picks)?,
        })
    }

    fn add(
        &self,
        k: &CalibKernels,
        [mlp, inner]: [(u64, bool); 2],
        ids_ptr: u64,
        picks: usize,
        top_k: usize,
    ) -> Result<(), String> {
        let (picks, e) = (picks as u32, self.experts as u32);
        let (perm, off) = (self.sort.perm.ptr, self.sort.offsets.ptr);
        self.sort.sort(k, ids_ptr, picks, e, self.counts.ptr, 0)?;
        let gu = (mlp.0, perm, off, self.gu.ptr);
        k.launch_syrk(mlp.1, gu, (picks, self.hidden as u32, top_k as u32, 0, e))?;
        let down = (inner.0, perm, off, self.down.ptr);
        k.launch_syrk(inner.1, down, (picks, self.ffn as u32, 1, 0, e))
    }

    fn flush(self, files: &Files, per_file: usize) -> Result<(), String> {
        let counts = self.counts.download_i64()?;
        let (gu_b, dn_b) = (tiled(self.hidden, TILE) * 4, tiled(self.ffn, TILE) * 4);
        for a in (0..self.experts).step_by(per_file) {
            let b = (a + per_file).min(self.experts);
            let shard = TileShard {
                first: a,
                dims: (self.hidden, self.ffn),
                gu: self.gu.download_at(a * gu_b, (b - a) * gu_b)?,
                down: self.down.download_at(a * dn_b, (b - a) * dn_b)?,
                counts: counts[a..b].to_vec(),
            };
            submit_experts(files, TILE, shard)?;
        }
        Ok(())
    }
}

/// GPU-resident full-symmetric dense Gram accumulators for one layer.
struct GpuDense {
    ctx: &'static Context,
    mats: BTreeMap<&'static str, (usize, DevBuf, i64)>,
}

impl GpuDense {
    fn new(ctx: &'static Context) -> Self {
        Self {
            ctx,
            mats: BTreeMap::new(),
        }
    }

    fn add(
        &mut self,
        k: &CalibKernels,
        name: &'static str,
        x: (u64, bool),
        rows: usize,
        dim: usize,
    ) -> Result<(), String> {
        if !dim.is_multiple_of(TILE) {
            return Err(format!(
                "tap {name} width {dim} is not a multiple of {TILE}"
            ));
        }
        if !self.mats.contains_key(name) {
            self.mats
                .insert(name, (dim, DevBuf::zeroed(self.ctx, dim * dim * 4)?, 0));
        }
        let entry = self.mats.get_mut(name).ok_or("dense sum missing")?;
        k.launch_syrk(
            x.1,
            (x.0, 0, 0, entry.1.ptr),
            (rows as u32, dim as u32, 1, 1, 1),
        )?;
        entry.2 += rows as i64;
        Ok(())
    }

    fn flush(self, files: &Files, heldout: bool) -> Result<(), String> {
        let mut mats: Vec<Dense> = Vec::with_capacity(self.mats.len());
        for (name, (dim, buf, count)) in self.mats {
            mats.push((name, dim, buf.download_f32()?, count));
        }
        submit_dense(files, heldout, mats)
    }
}

/// Whether a tap slot holds F32 rather than BF16.
fn slot_f32(slot: &Slot) -> Result<bool, String> {
    match slot.form.name() {
        "bf16" => Ok(false),
        "f32" => Ok(true),
        other => Err(format!("tap form {other} is not BF16 or F32")),
    }
}

fn floats(runner: &dyn Run, slot: Slot) -> Result<Vec<f32>, String> {
    let read = runner.read(slot).map_err(|e| e.to_string())?;
    Ok(read.floats().map_err(|e| e.to_string())?.to_vec())
}

/// One layer's runner, plan cache and device sums.
struct GpuLayer {
    lg: Graph,
    dev: LowerDev,
    cache: Cache,
    runner: Box<dyn Run>,
    seqs: usize,
    experts: Option<GpuExperts>,
    dense: Option<GpuDense>,
    held: Option<GpuDense>,
    started: Instant,
    times: [f64; 2],
}

/// Runs layer plans and Gram sums on CUDA.
struct Gpu<'a> {
    ctx: &'static Context,
    kernels: CalibKernels,
    settings: Settings,
    offer: &'a dyn Offer,
    rules: Vec<tessera_bundle::names::Rule>,
    mla: &'a mla::Config,
    seq_len: usize,
    first_seqs: usize,
    per_file: usize,
    pinned: PinnedBuf,
    layer: Option<GpuLayer>,
}

impl Engine for Gpu<'_> {
    fn open(&mut self, layer: usize, lg: &Graph, wants: &Wants) -> Result<(), String> {
        let started = Instant::now();
        let (rules, layers) = (&[self.rules.as_slice()], self.mla.stack.layers);
        let pins = tessera_bundle::pins(lg, self.offer, layers, rules)
            .map_err(|e| format!("layer {layer} pins: {e}"))?;
        let dev = LowerDev::stored(self.ctx.platform(), &self.settings)
            .unwrap_or_else(|_| LowerDev::guess(&self.settings).on(self.ctx.platform()))
            .pinned(calib_pins(pins))
            .with_compute(Compute::Bf16);
        let mut cache = Cache::default();
        let sizes = layer_sizes(self.first_seqs, self.seq_len);
        let plan = (cache.lower(lg, 0, &sizes, &dev))
            .map_err(|e| format!("lower layer {layer}: {e}"))?
            .clone();
        let mut runner = open_runner(plan, &self.settings).map_err(|e| format!("runner: {e}"))?;
        let mut at = 0u32;
        tessera_bundle::each(lg, self.offer, rules, &|_| None, |w, _| {
            runner.load_weight(at, &w)?;
            at += 1;
            Ok(())
        })
        .map_err(|e| format!("load layer {layer}: {e}"))?;
        let (hidden, ffn) = (self.mla.stack.hidden as usize, self.mla.stack.ffn as usize);
        let picks = self.first_seqs * self.seq_len * self.mla.top_k as usize;
        let experts = match wants.experts {
            true => Some(GpuExperts::new(
                self.ctx,
                self.mla.experts as usize,
                (hidden, ffn),
                picks,
            )?),
            false => None,
        };
        self.layer = Some(GpuLayer {
            lg: lg.clone(),
            dev,
            cache,
            runner,
            seqs: self.first_seqs,
            experts,
            dense: wants.dense.then(|| GpuDense::new(self.ctx)),
            held: wants.held.then(|| GpuDense::new(self.ctx)),
            started,
            times: [started.elapsed().as_secs_f64(), 0.0],
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
        let (seq_len, hidden) = (self.seq_len, self.mla.stack.hidden as usize);
        if seqs != l.seqs {
            let sizes = layer_sizes(seqs, seq_len);
            let plan = (l.cache.lower(&l.lg, 0, &sizes, &l.dev))
                .map_err(|e| format!("lower tail: {e}"))?;
            l.runner.replan(&plan).map_err(|e| format!("replan: {e}"))?;
            l.seqs = seqs;
        }
        let n_tok = seqs * seq_len;
        let x_t = Tensor::f32(vec![seqs, seq_len, hidden], x);
        let pos_t = Tensor::u32(vec![seqs, seq_len], positions(seqs, seq_len));
        let feeds: Vec<View> = (l.lg.entries[0].inputs.iter())
            .map(|port| match port.role {
                Role::Positions => pos_t.view(),
                _ => x_t.view(),
            })
            .collect();
        let started = Instant::now();
        l.runner.feed(&feeds).map_err(|e| format!("feed: {e}"))?;
        let steps = 0..l.runner.plan().steps.len();
        l.runner.submit(steps).map_err(|e| format!("submit: {e}"))?;
        let out_slot = l.runner.plan().outputs[0].slot;
        let out = match out_slot.form.name() {
            "bf16" => {
                let addr = l.runner.address(out_slot).ok_or("missing output address")?;
                let buf = &mut self.pinned.as_mut_slice()[..n_tok * hidden * 2];
                self.ctx
                    .download(addr, buf)
                    .map_err(|e| format!("output download: {e}"))?;
                buf.to_vec()
            }
            _ => crate::state::bf16_bytes(&floats(l.runner.as_ref(), out_slot)?),
        };
        let slots = l
            .runner
            .plan()
            .outputs
            .iter()
            .map(|o| o.slot)
            .collect::<Vec<_>>();
        let runner = l.runner.as_ref();
        let tapped = calib_outputs(&l.lg).map(|(i, tap)| (tap, slots[i]));
        let (k, dense, held) = (&self.kernels, &mut l.dense, &mut l.held);
        let taps = Taps::sort(tapped, |name, &slot| {
            let x = (
                runner.address(slot).ok_or("missing tap address")?,
                slot_f32(&slot)?,
            );
            for acc in [
                dense.as_mut().filter(|_| part.cal),
                held.as_mut().filter(|_| part.held),
            ]
            .into_iter()
            .flatten()
            {
                acc.add(k, name, x, n_tok, slot.len / n_tok)?;
            }
            Ok(())
        })?;
        let addr = |slot: Slot| -> Result<(u64, bool), String> {
            Ok((
                runner.address(slot).ok_or("missing tap address")?,
                slot_f32(&slot)?,
            ))
        };
        if part.cal
            && let (Some(acc), Some(m), Some(inner), Some(ids)) =
                (l.experts.as_ref(), taps.mlp, taps.inner, taps.ids)
        {
            let top_k = self.mla.top_k as usize;
            acc.add(
                k,
                [addr(m)?, addr(inner)?],
                addr(ids)?.0,
                n_tok * top_k,
                top_k,
            )?;
        }
        self.ctx.finish().map_err(|e| format!("gpu finish: {e}"))?;
        l.times[1] += started.elapsed().as_secs_f64();
        let routing = match (route, taps.mlp, taps.ids, taps.gates) {
            (true, Some(m), Some(ids), Some(g)) => Some(Routing {
                x: floats(runner, m)?,
                ids: runner.ids(ids).map_err(|e| e.to_string())?,
                gates: floats(runner, g)?,
            }),
            _ => None,
        };
        Ok((out, routing))
    }

    fn close(&mut self, files: &Files) -> Result<(), String> {
        let l = self.layer.take().ok_or("no open layer")?;
        drop(l.runner);
        if let Some(acc) = l.experts {
            acc.flush(files, self.per_file)?;
        }
        if let Some(acc) = l.dense {
            acc.flush(files, false)?;
        }
        if let Some(acc) = l.held {
            acc.flush(files, true)?;
        }
        let [load, run] = l.times;
        let total = l.started.elapsed().as_secs_f64();
        println!(
            "layer {:03}: load={load:.2}s run={run:.2}s total={total:.2}s",
            files.layer
        );
        Ok(())
    }
}

/// Runs layer-streaming BF16 MLA calibration on CUDA with GPU SYRK.
pub(crate) fn run_gpu(cfg: &Config, mla_cfg: &mla::Config, toks: Tokens) -> Result<(), String> {
    let (hidden, ffn) = (mla_cfg.stack.hidden as usize, mla_cfg.stack.ffn as usize);
    if cfg.tile != TILE || !hidden.is_multiple_of(TILE) || !ffn.is_multiple_of(TILE) {
        return Err(format!(
            "--gpu needs hidden {hidden} and expert width {ffn} in steps of {TILE}"
        ));
    }
    if mla_cfg.experts as usize > MAX_EXPERTS {
        return Err(format!("--gpu takes at most {MAX_EXPERTS} experts"));
    }
    if let Some(ord) = cfg.device {
        select_gpu(ord);
    }
    let plan = prepare_run(cfg, mla_cfg, toks)?;
    if plan.todo.is_empty() {
        return Ok(());
    }
    let offer = open_ckpt(&cfg.model, None).map_err(|e| format!("checkpoint: {e}"))?;
    let rules = mla_rules(mla_cfg, false);
    first_hidden(cfg, &plan, &mut |sub| {
        layer_from_offer(sub, offer.as_ref(), &rules)
    })?;
    let ctx = cached_context::<Context>().map_err(|e| format!("cuda context: {e}"))?;
    let first_seqs = plan.batches[0].1 - plan.batches[0].0;
    let mut gpu = Gpu {
        ctx,
        kernels: CalibKernels::compile(ctx)?,
        settings: Settings {
            backend: Pick::Gpu(Backend::Cuda),
            device: cfg.device,
            ..Settings::default()
        },
        offer: offer.as_ref(),
        rules,
        mla: mla_cfg,
        seq_len: plan.seq_len,
        first_seqs,
        per_file: cfg.experts_per_file,
        pinned: PinnedBuf::new(ctx, cfg.batch * plan.seq_len * hidden * 2)?,
        layer: None,
    };
    run_layers(cfg, plan, &mut gpu)
}

fn fill_bf16(len: usize, seed: u64) -> (Vec<u16>, Vec<f32>) {
    let mut s = seed | 1;
    let mut raw = Vec::with_capacity(len);
    let mut f32s = Vec::with_capacity(len);
    for _ in 0..len {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = ((s & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.25;
        let b = Bf16::from(v);
        raw.push(b.0);
        f32s.push(f32::from(b));
    }
    (raw, f32s)
}

fn max_err(got: &[f32], want: &[f64]) -> f64 {
    let pairs = got.iter().zip(want);
    let errs = pairs.map(|(&a, &b)| (f64::from(a) - b).abs());
    errs.fold(0.0, |m, d| if d.is_nan() || d > m { d } else { m })
}

/// Checks device sort and SYRK sums against the host references.
fn check_small(k: &CalibKernels) -> Result<(), String> {
    let (n_tok, top_k, experts, dim) = (64usize, 2usize, 4usize, TILE);
    let picks = n_tok * top_k;
    let ids: Vec<u32> = (0..picks).map(|i| ((i * 3 + 1) % experts) as u32).collect();
    let (x_raw, x_f32) = fill_bf16(n_tok * dim, 17);
    let (in_raw, in_f32) = fill_bf16(picks * dim, 29);
    let d_ids = DevBuf::alloc(k.ctx, picks * 4)?;
    d_ids.upload_le(&ids, u32::to_le_bytes)?;
    let d_x = DevBuf::alloc(k.ctx, n_tok * dim * 2)?;
    d_x.upload_le(&x_raw, u16::to_le_bytes)?;
    let d_in = DevBuf::alloc(k.ctx, picks * dim * 2)?;
    d_in.upload_le(&in_raw, u16::to_le_bytes)?;

    let exp = GpuExperts::new(k.ctx, experts, (dim, dim), picks)?;
    exp.add(
        k,
        [(d_x.ptr, false), (d_in.ptr, false)],
        d_ids.ptr,
        picks,
        top_k,
    )?;
    let mut dense = GpuDense::new(k.ctx);
    dense.add(k, "x", (d_x.ptr, false), n_tok, dim)?;
    k.ctx.finish().map_err(|e| format!("finish: {e}"))?;

    let (offsets, perm, want_cnt) = sort_cpu(&ids, experts, 0);
    let got_cnt = exp.counts.download_i64()?;
    if got_cnt != want_cnt {
        return Err(format!("count mismatch: {got_cnt:?} vs {want_cnt:?}"));
    }
    let mut want_gu = vec![0.0f64; experts * tiled(dim, TILE)];
    syrk_tiles_cpu(&x_f32, &perm, &offsets, dim, TILE, top_k, &mut want_gu);
    let mut want_dn = vec![0.0f64; experts * tiled(dim, TILE)];
    syrk_tiles_cpu(&in_f32, &perm, &offsets, dim, TILE, 1, &mut want_dn);
    let mut want_d = vec![0.0f64; dim * dim];
    syrk_dense_cpu(&x_f32, n_tok, dim, &mut want_d);
    let got_d = dense.mats.remove("x").ok_or("dense sum missing")?.1;
    let errs = [
        ("grouped gu", max_err(&exp.gu.download_f32()?, &want_gu)),
        ("grouped down", max_err(&exp.down.download_f32()?, &want_dn)),
        ("dense", max_err(&got_d.download_f32()?, &want_d)),
    ];
    for (name, err) in errs {
        if err.is_nan() || err > 1e-2 {
            return Err(format!("{name} SYRK parity failed: max_err={err:.3e}"));
        }
        println!("{name} SYRK max_err={err:.3e}");
    }
    Ok(())
}

fn time_case(
    ctx: &'static Context,
    iters: usize,
    mut body: impl FnMut() -> Result<(), String>,
) -> Result<f64, String> {
    body()?;
    ctx.finish().map_err(|e| format!("warmup: {e}"))?;
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        ctx.stamp().map_err(|e| format!("stamp: {e}"))?;
        body()?;
        ctx.stamp().map_err(|e| format!("stamp: {e}"))?;
        let s = ctx.stamps().map_err(|e| format!("stamps: {e}"))?;
        times.push(*s.first().ok_or("the timer gave an empty stamp list")?);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    Ok(times[times.len() / 2])
}

fn report(what: &str, dt: f64, flops: f64) {
    println!(
        "{what}: {:.2} ms | {:.1} TFLOP/s",
        dt * 1e3,
        flops / dt / 1e12
    );
}

/// Checks the kernels on a small case, then times `shape`.
pub fn bench_kernels(device: Option<usize>, shape: BenchShape) -> Result<(), String> {
    if let Some(ord) = device {
        select_gpu(ord);
    }
    let ctx = cached_context::<Context>().map_err(|e| format!("cuda context: {e}"))?;
    println!("device: {}", ctx.name());
    let k = CalibKernels::compile(ctx)?;
    check_small(&k)?;

    let BenchShape {
        experts: e,
        hidden: h,
        ffn,
        tokens: t,
        top_k,
        dense: d_o,
    } = shape;
    if !h.is_multiple_of(TILE)
        || !ffn.is_multiple_of(TILE)
        || !d_o.is_multiple_of(TILE)
        || e > MAX_EXPERTS
    {
        return Err(format!(
            "--shape widths must be multiples of {TILE}, experts at most {MAX_EXPERTS}"
        ));
    }
    let picks = t * top_k;
    let ids: Vec<u32> = (0..picks).map(|i| ((i * 7 + 3) % e) as u32).collect();
    let d_ids = DevBuf::alloc(ctx, picks * 4)?;
    d_ids.upload_le(&ids, u32::to_le_bytes)?;
    let d_x = DevBuf::zeroed(ctx, t * h * 2)?;
    let d_in = DevBuf::zeroed(ctx, picks * ffn * 2)?;
    let sort = GpuSort::new(ctx, e, picks)?;
    sort.sort(&k, d_ids.ptr, picks as u32, e as u32, 0, TILE as u32)?;
    ctx.finish().map_err(|e| format!("sort bench prep: {e}"))?;
    let (perm, off) = (sort.perm.ptr, sort.offsets.ptr);
    let (pu, eu) = (picks as u32, e as u32);

    let d_acc = DevBuf::zeroed(ctx, e * tiled(h, TILE) * 4)?;
    let dt = time_case(ctx, 5, || {
        k.launch_syrk(
            false,
            (d_x.ptr, perm, off, d_acc.ptr),
            (pu, h as u32, top_k as u32, 0, eu),
        )
    })?;
    report(
        &format!("calib_syrk_bf16 gu [E={e} H={h} T={t} k={top_k}]"),
        dt,
        (picks * h * h) as f64,
    );
    drop(d_acc);
    let d_acc = DevBuf::zeroed(ctx, e * tiled(ffn, TILE) * 4)?;
    let dt = time_case(ctx, 5, || {
        k.launch_syrk(
            false,
            (d_in.ptr, perm, off, d_acc.ptr),
            (pu, ffn as u32, 1, 0, eu),
        )
    })?;
    report(
        &format!("calib_syrk_bf16 down [E={e} F={ffn} T={t} k={top_k}]"),
        dt,
        (picks * ffn * ffn) as f64,
    );
    drop(d_acc);

    let d_wg = DevBuf::zeroed(ctx, e * ffn * h * 2)?;
    let d_wu = DevBuf::zeroed(ctx, e * ffn * h * 2)?;
    let dt = time_case(ctx, 5, || {
        let mut a = KernelArgs::new();
        a.device_ptr(d_x.ptr)
            .device_ptr(d_wg.ptr)
            .device_ptr(d_wu.ptr)
            .device_ptr(off)
            .device_ptr(perm)
            .device_ptr(sort.tile_meta.ptr)
            .device_ptr(d_in.ptr)
            .u32(ffn as u32)
            .u32(h as u32)
            .u32(ffn as u32)
            .u32(top_k as u32);
        let grid = ((ffn / 64) as u32, sort.max_tiles, 1);
        ctx.launch(&k.pbf16_gu, grid, (256, 1, 1), 0, &mut a)
            .map_err(|e| format!("pbf16_gu: {e}"))
    })?;
    report(
        &format!("pbf16_gu_bf16_bf16 [E={e} H={h} F={ffn} T={t} k={top_k}]"),
        dt,
        4.0 * (picks * h * ffn) as f64,
    );
    drop((d_wg, d_wu));

    let d_wd = DevBuf::zeroed(ctx, e * h * ffn * 2)?;
    let d_out = DevBuf::zeroed(ctx, picks * h * 2)?;
    let dt = time_case(ctx, 5, || {
        let mut a = KernelArgs::new();
        a.device_ptr(d_in.ptr)
            .device_ptr(d_wd.ptr)
            .device_ptr(off)
            .device_ptr(perm)
            .device_ptr(sort.tile_meta.ptr)
            .device_ptr(d_out.ptr)
            .u32(h as u32)
            .u32(ffn as u32)
            .u32(h as u32)
            .u32(0)
            .u32(1);
        let grid = ((h / TILE) as u32, sort.max_tiles, 1);
        ctx.launch(&k.pbf16_gemm, grid, (256, 1, 1), 0, &mut a)
            .map_err(|e| format!("pbf16_gemm: {e}"))
    })?;
    report(
        &format!("pbf16_gemm_bf16_bf16 [E={e} F={ffn} H={h} T={t} k={top_k}]"),
        dt,
        2.0 * (picks * ffn * h) as f64,
    );
    drop((d_x, d_in, d_out, d_wd));

    let d_ox = DevBuf::zeroed(ctx, t * d_o * 2)?;
    let d_oacc = DevBuf::zeroed(ctx, d_o * d_o * 4)?;
    let dt = time_case(ctx, 5, || {
        k.launch_syrk(
            false,
            (d_ox.ptr, 0, 0, d_oacc.ptr),
            (t as u32, d_o as u32, 1, 1, 1),
        )
    })?;
    report(
        &format!("calib_syrk_bf16 dense [d={d_o} T={t}]"),
        dt,
        (t * d_o * d_o) as f64,
    );
    Ok(())
}
