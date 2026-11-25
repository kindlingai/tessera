//! Host Gram sums, saved inputs, routing logs and their files.

use crate::state::AsyncWriter;
use std::collections::BTreeMap;
use std::path::Path;
use tessera_checkpoint::hessian::{
    DenseEntry, ExpertShard, HessianMeta, Inputs, dense_name, expert_name, inputs_name,
    pack_tiles_bytes, route_name, triu_len, write_dense, write_experts, write_inputs, write_route,
};
use tessera_device::calib::{sort_cpu, syrk_dense_cpu, syrk_tiles_cpu};
use tessera_graph::{Bf16, CalibTap, F16};

/// Maps a dense calibration tap to its `HESSIAN.md` tensor name.
pub(crate) fn tap_name(tap: CalibTap) -> Option<&'static str> {
    match tap {
        CalibTap::AttnIn => Some("attn_in"),
        CalibTap::QResid => Some("q_resid"),
        CalibTap::KvLatent => Some("kv_latent"),
        CalibTap::OIn => Some("o_in"),
        CalibTap::MlpIn => Some("mlp_in"),
        CalibTap::DenseDownIn => Some("dense_down_in"),
        CalibTap::SharedDownIn => Some("shared_down_in"),
        CalibTap::RouterIds | CalibTap::RouterGates | CalibTap::RoutedInner => None,
    }
}

/// Routing and expert taps of one batch, by kind.
pub(crate) struct Taps<T> {
    pub mlp: Option<T>,
    pub ids: Option<T>,
    pub gates: Option<T>,
    pub inner: Option<T>,
}

impl<T> Taps<T> {
    /// Sorts `taps` and passes each dense one to `dense`.
    pub fn sort(
        taps: impl IntoIterator<Item = (CalibTap, T)>,
        mut dense: impl FnMut(&'static str, &T) -> Result<(), String>,
    ) -> Result<Self, String> {
        let mut out = Self {
            mlp: None,
            ids: None,
            gates: None,
            inner: None,
        };
        for (tap, t) in taps {
            match tap {
                CalibTap::RouterIds => out.ids = Some(t),
                CalibTap::RouterGates => out.gates = Some(t),
                CalibTap::RoutedInner => out.inner = Some(t),
                other => {
                    if let Some(name) = tap_name(other) {
                        dense(name, &t)?;
                    }
                    if other == CalibTap::MlpIn {
                        out.mlp = Some(t);
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Where one layer's files go.
pub(crate) struct Files<'a> {
    pub out: &'a Path,
    pub layer: usize,
    pub meta: &'a HessianMeta,
    pub writer: &'a AsyncWriter,
}

/// A dense Gram matrix: name, width, values and rows.
pub(crate) type Dense = (&'static str, usize, Vec<f32>, i64);

/// Queues one dense file. It must hold `mlp_in`.
pub(crate) fn submit_dense(files: &Files, heldout: bool, mats: Vec<Dense>) -> Result<(), String> {
    if mats.is_empty() {
        return Ok(());
    }
    if !mats.iter().any(|m| m.0 == "mlp_in") {
        return Err(format!("layer {}: dense taps lack mlp_in", files.layer));
    }
    let (layer, meta) = (files.layer, files.meta.clone());
    let path = files.out.join(dense_name(layer, heldout));
    files.writer.submit(move || {
        let entries: Vec<DenseEntry<'_>> = (mats.iter())
            .map(|(name, dim, data, count)| DenseEntry {
                name,
                dim: *dim,
                data,
                count: *count,
            })
            .collect();
        write_dense(&path, layer, &entries, heldout, &meta).map_err(|e| e.to_string())
    })
}

/// Tile-ordered FP32 sums of experts `first..` for one file.
pub(crate) struct TileShard {
    pub first: usize,
    pub dims: (usize, usize),
    pub gu: Vec<u8>,
    pub down: Vec<u8>,
    pub counts: Vec<i64>,
}

/// Packs `shard` into upper triangles and queues its file.
pub(crate) fn submit_experts(files: &Files, tile: usize, shard: TileShard) -> Result<(), String> {
    let (layer, meta) = (files.layer, files.meta.clone());
    let n = shard.counts.len();
    let path = files
        .out
        .join(expert_name(layer, shard.first, shard.first + n - 1));
    files.writer.submit(move || {
        let (h, f) = shard.dims;
        let pack = |tiles: &[u8], d: usize| {
            let (tile_b, tri_b) = (tiles.len() / n, triu_len(d) * 4);
            let mut tri = vec![0u8; n * tri_b];
            for i in 0..n {
                let dst = &mut tri[i * tri_b..(i + 1) * tri_b];
                pack_tiles_bytes(&tiles[i * tile_b..(i + 1) * tile_b], d, tile, dst);
            }
            tri
        };
        let file = ExpertShard {
            layer,
            first: shard.first,
            gu: &pack(&shard.gu, h),
            down: &pack(&shard.down, f),
            counts: &shard.counts,
            dims: shard.dims,
        };
        write_experts(&path, &file, &meta).map_err(|e| e.to_string())
    })
}

fn f32_bytes(vals: &[f64]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&v| (v as f32).to_le_bytes())
        .collect()
}

/// Dense Gram matrices for one layer, summed in f64.
#[derive(Default)]
pub(crate) struct DenseAccum {
    mats: BTreeMap<&'static str, (usize, Vec<f64>, i64)>,
}

impl DenseAccum {
    pub fn add(&mut self, name: &'static str, x: &[f32], rows: usize, dim: usize) {
        let entry = (self.mats)
            .entry(name)
            .or_insert_with(|| (dim, vec![0.0; dim * dim], 0));
        assert_eq!(entry.0, dim);
        syrk_dense_cpu(x, rows, dim, &mut entry.1);
        entry.2 += rows as i64;
    }

    pub fn flush(self, files: &Files, heldout: bool) -> Result<(), String> {
        let mats = (self.mats.into_iter())
            .map(|(name, (dim, sum, count))| {
                (name, dim, sum.iter().map(|&v| v as f32).collect(), count)
            })
            .collect();
        submit_dense(files, heldout, mats)
    }
}

/// Packed upper-triangle tile sums for routed experts, in f64.
pub(crate) struct ExpertAccum {
    experts: usize,
    dims: (usize, usize),
    tile: usize,
    gu: Vec<f64>,
    down: Vec<f64>,
    counts: Vec<i64>,
}

/// Elements in the upper-triangle tiles of a `d`-wide matrix.
pub(crate) fn tiled(d: usize, tile: usize) -> usize {
    let blocks = d / tile;
    blocks * (blocks + 1) / 2 * tile * tile
}

impl ExpertAccum {
    pub fn new(experts: usize, hidden: usize, ffn: usize, tile: usize) -> Self {
        Self {
            experts,
            dims: (hidden, ffn),
            tile,
            gu: vec![0.0; experts * tiled(hidden, tile)],
            down: vec![0.0; experts * tiled(ffn, tile)],
            counts: vec![0; experts],
        }
    }

    pub fn add(&mut self, mlp_in: &[f32], inner: &[f32], ids: &[u32], top_k: usize) {
        let (offsets, perm, chunk_counts) = sort_cpu(ids, self.experts, 0);
        for (dst, src) in self.counts.iter_mut().zip(chunk_counts) {
            *dst += src;
        }
        let (h, f) = self.dims;
        syrk_tiles_cpu(mlp_in, &perm, &offsets, h, self.tile, top_k, &mut self.gu);
        syrk_tiles_cpu(inner, &perm, &offsets, f, self.tile, 1, &mut self.down);
    }

    pub fn flush(self, files: &Files, per_file: usize) -> Result<(), String> {
        let (gu_t, dn_t) = (tiled(self.dims.0, self.tile), tiled(self.dims.1, self.tile));
        for a in (0..self.experts).step_by(per_file) {
            let b = (a + per_file).min(self.experts);
            let shard = TileShard {
                first: a,
                dims: self.dims,
                gu: f32_bytes(&self.gu[a * gu_t..b * gu_t]),
                down: f32_bytes(&self.down[a * dn_t..b * dn_t]),
                counts: self.counts[a..b].to_vec(),
            };
            submit_experts(files, self.tile, shard)?;
        }
        Ok(())
    }
}

/// Held-out MoE inputs (`inputs-LLL.safetensors`).
#[derive(Default)]
pub(crate) struct SavedInputs {
    x_bf16: Vec<u16>,
    idx: Vec<i32>,
    w: Vec<f32>,
}

impl SavedInputs {
    pub fn push(&mut self, x: &[f32], ids: &[u32], gates: &[f32]) {
        self.x_bf16.extend(x.iter().map(|&v| Bf16::from(v).0));
        self.idx.extend(ids.iter().map(|&id| id as i32));
        self.w.extend_from_slice(gates);
    }

    pub fn flush(
        self,
        files: &Files,
        heldout_seqs: usize,
        dims: (usize, usize),
    ) -> Result<(), String> {
        if self.x_bf16.is_empty() {
            return Ok(());
        }
        let (layer, meta) = (files.layer, files.meta.clone());
        let path = files.out.join(inputs_name(layer));
        files.writer.submit(move || {
            let inputs = Inputs {
                layer,
                heldout_seqs,
                x_bf16: &self.x_bf16,
                topk_idx: &self.idx,
                topk_w: &self.w,
                hidden: dims.0,
                top_k: dims.1,
            };
            write_inputs(&path, &inputs, &meta).map_err(|e| e.to_string())
        })
    }
}

/// Per-token routing log (`route-LLL.safetensors`).
#[derive(Default)]
pub(crate) struct RouteLog {
    idx: Vec<i16>,
    w_f16: Vec<u16>,
}

impl RouteLog {
    pub fn push(&mut self, ids: &[u32], gates: &[f32]) {
        self.idx.extend(ids.iter().map(|&id| id as i16));
        self.w_f16.extend(gates.iter().map(|&g| F16::from(g).0));
    }

    pub fn flush(self, files: &Files, shape: [usize; 3]) -> Result<(), String> {
        if self.idx.is_empty() {
            return Ok(());
        }
        let (layer, meta) = (files.layer, files.meta.clone());
        let path = files.out.join(route_name(layer));
        files.writer.submit(move || {
            write_route(&path, layer, "all", &self.idx, &self.w_f16, shape, &meta)
                .map_err(|e| e.to_string())
        })
    }
}
