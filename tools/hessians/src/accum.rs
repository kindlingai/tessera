//! Runner-owned packed upper-triangle and dense Gram accumulators.

use crate::state::AsyncWriter;
use std::collections::BTreeMap;
use std::path::Path;
use tessera_checkpoint::hessian::{
    DenseEntry, HessianMeta, dense_name, expert_name, inputs_name, pack_tiles, route_name,
    triu_len, write_dense, write_experts, write_inputs, write_route,
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
        CalibTap::EhIn => Some("eh_in"),
        CalibTap::RouterIds | CalibTap::RouterGates | CalibTap::RoutedInner => None,
    }
}

/// Full symmetric dense Gram matrices accumulated for one layer.
#[derive(Default)]
pub(crate) struct DenseAccum {
    pub mats: BTreeMap<String, (usize, Vec<f32>, i64)>,
}

impl DenseAccum {
    pub fn add(&mut self, name: &str, x: &[f32], rows: usize, dim: usize) {
        let entry = self
            .mats
            .entry(name.to_string())
            .or_insert_with(|| (dim, vec![0.0f32; dim * dim], 0));
        assert_eq!(entry.0, dim);
        syrk_dense_cpu(x, rows, dim, &mut entry.1);
        entry.2 += rows as i64;
    }

    pub fn flush(
        self,
        out_dir: &Path,
        layer: usize,
        heldout: bool,
        meta: &HessianMeta,
        writer: &AsyncWriter,
    ) -> Result<(), String> {
        if self.mats.is_empty() {
            return Ok(());
        }
        let path = out_dir.join(dense_name(layer, heldout));
        let meta = meta.clone();
        writer.submit(move || {
            let entries: Vec<DenseEntry<'_>> = self
                .mats
                .iter()
                .map(|(name, (dim, data, count))| DenseEntry {
                    name,
                    dim: *dim,
                    data,
                    count: *count,
                })
                .collect();
            write_dense(&path, layer, &entries, heldout, &meta).map_err(|e| e.to_string())
        })
    }
}

/// Packed upper-triangle tile accumulators for routed MoE experts.
pub(crate) struct ExpertAccum {
    pub experts: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub tile: usize,
    pub gu: Vec<f32>,
    pub down: Vec<f32>,
    pub counts: Vec<i64>,
}

impl ExpertAccum {
    pub fn new(experts: usize, hidden: usize, ffn: usize, tile: usize) -> Self {
        let gu_tiles = (hidden / tile) * (hidden / tile + 1) / 2;
        let dn_tiles = (ffn / tile) * (ffn / tile + 1) / 2;
        Self {
            experts,
            hidden,
            ffn,
            tile,
            gu: vec![0.0f32; experts * gu_tiles * tile * tile],
            down: vec![0.0f32; experts * dn_tiles * tile * tile],
            counts: vec![0i64; experts],
        }
    }

    pub fn add(&mut self, mlp_in: &[f32], inner: &[f32], ids: &[u32], top_k: usize) {
        let (offsets, perm, chunk_counts) = sort_cpu(ids, self.experts, 0);
        for (dst, src) in self.counts.iter_mut().zip(chunk_counts) {
            *dst += src;
        }
        syrk_tiles_cpu(
            mlp_in,
            &perm,
            &offsets,
            self.hidden,
            self.tile,
            top_k,
            &mut self.gu,
        );
        syrk_tiles_cpu(
            inner,
            &perm,
            &offsets,
            self.ffn,
            self.tile,
            1,
            &mut self.down,
        );
    }

    pub fn flush(
        self,
        out_dir: &Path,
        layer: usize,
        per_file: usize,
        meta: &HessianMeta,
        writer: &AsyncWriter,
    ) -> Result<(), String> {
        let gu_t = (self.hidden / self.tile) * (self.hidden / self.tile + 1) / 2;
        let dn_t = (self.ffn / self.tile) * (self.ffn / self.tile + 1) / 2;
        let tile_area = self.tile * self.tile;
        let th = triu_len(self.hidden);
        let ti = triu_len(self.ffn);
        let mut a = 0usize;
        while a < self.experts {
            let b = (a + per_file).min(self.experts);
            let n = b - a;
            let mut gu_shard = Vec::with_capacity(n * th);
            let mut dn_shard = Vec::with_capacity(n * ti);
            for e in a..b {
                let g0 = e * gu_t * tile_area;
                let d0 = e * dn_t * tile_area;
                gu_shard.extend(pack_tiles(
                    &self.gu[g0..g0 + gu_t * tile_area],
                    self.hidden,
                    self.tile,
                ));
                dn_shard.extend(pack_tiles(
                    &self.down[d0..d0 + dn_t * tile_area],
                    self.ffn,
                    self.tile,
                ));
            }
            let counts = self.counts[a..b].to_vec();
            let path = out_dir.join(expert_name(layer, a, b - 1));
            let meta = meta.clone();
            let dims = (self.hidden, self.ffn);
            writer.submit(move || {
                write_experts(&path, layer, a, &gu_shard, &dn_shard, &counts, dims, &meta)
                    .map_err(|e| e.to_string())
            })?;
            a = b;
        }
        Ok(())
    }
}

/// Captured held-out MoE inputs (`inputs-LLL.safetensors`).
#[derive(Default)]
pub(crate) struct SavedInputs {
    pub x_bf16: Vec<u16>,
    pub idx: Vec<i32>,
    pub w: Vec<f32>,
}

impl SavedInputs {
    pub fn push(&mut self, x: &[f32], ids: &[u32], gates: &[f32]) {
        for &v in x {
            self.x_bf16.push(Bf16::from(v).0);
        }
        for &id in ids {
            self.idx.push(id as i32);
        }
        self.w.extend_from_slice(gates);
    }

    pub fn flush(
        self,
        out_dir: &Path,
        layer: usize,
        heldout_seqs: usize,
        dims: (usize, usize),
        meta: &HessianMeta,
        writer: &AsyncWriter,
    ) -> Result<(), String> {
        if self.x_bf16.is_empty() {
            return Ok(());
        }
        let (hidden, top_k) = dims;
        let path = out_dir.join(inputs_name(layer));
        let meta = meta.clone();
        writer.submit(move || {
            write_inputs(
                &path,
                layer,
                heldout_seqs,
                &self.x_bf16,
                &self.idx,
                &self.w,
                hidden,
                top_k,
                &meta,
            )
            .map_err(|e| e.to_string())
        })
    }
}

/// Optional per-token routing log (`route-LLL.safetensors`).
#[derive(Default)]
pub(crate) struct RouteLog {
    pub idx: Vec<i16>,
    pub w_f16: Vec<u16>,
}

impl RouteLog {
    pub fn push(&mut self, ids: &[u32], gates: &[f32]) {
        for &id in ids {
            self.idx.push(id as i16);
        }
        for &g in gates {
            self.w_f16.push(F16::from(g).0);
        }
    }

    pub fn flush(
        self,
        out_dir: &Path,
        layer: usize,
        shape: [usize; 3],
        meta: &HessianMeta,
        writer: &AsyncWriter,
    ) -> Result<(), String> {
        if self.idx.is_empty() {
            return Ok(());
        }
        let path = out_dir.join(route_name(layer));
        let meta = meta.clone();
        writer.submit(move || {
            write_route(&path, layer, "all", &self.idx, &self.w_f16, shape, &meta)
                .map_err(|e| e.to_string())
        })
    }
}
