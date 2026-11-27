//! Host-side n-gram embedding table for layer PLE inputs.

use crate::Row;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use tessera_graph::{Bf16, E4M3};
use tessera_mem::Tensor;

/// Number of n-gram heads (8 bigram + 8 trigram).
pub const HEADS: usize = 16;

/// Configuration for an on-disk PLE table.
#[derive(Clone, Debug)]
pub struct PleConfig {
    pub path: PathBuf,
    /// Byte offset where table rows begin in `path`.
    pub data_offset: u64,
    /// Scale for FP8 E4M3 rows, or `None` for BF16.
    pub fp8_scale: Option<f32>,
    pub eos_token: i32,
    pub multipliers: [i64; 3],
    pub offsets: [i64; HEADS],
    pub vocab_sizes: [i64; HEADS],
    pub row_dim: usize,
    pub cache_rows: usize,
}

impl PleConfig {
    /// Default cache size: 524,288 rows (~168 MB at 320 B/row).
    pub const DEFAULT_CACHE_ROWS: usize = 1 << 19;

    pub fn embed_dim(&self) -> usize {
        self.row_dim * HEADS
    }
}

/// Direct-mapped host cache and pread reader for PLE rows.
pub struct PleStore {
    file: File,
    cfg: PleConfig,
    row_bytes: usize,
    tags: Vec<u64>,
    data: Vec<u16>,
    mask: usize,
    scratch: Vec<u8>,
    tokens: Vec<Vec<i32>>,
    prev_pos: Vec<u32>,
    last_row: Vec<Vec<f32>>,
}

impl PleStore {
    pub fn open(cfg: PleConfig) -> io::Result<Self> {
        let file = File::open(&cfg.path)?;
        let cap = cfg.cache_rows.max(1).next_power_of_two();
        let row_bytes = if cfg.fp8_scale.is_some() {
            cfg.row_dim
        } else {
            cfg.row_dim * 2
        };
        Ok(Self {
            file,
            row_bytes,
            tags: vec![u64::MAX; cap],
            data: vec![0u16; cap * cfg.row_dim],
            mask: cap - 1,
            scratch: vec![0u8; row_bytes],
            tokens: Vec::new(),
            prev_pos: Vec::new(),
            last_row: Vec::new(),
            cfg,
        })
    }

    /// Computes global row ids for `tok` and `ctx`.
    pub fn hash_token(&self, tok: i32, ctx: [i32; 2]) -> [u64; HEADS] {
        let eos = self.cfg.eos_token;
        let s0 = i64::from(tok);
        let (s1, s2) = if ctx[1] == eos {
            (i64::from(eos), i64::from(eos))
        } else if ctx[0] == eos {
            (i64::from(ctx[1]), i64::from(eos))
        } else {
            (i64::from(ctx[1]), i64::from(ctx[0]))
        };
        let m = &self.cfg.multipliers;
        let m2 = s0.wrapping_mul(m[0]) ^ s1.wrapping_mul(m[1]);
        let m3 = m2 ^ s2.wrapping_mul(m[2]);
        std::array::from_fn(|h| {
            let mix = if h < 8 { m2 } else { m3 };
            (mix.rem_euclid(self.cfg.vocab_sizes[h]) + self.cfg.offsets[h]) as u64
        })
    }

    fn fetch_row(&mut self, row_id: u64, dst: &mut [f32]) -> io::Result<()> {
        let dim = self.cfg.row_dim;
        let idx = (row_id.wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize) & self.mask;
        let off = idx * dim;
        if self.tags[idx] != row_id {
            let at = self.cfg.data_offset + row_id * self.row_bytes as u64;
            self.file.read_exact_at(&mut self.scratch, at)?;
            if self.cfg.fp8_scale.is_some() {
                for (d, &b) in self.data[off..off + dim].iter_mut().zip(&self.scratch) {
                    *d = u16::from(b);
                }
            } else {
                let (pairs, _) = self.scratch.as_chunks::<2>();
                for (d, b) in self.data[off..off + dim].iter_mut().zip(pairs) {
                    *d = u16::from_le_bytes(*b);
                }
            }
            self.tags[idx] = row_id;
        }
        if let Some(scale) = self.cfg.fp8_scale {
            for (o, &bits) in dst.iter_mut().zip(&self.data[off..off + dim]) {
                *o = f32::from(E4M3(bits as u8)) * scale;
            }
        } else {
            for (o, &bits) in dst.iter_mut().zip(&self.data[off..off + dim]) {
                *o = f32::from(Bf16(bits));
            }
        }
        Ok(())
    }

    fn ensure_slot(&mut self, slot: usize) {
        if slot < self.tokens.len() {
            return;
        }
        let n = slot + 1;
        let width = self.cfg.embed_dim();
        self.tokens.resize_with(n, Vec::new);
        self.prev_pos.resize(n, u32::MAX);
        self.last_row.resize_with(n, || vec![0.0; width]);
    }

    /// Fills `[filled, embed_dim]` f32 embeddings for `rows`, padding the tail.
    pub fn fill(&mut self, rows: &[Row], filled: usize) -> io::Result<Tensor> {
        let width = self.cfg.embed_dim();
        let dim = self.cfg.row_dim;
        let mut out = vec![0.0f32; filled * width];
        let mut row_buf = vec![0.0f32; width];
        // Repeats count only within one launch.
        self.prev_pos.fill(u32::MAX);
        for (r, row) in rows.iter().enumerate() {
            let s = row.slot as usize;
            self.ensure_slot(s);
            let dst = &mut out[r * width..(r + 1) * width];
            let pos = row.position as usize;
            let tok = row.token as i32;
            if row.position == self.prev_pos[s] && self.tokens[s].get(pos).copied() == Some(tok) {
                dst.copy_from_slice(&self.last_row[s]);
                continue;
            }
            let eos = self.cfg.eos_token;
            if pos == 0 {
                self.tokens[s].clear();
            }
            if self.tokens[s].len() <= pos {
                self.tokens[s].resize(pos + 1, eos);
            }
            let ctx = [
                if pos >= 2 {
                    self.tokens[s][pos - 2]
                } else {
                    eos
                },
                if pos >= 1 {
                    self.tokens[s][pos - 1]
                } else {
                    eos
                },
            ];
            self.tokens[s][pos] = tok;
            let ids = self.hash_token(tok, ctx);
            for (h, &id) in ids.iter().enumerate() {
                self.fetch_row(id, &mut row_buf[h * dim..(h + 1) * dim])?;
            }
            self.last_row[s].copy_from_slice(&row_buf);
            self.prev_pos[s] = row.position;
            dst.copy_from_slice(&row_buf);
        }
        if let Some(last) = rows.len().checked_sub(1) {
            let last_off = last * width;
            for r in rows.len()..filled {
                out.copy_within(last_off..last_off + width, r * width);
            }
        }
        Ok(Tensor::f32(vec![filled, width], out))
    }
}
