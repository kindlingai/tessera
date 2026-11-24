//! Host-side n-gram embedding table for layer PLE inputs.

use crate::Row;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use tessera_graph::Bf16;
use tessera_mem::Tensor;

/// Configuration for an on-disk PLE table.
#[derive(Clone, Debug)]
pub struct PleConfig {
    pub path: PathBuf,
    pub eos_token: i32,
    /// One per n-gram place, the current token first.
    pub multipliers: Vec<i64>,
    /// Per head, its first row. Bigram heads come first.
    pub offsets: Vec<i64>,
    /// Per head, its row count.
    pub vocab_sizes: Vec<i64>,
    pub row_dim: usize,
    pub cache_rows: usize,
}

impl PleConfig {
    /// Default cache size: 524,288 rows (~168 MB at 320 B/row).
    pub const DEFAULT_CACHE_ROWS: usize = 1 << 19;

    pub fn heads(&self) -> usize {
        self.offsets.len()
    }

    pub fn embed_dim(&self) -> usize {
        self.row_dim * self.heads()
    }

    /// Why the hash parameters disagree, if they do.
    fn misfit(&self) -> Option<String> {
        let (heads, orders) = (self.heads(), self.multipliers.len().saturating_sub(1));
        if orders == 0 {
            return Some(format!(
                "{} multipliers, want 2 or more",
                self.multipliers.len()
            ));
        }
        if self.vocab_sizes.len() != heads {
            return Some(format!(
                "{heads} offsets but {} vocab sizes",
                self.vocab_sizes.len()
            ));
        }
        if heads == 0 || !heads.is_multiple_of(orders) {
            return Some(format!(
                "{heads} heads do not split over {orders} n-gram orders"
            ));
        }
        self.vocab_sizes
            .iter()
            .any(|v| *v <= 0)
            .then(|| "a head has no rows".into())
    }
}

/// Direct-mapped host cache and pread reader for BF16 PLE rows.
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
        if let Some(why) = cfg.misfit() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
        }
        let file = File::open(&cfg.path)?;
        let cap = cfg.cache_rows.max(1).next_power_of_two();
        let row_bytes = cfg.row_dim * 2;
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

    /// Each head's row id for the last token of `history`.
    pub fn hash_token(&self, history: &[i32], ids: &mut [u64]) {
        let eos = i64::from(self.cfg.eos_token);
        let m = &self.cfg.multipliers;
        let per = self.cfg.heads() / (m.len() - 1);
        let (mut mix, mut ended) = (0i64, false);
        for (j, &mj) in m.iter().enumerate() {
            // Tokens at or before an EOS read as EOS.
            let s = match history.len().checked_sub(j + 1) {
                Some(at) if !ended => i64::from(history[at]),
                _ => eos,
            };
            ended |= j > 0 && s == eos;
            mix ^= s.wrapping_mul(mj);
            let heads = j.saturating_sub(1) * per..j * per;
            let sizes = self.cfg.vocab_sizes[heads.clone()].iter();
            let rows = sizes.zip(&self.cfg.offsets[heads.clone()]);
            for (id, (size, offset)) in ids[heads].iter_mut().zip(rows) {
                *id = (mix.rem_euclid(*size) + offset) as u64;
            }
        }
    }

    fn fetch_bf16(&mut self, row_id: u64, dst: &mut [f32]) -> io::Result<()> {
        let dim = self.cfg.row_dim;
        let idx = (row_id.wrapping_mul(0x9E37_79B9_7F4A_7C15) as usize) & self.mask;
        let off = idx * dim;
        if self.tags[idx] != row_id {
            self.file
                .read_exact_at(&mut self.scratch, row_id * self.row_bytes as u64)?;
            let (pairs, _) = self.scratch.as_chunks::<2>();
            for (d, b) in self.data[off..off + dim].iter_mut().zip(pairs) {
                *d = u16::from_le_bytes(*b);
            }
            self.tags[idx] = row_id;
        }
        for (o, &bits) in dst.iter_mut().zip(&self.data[off..off + dim]) {
            *o = f32::from(Bf16(bits));
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
        let mut ids = vec![0u64; self.cfg.heads()];
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
            self.tokens[s][pos] = tok;
            self.hash_token(&self.tokens[s][..=pos], &mut ids);
            for (h, &id) in ids.iter().enumerate() {
                self.fetch_bf16(id, &mut row_buf[h * dim..(h + 1) * dim])?;
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
