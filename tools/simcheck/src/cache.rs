//! Embeddings keyed by window text, kept between runs.

use crate::fetch::REVISION;
use anyhow::{Result, ensure};
use std::collections::HashMap;
use std::path::Path;

/// Changes with the file layout.
const MAGIC: &[u8; 8] = b"simck001";
/// Keep in step with eg2cpu's `rev` in Cargo.toml.
const EG2CPU: &str = "a80156abbf8ec2e5a9b85ed0e5ff98454c1aaa09";

/// FNV-1a over the model revisions and the window text.
pub fn key(text: &str) -> u64 {
    let parts = [EG2CPU.as_bytes(), REVISION.as_bytes(), text.as_bytes()];
    parts
        .iter()
        .flat_map(|p| p.iter())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
        })
}

#[derive(Default)]
pub struct Cache {
    dim: usize,
    map: HashMap<u64, Vec<f32>>,
}

impl Cache {
    /// Empty when the file is missing or has another layout.
    pub fn load(path: &Path) -> Result<Cache> {
        let Ok(bytes) = std::fs::read(path) else {
            return Ok(Cache::default());
        };
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Ok(Cache::default());
        }
        let dim = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
        let entry = 8 + dim * 4;
        let body = &bytes[12..];
        ensure!(
            dim > 0 && body.len() % entry == 0,
            "{}: truncated",
            path.display()
        );
        let map = body
            .chunks_exact(entry)
            .map(|e| {
                let k = u64::from_le_bytes(e[..8].try_into().expect("8 bytes"));
                let v = e[8..].as_chunks::<4>().0.iter();
                (k, v.map(|f| f32::from_le_bytes(*f)).collect())
            })
            .collect();
        Ok(Cache { dim, map })
    }

    pub fn get(&self, k: u64) -> Option<&Vec<f32>> {
        self.map.get(&k)
    }

    pub fn insert(&mut self, k: u64, v: Vec<f32>) {
        self.dim = v.len();
        self.map.insert(k, v);
    }

    /// Writes the entries in `keep`, dropping the rest.
    pub fn save(&self, path: &Path, keep: &[u64]) -> Result<()> {
        let mut out = MAGIC.to_vec();
        out.extend((self.dim as u32).to_le_bytes());
        let mut keys: Vec<u64> = keep.to_vec();
        keys.sort_unstable();
        keys.dedup();
        for k in keys.into_iter().filter(|k| self.map.contains_key(k)) {
            out.extend(k.to_le_bytes());
            out.extend(self.map[&k].iter().flat_map(|f| f.to_le_bytes()));
        }
        Ok(std::fs::write(path, out)?)
    }
}
