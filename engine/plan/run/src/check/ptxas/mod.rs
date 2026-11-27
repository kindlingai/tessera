//! PTX entry listing and `ptxas` resource statistics.

#[cfg(test)]
mod ptxas_test;

use super::CheckConfig;
use crate::Error;
use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::fs::OpenOptions;
use std::hash::BuildHasher;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use tessera_gpukit::Error as GpuError;

static NEXT_TMP: AtomicU64 = AtomicU64::new(0);

/// Resource counts reported by `ptxas -v` for one entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PtxStats {
    pub regs: Option<u32>,
    pub smem: u32,
    pub stack_bytes: Option<u32>,
    pub spill_stores: Option<u32>,
    pub spill_loads: Option<u32>,
    pub barriers: u32,
}

impl PtxStats {
    /// Worst measured spill footprint when all spill fields are present.
    pub fn spill_bytes(self) -> Option<u32> {
        let stack = self.stack_bytes?;
        let stores = self.spill_stores?;
        let loads = self.spill_loads?;
        Some(stack.max(stores).max(loads))
    }

    /// True when any measured spill field is non-zero.
    pub(super) fn has_spill(self) -> bool {
        self.stack_bytes.is_some_and(|n| n > 0)
            || self.spill_stores.is_some_and(|n| n > 0)
            || self.spill_loads.is_some_and(|n| n > 0)
    }
}

/// Extracts `.entry <name>` symbols from compiled PTX text.
pub(super) fn ptx_entries(ptx: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in ptx.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed
            .strip_prefix(".visible .entry ")
            .or_else(|| trimmed.strip_prefix(".entry "))
        else {
            continue;
        };
        let name: String = rest
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '$')
            .collect();
        if !name.is_empty() && !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

pub(super) fn resolve_ptxas(cfg: &CheckConfig) -> Result<Option<PathBuf>, Error> {
    if !cfg.run_ptxas {
        if cfg.max_spills.is_some() {
            return Err(
                GpuError::Cuda("--max-spills cannot be combined with --no-ptxas".into()).into(),
            );
        }
        return Ok(None);
    }
    if let Some(path) = &cfg.ptxas {
        if !path.is_file() {
            return Err(GpuError::Cuda(format!("ptxas not found at {}", path.display())).into());
        }
        return Ok(Some(path.clone()));
    }
    let found = find_ptxas();
    if found.is_none() && cfg.max_spills.is_some() {
        return Err(GpuError::Cuda(
            "ptxas not found; pass --ptxas <path> to enforce --max-spills".into(),
        )
        .into());
    }
    Ok(found)
}

/// Locates `ptxas` on `PATH` or under standard CUDA directories.
pub(super) fn find_ptxas() -> Option<PathBuf> {
    if Command::new("ptxas")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        return Some(PathBuf::from("ptxas"));
    }
    let direct = PathBuf::from("/usr/local/cuda/bin/ptxas");
    if direct.is_file() {
        return Some(direct);
    }
    let mut versions: Vec<PathBuf> = std::fs::read_dir("/usr/local")
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("cuda-"))
        })
        .map(|e| e.path().join("bin/ptxas"))
        .filter(|p| p.is_file())
        .collect();
    versions.sort();
    versions.pop()
}

pub(super) struct TempDir(pub(super) PathBuf);

impl TempDir {
    /// Creates an exclusive private directory under `base`.
    pub(super) fn create(base: &Path) -> Result<Self, String> {
        let pid = std::process::id();
        for attempt in 0..64u32 {
            let seq = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
            let tag = RandomState::new().hash_one((pid, seq, attempt));
            let dir = base.join(format!("tessera-check-{pid}-{tag:016x}"));
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(&dir) {
                Ok(()) => return Ok(Self(dir)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(format!("create ptxas temp dir: {e}")),
            }
        }
        Err("create ptxas temp dir: exhausted unique names".into())
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) fn run_ptxas(
    ptxas: &Path,
    sm_arch: &str,
    ptx: &[u8],
    temp: Option<&Path>,
) -> Result<HashMap<String, PtxStats>, String> {
    let dir = TempDir::create(temp.unwrap_or_else(|| Path::new("/tmp")))?;
    let ptx_path = dir.0.join("input.ptx");
    let cubin_path = dir.0.join("output.cubin");
    // NVRTC includes a trailing NUL byte that ptxas rejects.
    let end = ptx.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&ptx_path)
        .and_then(|mut file| file.write_all(&ptx[..end]))
        .map_err(|e| format!("write ptx temp: {e}"))?;
    let output = Command::new(ptxas)
        .args(["--gpu-name", sm_arch, "-v", "-o"])
        .arg(&cubin_path)
        .arg(&ptx_path)
        .output()
        .map_err(|e| format!("spawn ptxas: {e}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "ptxas ({sm_arch}) failed: {}",
            format!("{stderr}{stdout}").trim()
        ));
    }
    Ok(parse_ptxas(&stderr))
}

/// Extracts the unsigned integer directly before `marker` in `segment`.
fn num_before(segment: &str, marker: &str) -> Option<u32> {
    segment
        .split(marker)
        .next()?
        .split_whitespace()
        .last()?
        .parse()
        .ok()
}

fn merge_max(slot: &mut Option<u32>, value: u32) {
    *slot = Some(slot.map_or(value, |prev| prev.max(value)));
}

/// Parses `ptxas -v` stderr lines into per-entry resource stats.
pub(super) fn parse_ptxas(stderr: &str) -> HashMap<String, PtxStats> {
    let mut map: HashMap<String, PtxStats> = HashMap::new();
    let mut current: Option<(String, PtxStats, bool)> = None;
    let mut in_props = false;
    let flush = |cur: &mut Option<(String, PtxStats, bool)>,
                 map: &mut HashMap<String, PtxStats>| {
        if let Some((name, stats, true)) = cur.take() {
            map.insert(name, stats);
        }
    };
    for line in stderr.lines().map(str::trim) {
        if let Some(name) = line
            .split("Compiling entry function '")
            .nth(1)
            .and_then(|r| r.split('\'').next())
        {
            flush(&mut current, &mut map);
            current = Some((name.to_string(), PtxStats::default(), false));
            in_props = false;
            continue;
        }
        let Some((ref entry, ref mut stats, ref mut seen)) = current else {
            continue;
        };
        if let Some(prop_fn) = line.split("Function properties for ").nth(1) {
            in_props = prop_fn.trim() == entry;
            continue;
        }
        if in_props && line.contains("stack frame") {
            for part in line.split(',') {
                for (tag, dst) in [
                    ("bytes stack frame", &mut stats.stack_bytes),
                    ("bytes spill stores", &mut stats.spill_stores),
                    ("bytes spill loads", &mut stats.spill_loads),
                ] {
                    if let Some(n) = num_before(part, tag) {
                        merge_max(dst, n);
                        *seen = true;
                    }
                }
            }
            in_props = false;
            continue;
        }
        if let Some(rest) = line.split("Used ").nth(1) {
            for part in rest.split(',') {
                if let Some(n) = num_before(part, "registers") {
                    stats.regs = Some(n);
                    *seen = true;
                } else if let Some(n) = num_before(part, "bytes smem") {
                    stats.smem = n;
                    *seen = true;
                } else if let Some(n) = num_before(part, "barriers") {
                    stats.barriers = n;
                    *seen = true;
                } else if let Some(n) = num_before(part, "bytes cumulative stack size") {
                    merge_max(&mut stats.stack_bytes, n);
                    *seen = true;
                }
            }
        }
    }
    flush(&mut current, &mut map);
    map
}
