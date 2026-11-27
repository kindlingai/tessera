//! Offline CUDA kernel compilation and spill verification without a GPU.

#[cfg(test)]
mod check_test;

use crate::Error;
use crate::kernels::group;
use crate::overrides;
use crate::run::Run;
use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tessera_engine_settings::{Backend, Settings};
use tessera_gpukit::platform::{Asked, Platform};
use tessera_gpukit::{Error as GpuError, Limits};
use tessera_lower::{Plan, Source, Step};

/// Resource counts reported by `ptxas -v` for one entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PtxStats {
    pub regs: u32,
    pub smem: u32,
    pub stack_bytes: u32,
    pub spill_stores: u32,
    pub spill_loads: u32,
    pub barriers: u32,
}

impl PtxStats {
    /// Worst local-memory spill footprint reported for this entry.
    pub fn spill_bytes(self) -> u32 {
        self.stack_bytes
            .max(self.spill_stores)
            .max(self.spill_loads)
    }
}

/// Verification result for one compiled kernel entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelResult {
    pub entry: String,
    pub rule: String,
    pub kind: &'static str,
    pub dynamic_shared: usize,
    pub threads: usize,
    pub ptx_bytes: usize,
    pub stats: Option<PtxStats>,
    pub error: Option<String>,
}

/// Options for offline NVRTC and `ptxas` verification.
#[derive(Clone, Debug)]
pub struct CheckConfig {
    pub arch: String,
    pub max_spills: Option<u32>,
    pub ptxas: Option<PathBuf>,
    pub run_ptxas: bool,
    pub overrides: Option<PathBuf>,
    pub temp: Option<PathBuf>,
}

impl Default for CheckConfig {
    fn default() -> Self {
        Self {
            arch: "sm_121".into(),
            max_spills: None,
            ptxas: None,
            run_ptxas: true,
            overrides: None,
            temp: None,
        }
    }
}

/// Summary of an offline kernel verification pass.
#[derive(Clone, Debug)]
pub struct CheckReport {
    pub arch: String,
    pub nvrtc: Option<String>,
    pub ptxas: Option<String>,
    pub kernels: Vec<KernelResult>,
}

impl CheckReport {
    /// True when every verified kernel succeeded within spill limits.
    pub fn ok(&self) -> bool {
        self.kernels.iter().all(|k| k.error.is_none())
    }

    /// Count of kernels that failed compilation or spill checks.
    pub fn failed(&self) -> usize {
        self.kernels.iter().filter(|k| k.error.is_some()).count()
    }

    /// Count of kernels with non-zero local spills.
    pub fn spilling(&self) -> usize {
        self.kernels
            .iter()
            .filter(|k| k.stats.is_some_and(|s| s.spill_bytes() > 0))
            .count()
    }
}

/// Applies disk-cache and NVRTC flag settings before offline checks.
pub fn configure(settings: &Settings) {
    crate::configure(settings);
}

/// Target architecture string passed to `ptxas --gpu-name`.
fn sm_target(target: &str) -> Result<String, Error> {
    let (major, minor, suffix) = tessera_gpukit::backend::cuda_arch(target)?;
    Ok(format!("sm_{major}{minor}{suffix}"))
}

/// Builds a synthetic CUDA `Platform` for offline lowering.
fn target_platform(target: &str) -> Result<Platform, Error> {
    let (major, minor, _) = tessera_gpukit::backend::cuda_arch(target)?;
    let arch = format!("sm_{major}{minor}");
    let integrated = major == 12 && minor == 1;
    let limits = Limits {
        lanes: 32,
        group_bytes: 48 * 1024,
        dynamic_group_bytes: 99 * 1024,
        group_threads: 1024,
        compiler: true,
        pdl: false,
    };
    Ok(Platform::new(Asked {
        backend: Backend::Cuda.name(),
        arch: arch.clone(),
        name: format!("offline {arch}"),
        driver: "13.0".into(),
        limits,
        units: 48,
        bandwidth: 273e9,
        integrated,
        memory: 128u64 << 30,
    }))
}

#[derive(Clone)]
struct Collected {
    entry: String,
    rule: String,
    kind: &'static str,
    source: String,
    dynamic_shared: usize,
    threads: usize,
}

struct Captured {
    platform: Platform,
    entries: BTreeMap<(String, u64), Collected>,
}

static ACTIVE: Mutex<Option<Captured>> = Mutex::new(None);
static NEXT_TMP: AtomicU64 = AtomicU64::new(0);

/// Guard holding an active offline kernel collection session.
pub struct Session {
    live: bool,
}

impl Session {
    /// Ends collection and verifies all captured kernels.
    pub fn finish(mut self, cfg: &CheckConfig) -> Result<CheckReport, Error> {
        self.live = false;
        let captured = ACTIVE.lock().expect("check session").take();
        let entries: Vec<Collected> = captured
            .map(|c| c.entries.into_values().collect())
            .unwrap_or_default();
        verify_collected(entries, cfg)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if self.live
            && let Ok(mut guard) = ACTIVE.lock()
        {
            *guard = None;
        }
    }
}

/// Starts offline kernel collection for CUDA `target`.
pub fn start(target: &str) -> Result<Session, Error> {
    let platform = target_platform(target)?;
    let mut guard = ACTIVE.lock().expect("check session");
    *guard = Some(Captured {
        platform,
        entries: BTreeMap::new(),
    });
    Ok(Session { live: true })
}

/// Whether offline kernel verification mode is active.
pub(crate) fn active() -> bool {
    ACTIVE.lock().expect("check session").is_some()
}

/// Active synthetic GPU backend while checking offline.
pub(crate) fn backend() -> Option<Backend> {
    active().then_some(Backend::Cuda)
}

/// Active synthetic GPU platform while checking offline.
pub(crate) fn platform() -> Option<Platform> {
    ACTIVE
        .lock()
        .expect("check session")
        .as_ref()
        .map(|c| c.platform.clone())
}

/// Opens a capture runner if offline verification is active.
pub(crate) fn open(plan: &Plan) -> Option<Box<dyn Run>> {
    let platform = platform()?;
    Some(crate::sizer::passive(
        plan,
        platform,
        "offline check",
        |p| {
            let mut guard = ACTIVE.lock().expect("check session");
            if let Some(captured) = guard.as_mut() {
                collect_plan(p, &mut captured.entries);
            }
        },
    ))
}

fn source_hash(source: &str) -> u64 {
    let mut h = DefaultHasher::new();
    source.hash(&mut h);
    h.finish()
}

fn collect_step(step: &Step, out: &mut BTreeMap<(String, u64), Collected>) {
    let src = step.kernel.source();
    let entry = overrides::entry(&src).to_string();
    let kind = match src {
        Source::Fixed { .. } => "fixed",
        Source::Made(_) => "made",
    };
    let source = overrides::text(step, Backend::Cuda);
    if entry.is_empty() && source.is_empty() {
        return;
    }
    let key = (entry.clone(), source_hash(&source));
    let dynamic_shared = step
        .kernel
        .resident()
        .map_or(step.kernel.needs().dynamic_shared, |r| r.shared as usize);
    let threads = step
        .kernel
        .resident()
        .map_or_else(|| group(step), |r| r.threads as usize);
    out.entry(key)
        .and_modify(|item| {
            item.dynamic_shared = item.dynamic_shared.max(dynamic_shared);
            item.threads = item.threads.max(threads);
        })
        .or_insert(Collected {
            entry,
            rule: step.rule.to_string(),
            kind,
            source,
            dynamic_shared,
            threads,
        });
}

fn collect_plan(plan: &Plan, out: &mut BTreeMap<(String, u64), Collected>) {
    for step in &plan.steps {
        collect_step(step, out);
    }
}

/// Verifies one standalone CUDA file offline.
pub fn check_file(
    path: &Path,
    entry: Option<&str>,
    cfg: &CheckConfig,
) -> Result<CheckReport, Error> {
    let source =
        std::fs::read_to_string(path).map_err(|e| Error::KernelFile(path.to_path_buf(), e))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("kernel.cu");
    check_source(name, entry, &source, cfg)
}

/// Verifies one standalone CUDA source string offline.
fn check_source(
    name: &str,
    entry: Option<&str>,
    source: &str,
    cfg: &CheckConfig,
) -> Result<CheckReport, Error> {
    let sm = sm_target(&cfg.arch)?;
    let ptxas = resolve_ptxas(cfg)?;
    let nvrtc = nvrtc_tag();
    let ptxas_str = ptxas.as_ref().map(|p| p.display().to_string());
    let mut kernels = Vec::new();
    let unit = match nvrtc_compile(&cfg.arch, name, source) {
        Ok(ptx) => ptx,
        Err(err) => {
            kernels.push(KernelResult {
                entry: entry.unwrap_or(name).to_string(),
                rule: "file".into(),
                kind: "file",
                dynamic_shared: 0,
                threads: 0,
                ptx_bytes: 0,
                stats: None,
                error: Some(err),
            });
            return Ok(CheckReport {
                arch: sm,
                nvrtc,
                ptxas: ptxas_str,
                kernels,
            });
        }
    };
    let ptx_text = String::from_utf8_lossy(&unit);
    let found = ptx_entries(&ptx_text);
    let wanted: Vec<String> = match entry {
        Some(e) => vec![e.to_string()],
        None if !found.is_empty() => found.clone(),
        None => vec![name.to_string()],
    };
    let assembled = ptxas
        .as_deref()
        .map(|bin| run_ptxas(bin, &sm, &unit, cfg.temp.as_deref()));
    for want in wanted {
        let missing = entry.is_some() && !found.iter().any(|e| e == &want);
        let (stats, err) = eval_entry(&want, missing, &found, assembled.as_ref(), cfg.max_spills);
        kernels.push(KernelResult {
            entry: want,
            rule: "file".into(),
            kind: "file",
            dynamic_shared: 0,
            threads: 0,
            ptx_bytes: unit.len(),
            stats,
            error: err,
        });
    }
    Ok(CheckReport {
        arch: sm,
        nvrtc,
        ptxas: ptxas_str,
        kernels,
    })
}

fn verify_collected(mut entries: Vec<Collected>, cfg: &CheckConfig) -> Result<CheckReport, Error> {
    let sm = sm_target(&cfg.arch)?;
    let ptxas = resolve_ptxas(cfg)?;
    let nvrtc = nvrtc_tag();
    let ptxas_str = ptxas.as_ref().map(|p| p.display().to_string());
    if let Some(dir) = &cfg.overrides {
        for item in &mut entries {
            let path = dir.join(format!("{}.cu", item.entry));
            if let Ok(text) = std::fs::read_to_string(&path) {
                item.source = text;
                item.kind = "override";
            }
        }
    }
    // Group entries sharing identical source text to compile once.
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (idx, item) in entries.iter().enumerate() {
        groups
            .entry(source_hash(&item.source))
            .or_default()
            .push(idx);
    }
    let mut results: Vec<Option<KernelResult>> = vec![None; entries.len()];
    for indices in groups.values() {
        let first = &entries[indices[0]];
        if first.source.is_empty() {
            for &idx in indices {
                let item = &entries[idx];
                results[idx] = Some(KernelResult {
                    entry: item.entry.clone(),
                    rule: item.rule.clone(),
                    kind: item.kind,
                    dynamic_shared: item.dynamic_shared,
                    threads: item.threads,
                    ptx_bytes: 0,
                    stats: None,
                    error: Some(format!("kernel `{}` has empty CUDA source", item.entry)),
                });
            }
            continue;
        }
        let unit = nvrtc_compile(&cfg.arch, &first.entry, &first.source);
        let (ptx_bytes, found, assembled, nvrtc_err) = match unit {
            Ok(ptx) => {
                let found = ptx_entries(&String::from_utf8_lossy(&ptx));
                let asm = ptxas
                    .as_deref()
                    .map(|bin| run_ptxas(bin, &sm, &ptx, cfg.temp.as_deref()));
                (ptx.len(), found, asm, None)
            }
            Err(e) => (0, Vec::new(), None, Some(e)),
        };
        for &idx in indices {
            let item = &entries[idx];
            let (stats, error) = match &nvrtc_err {
                Some(e) => (None, Some(e.clone())),
                None => {
                    let missing = !found.iter().any(|e| e == &item.entry);
                    eval_entry(
                        &item.entry,
                        missing,
                        &found,
                        assembled.as_ref(),
                        cfg.max_spills,
                    )
                }
            };
            results[idx] = Some(KernelResult {
                entry: item.entry.clone(),
                rule: item.rule.clone(),
                kind: item.kind,
                dynamic_shared: item.dynamic_shared,
                threads: item.threads,
                ptx_bytes,
                stats,
                error,
            });
        }
    }
    let kernels = results.into_iter().flatten().collect();
    Ok(CheckReport {
        arch: sm,
        nvrtc,
        ptxas: ptxas_str,
        kernels,
    })
}

fn eval_entry(
    entry: &str,
    missing: bool,
    found: &[String],
    assembled: Option<&Result<HashMap<String, PtxStats>, String>>,
    max_spills: Option<u32>,
) -> (Option<PtxStats>, Option<String>) {
    if missing {
        return (
            None,
            Some(format!(
                "entry `{entry}` not found in PTX (found: [{}])",
                found.join(", ")
            )),
        );
    }
    let stats = match assembled {
        None => None,
        Some(Err(e)) => return (None, Some(e.clone())),
        Some(Ok(map)) => map.get(entry).copied(),
    };
    if let (Some(limit), Some(s)) = (max_spills, stats) {
        let spill = s.spill_bytes();
        if spill > limit {
            return (
                Some(s),
                Some(format!(
                    "spill {spill} B exceeds --max-spills {limit} B (stack={} B, stores={} B, loads={} B)",
                    s.stack_bytes, s.spill_stores, s.spill_loads
                )),
            );
        }
    }
    (stats, None)
}

#[cfg(feature = "cuda")]
fn nvrtc_tag() -> Option<String> {
    tessera_gpukit::cuda::nvrtc_version().map(|(maj, min)| format!("{maj}.{min}"))
}

#[cfg(not(feature = "cuda"))]
fn nvrtc_tag() -> Option<String> {
    None
}

#[cfg(feature = "cuda")]
fn nvrtc_compile(arch: &str, name: &str, source: &str) -> Result<Vec<u8>, String> {
    tessera_gpukit::cuda::compile_ptx(arch, name, source).map_err(|e| e.to_string())
}

#[cfg(not(feature = "cuda"))]
fn nvrtc_compile(_arch: &str, _name: &str, _source: &str) -> Result<Vec<u8>, String> {
    Err("this build has no CUDA support (enable feature `cuda`)".into())
}

/// Extracts `.entry <name>` symbols from compiled PTX text.
fn ptx_entries(ptx: &str) -> Vec<String> {
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

fn resolve_ptxas(cfg: &CheckConfig) -> Result<Option<PathBuf>, Error> {
    if !cfg.run_ptxas {
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
fn find_ptxas() -> Option<PathBuf> {
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

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn run_ptxas(
    ptxas: &Path,
    sm_arch: &str,
    ptx: &[u8],
    temp: Option<&Path>,
) -> Result<HashMap<String, PtxStats>, String> {
    let dir = temp.unwrap_or_else(|| Path::new("/tmp"));
    let pid = std::process::id();
    let seq = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
    let ptx_path = TempFile(dir.join(format!(".tessera-check-{pid}-{seq}.ptx")));
    let cubin_path = TempFile(dir.join(format!(".tessera-check-{pid}-{seq}.cubin")));
    // NVRTC includes a trailing NUL byte that ptxas rejects.
    let end = ptx.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    std::fs::write(&ptx_path.0, &ptx[..end]).map_err(|e| format!("write ptx temp: {e}"))?;
    let output = Command::new(ptxas)
        .arg("--gpu-name")
        .arg(sm_arch)
        .arg("-v")
        .arg("-o")
        .arg(&cubin_path.0)
        .arg(&ptx_path.0)
        .output()
        .map_err(|e| format!("spawn ptxas: {e}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let msg = format!("{stderr}{stdout}").trim().to_string();
        return Err(format!("ptxas ({sm_arch}) failed: {msg}"));
    }
    Ok(parse_ptxas(&stderr))
}

/// Extracts the unsigned integer directly before `marker` in `segment`.
fn num_before(segment: &str, marker: &str) -> Option<u32> {
    let head = segment.split(marker).next()?;
    head.split_whitespace().last()?.parse().ok()
}

/// Parses `ptxas -v` stderr lines into per-entry resource stats.
fn parse_ptxas(stderr: &str) -> HashMap<String, PtxStats> {
    let mut map: HashMap<String, PtxStats> = HashMap::new();
    let mut current: Option<String> = None;
    let mut in_props = false;
    for line in stderr.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.split("Compiling entry function '").nth(1)
            && let Some(name) = rest.split('\'').next()
        {
            current = Some(name.to_string());
            map.entry(name.to_string()).or_default();
            in_props = false;
            continue;
        }
        let Some(ref entry) = current else {
            continue;
        };
        if let Some(prop_fn) = trimmed.split("Function properties for ").nth(1) {
            in_props = prop_fn.trim() == entry;
            continue;
        }
        let stats = map.entry(entry.clone()).or_default();
        if in_props && trimmed.contains("stack frame") {
            for part in trimmed.split(',') {
                if let Some(n) = num_before(part, "bytes stack frame") {
                    stats.stack_bytes = stats.stack_bytes.max(n);
                } else if let Some(n) = num_before(part, "bytes spill stores") {
                    stats.spill_stores = stats.spill_stores.max(n);
                } else if let Some(n) = num_before(part, "bytes spill loads") {
                    stats.spill_loads = stats.spill_loads.max(n);
                }
            }
            in_props = false;
            continue;
        }
        if let Some(rest) = trimmed.split("Used ").nth(1) {
            for part in rest.split(',') {
                if let Some(n) = num_before(part, "registers") {
                    stats.regs = n;
                } else if let Some(n) = num_before(part, "bytes smem") {
                    stats.smem = n;
                } else if let Some(n) = num_before(part, "barriers") {
                    stats.barriers = n;
                } else if let Some(n) = num_before(part, "bytes cumulative stack size") {
                    stats.stack_bytes = stats.stack_bytes.max(n);
                }
            }
        }
    }
    map
}
