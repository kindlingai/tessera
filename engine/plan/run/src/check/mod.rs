//! Offline CUDA kernel compilation and spill verification without a GPU.

#[cfg(test)]
mod check_test;

use crate::Error;
use crate::kernels::group;
use crate::overrides;
use crate::run::Run;
use std::cell::RefCell;
use std::collections::hash_map::RandomState;
use std::collections::{BTreeMap, HashMap};
use std::fs::OpenOptions;
use std::hash::BuildHasher;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tessera_engine_settings::{Backend, Settings};
use tessera_gpukit::platform::{Asked, Platform};
use tessera_gpukit::{Error as GpuError, Limits};
use tessera_lower::{Plan, Source, Step};

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
    fn has_spill(self) -> bool {
        self.stack_bytes.is_some_and(|n| n > 0)
            || self.spill_stores.is_some_and(|n| n > 0)
            || self.spill_loads.is_some_and(|n| n > 0)
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
    /// True when kernels ran and every kernel succeeded.
    pub fn ok(&self) -> bool {
        !self.kernels.is_empty() && self.kernels.iter().all(|k| k.error.is_none())
    }

    /// Count of kernels that failed compilation or spill checks.
    pub fn failed(&self) -> usize {
        self.kernels.iter().filter(|k| k.error.is_some()).count()
    }

    /// Count of kernels with non-zero local spills.
    pub fn spilling(&self) -> usize {
        self.kernels
            .iter()
            .filter(|k| k.stats.is_some_and(PtxStats::has_spill))
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

/// Hardware defaults for known CUDA architectures in offline checks.
fn arch_spec(major: u32, minor: u32) -> (Option<&'static str>, usize, f64, usize, bool) {
    match (major, minor) {
        // Match the calibrated GB10 chip model on sm_121.
        (12, 1) => (Some("NVIDIA GB10"), 48, 273e9, 99 * 1024, true),
        (12, 0) => (None, 170, 1792e9, 99 * 1024, false),
        (10, _) => (None, 148, 8000e9, 227 * 1024, false),
        (9, _) => (None, 132, 3350e9, 227 * 1024, false),
        (8, 0) => (None, 108, 2039e9, 163 * 1024, false),
        _ => (None, 48, 273e9, 99 * 1024, false),
    }
}

/// Builds a synthetic CUDA `Platform` for offline lowering.
fn target_platform(target: &str) -> Result<Platform, Error> {
    let (major, minor, _) = tessera_gpukit::backend::cuda_arch(target)?;
    let arch = format!("sm_{major}{minor}");
    let (known, units, bandwidth, dynamic_group_bytes, integrated) = arch_spec(major, minor);
    let limits = Limits {
        lanes: 32,
        group_bytes: 48 * 1024,
        dynamic_group_bytes,
        group_threads: 1024,
        compiler: true,
        pdl: false,
    };
    Ok(Platform::new(Asked {
        backend: Backend::Cuda.name(),
        arch: arch.clone(),
        name: known.map_or_else(|| format!("offline {arch}"), str::to_string),
        driver: "13.0".into(),
        limits,
        units,
        bandwidth,
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
    flags: Vec<String>,
    prep_error: Option<String>,
    resident: bool,
    dynamic_shared: usize,
    threads: usize,
}

struct Captured {
    id: u64,
    closed: bool,
    platform: Platform,
    entries: BTreeMap<(String, String), Collected>,
}

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
static NEXT_TMP: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static ACTIVE: RefCell<Option<Arc<Mutex<Captured>>>> = const { RefCell::new(None) };
}

fn slot_is(slot: &Option<Arc<Mutex<Captured>>>, id: u64) -> bool {
    slot.as_ref()
        .is_some_and(|c| c.lock().is_ok_and(|g| g.id == id))
}

/// Check session handle forwarded to an intended loader thread.
#[derive(Clone)]
pub struct Scope(Arc<Mutex<Captured>>);

struct ScopeGuard {
    id: u64,
    prev: Option<Arc<Mutex<Captured>>>,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let _ = ACTIVE.try_with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot_is(&slot, self.id) {
                *slot = self.prev.take();
            }
        });
    }
}

impl Scope {
    /// Runs `f` with this check session active on this thread.
    pub fn run<R>(self, f: impl FnOnce() -> R) -> R {
        let id = self.0.lock().expect("check session").id;
        let prev = ACTIVE.with(|cell| cell.borrow_mut().replace(self.0));
        let _guard = ScopeGuard { id, prev };
        f()
    }
}

/// Returns the current thread's check session scope, if active.
pub fn current_scope() -> Option<Scope> {
    ACTIVE.with(|cell| {
        let slot = cell.borrow();
        let arc = slot.as_ref()?;
        (!arc.lock().ok()?.closed).then(|| Scope(arc.clone()))
    })
}

/// Guard holding an active offline kernel collection session.
pub struct Session {
    id: u64,
    handle: Arc<Mutex<Captured>>,
    live: bool,
}

impl Session {
    /// Ends collection and verifies all captured kernels.
    pub fn finish(mut self, cfg: &CheckConfig) -> Result<CheckReport, Error> {
        self.live = false;
        let owned = ACTIVE.with(|cell| {
            let mut slot = cell.borrow_mut();
            let same = slot_is(&slot, self.id);
            if same {
                slot.take();
            }
            same
        });
        let mut captured = self.handle.lock().expect("check session");
        let was_closed = captured.closed;
        captured.closed = true;
        if !owned || was_closed {
            return Err(GpuError::Cuda(
                "offline kernel check session was lost before finish".into(),
            )
            .into());
        }
        let entries: Vec<Collected> = std::mem::take(&mut captured.entries)
            .into_values()
            .collect();
        drop(captured);
        verify_collected(entries, cfg)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.live {
            return;
        }
        if let Ok(mut captured) = self.handle.lock() {
            captured.closed = true;
        }
        let _ = ACTIVE.try_with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot_is(&slot, self.id) {
                *slot = None;
            }
        });
    }
}

/// Starts offline kernel collection for CUDA `target`.
pub fn start(target: &str) -> Result<Session, Error> {
    let platform = target_platform(target)?;
    ACTIVE.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot
            .as_ref()
            .is_some_and(|c| c.lock().is_ok_and(|g| !g.closed))
        {
            return Err(
                GpuError::Cuda("an offline kernel check session is already active".into()).into(),
            );
        }
        let id = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
        let handle = Arc::new(Mutex::new(Captured {
            id,
            closed: false,
            platform,
            entries: BTreeMap::new(),
        }));
        *slot = Some(handle.clone());
        Ok(Session {
            id,
            handle,
            live: true,
        })
    })
}

/// Whether offline kernel verification mode is active on this thread.
pub fn active() -> bool {
    ACTIVE.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(|c| c.lock().is_ok_and(|g| !g.closed))
    })
}

/// Active synthetic GPU backend while checking offline.
pub(crate) fn backend() -> Option<Backend> {
    active().then_some(Backend::Cuda)
}

/// Active synthetic GPU platform while checking offline.
pub(crate) fn platform() -> Option<Platform> {
    ACTIVE.with(|cell| {
        let slot = cell.borrow();
        let arc = slot.as_ref()?;
        let guard = arc.lock().ok()?;
        (!guard.closed).then(|| guard.platform.clone())
    })
}

/// Opens a capture runner if offline verification is active.
pub(crate) fn open(plan: &Plan) -> Option<Box<dyn Run>> {
    let arc = ACTIVE.with(|cell| cell.borrow().clone())?;
    let platform = {
        let guard = arc.lock().expect("check session");
        if guard.closed {
            return None;
        }
        guard.platform.clone()
    };
    Some(crate::sizer::passive(
        plan,
        platform,
        "offline check",
        move |p| {
            let mut guard = arc.lock().expect("check session");
            if !guard.closed {
                collect_plan(p, &mut guard.entries);
            }
        },
    ))
}

/// Normalized compilation unit name passed to NVRTC.
fn unit_key(name: &str) -> String {
    match name.is_empty() {
        true => "kernel.cu".into(),
        false if name.contains('.') => name.to_string(),
        false => format!("{name}.cu"),
    }
}

type GroupKey<'a> = (String, &'a str, &'a [String], Option<&'a str>);

/// Groups entries by compile unit, source, flags and error.
fn compile_groups(entries: &[Collected]) -> BTreeMap<GroupKey<'_>, Vec<usize>> {
    let mut groups: BTreeMap<GroupKey<'_>, Vec<usize>> = BTreeMap::new();
    for (idx, item) in entries.iter().enumerate() {
        groups
            .entry((
                unit_key(&item.entry),
                item.source.as_str(),
                item.flags.as_slice(),
                item.prep_error.as_deref(),
            ))
            .or_default()
            .push(idx);
    }
    groups
}

fn collect_step(step: &Step, out: &mut BTreeMap<(String, String), Collected>) {
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
    let key = (entry.clone(), source.clone());
    let resident = step.kernel.resident().is_some();
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
            item.resident |= resident;
            item.dynamic_shared = item.dynamic_shared.max(dynamic_shared);
            item.threads = item.threads.max(threads);
        })
        .or_insert(Collected {
            entry,
            rule: step.rule.to_string(),
            kind,
            source,
            flags: Vec::new(),
            prep_error: None,
            resident,
            dynamic_shared,
            threads,
        });
}

fn collect_plan(plan: &Plan, out: &mut BTreeMap<(String, String), Collected>) {
    for step in &plan.steps {
        collect_step(step, out);
    }
}

fn file_error_report(
    entry: String,
    arch: String,
    nvrtc: Option<String>,
    ptxas: Option<String>,
    ptx_bytes: usize,
    err: String,
) -> CheckReport {
    let k = KernelResult {
        entry,
        rule: "file".into(),
        kind: "file",
        dynamic_shared: 0,
        threads: 0,
        ptx_bytes,
        stats: None,
        error: Some(err),
    };
    CheckReport {
        arch,
        nvrtc,
        ptxas,
        kernels: vec![k],
    }
}

/// Verifies one standalone CUDA file offline.
pub fn check_file(
    path: &Path,
    entry: Option<&str>,
    cfg: &CheckConfig,
) -> Result<CheckReport, Error> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("kernel.cu");
    if !path.exists() {
        return Err(Error::KernelFile(
            path.to_path_buf(),
            std::fs::read_to_string(path).unwrap_err(),
        ));
    }
    match overrides::load_file(path, Backend::Cuda) {
        Ok((source, launch)) => {
            let flags = launch.as_ref().map_or_else(Vec::new, |l| l.flags());
            check_source(name, entry, &source, &flags, cfg)
        }
        Err(err) => Ok(file_error_report(
            entry.unwrap_or(name).to_string(),
            sm_target(&cfg.arch)?,
            nvrtc_tag(),
            resolve_ptxas(cfg)?.map(|p| p.display().to_string()),
            0,
            err,
        )),
    }
}

/// Verifies one standalone CUDA source string offline.
fn check_source(
    name: &str,
    entry: Option<&str>,
    source: &str,
    flags: &[&str],
    cfg: &CheckConfig,
) -> Result<CheckReport, Error> {
    let (sm, ptxas, nvrtc) = (sm_target(&cfg.arch)?, resolve_ptxas(cfg)?, nvrtc_tag());
    let ptxas_str = ptxas.as_ref().map(|p| p.display().to_string());
    let unit = match nvrtc_compile(&cfg.arch, name, source, flags) {
        Ok(ptx) => ptx,
        Err(err) => {
            let want = entry.unwrap_or(name).to_string();
            return Ok(file_error_report(want, sm, nvrtc, ptxas_str, 0, err));
        }
    };
    let found = ptx_entries(&String::from_utf8_lossy(&unit));
    if entry.is_none() && found.is_empty() {
        let msg = format!("missing `.entry` kernel in PTX for `{name}`");
        return Ok(file_error_report(
            name.into(),
            sm,
            nvrtc,
            ptxas_str,
            unit.len(),
            msg,
        ));
    }
    let wanted = entry.map_or_else(|| found.clone(), |e| vec![e.to_string()]);
    let asm = ptxas
        .as_deref()
        .map(|b| run_ptxas(b, &sm, &unit, cfg.temp.as_deref()));
    let kernels = wanted
        .into_iter()
        .map(|want| {
            let missing = entry.is_some() && !found.iter().any(|e| e == &want);
            let (stats, error) = eval_entry(&want, missing, &found, asm.as_ref(), cfg.max_spills);
            KernelResult {
                entry: want,
                rule: "file".into(),
                kind: "file",
                dynamic_shared: 0,
                threads: 0,
                ptx_bytes: unit.len(),
                stats,
                error,
            }
        })
        .collect();
    Ok(CheckReport {
        arch: sm,
        nvrtc,
        ptxas: ptxas_str,
        kernels,
    })
}

fn verify_collected(mut entries: Vec<Collected>, cfg: &CheckConfig) -> Result<CheckReport, Error> {
    let (sm, ptxas, nvrtc) = (sm_target(&cfg.arch)?, resolve_ptxas(cfg)?, nvrtc_tag());
    let ptxas_str = ptxas.as_ref().map(|p| p.display().to_string());
    if let Some(dir) = &cfg.overrides {
        for item in &mut entries {
            let planned = item.source.clone();
            match overrides::load_entry(dir, Backend::Cuda, &item.entry, item.resident, || planned)
            {
                Ok(Some(prep)) => {
                    item.source = prep.source;
                    item.kind = "override";
                    if let Some(launch) = prep.launch {
                        item.flags = launch.nvrtc_flags;
                        item.threads = launch.threads.map_or(item.threads, |t| t as usize);
                        item.dynamic_shared = launch
                            .dynamic_shared
                            .map_or(item.dynamic_shared, |s| s as usize);
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    item.kind = "override";
                    item.prep_error = Some(err);
                }
            }
        }
    }
    // Group entries by exact compile unit name and source text.
    let groups: Vec<Vec<usize>> = compile_groups(&entries).into_values().collect();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16)
        .min(groups.len().max(1));
    let next = AtomicUsize::new(0);
    let batch = Mutex::new(Vec::with_capacity(entries.len()));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= groups.len() {
                        break;
                    }
                    let done = verify_group(&entries, &groups[idx], cfg, &sm, ptxas.as_deref());
                    batch.lock().expect("check results").extend(done);
                }
            });
        }
    });
    let mut results: Vec<Option<KernelResult>> = vec![None; entries.len()];
    for (idx, res) in batch.into_inner().expect("check results") {
        results[idx] = Some(res);
    }
    let kernels = results.into_iter().flatten().collect();
    Ok(CheckReport {
        arch: sm,
        nvrtc,
        ptxas: ptxas_str,
        kernels,
    })
}

fn item_result(
    item: &Collected,
    ptx_bytes: usize,
    stats: Option<PtxStats>,
    error: Option<String>,
) -> KernelResult {
    KernelResult {
        entry: item.entry.clone(),
        rule: item.rule.clone(),
        kind: item.kind,
        dynamic_shared: item.dynamic_shared,
        threads: item.threads,
        ptx_bytes,
        stats,
        error,
    }
}

fn verify_group(
    entries: &[Collected],
    indices: &[usize],
    cfg: &CheckConfig,
    sm: &str,
    ptxas: Option<&Path>,
) -> Vec<(usize, KernelResult)> {
    let first = &entries[indices[0]];
    let pre_err = first.prep_error.clone().or_else(|| {
        first
            .source
            .is_empty()
            .then(|| format!("kernel `{}` has empty CUDA source", first.entry))
    });
    if let Some(err) = pre_err {
        return indices
            .iter()
            .map(|&idx| (idx, item_result(&entries[idx], 0, None, Some(err.clone()))))
            .collect();
    }
    let flags: Vec<&str> = first.flags.iter().map(String::as_str).collect();
    let unit = nvrtc_compile(&cfg.arch, &first.entry, &first.source, &flags);
    let (ptx_bytes, found, assembled, nvrtc_err) = match unit {
        Ok(ptx) => {
            let found = ptx_entries(&String::from_utf8_lossy(&ptx));
            let asm = ptxas.map(|bin| run_ptxas(bin, sm, &ptx, cfg.temp.as_deref()));
            (ptx.len(), found, asm, None)
        }
        Err(e) => (0, Vec::new(), None, Some(e)),
    };
    indices
        .iter()
        .map(|&idx| {
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
            (idx, item_result(item, ptx_bytes, stats, error))
        })
        .collect()
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
        None => {
            if let Some(limit) = max_spills {
                return (
                    None,
                    Some(format!(
                        "--max-spills {limit} B requested without ptxas stats for `{entry}`"
                    )),
                );
            }
            return (None, None);
        }
        Some(Err(e)) => return (None, Some(e.clone())),
        Some(Ok(map)) => map.get(entry).copied(),
    };
    if let Some(limit) = max_spills {
        let Some(s) = stats else {
            return (
                None,
                Some(format!(
                    "missing ptxas stats for `{entry}` under --max-spills {limit} B"
                )),
            );
        };
        let Some(spill) = s.spill_bytes() else {
            let msg = format!(
                "ptxas reported partial spill stats for `{entry}` under --max-spills {limit} B"
            );
            return (Some(s), Some(msg));
        };
        if spill > limit {
            let (st, wr, rd) = (
                s.stack_bytes.unwrap_or(0),
                s.spill_stores.unwrap_or(0),
                s.spill_loads.unwrap_or(0),
            );
            let msg = format!(
                "spill {spill} B exceeds --max-spills {limit} B (stack={st} B, stores={wr} B, loads={rd} B)"
            );
            return (Some(s), Some(msg));
        }
    }
    (stats, None)
}

fn nvrtc_tag() -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        tessera_gpukit::cuda::nvrtc_version().map(|(maj, min)| format!("{maj}.{min}"))
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

fn nvrtc_compile(arch: &str, name: &str, source: &str, flags: &[&str]) -> Result<Vec<u8>, String> {
    #[cfg(feature = "cuda")]
    {
        tessera_gpukit::cuda::compile_ptx_with(arch, name, source, flags).map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (arch, name, source, flags);
        Err("this build has no CUDA support (enable feature `cuda`)".into())
    }
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

struct TempDir(PathBuf);

impl TempDir {
    /// Creates an exclusive private directory under `base`.
    fn create(base: &Path) -> Result<Self, String> {
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

fn run_ptxas(
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
fn parse_ptxas(stderr: &str) -> HashMap<String, PtxStats> {
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
