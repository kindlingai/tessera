//! Watch table, JSON report and baseline comparison formatting.

use serde::{Deserialize, Deserializer, Serialize};
use std::error::Error;
use std::fmt::Write as _;
use std::path::Path;
use tessera_run::KernelAttrs;

/// One kernel entry: its steps, cost and hardware attributes.
pub(crate) struct Row {
    pub entry: String,
    pub rule: &'static str,
    pub steps: Vec<usize>,
    pub seconds: f64,
    pub planned: f64,
    pub bound: Option<f64>,
    /// Whether a kernel file runs in place of the plan's.
    pub loaded: bool,
    pub attrs: Option<KernelAttrs>,
}

/// Static run metadata recorded in watch reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Meta {
    pub device: String,
    pub parts: Vec<u32>,
    pub rank: u32,
    pub region: Option<String>,
    pub recipe: Option<String>,
}

/// Deserializes JSON `null` from non-finite errors as infinity.
fn err_or_inf<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    Ok(Option::<f32>::deserialize(d)?.unwrap_or(f32::INFINITY))
}

/// One kernel entry in a serialized watch report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct KernelReport {
    pub entry: String,
    pub rule: String,
    pub us: f64,
    pub planned_us: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bound_us: Option<f64>,
    pub steps: usize,
    pub loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smem: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lmem: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctas: Option<u32>,
}

/// Machine-readable report written by `watch --json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Report {
    pub device: String,
    pub parts: Vec<u32>,
    pub rank: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    pub plan_us: f64,
    pub alone_us: f64,
    #[serde(deserialize_with = "err_or_inf")]
    pub err: f32,
    #[serde(deserialize_with = "err_or_inf")]
    pub ref_max: f32,
    pub kernels: Vec<KernelReport>,
}

/// Builds a sorted report from measured rows and plan totals.
pub(crate) fn build_report(
    meta: &Meta,
    rows: &[Row],
    whole_seconds: f64,
    err: f32,
    ref_max: f32,
) -> Report {
    let mut order: Vec<&Row> = rows.iter().collect();
    order.sort_by(|a, b| b.seconds.total_cmp(&a.seconds));
    let kernels = order
        .into_iter()
        .map(|r| KernelReport {
            entry: r.entry.clone(),
            rule: r.rule.to_string(),
            us: r.seconds * 1e6,
            planned_us: r.planned * 1e6,
            bound_us: r.bound.map(|b| b * 1e6),
            steps: r.steps.len(),
            loaded: r.loaded,
            regs: r.attrs.map(|a| a.regs),
            smem: r.attrs.map(|a| a.shared_bytes),
            lmem: r.attrs.map(|a| a.local_bytes),
            ctas: r.attrs.map(|a| a.ctas_per_sm),
        })
        .collect();
    let alone: f64 = rows.iter().map(|r| r.seconds).sum();
    Report {
        device: meta.device.clone(),
        parts: meta.parts.clone(),
        rank: meta.rank,
        region: meta.region.clone(),
        recipe: meta.recipe.clone(),
        plan_us: whole_seconds * 1e6,
        alone_us: alone * 1e6,
        err,
        ref_max,
        kernels,
    }
}

fn opt_u32(v: Option<u32>) -> String {
    v.map_or("-".into(), |n| n.to_string())
}

/// Formats `report` as an aligned terminal table.
pub(crate) fn format_table(report: &Report) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:>9} {:>9} {:>9} {:>5} {:>5} {:>6} {:>6} {:>5}  kernel",
        "us", "planned", "bound", "steps", "regs", "smem", "lmem", "ctas"
    );
    for k in &report.kernels {
        let bound = k.bound_us.map_or("-".into(), |b| format!("{b:.1}"));
        let mark = if k.loaded { "*" } else { " " };
        let (regs, smem) = (opt_u32(k.regs), opt_u32(k.smem));
        let (lmem, ctas) = (opt_u32(k.lmem), opt_u32(k.ctas));
        let _ = writeln!(
            out,
            "{:>9.1} {:>9.1} {bound:>9} {:>5} {regs:>5} {smem:>6} {lmem:>6} {ctas:>5} {mark}{}",
            k.us, k.planned_us, k.steps, k.entry
        );
    }
    let _ = writeln!(
        out,
        "plan {:.1} us, steps alone {:.1} us, error {:.3e} of {:.3e}",
        report.plan_us, report.alone_us, report.err, report.ref_max
    );
    out
}

fn speed(base: f64, now: f64) -> String {
    if now > 0.0 && base > 0.0 {
        format!("{:.2}x", base / now)
    } else {
        "-".into()
    }
}

fn attr_delta(base: &KernelReport, now: &KernelReport) -> String {
    let mut notes = Vec::new();
    if let (Some(a), Some(b)) = (base.regs, now.regs)
        && a != b
    {
        notes.push(format!("r{a}->{b}"));
    }
    if let (Some(a), Some(b)) = (base.lmem, now.lmem)
        && a != b
    {
        notes.push(format!("lmem {a}->{b}"));
    }
    if let (Some(a), Some(b)) = (base.ctas, now.ctas)
        && a != b
    {
        notes.push(format!("ctas {a}->{b}"));
    }
    if notes.is_empty() {
        "-".into()
    } else {
        notes.join(",")
    }
}

/// Formats per-kernel and whole-plan deltas against `base`.
pub(crate) fn format_diff(base: &Report, now: &Report) -> String {
    let mut out = String::new();
    let delta = now.plan_us - base.plan_us;
    let plan_x = speed(base.plan_us, now.plan_us);
    let _ = writeln!(
        out,
        "against baseline: plan {:.1} -> {:.1} us ({delta:+.1} us, {plan_x}), \
         alone {:.1} -> {:.1} us, err {:.3e}",
        base.plan_us, now.plan_us, base.alone_us, now.alone_us, now.err
    );
    let _ = writeln!(
        out,
        "  {:>9} {:>9} {:>9} {:>7}  {:18} kernel",
        "base_us", "now_us", "delta_us", "speed", "attrs"
    );
    for cur in &now.kernels {
        match base.kernels.iter().find(|k| k.entry == cur.entry) {
            Some(prev) => {
                let d = cur.us - prev.us;
                let x = speed(prev.us, cur.us);
                let note = attr_delta(prev, cur);
                let _ = writeln!(
                    out,
                    "  {:>9.1} {:>9.1} {:>+9.1} {:>7}  {note:18} {}",
                    prev.us, cur.us, d, x, cur.entry
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "  {:>9} {:>9.1} {:>9} {:>7}  {:18} {}",
                    "-", cur.us, "new", "-", "-", cur.entry
                );
            }
        }
    }
    for prev in &base.kernels {
        if !now.kernels.iter().any(|k| k.entry == prev.entry) {
            let _ = writeln!(
                out,
                "  {:>9.1} {:>9} {:>9} {:>7}  {:18} {}",
                prev.us, "-", "gone", "-", "-", prev.entry
            );
        }
    }
    out
}

/// Writes `report` as pretty JSON to `path`, or stdout for `-`.
pub(crate) fn write_json(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    let text = serde_json::to_string_pretty(report)?;
    if path == Path::new("-") {
        println!("{text}");
        return Ok(());
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, format!("{text}\n"))?;
    Ok(())
}

/// Loads a baseline report written by `watch --json`.
pub(crate) fn load_report(path: &Path) -> Result<Report, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}
