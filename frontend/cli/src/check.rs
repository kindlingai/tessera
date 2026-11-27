//! `check-kernels`: compile planned or standalone CUDA kernels offline.

use crate::source::{Origin, Source, Stop};
use serde_json::{Value, json};
use std::path::PathBuf;
use tessera_engine::check::{self, CheckConfig, CheckReport, KernelResult};
use tessera_recipe::Files;

struct Args {
    file: Option<PathBuf>,
    entry: Option<String>,
    arch: String,
    max_spills: Option<u32>,
    ptxas: Option<PathBuf>,
    run_ptxas: bool,
    json: bool,
    serve: Vec<String>,
}

fn parse(rest: &[String]) -> Result<Args, Stop> {
    let mut out = Args {
        file: None,
        entry: None,
        arch: "sm_121".into(),
        max_spills: None,
        ptxas: None,
        run_ptxas: true,
        json: false,
        serve: Vec::new(),
    };
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--file" => out.file = Some(PathBuf::from(it.next().ok_or(Stop::Usage)?)),
            "--entry" => out.entry = Some(it.next().ok_or(Stop::Usage)?.clone()),
            "--arch" => out.arch = it.next().ok_or(Stop::Usage)?.clone(),
            "--max-spills" => {
                let raw = it.next().ok_or(Stop::Usage)?;
                let n = raw
                    .parse::<u32>()
                    .map_err(|_| Stop::Failed(format!("invalid --max-spills {raw:?}")))?;
                out.max_spills = Some(n);
            }
            "--ptxas" => out.ptxas = Some(PathBuf::from(it.next().ok_or(Stop::Usage)?)),
            "--no-ptxas" => out.run_ptxas = false,
            "--json" => out.json = true,
            "--ctx" => out.serve.push("--max-model-len".into()),
            _ => out.serve.push(arg.clone()),
        }
    }
    Ok(out)
}

pub fn run(origin: Option<Origin>, rest: &[String]) -> Result<(), Stop> {
    tessera_engine::flatten_overrides();
    let parsed = parse(rest)?;
    let report = match parsed.file.as_deref() {
        Some(path) => {
            let settings = tessera_frontend_settings::read();
            check::configure(&settings);
            let cfg = config(&parsed, &settings);
            check::check_file(path, parsed.entry.as_deref(), &cfg)
                .map_err(|e| Stop::Failed(e.to_string()))?
        }
        None => run_plan(origin, &parsed)?,
    };
    emit(&report, parsed.json);
    finish_outcome(&report)
}

pub(crate) fn finish_outcome(report: &CheckReport) -> Result<(), Stop> {
    if report.kernels.is_empty() {
        return Err(Stop::Failed("empty plan: 0 CUDA kernels captured".into()));
    }
    if !report.ok() {
        return Err(Stop::Failed(format!(
            "{} of {} kernel(s) failed offline check",
            report.failed(),
            report.kernels.len()
        )));
    }
    Ok(())
}

fn config(parsed: &Args, settings: &tessera_engine::Settings) -> CheckConfig {
    CheckConfig {
        arch: parsed.arch.clone(),
        max_spills: parsed.max_spills,
        ptxas: parsed.ptxas.clone(),
        run_ptxas: parsed.run_ptxas,
        overrides: settings.kernels.clone(),
        temp: settings.temp.clone(),
    }
}

fn run_plan(origin: Option<Origin>, parsed: &Args) -> Result<CheckReport, Stop> {
    let session = check::start(&parsed.arch).map_err(|e| Stop::Failed(e.to_string()))?;
    let at = parsed.serve.iter().position(|f| f == "--manifest");
    let manifest = at.and_then(|i| parsed.serve.get(i + 1)).map(String::as_str);
    let source = Source::open_as(
        origin,
        crate::progress::stderr(),
        (Files::Headers, manifest),
    )?;
    let options = source.options(&parsed.serve)?;
    let settings = tessera_frontend_settings::read();
    let cfg = config(parsed, &settings);
    crate::task::load(options)?;
    session
        .finish(&cfg)
        .map_err(|e| Stop::Failed(e.to_string()))
}

fn kernel_json(k: &KernelResult) -> Value {
    let stats = k.stats.map(|s| {
        json!({
            "regs": s.regs,
            "smem": s.smem,
            "stack_bytes": s.stack_bytes,
            "spill_stores": s.spill_stores,
            "spill_loads": s.spill_loads,
            "spill_bytes": s.spill_bytes(),
            "barriers": s.barriers,
        })
    });
    json!({
        "entry": k.entry,
        "rule": k.rule,
        "kind": k.kind,
        "dynamic_shared": k.dynamic_shared,
        "threads": k.threads,
        "ptx_bytes": k.ptx_bytes,
        "stats": stats,
        "ok": k.error.is_none(),
        "error": k.error,
    })
}

fn fmt_opt(v: Option<u32>, suffix: &str) -> String {
    match v {
        Some(n) => format!("{n}{suffix}"),
        None => "?".into(),
    }
}

pub(crate) fn report_json(report: &CheckReport) -> Value {
    json!({
        "arch": report.arch,
        "nvrtc": report.nvrtc,
        "ptxas": report.ptxas,
        "ok": report.ok(),
        "total": report.kernels.len(),
        "failed": report.failed(),
        "spilling": report.spilling(),
        "kernels": report.kernels.iter().map(kernel_json).collect::<Vec<_>>(),
    })
}

fn emit(report: &CheckReport, as_json: bool) {
    if as_json {
        println!("{}", report_json(report));
        return;
    }
    let nvrtc = report.nvrtc.as_deref().unwrap_or("none");
    let ptxas = report.ptxas.as_deref().unwrap_or("off");
    println!(
        "check-kernels: arch={} nvrtc={} ptxas={} total={} failed={} spilling={}",
        report.arch,
        nvrtc,
        ptxas,
        report.kernels.len(),
        report.failed(),
        report.spilling()
    );
    for k in &report.kernels {
        let status = if k.error.is_none() { "OK" } else { "FAIL" };
        let res = match k.stats {
            Some(s) => format!(
                "regs={} smem={}B lmem={} spill={}/{}",
                fmt_opt(s.regs, ""),
                s.smem,
                fmt_opt(s.stack_bytes, "B"),
                fmt_opt(s.spill_stores, ""),
                fmt_opt(s.spill_loads, "B")
            ),
            None => format!("ptx={}B", k.ptx_bytes),
        };
        println!("  [{status}] {} ({}, {}) {res}", k.entry, k.rule, k.kind);
        if let Some(err) = &k.error {
            eprintln!("    error: {err}");
        }
    }
}
