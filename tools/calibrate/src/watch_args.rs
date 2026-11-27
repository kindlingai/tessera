//! Argument, recipe and option parsing for `watch`.

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use tessera_bundle::names::Rule as NameRule;
use tessera_engine::progress::Progress;
use tessera_engine::wasm::{Module, Trace};
use tessera_graph::Parts;
use tessera_lower::{Knobs, Pins, Recipe, Settings};
use tessera_recipe::Files;
use tessera_serve::start::Options;

pub(crate) const USAGE: &str = "usage: tessera-calibrate watch <dir> \
    [--recipe <path-or-name>] [--parts <n|a,b,...>] [--rank <n>] \
    [--once] [--json <path>] [--against <base.json>] \
    [--favor <rule>] [--bar <kernel>] [<serve flags>...] \
    gemm <m> <k> <n> | attend <heads> <kv heads> <width> <queries> <keys> | \
    mla <heads> <rows> <context> [--kv f32|bf16|fp8] | \
    gqa <heads> <kv heads> <rows> <context> [--kv f32|bf16|fp8] | \
    moe <experts> <top k> <hidden> <ffn> <rows> [--form <name>] [--skew <n>] [--plain] | \
    model <model dir> [<part>] [--region <kind>.<n>] [--list] [--skew <n>] [<Dim>=<n> ...]";

/// Parsed arguments for `tessera-calibrate watch`.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct WatchArgs {
    pub(crate) dir: PathBuf,
    pub(crate) recipe: Option<String>,
    pub(crate) parts: Option<Parts>,
    pub(crate) rank: Option<u32>,
    pub(crate) once: bool,
    pub(crate) json: Option<PathBuf>,
    pub(crate) against: Option<PathBuf>,
    pub(crate) favor: Option<String>,
    pub(crate) barred: Vec<String>,
    pub(crate) serve_flags: Vec<(String, Option<String>)>,
    pub(crate) subject_args: Vec<String>,
}

/// Parses `--parts` as either a rank count or comma shares.
pub(crate) fn parse_parts(value: &str) -> Result<Parts, Box<dyn Error>> {
    if value.contains(',') {
        let mut shares = Vec::new();
        for tok in value.split(',') {
            let n: u32 = tok
                .parse()
                .map_err(|_| format!("invalid --parts share `{tok}`"))?;
            if n == 0 {
                return Err("--parts shares must be > 0".into());
            }
            shares.push(n);
        }
        return Ok(Parts(shares));
    }
    let count: usize = value
        .parse()
        .map_err(|_| format!("invalid --parts count `{value}`"))?;
    if count == 0 {
        return Err("--parts count must be >= 1".into());
    }
    Ok(Parts::even(count))
}

/// Splits watch and serve flags out of `args`.
pub(crate) fn parse_watch(args: &[String]) -> Result<WatchArgs, Box<dyn Error>> {
    let (dir, rest) = args.split_first().ok_or(USAGE)?;
    let mut out = WatchArgs {
        dir: PathBuf::from(dir),
        ..WatchArgs::default()
    };
    let mut probe = Options::default();
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--once" => out.once = true,
            "--recipe" => out.recipe = Some(it.next().ok_or(USAGE)?.clone()),
            "--parts" => out.parts = Some(parse_parts(it.next().ok_or(USAGE)?)?),
            "--rank" => {
                let raw = it.next().ok_or(USAGE)?;
                out.rank = Some(
                    raw.parse::<u32>()
                        .map_err(|_| format!("bad --rank `{raw}`"))?,
                );
            }
            "--json" => out.json = Some(PathBuf::from(it.next().ok_or(USAGE)?)),
            "--against" => out.against = Some(PathBuf::from(it.next().ok_or(USAGE)?)),
            "--favor" => out.favor = Some(it.next().ok_or(USAGE)?.clone()),
            "--bar" => out.barred.push(it.next().ok_or(USAGE)?.clone()),
            "--list" | "--plain" => out.subject_args.push(arg.clone()),
            "--region" | "--skew" | "--form" | "--kv" => {
                let val = it.next().ok_or(USAGE)?;
                out.subject_args.push(arg.clone());
                out.subject_args.push(val.clone());
            }
            "--shape" => {
                let val = it.next().ok_or(USAGE)?;
                Knobs::split_flag(val)
                    .map_err(|_| format!("invalid flag `--shape {val}`\n{USAGE}"))?;
                out.serve_flags.push((arg.clone(), Some(val.clone())));
            }
            _ if probe.switch(arg) => out.serve_flags.push((arg.clone(), None)),
            _ if arg.starts_with("--") => {
                let val = it
                    .next()
                    .ok_or_else(|| format!("invalid flag `{arg}`\n{USAGE}"))?;
                probe
                    .flag(arg, val.clone())
                    .ok_or_else(|| format!("invalid flag `{arg} {val}`\n{USAGE}"))?;
                out.serve_flags.push((arg.clone(), Some(val.clone())));
            }
            _ => out.subject_args.push(arg.clone()),
        }
    }
    if let (Some(parts), Some(rank)) = (&out.parts, out.rank)
        && rank as usize >= parts.ranks()
    {
        return Err(format!("--rank {rank} out of range for {} parts", parts.ranks()).into());
    }
    Ok(out)
}

/// Effective model dimensions set by recipe or CLI serve flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ServeDims {
    pub(crate) slots: Option<u64>,
    pub(crate) context: Option<u64>,
    pub(crate) max_batch_rows: Option<u64>,
}

/// Applies one serve switch or flag-value pair to `options`.
fn apply_serve_flag(
    options: &mut Options,
    dims: &mut ServeDims,
    flag: &str,
    value: Option<String>,
) -> Result<(), Box<dyn Error>> {
    match value {
        None if options.switch(flag) => Ok(()),
        Some(v) => {
            options
                .flag(flag, v)
                .ok_or_else(|| format!("invalid serve flag `{flag}`"))?;
            match flag {
                "--slots" => dims.slots = Some(options.slots as u64),
                "--max-model-len" => dims.context = Some(options.max_model_len as u64),
                "--max-batch-rows" => {
                    dims.max_batch_rows = options.max_batch_rows.map(|n| n as u64)
                }
                _ => {}
            }
            Ok(())
        }
        None => Err(format!("invalid serve flag `{flag}`").into()),
    }
}

/// Opens any `--recipe` and merges CLI serve flags on top.
pub(crate) fn build_options(
    args: &WatchArgs,
    settings: &Settings,
    has_model: bool,
) -> Result<(Options, ServeDims, Option<PathBuf>), Box<dyn Error>> {
    let mut options = Options::default();
    let mut dims = ServeDims::default();
    let mut weights_dir = None;
    if let Some(src) = &args.recipe {
        let cache = settings.recipes.as_deref();
        let r = tessera_recipe::Recipe::open(src, cache)?;
        if !has_model {
            let dir = r.weights_as(
                Files::Whole,
                cache,
                settings.hf_token.as_deref(),
                &Progress::default(),
            )?;
            weights_dir = Some(dir);
        }
        options.name = Some(r.name.clone());
        options.lower = r.lower.clone();
        options.names = r.names();
        options.config = tessera_bundle::Overrides::new(r.overrides());
        if let Some(bytes) = r.module_bytes(cache)? {
            let trace = Trace {
                dump: settings.dump_sites.clone(),
                favor: args
                    .favor
                    .clone()
                    .or_else(|| r.lower.as_ref().and_then(|l| l.favor.clone())),
            };
            let module = Module::open_with(&bytes, trace)?;
            let rules = module.rules.iter().copied();
            options.lower.get_or_insert_default().rules.extend(rules);
            options.module = Some(Arc::new(module));
        }
        for (flag, value) in r.flags() {
            apply_serve_flag(&mut options, &mut dims, &flag, value)?;
        }
    }
    for (flag, value) in &args.serve_flags {
        apply_serve_flag(&mut options, &mut dims, flag, value.clone())?;
    }
    Ok((options, dims, weights_dir))
}

/// Builds a lowering recipe from options and CLI flags.
pub(crate) fn build_recipe(
    options: &Options,
    pins: Option<Pins>,
    config: Option<String>,
    args: &WatchArgs,
) -> Recipe {
    let mut recipe = options.lowering(pins);
    recipe.config = config;
    if let Some(favor) = &args.favor {
        recipe.favor = Some(favor.clone());
    }
    recipe.barred.extend(args.barred.iter().cloned());
    recipe.settle();
    recipe
}

/// Loads user weight-name rules from recipe or settings.
pub(crate) fn user_rules(
    options: &Options,
    settings: &Settings,
) -> Result<Vec<NameRule>, Box<dyn Error>> {
    match options.names.as_deref().or(settings.names.as_deref()) {
        Some(path) => Ok(tessera_bundle::names::rules(&std::fs::read_to_string(
            path,
        )?)?),
        None => Ok(Vec::new()),
    }
}
