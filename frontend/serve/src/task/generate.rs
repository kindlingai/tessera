//! Text generation, batched over slots and ranks.

use super::{Entry, Flag, Load, Pick};
use crate::start::Options;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;
use tessera_engine::{Batch, Engine, Fair, Pins, PleStore, Recipe, Settings, Steps};
use tessera_tasks::api::Score;
use tessera_tasks::{Generate, Job, Queued};

pub static ENTRY: Entry = Entry {
    task: "generate",
    pick: Pick::Plain,
    flags: &[
        Flag::switch("--guided"),
        Flag::switch("--reuse-prefix"),
        Flag::text("--manifest", "<name>"),
    ],
    open,
};

/// Applies any `--kv-dtype` override to `pins`.
pub(super) fn kv_pin(options: &Options, pins: Pins) -> Pins {
    match options
        .kv_dtype
        .as_deref()
        .and_then(tessera_engine::form::named)
    {
        Some(form) => pins.kept(form),
        None => pins,
    }
}

/// The lowering choices, with the model's pins by default.
pub(super) fn lowering(options: &Options, pins: Option<Pins>) -> Recipe {
    let mut recipe = options.lower.clone().unwrap_or_default();
    let base = pins.unwrap_or_else(|| Pins::all(tessera_engine::form::BF16_FORM));
    if recipe.pins == Pins::default() {
        recipe.pins = base;
    } else if recipe.pins.weights.name() == "f32" && recipe.pins.named.is_empty() {
        // Keep checkpoint weight forms when recipe overrides float pins.
        let over = recipe.pins;
        recipe.pins = base;
        if over.activations.name() != "f32" {
            recipe.pins = recipe.pins.activations(over.activations);
        }
        if over.kept.name() != "f32" {
            recipe.pins = recipe.pins.kept(over.kept);
        }
        if over.outputs.name() != "f32" {
            recipe.pins = recipe.pins.outputs(over.outputs);
        }
        for (site, form) in over.sites {
            recipe.pins = recipe.pins.site(site, form);
        }
    }
    recipe.pins = kv_pin(options, recipe.pins);
    if options.require_favored {
        recipe.require_favored = true;
    }
    recipe
}

/// Loads optional user weight-name rules from `settings`.
pub(super) fn user_rules(settings: &Settings) -> Result<Vec<tessera_bundle::names::Rule>, String> {
    match &settings.names {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|text| tessera_bundle::names::rules(&text).map_err(|e| e.to_string())),
        None => Ok(Vec::new()),
    }
}

/// Prints lowering details or a boot warning on rank 0.
pub(super) fn check_lowering(o: &Options, engine: &mut Engine, extra_rows: Option<usize>) {
    if o.ranks.follows() {
        return;
    }
    match engine.lowering_report(1) {
        Ok(report) if o.explain_lowering => eprintln!("{}", report.format(12)),
        Ok(report) => {
            if let Some(warning) = report.warning("--explain-lowering") {
                eprintln!("{warning}");
            }
        }
        Err(e) => eprintln!("warning: lowering report failed: {e}"),
    }
    if o.explain_lowering
        && let Some(rows) = extra_rows
        && rows > 1
        && let Ok(report) = engine.lowering_report(rows)
    {
        eprintln!("{}", report.format(12));
    }
}

/// Fills the slot with `generate`. Returns the job queue.
pub(super) fn queue(load: &Load, generate: Generate, score: bool) -> mpsc::Receiver<Job> {
    let generate = Arc::new(generate);
    let (queue, jobs) = mpsc::channel::<Job>();
    let queued = Arc::new(Queued {
        generate: generate.clone(),
        jobs: Mutex::new(queue),
    });
    load.fill(generate.clone(), |m| {
        m.chat = Some(generate);
        m.generator = Some(queued.clone());
        m.score = score.then_some(queued as Arc<dyn Score>);
    });
    jobs
}

/// Loads the generator, fills the slot, then steps jobs.
fn open(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    let guided = load.on("--guided");
    let start = Instant::now();
    let links = o.ranks.join(&settings)?;
    let rules = user_rules(&settings)?;
    let (slots, context) = (o.slots as u64, o.max_model_len as u64);
    // Ranks keep one entry, as they did before scoring.
    let steps = match (guided, o.ranks.alone()) {
        (true, _) => Steps::Masked,
        (false, true) => Steps::Scored,
        (false, false) => Steps::Plain,
    };
    let at = [slots, context];
    let manifest = load.given("--manifest");
    let module = o.module.as_deref();
    let built = module.map(|m| tessera_bundle::load_module(&o.model, at, &rules, &o.config, m));
    let loaded = match built.transpose().map_err(|e| e.to_string())?.flatten() {
        Some(bundle) => bundle,
        None => tessera_bundle::load_named(&o.model, at, &rules, steps, &o.config, manifest)
            .map_err(|e| e.to_string())?,
    };
    let scores = loaded.graph.entries.len() > 1;
    let mut recipe = lowering(o, loaded.pins);
    if recipe.config.is_none() {
        recipe.config = loaded.config;
    }
    let at = (&recipe, &settings);
    let count = loaded.graph.weights.len() as u32;
    let mut engine = o.ranks.open(links, loaded.graph, o.max_model_len, at)?;
    if let Some(ple) = loaded.ple {
        let store = PleStore::open(ple).map_err(|e| format!("ple store: {e}"))?;
        engine.set_ple(store);
    }
    let shares: Vec<_> = (0..count).map(|at| engine.share(at)).collect();
    let shares = |at: u32| shares[at as usize];
    let loading = (loaded.weights).each_share(shares, |at, w| engine.load_weight(at, w));
    loading.map_err(|e| e.to_string())?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!("{}: loaded in {seconds:.1} s", engine.device());
    check_lowering(o, &mut engine, (o.slots > 1).then(|| o.slots.min(4)));
    if o.ranks.follows() {
        return engine.follow().map_err(|e| e.to_string());
    }
    let guides = match (guided, engine.words()) {
        (false, _) => None,
        (true, None) => return Err("the model has no mask port".into()),
        (true, Some(words)) => Some(
            tessera_tasks::guides(&loaded.tokenizer, &loaded.stop, words)
                .map_err(|e| e.to_string())?,
        ),
    };
    let generate = Generate {
        tokenizer: loaded.tokenizer,
        template: loaded.template,
        stop: loaded.stop,
        guides,
    };
    let jobs = queue(load, generate, scores);
    let mut batch = Batch::new(&mut engine, Box::new(Fair));
    if let Some(rows) = o.max_batch_rows {
        batch.budget(rows);
    }
    if load.on("--reuse-prefix") {
        batch.reuse();
    }
    loop {
        // Takes every job that has arrived. Waits only when idle.
        let job = match batch.busy() {
            true => jobs.try_recv().ok(),
            false => match jobs.recv() {
                Ok(job) => Some(job),
                Err(_) => return Ok(()),
            },
        };
        if let Some((request, sink)) = job {
            batch.submit(request, sink);
            continue;
        }
        batch.step().map_err(|e| e.to_string())?;
    }
}
