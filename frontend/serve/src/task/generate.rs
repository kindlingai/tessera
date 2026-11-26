//! Text generation, batched over slots and ranks.

use super::{Entry, Flag, Load, Pick};
use crate::ranks::Links;
use crate::start::Options;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;
use tessera_bundle::Weights;
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::kv::KvConfig;
use tessera_engine::progress::{Progress, Unit};
use tessera_engine::turns::{Asker, Seated, Turn};
use tessera_engine::{Batch, Engine, Error, Fair, Graph, Parked, Pins, PleConfig, PleStore};
use tessera_engine::{Recipe, Settings, Steps};
use tessera_tasks::api::Score;
use tessera_tasks::{Generate, Guides, Job, Queued, Steer};

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

/// Whether KV store settings require the paged KV layout.
pub(super) fn wants_paged(o: &Options, settings: &Settings) -> bool {
    if !o.kv_stores.is_empty() {
        return KvConfig::parse_all(&o.kv_stores, o.kv_keep)
            .is_ok_and(|c| !c.disabled && !c.tiers.is_empty());
    }
    settings.gpu().is_some() && !tessera_engine::sizer::platform(settings).integrated
}

/// Whether sparse MLA models should enable the host tier.
pub(super) fn wants_host_tier(o: &Options, settings: &Settings, multi_plain: bool) -> bool {
    if multi_plain || o.kv_cache_tokens.is_none() || settings.gpu().is_none() {
        return false;
    }
    if tessera_engine::sizer::platform(settings).integrated {
        return false;
    }
    if o.kv_stores.is_empty() {
        return true;
    }
    KvConfig::parse_all(&o.kv_stores, o.kv_keep).is_ok_and(|c| {
        !c.disabled
            && c.tiers.len() == 1
            && matches!(c.tiers[0].store, tessera_engine::kv::StoreKind::Ram { .. })
    })
}

/// Splits sparse MLA KV and index VRAM budgets.
pub(super) fn apply_host_tier(
    o: &Options,
    settings: &Settings,
    recipe: &mut Recipe,
    multi_plain: bool,
) {
    if !wants_host_tier(o, settings, multi_plain) {
        return;
    }
    let Some(tokens) = o.kv_cache_tokens else {
        return;
    };
    let full = o.slots.saturating_mul(o.max_model_len);
    if full <= tokens {
        return;
    }
    let cap = full.min(tokens.saturating_mul(5) / 2);
    let kv_vram = tokens
        .saturating_mul(704)
        .saturating_sub(cap.saturating_mul(128))
        / 576;
    let kv_vram = kv_vram.max(tokens / 2).min(tokens);
    recipe.share_index(kv_vram, cap);
}

/// Resolves explicit `--kv-store` flags or default discrete-GPU RAM tier.
pub(super) fn resolve_kv(
    o: &Options,
    settings: &Settings,
    eng: &Engine,
) -> Result<Option<KvConfig>, String> {
    if !o.kv_stores.is_empty() {
        let mut cfg = KvConfig::parse_all(&o.kv_stores, o.kv_keep)?;
        if let Some(n) = o.kv_system_after {
            cfg.system_after = n;
        }
        return Ok((!cfg.disabled && !cfg.tiers.is_empty()).then_some(cfg));
    }
    if settings.gpu().is_none() || eng.integrated() {
        return Ok(None);
    }
    let [total, avail] = tessera_engine::sizer::host();
    let floor = (8u64 << 30).min(total / 4);
    let ranks = eng.rank_count().max(1) as u64;
    let bytes = (avail.saturating_sub(floor) / (2 * ranks)).min(64u64 << 30);
    if bytes < (256u64 << 20) {
        return Ok(None);
    }
    let mut cfg = KvConfig::default_ram(bytes);
    if let Some(keep) = o.kv_keep {
        cfg.keep = keep;
        cfg.tiers[0].keep = Some(keep);
    }
    if let Some(n) = o.kv_system_after {
        cfg.system_after = n;
    }
    Ok(Some(cfg))
}

/// Attaches the resolved KV store to `eng`. Returns whether active.
pub(super) fn attach_kv(
    o: &Options,
    settings: &Settings,
    eng: &mut Engine,
) -> Result<bool, String> {
    let explicit = !o.kv_stores.is_empty();
    let Some(cfg) = resolve_kv(o, settings, eng)? else {
        return Ok(false);
    };
    match eng.set_kv_store(&cfg) {
        Ok(()) => {
            if !o.ranks.follows() {
                if let Some((vram, host, cap)) = eng.host_tier_info() {
                    eprintln!(
                        "kv host tier: {vram} vram + {host} pinned host ({cap} active tokens)"
                    );
                } else {
                    eprintln!("kv store: {}", cfg.describe());
                }
            }
            Ok(true)
        }
        Err(Error::Stateful) if !explicit => Ok(false),
        Err(e) => Err(format!("kv store: {e}")),
    }
}

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

/// Folds the recipe's dtype and shape rules, then the flags'.
pub(super) fn dtyped(options: &Options, mut recipe: Recipe) -> Recipe {
    let filed = options.lower.iter().flat_map(|r| r.dtype.iter());
    recipe.dtype = filed.chain(&options.dtypes).cloned().collect();
    recipe.shape = options.knobs();
    recipe.settle();
    recipe
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
    if let Some(tokens) = options.kv_cache_tokens {
        recipe.share_kept(tokens);
    }
    if options.require_favored {
        recipe.require_favored = true;
    }
    dtyped(options, recipe)
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

/// Distinct decode row buckets to warm at boot.
pub(crate) fn warm_rows(slots: usize, extra_rows: Option<usize>) -> Vec<usize> {
    let mut rows = vec![1];
    let mut n = 2usize;
    while n < slots {
        rows.push(n);
        n = n.saturating_mul(2);
    }
    if slots > 1 {
        rows.push(slots);
    }
    if let Some(extra) = extra_rows.filter(|&r| r > 1)
        && !rows.contains(&extra)
    {
        rows.push(extra);
    }
    rows
}

/// Reports lowering on rank 0 and warms plans.
pub(super) fn check_lowering(
    o: &Options,
    engine: &mut Engine,
    extra_rows: Option<usize>,
) -> Result<(), String> {
    let show = !o.ranks.follows();
    let chunk = engine.chunk(o.max_batch_rows);
    if show {
        let mut fell = false;
        match engine.lowering_report(1) {
            Ok(report) => {
                fell |= report.has_fallbacks();
                if let Some(text) = report.notice(o.explain_lowering, "--explain-lowering") {
                    eprintln!("{text}");
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
        // A prefill step off the fast path runs far slower.
        match engine.prefill_report(chunk) {
            Ok(_) if chunk == 1 => {}
            Ok(report) => {
                fell |= report.has_fallbacks();
                if let Some(text) = report.notice(o.explain_lowering, "--explain-lowering") {
                    eprintln!("{text}");
                }
            }
            Err(e) => eprintln!("warning: prefill lowering report failed: {e}"),
        }
        if fell && !o.allow_failed_lowering {
            return Err(
                "lowering fell off fast path - pass --allow-failed-lowering to boot".into(),
            );
        }
    }
    if chunk > 1
        && let Err(e) = engine.warm(chunk, true)
        && show
    {
        eprintln!("warning: prefill warm failed: {e}");
    }
    for rows in warm_rows(o.slots, extra_rows) {
        if let Err(e) = engine.warm(rows, false)
            && show
        {
            eprintln!("warning: decode warm ({rows}) failed: {e}");
        }
    }
    Ok(())
}

/// Guides under `--guided`, sized by the engine's mask port.
pub(super) fn guides(
    load: &Load,
    engine: &Engine,
    tokenizer: &Hf,
    stop: &[u32],
) -> Result<Option<Arc<dyn Guides>>, String> {
    match (load.on("--guided"), engine.words()) {
        (false, _) => Ok(None),
        (true, None) => Err("the model has no mask port".into()),
        (true, Some(words)) => tessera_tasks::guides(tokenizer, stop, words)
            .map(Some)
            .map_err(|e| e.to_string()),
    }
}

/// Fills the slot with `generate`. Returns the job queue.
pub(super) fn queue(load: &Load, generate: Generate, score: bool) -> mpsc::Receiver<Job> {
    seated(load, generate, score, None)
}

/// As `queue`. Each job that enters the queue tells `seat`.
fn seated(
    load: &Load,
    generate: Generate,
    score: bool,
    seat: Option<Asker>,
) -> mpsc::Receiver<Job> {
    let generate = Arc::new(generate);
    let (queue, jobs) = mpsc::channel::<Job>();
    let sent = seat.map(|s| Box::new(move || s.asked()) as Box<dyn Fn() + Send + Sync>);
    let queued = Arc::new(Queued {
        generate: generate.clone(),
        jobs: Mutex::new(queue),
        sent,
    });
    load.fill(generate.clone(), |m| {
        m.chat = Some(generate);
        m.generator = Some(queued.clone());
        m.score = score.then_some(queued as Arc<dyn Score>);
    });
    jobs
}

/// A model's graph and weights, kept for each reload.
struct Source<'a> {
    o: &'a Options,
    graph: Graph,
    weights: Weights,
    recipe: Recipe,
    settings: Settings,
    ple: Option<PleConfig>,
}

impl Source<'_> {
    /// Opens the engine and loads every weight.
    fn engine(&self, links: Option<Links>, progress: &Progress) -> Result<Engine, String> {
        let o = self.o;
        let at = (&self.recipe, &self.settings);
        let count = self.graph.weights.len() as u32;
        let compiling = progress.step("compile", Unit::Items, None);
        let mut engine = o.ranks.open(links, self.graph.clone(), o.max_model_len, at)?;
        compiling.finish();
        if let Some(ple) = self.ple.clone() {
            let store = PleStore::open(ple).map_err(|e| format!("ple store: {e}"))?;
            engine.set_ple(store);
        }
        let shares: Vec<_> = (0..count).map(|at| engine.share(at)).collect();
        let shares = |at: u32| shares[at as usize];
        let take = |at, w: &_| engine.load_weight(at, w);
        let loading = (self.weights).each_counted(progress, count, shares, take);
        loading.map_err(|e| e.to_string())?;
        engine.ready().map_err(|e| e.to_string())?;
        Ok(engine)
    }

    /// Waits for room, loads, and tells the seat the cost.
    fn enter(&self, seat: &Seated, links: Option<Links>, p: &Progress) -> Result<Engine, String> {
        while seat.next() != Turn::Load {}
        let start = Instant::now();
        let engine = self.engine(links, p)?;
        let seconds = start.elapsed().as_secs_f64();
        seat.loaded(engine.bytes(), seconds, engine.memory());
        let name = self.o.name.as_deref().unwrap_or("model");
        eprintln!("{name} on {}: loaded in {seconds:.1} s", engine.device());
        Ok(engine)
    }
}

/// Steps `batch` on its turns until the model must leave.
fn serve(mut batch: Batch, jobs: &mpsc::Receiver<Job>, seat: &Seated) -> Result<Parked, String> {
    loop {
        match seat.next() {
            Turn::Step { admit } => batch.admitting(admit),
            Turn::Leave => return Ok(batch.park()),
            Turn::Load => continue,
        }
        // Takes every job that has arrived.
        for (request, sink) in jobs.try_iter() {
            batch.submit(request, sink);
        }
        batch.step().map_err(|e| e.to_string())?;
        seat.stepped((batch.busy(), batch.running()), batch.bytes());
    }
}

/// Loads the generator, fills the slot, then steps jobs.
fn open(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    let guided = load.on("--guided");
    let links = o.ranks.join(&settings, &o.knobs())?;
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
    let reading = o.progress.step("read model", Unit::Items, None);
    let built = module.map(|m| tessera_bundle::load_module(&o.model, at, &rules, &o.config, manifest, m));
    let loaded = match built.transpose().map_err(|e| e.to_string())?.flatten() {
        Some(bundle) => bundle,
        None => tessera_bundle::load_named(&o.model, at, &rules, steps, &o.config, manifest)
            .map_err(|e| e.to_string())?,
    };
    reading.finish();
    let scores = loaded.graph.entries.len() > 1;
    let mut recipe = lowering(o, loaded.pins);
    if wants_paged(o, &settings) && recipe.kept.page().is_none() {
        recipe.kept = Engine::paged_layout();
    }
    apply_host_tier(o, &settings, &mut recipe, !o.ranks.alone());
    if recipe.config.is_none() {
        recipe.config = loaded.config;
    }
    let source = Source {
        o,
        graph: loaded.graph,
        weights: loaded.weights,
        recipe,
        settings: settings.clone(),
        ple: loaded.ple,
    };
    let seat = load.seat();
    let mut engine = source.enter(&seat, links, &o.progress)?;
    let kv_on = attach_kv(o, &settings, &mut engine)?;
    check_lowering(o, &mut engine, (o.slots > 1).then(|| o.slots.min(4)))?;
    if o.ranks.follows() {
        return engine.follow().map_err(|e| e.to_string());
    }
    let generate = Generate {
        steer: Steer::Guided(guides(load, &engine, &loaded.tokenizer, &loaded.stop)?),
        tokenizer: loaded.tokenizer,
        template: loaded.template,
        response: loaded.response,
        stop: loaded.stop,
        sampling: true,
    };
    let jobs = seated(load, generate, scores, Some(seat.asker()));
    let mut parked = Parked::default();
    loop {
        let mut batch = Batch::resumed(&mut engine, Box::new(Fair), parked);
        if let Some(rows) = o.max_batch_rows {
            batch.budget(rows);
        }
        if load.on("--reuse-prefix") || kv_on {
            batch.reuse();
        }
        parked = serve(batch, &jobs, &seat)?;
        drop(engine);
        seat.left();
        engine = source.enter(&seat, None, &Progress::default())?;
        let _ = attach_kv(o, &settings, &mut engine)?;
    }
}
