//! Generation with a DFlash drafter, when `--draft` is given.

use super::generate::{check_lowering, guides, lowering, queue, user_rules};
use super::{Entry, Flag, Load, Pick};
use crate::ranks::{Grids, Links};
use crate::start::Options;
use std::time::Instant;
use tessera_bundle::Bundle;
use tessera_bundle::dflash::DFlash;
use tessera_engine::comm::Transport;
use tessera_engine::sums::Sum;
use tessera_engine::{
    CostModel, Drafter, Engine, Place, PleStore, Recipe, Request, Settings, Sink, SpecLen,
    SpecStats, Speculate, Steps,
};
use tessera_tasks::{Generate, Steer};

pub static ENTRY: Entry = Entry {
    task: "generate",
    pick: Pick::Given(&["--draft"]),
    flags: &[Flag::text("--draft", "<dir>"), ADAPTIVE, SPEC_LEN],
    open,
};

/// Picks the draft length from measured acceptance.
pub(super) const ADAPTIVE: Flag = Flag::switch("--spec-adaptive");

/// A fixed or largest draft length.
pub(super) const SPEC_LEN: Flag = Flag {
    name: "--spec-len",
    value: "<k>",
    check: |v| v.parse::<usize>().is_ok_and(|k| k > 0),
};

/// The draft-length policy, at most `limit` tokens.
pub(super) fn spec_len(load: &Load, limit: usize, cost: CostModel, mtp_seeds: bool) -> SpecLen {
    let cap = load.count("--spec-len", limit).min(limit);
    let spec = if load.on("--spec-adaptive") {
        SpecLen::adaptive(cap, cost)
    } else {
        SpecLen::fixed(cap).with_cost(cost)
    };
    if mtp_seeds {
        spec.with_mtp_seeds()
    } else {
        spec
    }
}

/// Opens a ranked target engine and loads its sharded weights.
pub(super) fn open_target(
    o: &Options,
    links: Option<Links>,
    target: Bundle,
    settings: &Settings,
) -> Result<(Engine, Bundle), String> {
    let mut recipe = lowering(o, target.pins.clone());
    if recipe.config.is_none() {
        recipe.config = target.config.clone();
    }
    let count = target.graph.weights.len() as u32;
    let mut engine = o.ranks.open(
        links,
        target.graph.clone(),
        o.max_model_len,
        (&recipe, settings),
    )?;
    if let Some(ple) = target.ple.clone() {
        let store = PleStore::open(ple).map_err(|e| format!("ple store: {e}"))?;
        engine.set_ple(store);
    }
    let shares: Vec<_> = (0..count).map(|at| engine.share(at)).collect();
    let shares = |at: u32| shares[at as usize];
    (target.weights)
        .each_counted(&o.progress, count, shares, |at, w| engine.load_weight(at, w))
        .map_err(|e| e.to_string())?;
    Ok((engine, target))
}

/// The timer, links and rules a speculative entry starts with.
pub(super) struct Begin<'a> {
    pub start: Instant,
    pub links: Option<Links>,
    pub rules: Vec<tessera_bundle::names::Rule>,
    pub at: [u64; 2],
    pub manifest: Option<&'a str>,
    /// The serving mode that refuses guides, if any.
    pub mode: Option<&'static str>,
    pub steps: Steps,
}

/// Refuses `--guided` where guides cannot run, then joins the ranks.
pub(super) fn begin<'a>(load: &'a Load, settings: &Settings, batched: bool) -> Result<Begin<'a>, String> {
    let mode = match (batched, load.options.ranks.alone()) {
        (true, _) => Some("batched speculative decoding"),
        (false, false) => Some("speculative decoding over ranks"),
        (false, true) => None,
    };
    let guided = load.on("--guided");
    if let Some(mode) = mode
        && guided
    {
        return Err(format!("{mode} cannot combine with --guided"));
    }
    let start = Instant::now();
    let links = load.options.ranks.join(settings, &load.options.knobs())?;
    Ok(Begin {
        start,
        links,
        rules: user_rules(settings)?,
        at: [1, load.options.max_model_len as u64],
        manifest: load.given("--manifest"),
        mode,
        steps: if guided { Steps::Masked } else { Steps::Plain },
    })
}

/// The front for `target`, with guides when serving solo.
pub(super) fn front(
    load: &Load,
    mode: Option<&'static str>,
    engine: &Engine,
    target: Bundle,
) -> Result<Generate, String> {
    let steer = match mode {
        Some(mode) => Steer::Unguided(mode),
        None => Steer::Guided(guides(load, engine, &target.tokenizer, &target.stop)?),
    };
    Ok(Generate {
        tokenizer: target.tokenizer,
        template: target.template,
        response: target.response,
        stop: target.stop,
        steer,
        sampling: false,
    })
}

/// Transport and optional sum grid handed to a ranked drafter.
pub(super) type DraftLinks = (Box<dyn Transport>, Option<Box<dyn Sum>>);

/// Shares `links`'s sums and grid across target and drafter.
pub(super) fn split_links(links: Option<Links>) -> (Option<Links>, Option<DraftLinks>) {
    let Some(([sums, control], grids)) = links else {
        return (None, None);
    };
    let [target_sums, draft_sums] = Drafter::share_transport(sums);
    let (target_grids, draft_grid) = match grids {
        Grids::Sums(grid) => {
            let [g0, g1] = Drafter::share_sum(grid);
            (Grids::Sums(g0), Some(g1))
        }
        other => (other, None),
    };
    (
        Some(([target_sums, control], target_grids)),
        Some((draft_sums, draft_grid)),
    )
}

/// Opens a single-rank or TP-sharded drafter and streams its weights.
pub(super) fn open_drafter(
    o: &Options,
    links: Option<DraftLinks>,
    pair: &DFlash,
    settings: &Settings,
) -> Result<Drafter, String> {
    let base = Recipe {
        pins: super::generate::kv_pin(o, pair.pins.clone()),
        sums: o.lower.as_ref().map(|r| r.sums.clone()).unwrap_or_default(),
        ..Recipe::default()
    };
    let recipe = super::generate::dtyped(o, base);
    let mut drafter = match links {
        Some((sums, grid)) => {
            let gpu = settings.gpu().is_some();
            let place = Place::mesh(sums.ranks() as usize, o.ranks.parts().to_vec(), gpu)
                .with_wire(o.ranks.sum_wire());
            Drafter::open_ranked(
                pair.graph.clone(),
                o.max_model_len,
                pair.block,
                pair.mask_token,
                (&recipe, settings),
                place,
                sums,
                grid,
            )
        }
        None => Drafter::open(
            pair.graph.clone(),
            o.max_model_len,
            pair.block,
            pair.mask_token,
            &recipe,
            settings,
        ),
    }
    .map_err(|e| e.to_string())?;
    drafter.set_causal(pair.causal);
    let count = drafter.weight_count();
    let shares: Vec<_> = (0..count).map(|at| drafter.share(at)).collect();
    let take = |at, w: &_| drafter.load_weight(at, w);
    pair.weights
        .each_counted(&o.progress, count, |at| shares[at as usize], take)
        .map_err(|e| e.to_string())?;
    Ok(drafter)
}

/// Loads a target and DFlash drafter, then answers requests.
fn open(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    let draft_dir = std::path::Path::new(load.given("--draft").unwrap_or_default());
    let b = begin(load, &settings, false)?;
    let (target_links, draft_links) = split_links(b.links);
    let (start, rules, at, manifest) = (b.start, b.rules, b.at, b.manifest);
    let pair = tessera_bundle::dflash::load(
        [&o.model, draft_dir],
        (at, b.steps),
        &rules,
        &o.config,
        manifest,
    )
    .map_err(|e| e.to_string())?;
    let mut drafter = open_drafter(o, draft_links, &pair, &settings)?;
    let (mut engine, target) = open_target(o, target_links, pair.target, &settings)?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{}: loaded target + dflash (block {}) in {seconds:.1} s",
        engine.device(),
        pair.block
    );
    check_lowering(o, &mut engine, Some(pair.block.min(16)));
    let generate = front(load, b.mode, &engine, target)?;
    let mut spec = Speculate::new(&mut engine, &mut drafter).map_err(|e| e.to_string())?;
    if let Some(rows) = o.max_batch_rows {
        spec = spec.with_prefill_chunk(rows);
    }
    if o.ranks.follows() {
        return spec.follow().map_err(|e| e.to_string());
    }
    let sl = spec_len(load, pair.block - 1, CostModel::dflash(), false);
    let mut spec = spec.with_spec_len(sl);
    serve(load, &settings, generate, "dflash", |req, sink| {
        spec.generate(req, sink).map_err(|e| e.to_string())
    })?;
    let _ = engine.dismiss();
    Ok(())
}

/// Serves speculative decode jobs over one target and drafter.
pub(super) fn serve(
    load: &Load,
    settings: &Settings,
    generate: Generate,
    label: &str,
    mut run: impl FnMut(Request, &mut Sink) -> Result<SpecStats, String>,
) -> Result<(), String> {
    let jobs = queue(load, generate, false);
    while let Ok((request, mut sink)) = jobs.recv() {
        let t0 = Instant::now();
        let stats = run(request, &mut sink)?;
        if settings.profile.is_some() && stats.steps > 0 {
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "{label}: {} tok in {} steps ({:.2} tok/step, {:.1} ms/step, {:.1} ms, dflash {}, mtp {}, sw {})",
                stats.emitted,
                stats.steps,
                stats.mean_step(),
                ms / stats.steps as f64,
                ms,
                stats.dflash_steps,
                stats.mtp_steps,
                stats.switches,
            );
        }
    }
    Ok(())
}
