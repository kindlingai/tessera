//! Generation with a DFlash drafter, when `--draft` is given.

use super::generate::{lowering, queue, user_rules};
use super::{Entry, Flag, Load, Pick};
use crate::ranks::{Grids, Links};
use crate::start::Options;
use std::time::Instant;
use tessera_bundle::Bundle;
use tessera_bundle::dflash::DFlash;
use tessera_engine::comm::Transport;
use tessera_engine::sums::Sum;
use tessera_engine::{
    Drafter, Engine, Place, Recipe, Request, Settings, Sink, SpecStats, Speculate,
};
use tessera_tasks::Generate;

pub static ENTRY: Entry = Entry {
    task: "generate",
    pick: Pick::Given("--draft"),
    flags: &[Flag::text("--draft", "<dir>")],
    open,
};

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
    let shares: Vec<_> = (0..count).map(|at| engine.share(at)).collect();
    let shares = |at: u32| shares[at as usize];
    (target.weights)
        .each_share(shares, |at, w| engine.load_weight(at, w))
        .map_err(|e| e.to_string())?;
    Ok((engine, target))
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
fn open_drafter(
    o: &Options,
    links: Option<DraftLinks>,
    pair: &DFlash,
    settings: &Settings,
) -> Result<Drafter, String> {
    let recipe = Recipe {
        pins: pair.pins.clone(),
        ..Recipe::default()
    };
    let mut drafter = match links {
        Some((sums, grid)) => {
            let gpu = settings.gpu().is_some();
            let place = Place::mesh(sums.ranks() as usize, o.ranks.parts().to_vec(), gpu);
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
    pair.weights
        .each_share(|at| shares[at as usize], |at, w| drafter.load_weight(at, w))
        .map_err(|e| e.to_string())?;
    Ok(drafter)
}

/// Loads a target and DFlash drafter, then answers requests.
fn open(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    if load.on("--guided") {
        return Err("--draft cannot combine with --guided yet".into());
    }
    let draft_dir = std::path::Path::new(load.given("--draft").unwrap_or_default());
    let start = Instant::now();
    let (target_links, draft_links) = split_links(o.ranks.join(&settings)?);
    let rules = user_rules(&settings)?;
    let at = [1u64, o.max_model_len as u64];
    let manifest = load.given("--manifest");
    let pair = tessera_bundle::dflash::load([&o.model, draft_dir], at, &rules, &o.config, manifest)
        .map_err(|e| e.to_string())?;
    let mut drafter = open_drafter(o, draft_links, &pair, &settings)?;
    let (mut engine, target) = open_target(o, target_links, pair.target, &settings)?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{}: loaded target + dflash (block {}) in {seconds:.1} s",
        engine.device(),
        pair.block
    );
    if o.ranks.follows() {
        let mut spec = Speculate::new(&mut engine, &mut drafter).map_err(|e| e.to_string())?;
        return spec.follow().map_err(|e| e.to_string());
    }
    let mut spec = Speculate::new(&mut engine, &mut drafter).map_err(|e| e.to_string())?;
    serve(load, &settings, target, "dflash", |req, sink| {
        spec.generate(req, sink).map_err(|e| e.to_string())
    })?;
    let _ = engine.dismiss();
    Ok(())
}

/// Serves speculative decode jobs over one target and drafter.
pub(super) fn serve(
    load: &Load,
    settings: &Settings,
    target: Bundle,
    label: &str,
    mut run: impl FnMut(Request, &mut Sink) -> Result<SpecStats, String>,
) -> Result<(), String> {
    let generate = Generate {
        tokenizer: target.tokenizer,
        template: target.template,
        stop: target.stop,
        guides: None,
    };
    let jobs = queue(load, generate, false);
    while let Ok((request, mut sink)) = jobs.recv() {
        let t0 = Instant::now();
        let stats = run(request, &mut sink)?;
        if settings.profile.is_some() && stats.steps > 0 {
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "{label}: {} tok in {} steps ({:.2} tok/step, {:.1} ms/step, {:.1} ms total)",
                stats.emitted,
                stats.steps,
                stats.mean_step(),
                ms / stats.steps as f64,
                ms,
            );
        }
    }
    Ok(())
}
