//! Generation drafted by the model's own MTP layer, given `--mtp`.

use super::draft::{DraftLinks, open_target, serve, split_links};
use super::generate::user_rules;
use super::{Entry, Flag, Load, Pick};
use crate::start::Options;
use std::time::Instant;
use tessera_bundle::mtp::Mtp;
use tessera_engine::{MtpDrafter, MtpSpeculate, Place, Recipe, Settings};

pub static ENTRY: Entry = Entry {
    task: "generate",
    pick: Pick::Given("--mtp"),
    flags: &[Flag {
        name: "--mtp",
        value: "<k>",
        check: |v| v.parse::<usize>().is_ok_and(|depth| depth > 0),
    }],
    open,
};

/// Opens a single-rank or TP-sharded MTP drafter and loads weights.
fn open_mtp(
    o: &Options,
    links: Option<DraftLinks>,
    pair: &Mtp,
    settings: &Settings,
) -> Result<MtpDrafter, String> {
    let mut recipe = o.lower.clone().unwrap_or_default();
    recipe.pins = pair.pins.clone();
    recipe.kept = Recipe::default().kept;
    if o.require_favored {
        recipe.require_favored = true;
    }
    if recipe.config.is_none() {
        recipe.config = pair.target.config.clone();
    }
    let mut drafter = match links {
        Some((sums, grid)) => {
            let gpu = settings.gpu().is_some();
            let place = Place::mesh(sums.ranks() as usize, o.ranks.parts().to_vec(), gpu)
                .with_grid(grid)
                .with_moe_tp(o.ranks.moe_tp());
            MtpDrafter::open_ranked(
                pair.graph.clone(),
                o.max_model_len,
                (&recipe, settings),
                place,
                sums,
            )
        }
        None => MtpDrafter::open(pair.graph.clone(), o.max_model_len, &recipe, settings),
    }
    .map_err(|e| e.to_string())?;
    let count = drafter.weight_count();
    let shares: Vec<_> = (0..count).map(|at| drafter.share(at)).collect();
    pair.weights
        .each_share(|at| shares[at as usize], |at, w| drafter.load_weight(at, w))
        .map_err(|e| e.to_string())?;
    Ok(drafter)
}

/// Loads an MLA target and its MTP layer, then serves.
fn open(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    if load.on("--guided") {
        return Err("--mtp cannot combine with --guided yet".into());
    }
    let depth = load.count("--mtp", 1);
    let start = Instant::now();
    let (target_links, draft_links) = split_links(o.ranks.join(&settings)?);
    let rules = user_rules(&settings)?;
    let at = [1u64, o.max_model_len as u64];
    let manifest = load.given("--manifest");
    let pair = tessera_bundle::mtp::load(&o.model, at, &rules, &o.config, manifest)
        .map_err(|e| e.to_string())?;
    let mut drafter = open_mtp(o, draft_links, &pair, &settings)?;
    let (mut engine, target) = open_target(o, target_links, pair.target, &settings)?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{}: loaded target + mtp (layer {}, depth {depth}) in {seconds:.1} s",
        engine.device(),
        pair.layer
    );
    let mut spec =
        MtpSpeculate::new(&mut engine, &mut drafter, depth).map_err(|e| e.to_string())?;
    if o.ranks.follows() {
        return spec.follow().map_err(|e| e.to_string());
    }
    serve(load, &settings, target, "mtp", |req, sink| {
        spec.generate(req, sink).map_err(|e| e.to_string())
    })?;
    let _ = engine.dismiss();
    Ok(())
}
