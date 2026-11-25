//! MTP drafting, given `--mtp`. DFlash joins when `--draft` is given.

use super::draft::{
    ADAPTIVE, DraftLinks, SPEC_LEN, begin, open_drafter, open_target, serve, spec_len, split_links,
};
use super::generate::{check_lowering, queue};
use super::{Entry, Flag, Load, Pick};
use crate::ranks::{Grids, Links};
use crate::start::Options;
use tessera_bundle::Bundle;
use tessera_bundle::mtp::Mtp;
use tessera_engine::{
    BatchMtp, CostModel, Drafter, DualSelector, DualSpeculate, Engine, MtpDrafter, MtpSpeculate,
    Place, Recipe, Settings, Steps,
};
use tessera_tasks::Generate;

pub static ENTRY: Entry = Entry {
    task: "generate",
    pick: Pick::Given(&["--mtp"]),
    flags: &[
        Flag {
            name: "--mtp",
            value: "<k>",
            check: |v| v.parse::<usize>().is_ok_and(|depth| depth > 0),
        },
        ADAPTIVE,
        SPEC_LEN,
    ],
    open,
};

/// Opens a single-rank or TP-sharded MTP drafter and loads weights.
pub(super) fn open_mtp(
    o: &Options,
    links: Option<DraftLinks>,
    pair: &Mtp,
    settings: &Settings,
) -> Result<MtpDrafter, String> {
    let mut recipe = o.lower.clone().unwrap_or_default();
    recipe.pins = super::generate::kv_pin(o, pair.pins.clone());
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
                .with_moe_tp(o.ranks.moe_tp())
                .with_wire(o.ranks.sum_wire());
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
    let take = |at, w: &_| drafter.load_weight(at, w);
    pair.weights
        .each_counted(&o.progress, count, |at| shares[at as usize], take)
        .map_err(|e| e.to_string())?;
    Ok(drafter)
}

/// Runs rank 0's job loop or follower loop on `batch`.
fn run_batch_loop(load: &Load, mut batch: BatchMtp<'_>, target: Bundle) -> Result<(), String> {
    let o = load.options;
    if let Some(rows) = o.max_batch_rows {
        batch = batch.with_prefill_chunk(rows);
    }
    if o.ranks.follows() {
        return batch.follow().map_err(|e| e.to_string());
    }
    let generate = Generate {
        tokenizer: target.tokenizer,
        template: target.template,
        response: target.response,
        stop: target.stop,
        guides: None,
    };
    let jobs = queue(load, generate, false);
    loop {
        let job = match batch.busy() {
            true => jobs.try_recv().ok(),
            false => match jobs.recv() {
                Ok(job) => Some(job),
                Err(_) => break,
            },
        };
        if let Some((request, sink)) = job {
            batch.submit(request, sink);
            continue;
        }
        batch.step().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Serves continuous-batching MTP decoding across many slots.
fn serve_batch(
    load: &Load,
    engine: &mut Engine,
    drafter: &mut MtpDrafter,
    target: Bundle,
    depth: usize,
) -> Result<(), String> {
    let cap = load.count("--spec-len", depth).min(depth);
    let batch = BatchMtp::new(engine, drafter, cap).map_err(|e| e.to_string())?;
    run_batch_loop(load, batch, target)?;
    let _ = engine.dismiss();
    Ok(())
}

/// Loads an MLA target and its MTP layer, then serves.
fn open(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    let depth = load.count("--mtp", 1);
    let b = begin(load, &settings, "--mtp")?;
    let (target_links, draft_links) = split_links(b.links);
    let (start, rules, manifest) = (b.start, b.rules, b.manifest);
    let at = [o.slots as u64, o.max_model_len as u64];
    let pair = tessera_bundle::mtp::load(&o.model, at, &rules, &o.config, manifest)
        .map_err(|e| e.to_string())?;
    let mut drafter = open_mtp(o, draft_links, &pair, &settings)?;
    let (mut engine, target) = open_target(o, target_links, pair.target, &settings)?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{}: loaded target + mtp (layer {}, depth {depth}, slots {}) in {seconds:.1} s",
        engine.device(),
        pair.layer,
        o.slots,
    );
    check_lowering(o, &mut engine, Some((o.slots * (depth + 1)).min(16)));
    if o.slots > 1 {
        return serve_batch(load, &mut engine, &mut drafter, target, depth);
    }
    if o.ranks.follows() {
        let mut spec =
            MtpSpeculate::new(&mut engine, &mut drafter, depth).map_err(|e| e.to_string())?;
        return spec.follow().map_err(|e| e.to_string());
    }
    let sl = spec_len(load, depth, CostModel::mtp(), true);
    let mut spec = MtpSpeculate::new(&mut engine, &mut drafter, depth)
        .map_err(|e| e.to_string())?
        .with_spec_len(sl);
    serve(load, &settings, target, "mtp", |req, sink| {
        spec.generate(req, sink).map_err(|e| e.to_string())
    })?;
    let _ = engine.dismiss();
    Ok(())
}

pub static DUAL: Entry = Entry {
    task: "generate",
    pick: Pick::Given(&["--draft", "--mtp"]),
    flags: &[],
    open: open_dual,
};

/// Shares `links`'s sums and grid across target and both drafters.
fn split_dual(links: Option<Links>) -> (Option<Links>, Option<DraftLinks>, Option<DraftLinks>) {
    let Some(([sums, control], grids)) = links else {
        return (None, None, None);
    };
    let [target_sums, dflash_sums, mtp_sums] = Drafter::share_transports(sums);
    let (target_grids, dflash_grid, mtp_grid) = match grids {
        Grids::Sums(grid) => {
            let [g0, g1, g2] = Drafter::share_sums(grid);
            (Grids::Sums(g0), Some(g1), Some(g2))
        }
        other => (other, None, None),
    };
    (
        Some(([target_sums, control], target_grids)),
        Some((dflash_sums, dflash_grid)),
        Some((mtp_sums, mtp_grid)),
    )
}

/// Loads an MLA target with DFlash and MTP drafters.
fn open_dual(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    let draft_dir = std::path::Path::new(load.given("--draft").unwrap_or_default());
    let depth = load.count("--mtp", 1);
    let b = begin(load, &settings, "--draft and --mtp")?;
    let (target_links, dflash_links, mtp_links) = split_dual(b.links);
    let (start, rules, at_solo, manifest) = (b.start, b.rules, b.at, b.manifest);
    let at_batch = [o.slots as u64, o.max_model_len as u64];
    // Drop each pair before the next, to limit open files.
    let (mut drafter, dflash_block, mut taps) = {
        let pair =
            tessera_bundle::dflash::load([&o.model, draft_dir], at_solo, &rules, &o.config, manifest)
                .map_err(|e| e.to_string())?;
        let drafter = open_drafter(o, dflash_links, &pair, &settings)?;
        let taps: Vec<u64> = pair.target_layers.iter().map(|&l| l + 1).collect();
        (drafter, pair.block, taps)
    };
    let (mut mtp, mtp_layer) = {
        let pair = tessera_bundle::mtp::load(&o.model, at_batch, &rules, &o.config, manifest)
            .map_err(|e| e.to_string())?;
        let mtp = open_mtp(o, mtp_links, &pair, &settings)?;
        (mtp, pair.layer)
    };
    taps.push(mtp_layer + 1);
    let combined = tessera_bundle::load_tapped(
        &o.model,
        at_batch,
        &rules,
        Steps::Plain,
        &o.config,
        manifest,
        &taps,
    )
    .map_err(|e| e.to_string())?;
    let (mut engine, target) = open_target(o, target_links, combined, &settings)?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{}: loaded target + dflash (block {dflash_block}) + mtp (layer {mtp_layer}, depth {depth}, slots {}) in {seconds:.1} s",
        engine.device(),
        o.slots,
    );
    check_lowering(o, &mut engine, Some(dflash_block.min(16)));
    let d_sl = spec_len(load, dflash_block - 1, CostModel::dual_dflash(), false);
    let m_sl = spec_len(load, depth, CostModel::dual_mtp(), true);
    let selector = DualSelector::new(d_sl, m_sl);
    if o.slots > 1 {
        let batch = BatchMtp::new_dual(&mut engine, &mut drafter, &mut mtp, depth, selector)
            .map_err(|e| e.to_string())?;
        run_batch_loop(load, batch, target)?;
        let _ = engine.dismiss();
        return Ok(());
    }
    if o.ranks.follows() {
        let mut spec = DualSpeculate::new(&mut engine, &mut drafter, &mut mtp, depth)
            .map_err(|e| e.to_string())?;
        return spec.follow().map_err(|e| e.to_string());
    }
    let mut spec = DualSpeculate::new(&mut engine, &mut drafter, &mut mtp, depth)
        .map_err(|e| e.to_string())?
        .with_selector(selector);
    serve(load, &settings, target, "dual", |req, sink| {
        spec.generate(req, sink).map_err(|e| e.to_string())
    })?;
    let _ = engine.dismiss();
    Ok(())
}
