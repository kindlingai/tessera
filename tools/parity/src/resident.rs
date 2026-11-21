//! Plans for recipes whose step runs as one resident grid.

use tessera_engine::Row;
use tessera_graph::DimName::{Context, Picks, Rows, Segments, Seqs};
use tessera_graph::{Graph, Parts, shard_aligned, trim};
use tessera_lower::{Cluster, Device, Limits, Plan, Platform, Recipe, lower};

/// Rank `rank` of `ranks`' plan of `g` for two rows.
pub fn rank_plan(g: &Graph, recipe: &Recipe, ranks: u32, rank: u32) -> Plan {
    let mut g = trim(g, 0).unwrap();
    // A device with a compiler, so the kernel text fits.
    let mut device = Device::default().with(recipe);
    device.platform = Platform::assumed(Limits::assumed());
    if ranks > 1 {
        let parts = Parts::even(ranks as usize);
        let sharded = shard_aligned(&g, &parts, &recipe.pins.shares(&g));
        g = sharded.unwrap().graph;
        device = device.zoned(Cluster::two("a", "b"));
        device = device.ranked((0..ranks).collect()).as_rank(rank);
    }
    let sizes = [
        (Seqs, 4),
        (Rows, 2),
        (Context, 16),
        (Segments, 1),
        (Picks, 2),
    ];
    lower(&g, 0, &sizes, &device).unwrap()
}

/// The plan's resident rules, its sums outside them, and zoning.
pub fn taken(plan: &Plan) -> (Vec<&'static str>, usize, bool) {
    let sums = plan
        .steps
        .iter()
        .filter(|s| s.kernel.name() == "all_reduce");
    let resident = plan.steps.iter().filter(|s| s.kernel.resident().is_some());
    let rules = resident.map(|s| s.rule).collect();
    (rules, sums.count(), plan.zoned())
}

/// A `prompt` of rows, then `steps` launches over two slots.
pub fn launches(prompt: u32, steps: u32) -> Vec<Vec<Row>> {
    let row = |slot, token, position| Row {
        slot,
        token,
        position,
    };
    let mut all = vec![(0..prompt).map(|p| row(0, (p * 7 + 3) % 64, p)).collect()];
    all.push(vec![row(1, 5, 0), row(1, 9, 1)]);
    for p in 0..steps {
        all.push(vec![row(0, p + 11, prompt + p), row(1, p + 2, 2 + p)]);
    }
    all
}
