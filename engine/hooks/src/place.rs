//! Where a rank sits among several, and its runner.

use crate::Error;
use tessera_engine_comm::Transport;
use tessera_graph::{DType, Graph, Parts, Whole, shard_mode};
use tessera_lower::{Cluster, Device, Pins, Recipe, Settings, Wires, ZoneKind};
use tessera_run::{Rank, Run, Sum, sum_wire};

/// A rank's zones and each rank's share of split weights.
pub struct Place {
    pub cluster: Cluster,
    pub parts: Parts,
    pub rank: u32,
    /// A persistent grid for the sums, if any.
    pub grid: Option<Box<dyn Sum>>,
    /// Split routed experts along inner width, not experts.
    pub moe_tp: bool,
    /// The wire dtype for sums the model leaves unpinned.
    pub wire: Option<DType>,
}

impl Place {
    /// One zone per rank, in rank order.
    pub fn new(cluster: Cluster, parts: Parts) -> Self {
        let rank = 0;
        Self {
            cluster,
            parts,
            rank,
            grid: None,
            moe_tp: false,
            wire: None,
        }
    }

    /// Runs the sums through `grid`.
    pub fn with_grid(mut self, grid: Option<Box<dyn Sum>>) -> Self {
        self.grid = grid;
        self
    }

    /// Sends unpinned sums' parts as `wire`. `None`: the transport's dtype.
    pub fn with_wire(mut self, wire: Option<DType>) -> Self {
        self.wire = wire;
        self
    }

    /// Settles the wire dtype once the sums' transport is known.
    pub fn settle(&mut self, grid: Option<&dyn Sum>) {
        self.wire = Some(sum_wire(self.wire, grid));
    }

    /// Pins `g`'s sums: model pins, then `wires`, then setup.
    pub fn resolve(&self, g: Graph, wires: &Wires) -> Graph {
        let g = wires.pin(g);
        match self.wire {
            Some(wire) => g.wired(wire),
            None => g,
        }
    }

    /// Splits wide routed experts along inner width when `on`.
    pub fn with_moe_tp(mut self, on: bool) -> Self {
        self.moe_tp = on;
        self
    }

    /// A zone per rank, all linked. No `parts`: even shares.
    pub fn mesh(ranks: usize, parts: Vec<u32>, gpu: bool) -> Self {
        let kind = match gpu {
            true => ZoneKind::Gpu,
            false => ZoneKind::Cpu,
        };
        let parts = match parts.is_empty() {
            true => Parts::even(ranks),
            false => Parts(parts),
        };
        Self::new(Cluster::mesh(ranks, kind), parts)
    }

    /// Logs unsharded weights on rank 0.
    pub fn log_whole(&self, g: &Graph, tag: &str, whole: &[Whole]) {
        if self.rank != 0 {
            return;
        }
        for w in whole {
            let path = &g.weights[w.weight as usize].path;
            eprintln!("tessera: {tag}{path:?} stays whole: {}", w.why);
        }
    }

    pub fn split(&self, g: &Graph, (pins, wires): (&Pins, &Wires)) -> Result<Graph, Error> {
        let sharded = shard_mode(g, &self.parts, &pins.shares(g), self.moe_tp)?;
        self.log_whole(g, "", &sharded.whole);
        Ok(self.resolve(sharded.graph, wires))
    }

    /// `device`, lowering this rank's copy of the plan.
    pub fn device(&self, device: Device) -> Device {
        let ranks = self.parts.ranks() as u32;
        let device = device.zoned(self.cluster.clone());
        let device = device.ranked((0..ranks).collect());
        device.uneven(self.parts.0.clone()).as_rank(self.rank)
    }
}

/// Opens a ranked runner with an optional persistent sum grid.
pub fn open_rank<T: Transport + 'static>(
    plan: tessera_lower::Plan,
    settings: &Settings,
    sums: T,
    grid: Option<Box<dyn Sum>>,
) -> Result<Box<dyn Run>, Error> {
    Ok(Box::new(Rank::open(plan, settings, sums)?.with_grid(grid)?))
}

/// Configures `runner` and loads its stored device model.
pub fn stored_device(
    runner: &mut dyn Run,
    recipe: &Recipe,
    settings: &Settings,
    place: Option<&Place>,
    profile: bool,
) -> Result<Device, Error> {
    runner.launching(recipe.launch);
    if profile {
        runner.profiling(settings.profile.is_some());
    }
    let mut device = Device::stored(runner.platform(), settings)
        .map_err(tessera_lower::Error::from)?
        .with(recipe);
    if let Some(place) = place {
        device = place.device(device);
    }
    Ok(device)
}
