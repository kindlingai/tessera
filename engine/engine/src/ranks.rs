//! An engine as one rank of several.

use crate::{Engine, Error, Head, Row};
use tessera_engine_comm::{Reader, Transport, Writer};
use tessera_graph::{DType, Graph, Parts, shard_mode};
use tessera_lower::{Cluster, Device, Pins, Recipe, Settings, Wants, Wires, ZoneKind};
use tessera_run::{Profile, Rank, Ranks, Run, Sum, sum_wire};

/// A rank's zones and each rank's share of split weights.
pub struct Place {
    pub cluster: Cluster,
    pub parts: Parts,
    pub(crate) rank: u32,
    /// A persistent grid for the sums, if any.
    pub(crate) grid: Option<Box<dyn Sum>>,
    /// Split routed experts along inner width, not experts.
    pub(crate) moe_tp: bool,
    /// The wire dtype for sums the model leaves unpinned.
    pub(crate) wire: Option<DType>,
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
    pub(crate) fn settle(&mut self, grid: Option<&dyn Sum>) {
        self.wire = Some(sum_wire(self.wire, grid));
    }

    /// Pins `g`'s sums: model pins, then `wires`, then setup.
    pub(crate) fn resolve(&self, g: Graph, wires: &Wires) -> Graph {
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

    pub(crate) fn split(&self, g: &Graph, (pins, wires): (&Pins, &Wires)) -> Result<Graph, Error> {
        let sharded = shard_mode(g, &self.parts, &pins.shares(g), self.moe_tp)?;
        for w in sharded.whole.iter().filter(|_| self.rank == 0) {
            let path = &g.weights[w.weight as usize].path;
            eprintln!("tessera: {path:?} stays whole: {}", w.why);
        }
        Ok(self.resolve(sharded.graph, wires))
    }

    /// `device`, lowering this rank's copy of the plan.
    pub(crate) fn device(&self, device: Device) -> Device {
        let ranks = self.parts.ranks() as u32;
        let device = device.zoned(self.cluster.clone());
        let device = device.ranked((0..ranks).collect());
        device.uneven(self.parts.0.clone()).as_rank(self.rank)
    }
}

/// Opens a ranked runner with an optional persistent sum grid.
pub(crate) fn open_rank<T: Transport + 'static>(
    plan: tessera_lower::Plan,
    settings: &Settings,
    sums: T,
    grid: Option<Box<dyn Sum>>,
) -> Result<Box<dyn Run>, Error> {
    Ok(Box::new(Rank::open(plan, settings, sums)?.with_grid(grid)))
}

impl Engine {
    /// Opens the rank `sums` names. Launches cross on `control`.
    pub fn open_ranked<T: Transport + 'static>(
        graph: Graph,
        context: usize,
        (recipe, settings): (&Recipe, &Settings),
        mut place: Place,
        sums: T,
        control: Box<dyn Transport>,
    ) -> Result<Self, Error> {
        place.rank = sums.rank();
        let grid = place.grid.take();
        place.settle(grid.as_deref());
        let run = |plan| open_rank(plan, settings, sums, grid);
        let mut engine = Self::placed(graph, context, recipe, settings, Some(&place), run)?;
        engine.control = Some(control);
        Ok(engine)
    }

    /// Runs rank 0's launches until it dismisses the ranks.
    pub fn follow(&mut self) -> Result<(), Error> {
        loop {
            let held = self.held.take();
            let Some(told) = self.told(&held.map_or(Vec::new(), |(_, p)| p.to_bytes()))? else {
                return Ok(());
            };
            let Some(t) = decode(&told) else {
                return Ok(());
            };
            if let (Some(lanes), false) = (&mut self.lanes, t.table.is_empty()) {
                lanes.replace(t.table);
            }
            self.run(&t.rows, &t.picks, t.head, t.wants)?;
            self.report();
        }
    }

    /// Ends each other rank's `follow`.
    pub fn dismiss(&mut self) -> Result<(), Error> {
        self.told(&[]).map(|_| ())
    }

    /// Rank 0's bytes of one round. None without ranks.
    fn told(&mut self, mine: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let Some(control) = &mut self.control else {
            return Ok(None);
        };
        let lead = control.rank() == 0;
        let mut parts = control.exchange(mine).map_err(tessera_run::Error::from)?;
        if let (true, Some((launches, own))) = (lead, self.held.take()) {
            let mut profiles = vec![Some(own)];
            profiles.extend(parts[1..].iter().map(|p| Profile::from_bytes(p)));
            let ranks = Ranks { launches, profiles };
            eprintln!("profile of {launches} launches:\n{ranks}");
            self.ranks = Some(ranks);
        }
        Ok(Some(parts.swap_remove(0)))
    }

    /// The last profiles rank 0 gathered, one per rank.
    pub fn ranks(&self) -> Option<&Ranks> {
        self.ranks.as_ref()
    }

    /// Rank 0 tells the others its launch and changed table.
    pub(super) fn announce(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        head: &Head,
        wants: Wants,
    ) -> Result<(), Error> {
        match &self.control {
            Some(control) if control.rank() == 0 => {}
            _ => return Ok(()),
        }
        let table = self
            .lanes
            .as_ref()
            .and_then(crate::lanes::Lanes::pending)
            .unwrap_or(&[]);
        let bytes = encode(rows, picks, head, wants, table);
        self.told(&bytes).map(|_| ())
    }
}

/// A launch as words: head, wants, table, rows, picks, list.
fn encode(rows: &[Row], picks: &[u32], head: &Head, wants: Wants, table: &[u32]) -> Vec<u8> {
    let (tag, k, list): (u32, usize, &[u32]) = match head {
        Head::Full(None) => (0, 0, &[]),
        Head::Full(Some(masks)) => (1, 0, masks),
        Head::Sparse(ids, k) => (2, *k, ids),
    };
    let count = 7 + table.len() + rows.len() * 3 + picks.len() + list.len();
    let mut words = Writer(Vec::with_capacity(count * 4));
    words
        .u32(tag)
        .u32(k as u32)
        .u32(wants.0)
        .u32s(table)
        .u32((rows.len() * 3) as u32);
    for r in rows {
        words.u32(r.slot).u32(r.token).u32(r.position);
    }
    words.u32s(picks).u32s(list);
    words.0
}

/// One launch from rank 0.
struct Told {
    rows: Vec<Row>,
    picks: Vec<u32>,
    head: Head,
    wants: Wants,
    /// The page table, or empty if it did not change.
    table: Vec<u32>,
}

/// The launch `bytes` hold, if they hold one.
fn decode(bytes: &[u8]) -> Option<Told> {
    let mut words = Reader(bytes);
    let (tag, k) = (words.u32()?, words.u32()?);
    let (wants, table) = (Wants(words.u32()?), words.u32s()?);
    let count = words.u32()? as usize;
    let mut raw = Reader(words.take(count.checked_mul(4)?)?);
    let mut rows = Vec::with_capacity(count / 3);
    for _ in 0..count / 3 {
        rows.push(Row {
            slot: raw.u32()?,
            token: raw.u32()?,
            position: raw.u32()?,
        });
    }
    let (picks, list) = (words.u32s()?, words.u32s()?);
    let head = match tag {
        0 => Head::Full(None),
        1 => Head::Full(Some(list)),
        _ => Head::Sparse(list, k as usize),
    };
    Some(Told {
        rows,
        picks,
        head,
        wants,
        table,
    })
}
