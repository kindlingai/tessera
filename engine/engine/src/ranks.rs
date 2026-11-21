//! An engine as one rank of several.

use crate::{Engine, Error, Head, Row};
use tessera_engine_comm::Transport;
use tessera_graph::{Graph, Parts, shard};
use tessera_lower::{Cluster, Device, Recipe, Settings, ZoneKind};
use tessera_run::{Profile, Rank, Ranks, Run, Sum};

/// A rank's zones and each rank's share of split weights.
pub struct Place {
    pub cluster: Cluster,
    pub parts: Parts,
    rank: u32,
    /// A persistent grid for the sums, if any.
    grid: Option<Box<dyn Sum>>,
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
        }
    }

    /// Runs the sums through `grid`.
    pub fn with_grid(mut self, grid: Option<Box<dyn Sum>>) -> Self {
        self.grid = grid;
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

    pub(crate) fn split(&self, g: &Graph) -> Result<Graph, Error> {
        Ok(shard(g, &self.parts)?.graph)
    }

    /// `device`, lowering this rank's copy of the plan.
    pub(crate) fn device(&self, device: Device) -> Device {
        let ranks = self.parts.ranks() as u32;
        let device = device.zoned(self.cluster.clone());
        let device = device.ranked((0..ranks).collect());
        device.uneven(self.parts.0.clone()).as_rank(self.rank)
    }
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
        let run = |plan| {
            let rank = Rank::open(plan, settings, sums)?.with_grid(grid);
            Ok(Box::new(rank) as Box<dyn Run>)
        };
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
            self.launch(&t.rows, &t.picks, t.head)?;
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
        table: &[u32],
    ) -> Result<(), Error> {
        match &self.control {
            Some(control) if control.rank() == 0 => {}
            _ => return Ok(()),
        }
        self.told(&encode(rows, picks, head, table)).map(|_| ())
    }
}

/// A launch as words: counts, head, table, rows, picks, list.
fn encode(rows: &[Row], picks: &[u32], head: &Head, table: &[u32]) -> Vec<u8> {
    let (tag, k, list): (u32, usize, &[u32]) = match head {
        Head::Full(None) => (0, 0, &[]),
        Head::Full(Some(masks)) => (1, 0, masks),
        Head::Sparse(ids, k) => (2, *k, ids),
    };
    let mut words = vec![rows.len() as u32, picks.len() as u32, tag, k as u32];
    words.push(table.len() as u32);
    words.extend(table);
    for r in rows {
        words.extend([r.slot, r.token, r.position]);
    }
    words.extend(picks);
    words.extend(list);
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// One launch from rank 0.
struct Told {
    rows: Vec<Row>,
    picks: Vec<u32>,
    head: Head,
    /// The page table, or empty if it did not change.
    table: Vec<u32>,
}

/// The launch `bytes` hold, if they hold one.
fn decode(bytes: &[u8]) -> Option<Told> {
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let (&[rows, picks, tag, k, table], rest) = words.split_first_chunk::<5>()?;
    let (table, rest) = rest.split_at_checked(table as usize)?;
    let (rows, rest) = rest.split_at_checked(3 * rows as usize)?;
    let (picks, list) = rest.split_at_checked(picks as usize)?;
    let row = |r: &[u32]| Row {
        slot: r[0],
        token: r[1],
        position: r[2],
    };
    let head = match tag {
        0 => Head::Full(None),
        1 => Head::Full(Some(list.to_vec())),
        _ => Head::Sparse(list.to_vec(), k as usize),
    };
    Some(Told {
        rows: rows.chunks(3).map(row).collect(),
        picks: picks.to_vec(),
        head,
        table: table.to_vec(),
    })
}
