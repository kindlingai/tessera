//! An engine as one rank of several.

use crate::{Engine, Error, Head, Row};
use sensai_engine_comm::Transport;
use sensai_graph::{Graph, Parts, shard};
use sensai_lower::{Cluster, Device, Recipe, Settings};
use sensai_run::{Rank, Run};

/// A rank's zones and each rank's share of split weights.
pub struct Place {
    pub cluster: Cluster,
    pub parts: Parts,
    rank: u32,
}

impl Place {
    /// One zone per rank, in rank order.
    pub fn new(cluster: Cluster, parts: Parts) -> Self {
        let rank = 0;
        Self {
            cluster,
            parts,
            rank,
        }
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
        let run = |plan| Ok(Box::new(Rank::open(plan, settings, sums)?) as Box<dyn Run>);
        let mut engine = Self::placed(graph, context, recipe, settings, Some(&place), run)?;
        engine.control = Some(control);
        Ok(engine)
    }

    /// Runs rank 0's launches until it dismisses the ranks.
    pub fn follow(&mut self) -> Result<(), Error> {
        while let Some(told) = self.told(&[])? {
            let Some((rows, picks, head)) = decode(&told) else {
                return Ok(());
            };
            self.launch(&rows, &picks, head)?;
        }
        Ok(())
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
        let mut parts = control.exchange(mine).map_err(sensai_run::Error::from)?;
        Ok(Some(parts.swap_remove(0)))
    }

    /// Rank 0 tells the others its launch.
    pub(super) fn announce(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        head: &Head,
    ) -> Result<(), Error> {
        match &self.control {
            Some(control) if control.rank() == 0 => {}
            _ => return Ok(()),
        }
        self.told(&encode(rows, picks, head)).map(|_| ())
    }
}

/// A launch as words: counts, head, rows, picks, list.
fn encode(rows: &[Row], picks: &[u32], head: &Head) -> Vec<u8> {
    let (tag, k, list): (u32, usize, &[u32]) = match head {
        Head::Full(None) => (0, 0, &[]),
        Head::Full(Some(masks)) => (1, 0, masks),
        Head::Sparse(ids, k) => (2, *k, ids),
    };
    let mut words = vec![rows.len() as u32, picks.len() as u32, tag, k as u32];
    for r in rows {
        words.extend([r.slot, r.token, r.position]);
    }
    words.extend(picks);
    words.extend(list);
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The launch `bytes` hold, if they hold one.
fn decode(bytes: &[u8]) -> Option<(Vec<Row>, Vec<u32>, Head)> {
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let (&[rows, picks, tag, k], rest) = words.split_first_chunk::<4>()?;
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
    Some((rows.chunks(3).map(row).collect(), picks.to_vec(), head))
}
