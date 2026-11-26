//! An engine as one rank of several.

use crate::{Engine, Error, Head, Row};
use tessera_engine_comm::{Reader, Transport, Writer};
use tessera_graph::{DType, Graph, Parts, Whole, shard_mode};
use tessera_kv::{DiskKv, DiskMsg, KvConfig, StoreStats, decode_msg};
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

    /// Logs unsharded weights on rank 0.
    pub(crate) fn log_whole(&self, g: &Graph, tag: &str, whole: &[Whole]) {
        if self.rank != 0 {
            return;
        }
        for w in whole {
            let path = &g.weights[w.weight as usize].path;
            eprintln!("tessera: {tag}{path:?} stays whole: {}", w.why);
        }
    }

    pub(crate) fn split(&self, g: &Graph, (pins, wires): (&Pins, &Wires)) -> Result<Graph, Error> {
        let sharded = shard_mode(g, &self.parts, &pins.shares(g), self.moe_tp)?;
        self.log_whole(g, "", &sharded.whole);
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
    Ok(Box::new(Rank::open(plan, settings, sums)?.with_grid(grid)?))
}

/// Configures `runner` and loads its stored device model.
pub(crate) fn stored_device(
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
            if let Some(d) = decode_msg(&told) {
                let _ = self.disk_exec(d, false)?;
                continue;
            }
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

    /// Sets the minimum row count for split-batch overlap.
    pub fn set_split_rows(&mut self, rows: usize) {
        if let Some((_, min)) = &mut self.split {
            *min = rows;
        }
    }

    /// Rank 0's launch bytes of one round. None without ranks.
    pub(crate) fn told(&mut self, mine: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let Some(control) = &self.control else {
            return Ok(None);
        };
        // Other ranks act on profile tags here. Empty bytes dismiss.
        if control.rank() == 0 {
            let mut tagged = Vec::with_capacity(mine.len() + 4);
            if !mine.is_empty() {
                tagged.extend_from_slice(&LAUNCH.to_le_bytes());
                tagged.extend_from_slice(mine);
            }
            return self.round(&tagged).map(Some);
        }
        let mut mine = mine.to_vec();
        loop {
            let told = self.round(&mine)?;
            let Some((tag, rest)) = told.split_first_chunk::<4>() else {
                return Ok(Some(told));
            };
            match u32::from_le_bytes(*tag) {
                LAUNCH => return Ok(Some(rest.to_vec())),
                tag => self.profile_ranked(tag == PROFILE_ON),
            }
            mine = self
                .held
                .take()
                .map_or_else(Vec::new, |(_, p)| p.to_bytes());
        }
    }

    /// One exchange. Rank 0 prints its held profile.
    fn round(&mut self, mine: &[u8]) -> Result<Vec<u8>, Error> {
        let Some(control) = &mut self.control else {
            return Ok(Vec::new());
        };
        let mut parts = control.exchange(mine).map_err(tessera_run::Error::from)?;
        if control.rank() == 0 {
            self.gather(parts[1..].iter().map(|p| Profile::from_bytes(p)));
        }
        Ok(parts.swap_remove(0))
    }

    /// Prints rank 0's held profile with the other ranks'.
    pub(crate) fn gather(&mut self, others: impl Iterator<Item = Option<Profile>>) {
        let Some((launches, own)) = self.held.take() else {
            return;
        };
        let profiles = std::iter::once(Some(own)).chain(others).collect();
        let ranks = Ranks { launches, profiles };
        eprintln!("profile of {launches} launches:\n{ranks}");
        self.ranks = Some(ranks);
    }

    /// Starts or stops profiling. Stopping holds the profile.
    fn profile_ranked(&mut self, on: bool) {
        if on == self.profile.is_some() {
            return;
        }
        let launches = self.profile.take().map_or(0, |[_, done]| done);
        if let Some(profile) = self.take_profile().filter(|_| !on) {
            self.held = Some((launches, profile));
        }
        self.runner.profiling(on);
        self.profile = on.then_some([u64::MAX, 0]);
    }

    /// Profiles while requests ask. Stopping prints the profile.
    pub fn settle(&mut self) -> Result<(), Error> {
        let Some(asks) = &self.asks else {
            return Ok(());
        };
        let on = asks.wanted();
        let lead = self.control.as_ref().is_none_or(|c| c.rank() == 0);
        if !lead || on == self.profile.is_some() {
            return Ok(());
        }
        let tag = match on {
            true => PROFILE_ON,
            false => PROFILE_OFF,
        };
        let tag = tag.to_le_bytes();
        self.profile_ranked(on);
        if self.control.is_none() {
            if let Some((launches, profile)) = self.held.take() {
                eprintln!(
                    "profile of {launches} launches:\n{}",
                    profile.report(launches)
                );
            }
            return Ok(());
        }
        // Other ranks send their profiles one round after the tag.
        let held = self.held.take();
        self.round(&tag)?;
        if !on {
            self.held = held;
            self.round(&tag)?;
        }
        Ok(())
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

    /// Default paged KV layout for disk caching.
    pub fn paged_layout() -> &'static dyn tessera_lower::kept::Layout {
        tessera_lower::kept::named("paged").expect("paged layout")
    }

    /// Whether this engine runs on unified host/device memory.
    pub fn integrated(&self) -> bool {
        self.runner.platform().integrated
    }

    /// Number of ranks participating in this engine.
    pub fn rank_count(&self) -> usize {
        self.runner.plan().ranks as usize
    }

    /// Attaches a tiered KV store from `cfg`.
    pub fn set_kv_store(&mut self, cfg: &KvConfig) -> Result<(), Error> {
        if cfg.disabled || cfg.tiers.is_empty() {
            self.disk = None;
            return Ok(());
        }
        if self.stateful {
            return Err(Error::Stateful);
        }
        let page = self.lanes.as_ref().map(|l| l.page).ok_or(Error::RowPlace)?;
        let (r, n) = (self.runner.plan().rank, self.runner.plan().ranks);
        let meta = self.runner.kept_page_meta(page).ok_or(Error::RowPlace)?;
        let d = DiskKv::open_config(cfg, r, n, page, &meta).ok_or(Error::RowPlace)?;
        self.disk = Some(d.map_err(Error::Store)?);
        Ok(())
    }

    /// Disk KV cache stats, if a store is attached.
    pub fn kv_store_stats(&self) -> Option<StoreStats> {
        self.disk.as_ref().map(DiskKv::stats)
    }

    /// Runs a disk cache op locally, broadcasting when `lead`.
    pub(crate) fn disk_exec(&mut self, msg: DiskMsg, lead: bool) -> Result<usize, Error> {
        if msg.is_empty() || self.disk.is_none() {
            return Ok(0);
        }
        if lead && self.control.as_ref().is_some_and(|c| c.rank() == 0) {
            let _ = self.told(&msg.encode())?;
        }
        let (disk, run) = (self.disk.as_mut().expect("disk"), &mut self.runner);
        let (mine, sync) = disk.exec(&msg, Error::Store, |up, p, g, b| {
            let ok = if up {
                run.upload_kept_pages(p, g, b)?
            } else {
                run.download_kept_pages(p, g, b)?
            };
            ok.then_some(()).ok_or(Error::RowPlace)
        })?;
        let Some(c) = self.control.as_mut().filter(|_| sync) else {
            return Ok(if sync { mine as usize } else { 0 });
        };
        let parts = c
            .exchange(&mine.to_le_bytes())
            .map_err(tessera_run::Error::from)?;
        let f = |p: Vec<u8>| p.try_into().ok().map(u32::from_le_bytes);
        Ok(parts.into_iter().filter_map(f).fold(mine, u32::min) as usize)
    }

    /// Restores extra prefix pages from the KV store past `skip`.
    pub(crate) fn restore_kv(
        &mut self,
        kept: Option<&mut crate::kept::Kept>,
        slot: u32,
        tokens: &[u32],
        skip: usize,
        lead: bool,
    ) -> Result<usize, Error> {
        let Some(d) = self.disk.as_ref() else {
            return Ok(skip);
        };
        let (page, (vram, keys)) = (d.page, d.missing_keys(tokens, skip));
        let hit = self.disk_exec(DiskMsg::Prefix(keys.clone()), lead)?;
        let Some(kept) = kept.filter(|_| hit > 0) else {
            return Ok(skip);
        };
        if kept.grow(slot, (vram + hit) * page).is_err() {
            return Ok(skip);
        }
        let Some(pages) = kept.pages(slot).map(<[u32]>::to_vec) else {
            return Ok(skip);
        };
        let pairs = DiskKv::load_pairs_slice(&pages, keys, vram, hit);
        let loaded = self.disk_exec(DiskMsg::Load(pairs), lead)?;
        if let Some(lanes) = &mut self.lanes {
            lanes.set(slot, &pages);
        }
        Ok((vram + loaded) * page)
    }

    /// Spills full pages of `slot` to the KV store.
    pub(crate) fn spill_kv(
        &mut self,
        kept: Option<&crate::kept::Kept>,
        slot: u32,
        history: &[u32],
        lead: bool,
    ) {
        if history.is_empty() {
            return;
        }
        let (Some(d), Some(kept)) = (&self.disk, kept) else {
            return;
        };
        if let Some(pages) = kept.pages(slot) {
            let pairs = d.spill_pairs(pages, history);
            let _ = self.disk_exec(DiskMsg::Save(pairs), lead);
        }
    }
}

/// Tags on rank 0's control messages.
const LAUNCH: u32 = 0;
const PROFILE_ON: u32 = 1;
const PROFILE_OFF: u32 = 2;

/// A launch as words: head, wants, table, rows, picks, lists.
fn encode(rows: &[Row], picks: &[u32], head: &Head, wants: Wants, table: &[u32]) -> Vec<u8> {
    let (tag, k, list, edits): (u32, usize, &[u32], &[u32]) = match head {
        Head::Full(None) => (0, 0, &[], &[]),
        Head::Full(Some(masks)) => (1, 0, masks, &[]),
        Head::Sparse(ids, k) => (2, *k, ids, &[]),
        Head::Edited(masks, edits, k) => (3, *k, masks, edits),
    };
    let count = 8 + table.len() + rows.len() * 3 + picks.len() + list.len() + edits.len();
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
    words.u32s(picks).u32s(list).u32s(edits);
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
    let (picks, list, edits) = (words.u32s()?, words.u32s()?, words.u32s()?);
    let head = match tag {
        0 => Head::Full(None),
        1 => Head::Full(Some(list)),
        3 => Head::Edited(list, edits, k as usize),
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

/// Builds the two-microbatch overlap graph and its row threshold.
pub(crate) fn build_split(graph: &Graph, entry: usize, min: usize) -> Option<(Graph, usize)> {
    tessera_graph::split_batch(graph, entry)
        .ok()
        .map(|g| (g, min))
}

/// Picks the split graph when `filled` meets the overlap threshold.
pub(crate) fn pick_graph<'a>(
    base: &'a Graph,
    split: &'a Option<(Graph, usize)>,
    allow: bool,
    filled: usize,
    sizes: &mut Vec<(tessera_graph::DimName, u64)>,
) -> &'a Graph {
    match split {
        Some((g, min)) if allow && *min > 0 && filled >= *min && filled.is_multiple_of(2) => {
            sizes.push((tessera_graph::HALF_ROWS, (filled / 2) as u64));
            g
        }
        _ => base,
    }
}
