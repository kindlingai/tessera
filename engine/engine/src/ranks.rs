//! An engine as one rank of several.

use crate::kept::Kept;
use crate::lanes::{decode_patches, encode_patches};
use crate::{Asks, Engine, Error, Head, Place, Row};
use tessera_engine_comm::{Backed, HostPool, Reader, Transport, Writer};
use tessera_engine_hooks::open_rank;
use tessera_graph::Graph;
use tessera_kv::{DiskKv, DiskMsg, KvConfig, StoreStats, decode_msg};
use tessera_lower::{Device, Recipe, Settings, Wants};
use tessera_run::{Profile, Ranks, Run};

/// Profile state preserved while an engine is evicted.
pub(crate) struct ParkedProfile {
    pub(crate) carried: Option<(u64, Profile)>,
    pub(crate) asks: Option<Asks>,
    pub(crate) sink: Option<Box<dyn FnMut(Ranks) + Send>>,
}

impl ParkedProfile {
    /// Emits any carried profile if never resumed.
    pub(crate) fn flush(&mut self) {
        let Some((launches, profile)) = self.carried.take() else {
            return;
        };
        match &mut self.sink {
            Some(sink) => sink(Ranks {
                launches,
                profiles: vec![Some(profile)],
            }),
            None => eprintln!(
                "profile of {launches} launches:\n{}",
                profile.report(launches)
            ),
        }
    }
}

fn merge_carried(
    carried: Option<(u64, Profile)>,
    now: Option<(u64, Profile)>,
) -> Option<(u64, Profile)> {
    match (carried, now) {
        (Some((l0, mut p0)), Some((l1, p1))) => {
            p0.merge(p1);
            Some((l0 + l1, p0))
        }
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
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
        place.settle(grid.as_deref());
        let run = |plan| open_rank(plan, settings, sums, grid);
        let mut engine = Self::placed(graph, context, recipe, settings, Some(&place), run)?;
        engine.control = Some(control);
        Ok(engine)
    }

    /// Waits for every rank to finish loading.
    pub fn ready(&mut self) -> Result<(), Error> {
        if let Some(control) = self.control.as_mut() {
            control.exchange(&[1]).map_err(tessera_run::Error::from)?;
        }
        Ok(())
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
            if let (Some(lanes), false) = (&mut self.lanes, t.patches.is_empty()) {
                lanes.apply(&t.patches);
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
        if let Some((_, min)) = &mut self.split_owned {
            *min = rows;
        }
    }

    /// Sets the minimum row count for row ownership across ranks.
    pub fn set_own_rows(&mut self, rows: usize) {
        if let Some((_, min)) = &mut self.owned {
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
        match &mut self.profiled {
            Some(sink) => sink(ranks.clone()),
            None => eprintln!("profile of {launches} launches:\n{ranks}"),
        }
        self.ranks = Some(ranks);
    }

    /// Profiles while `asks` hold, and hands each profile to `sink`.
    pub fn profile_into(&mut self, asks: Asks, sink: Box<dyn FnMut(Ranks) + Send>) {
        self.asks = Some(asks);
        self.profiled = Some(sink);
    }

    /// Seconds per kernel and phase since the last take.
    pub fn take_profile(&mut self) -> Option<Profile> {
        self.runner.take_profile()
    }

    /// Gives one rank's profile to the sink, else prints it.
    fn hand(&mut self, launches: u64, profile: Profile) {
        match &mut self.profiled {
            Some(sink) => sink(Ranks {
                launches,
                profiles: vec![Some(profile)],
            }),
            None => eprintln!(
                "profile of {launches} launches:\n{}",
                profile.report(launches)
            ),
        }
    }

    /// Hands on the profile once every so many launches.
    pub(crate) fn report(&mut self) {
        let Some([every, done]) = &mut self.profile else {
            return;
        };
        *done += 1;
        if *done < *every {
            return;
        }
        let launches = std::mem::take(done);
        let now = self.take_profile().map(|p| (launches, p));
        let Some((launches, profile)) = merge_carried(self.carried.take(), now) else {
            return;
        };
        match self.control {
            Some(_) => self.held = Some((launches, profile)),
            None => self.hand(launches, profile),
        }
    }

    /// Starts or stops profiling. Stopping holds the profile.
    fn profile_ranked(&mut self, on: bool) {
        if on == self.profile.is_some() {
            return;
        }
        let launches = self.profile.take().map_or(0, |[_, done]| done);
        if !on {
            let now = self.take_profile().map(|p| (launches, p));
            self.held = merge_carried(self.carried.take(), now);
        }
        self.runner.profiling(on);
        self.profile = on.then_some([u64::MAX, 0]);
    }

    fn switch_profile(&mut self, on: bool) -> Result<(), Error> {
        let tag = match on {
            true => PROFILE_ON,
            false => PROFILE_OFF,
        };
        let tag = tag.to_le_bytes();
        self.profile_ranked(on);
        if self.control.is_none() {
            if let Some((launches, profile)) = self.held.take() {
                self.hand(launches, profile);
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

    /// Profiles while requests ask. Stopping prints the profile.
    pub fn settle(&mut self) -> Result<(), Error> {
        let Some(asks) = &self.asks else {
            return Ok(());
        };
        let on = asks.wanted();
        let lead = self.control.as_ref().is_none_or(|c| c.rank() == 0);
        if !lead {
            return Ok(());
        }
        if !on && self.profile.is_none() {
            if let Some((launches, profile)) = self.carried.take() {
                self.hand(launches, profile);
            }
            return Ok(());
        }
        if on == self.profile.is_some() {
            return Ok(());
        }
        self.switch_profile(on)
    }

    /// Saves or flushes active profile data before eviction.
    pub(crate) fn park_profile(&mut self) -> Result<Option<ParkedProfile>, Error> {
        self.settle()?;
        let carried = if self.control.is_some() {
            let lead = self.control.as_ref().is_some_and(|c| c.rank() == 0);
            if lead && self.profile.is_some() {
                self.switch_profile(false)?;
            }
            self.carried.take()
        } else {
            let now = if let Some([_, done]) = self.profile.take() {
                let taken = self.take_profile().map(|p| (done, p));
                self.runner.profiling(false);
                taken
            } else {
                None
            };
            merge_carried(self.carried.take(), now)
        };
        let asks = self.asks.clone();
        let sink = self.profiled.take();
        if carried.is_none() && asks.is_none() && sink.is_none() {
            return Ok(None);
        }
        Ok(Some(ParkedProfile {
            carried,
            asks,
            sink,
        }))
    }

    /// Restores profile state preserved across eviction.
    pub(crate) fn resume_profile(&mut self, parked: ParkedProfile) {
        if self.asks.is_none() {
            self.asks = parked.asks;
        }
        if self.profiled.is_none() {
            self.profiled = parked.sink;
        }
        self.carried = merge_carried(self.carried.take(), parked.carried);
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
        let patches = self
            .lanes
            .as_ref()
            .map(crate::lanes::Lanes::pending)
            .unwrap_or_default();
        let bytes = encode(rows, picks, head, wants, &patches);
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

    /// `(kv_vram_tokens, host_tokens, active_tokens)` when `TIER_HOST` is active.
    pub fn host_tier_info(&self) -> Option<(usize, usize, usize)> {
        let l = self.lanes.as_ref().filter(|l| l.tiered())?;
        Some((
            l.kv_pages() * l.page,
            l.host_pages() * l.page,
            l.pages() * l.page,
        ))
    }

    /// Opens the host KV pool one box's ranks share.
    fn open_host_pool(&mut self, pages: usize, page: usize) -> Result<bool, Error> {
        let open = |runner: &mut Box<dyn Run>, by| match pages {
            0 => Ok(HostPool::None),
            _ => runner.open_host_kv(pages, page, by),
        };
        let Some(control) = self.control.as_mut().filter(|c| c.ranks() > 1) else {
            return Ok(open(&mut self.runner, Backed::Own)? != HostPool::None);
        };
        // Every rank joins this round, with or without a pool.
        let lead = control.rank() == 0;
        let mut opened = match lead {
            true => open(&mut self.runner, Backed::Lead)?,
            false => HostPool::None,
        };
        let offer = match &opened {
            HostPool::Shared(offer) => offer.as_bytes(),
            _ => &[],
        };
        let parts = control.exchange(offer).map_err(tessera_run::Error::from)?;
        if !lead {
            let offer = parts.first().and_then(|p| std::str::from_utf8(p).ok());
            let by = offer
                .filter(|o| !o.is_empty())
                .map_or(Backed::Own, Backed::Follow);
            opened = open(&mut self.runner, by)?;
        }
        Ok(opened != HostPool::None)
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
        if self.lanes.as_ref().is_some_and(crate::lanes::Lanes::split)
            && cfg.tiers.len() == 1
            && let tessera_kv::StoreKind::Ram { bytes } = cfg.tiers[0].store
        {
            let sparse_page_bytes: usize = self
                .runner
                .plan()
                .kept
                .iter()
                .zip(&self.runner.plan().kept_info)
                .filter(|(_, i)| i.sparse && i.context > 1 && i.page_rows as usize == page)
                .map(|((_, slot), info)| slot.form.bytes(page * info.width as usize))
                .sum();
            let max_host = if sparse_page_bytes > 0 {
                (bytes / sparse_page_bytes as u64) as usize
            } else {
                0
            };
            let index_pages = self.lanes.as_ref().expect("lanes").index_pages();
            let host_pages = index_pages.min(max_host);
            if self.open_host_pool(host_pages, page)? {
                let lanes = self.lanes.as_mut().expect("lanes");
                lanes.enable_tier(host_pages, Box::new(tessera_kv::Aged::default()));
                self.disk = None;
                return Ok(());
            }
        }
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
        kept: Option<&mut Kept>,
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
    pub(crate) fn spill_kv(&mut self, kept: Option<&Kept>, slot: u32, history: &[u32], lead: bool) {
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

/// A launch as words: head, wants, patches, rows, picks, lists.
fn encode(
    rows: &[Row],
    picks: &[u32],
    head: &Head,
    wants: Wants,
    patches: &[(u32, &[u32])],
) -> Vec<u8> {
    let (tag, k, list, edits): (u32, usize, &[u32], &[u32]) = match head {
        Head::Full(None) => (0, 0, &[], &[]),
        Head::Full(Some(masks)) => (1, 0, masks, &[]),
        Head::Sparse(ids, k) => (2, *k, ids, &[]),
        Head::Edited(masks, edits, k) => (3, *k, masks, edits),
    };
    let pwords: usize = patches.iter().map(|(_, v)| 2 + v.len()).sum();
    let count = 8 + pwords + rows.len() * 3 + picks.len() + list.len() + edits.len();
    let mut words = Writer(Vec::with_capacity(count * 4));
    words.u32(tag).u32(k as u32).u32(wants.0);
    encode_patches(&mut words, patches);
    words.u32((rows.len() * 3) as u32);
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
    /// Changed page-table slices, empty if unchanged.
    patches: Vec<(u32, Vec<u32>)>,
}

/// The launch `bytes` hold, if they hold one.
fn decode(bytes: &[u8]) -> Option<Told> {
    let mut words = Reader(bytes);
    let (tag, k) = (words.u32()?, words.u32()?);
    let wants = Wants(words.u32()?);
    let patches = decode_patches(&mut words)?;
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
        patches,
    })
}

/// Builds the two-microbatch overlap graph and its row threshold.
pub(crate) fn build_split(graph: &Graph, entry: usize, min: usize) -> Option<(Graph, usize)> {
    tessera_graph::split_batch(graph, entry)
        .ok()
        .map(|g| (g, min))
}

/// Builds the row-owned graph and its minimum row threshold.
pub(crate) fn build_owned(graph: &Graph, min: usize, attn: bool) -> Option<(Graph, usize)> {
    let has_sum = graph.nodes.iter().any(|n| {
        matches!(
            &n.op,
            tessera_graph::Op::Row(r) if r.0.contains(&tessera_graph::Stage::AllReduce)
        )
    });
    if !has_sum {
        return None;
    }
    let out = if attn {
        tessera_graph::own_rows_attn(graph)
    } else {
        tessera_graph::own_rows(graph)
    };
    out.ok().map(|g| (g, min))
}

/// Optional transformed graph paired with its minimum row count.
pub(crate) type Variant = Option<(Graph, usize)>;

/// Graph variants selected per launch: base, owned, split, split-owned.
pub(crate) type Variants<'a> = (&'a Graph, &'a Variant, &'a Variant, &'a Variant);

fn even_rows(device: &Device, rows: usize) -> bool {
    let parts = device.division();
    let ranks = parts.ranks();
    let each = parts.rows(rows, 0).1;
    ranks > 1 && each > 0 && (1..ranks).all(|r| parts.rows(rows, r).1 == each)
}

/// Picks split-owned, split, owned, or base graph for `filled` rows.
pub(crate) fn pick_graph<'a>(
    (base, owned, split, split_owned): Variants<'a>,
    device: &Device,
    allow_split: bool,
    filled: usize,
    sizes: &mut Vec<(tessera_graph::DimName, u64)>,
) -> &'a Graph {
    if let Some((g, min)) = split
        && allow_split
        && *min > 0
        && filled >= *min
        && filled.is_multiple_of(2)
    {
        let half = filled / 2;
        sizes.push((tessera_graph::HALF_ROWS, half as u64));
        if let Some((sg, _)) = split_owned
            && owned.as_ref().is_some_and(|(_, om)| *om > 0 && half >= *om)
            && even_rows(device, half)
        {
            return sg;
        }
        return g;
    }
    if let Some((g, min)) = owned
        && *min > 0
        && filled >= *min
        && even_rows(device, filled)
    {
        return g;
    }
    base
}

/// Whether every wanted output of `entry` is indexed by `Picks`.
pub(crate) fn order_free(graph: &Graph, entry: usize, wants: Wants) -> bool {
    let e = &graph.entries[entry];
    let has_picks = e
        .inputs
        .iter()
        .any(|p| p.role == tessera_graph::Role::Picks);
    let picks_dim = tessera_graph::DimExpr::Dim(tessera_graph::DimName::Picks);
    e.outputs.iter().enumerate().all(|(i, o)| {
        wants.0 >> i & 1 == 0
            || (has_picks && graph.node(o.node).ty.shape.0.first() == Some(&picks_dim))
    })
}

/// Block-zigzag permutation `(perm, inv)` balancing causal lengths across ranks.
pub(crate) fn zigzag_order(
    filled: usize,
    chunks: usize,
    ranks: usize,
    page: usize,
) -> Option<(Vec<usize>, Vec<u32>)> {
    if ranks <= 1 || chunks == 0 || !filled.is_multiple_of(chunks) {
        return None;
    }
    let len = filled / chunks;
    if !len.is_multiple_of(ranks) {
        return None;
    }
    let per = len / ranks;
    if per < 2 || !per.is_multiple_of(2) {
        return None;
    }
    let half_per = per / 2;
    let mut block = 1usize;
    while block * 2 <= page && half_per.is_multiple_of(block * 2) {
        block *= 2;
    }
    let pairs = half_per / block;
    let stride = 2 * ranks * block;
    let mut perm = vec![0usize; filled];
    let mut inv = vec![0u32; filled];
    for c in 0..chunks {
        let base = c * len;
        for p in 0..pairs {
            let cycle = base + p * stride;
            for r in 0..ranks {
                let dst = base + r * per + 2 * p * block;
                let fwd = cycle + r * block;
                let rev = cycle + (2 * ranks - 1 - r) * block;
                for k in 0..block {
                    perm[dst + k] = fwd + k;
                    inv[fwd + k] = (dst + k) as u32;
                    perm[dst + block + k] = rev + k;
                    inv[rev + k] = (dst + block + k) as u32;
                }
            }
        }
    }
    Some((perm, inv))
}
