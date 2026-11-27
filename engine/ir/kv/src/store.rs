//! Content-addressed on-disk cache for whole KV pages.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const K0: u64 = 0x736f6d6570736575;
const K1: u64 = 0x646f72616e646f6d;
const K2: u64 = 0x6c7967656e657261;
const K3: u64 = 0x7465646279746573;

/// Content hash of one KV page and its prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PageKey(pub [u64; 2]);

impl PageKey {
    /// Lowercase 32-character hex encoding.
    pub fn hex(self) -> String {
        format!("{:016x}{:016x}", self.0[0], self.0[1])
    }

    /// Parses a 32-character lowercase or uppercase hex key.
    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 32 {
            return None;
        }
        let a = u64::from_str_radix(&s[..16], 16).ok()?;
        let b = u64::from_str_radix(&s[16..], 16).ok()?;
        Some(Self([a, b]))
    }
}

fn sip_round(v: &mut [u64; 4]) {
    v[0] = v[0].wrapping_add(v[1]);
    v[1] = v[1].rotate_left(13) ^ v[0];
    v[0] = v[0].rotate_left(32);
    v[2] = v[2].wrapping_add(v[3]);
    v[3] = v[3].rotate_left(16) ^ v[2];
    v[0] = v[0].wrapping_add(v[3]);
    v[3] = v[3].rotate_left(21) ^ v[0];
    v[2] = v[2].wrapping_add(v[1]);
    v[1] = v[1].rotate_left(17) ^ v[2];
    v[2] = v[2].rotate_left(32);
}

fn siphash24(k0: u64, k1: u64, words: &[u64]) -> u64 {
    let mut v = [k0 ^ K0, k1 ^ K1, k0 ^ K2, k1 ^ K3];
    for &m in words {
        v[3] ^= m;
        sip_round(&mut v);
        sip_round(&mut v);
        v[0] ^= m;
    }
    let last = ((words.len() as u64) & 0xff) << 56;
    v[3] ^= last;
    sip_round(&mut v);
    sip_round(&mut v);
    v[0] ^= last;
    v[2] ^= 0xff;
    for _ in 0..4 {
        sip_round(&mut v);
    }
    v[0] ^ v[1] ^ v[2] ^ v[3]
}

/// Chained content keys for each full `page` of `tokens`.
pub fn page_keys(seed: [u64; 2], tokens: &[u32], page: usize) -> Vec<PageKey> {
    if page == 0 {
        return Vec::new();
    }
    let mut cur = seed;
    let mut words = Vec::with_capacity(page.div_ceil(2) + 1);
    let mut out = Vec::with_capacity(tokens.len() / page);
    for chunk in tokens.chunks_exact(page) {
        words.clear();
        words.push(page as u64);
        for pair in chunk.chunks(2) {
            let lo = pair[0] as u64;
            let hi = pair.get(1).copied().unwrap_or(0) as u64;
            words.push(lo | (hi << 32));
        }
        let a = siphash24(cur[0], cur[1], &words);
        let b = siphash24(cur[1] ^ K1, cur[0] ^ K2, &words);
        cur = [a, b];
        out.push(PageKey(cur));
    }
    out
}

/// Counters for one on-disk KV page store.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreStats {
    pub pages: usize,
    pub bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub saves: u64,
    pub evictions: u64,
}

/// Stamp-based LRU order shared by RAM and disk stores.
#[derive(Default)]
pub(crate) struct LruClock {
    stamps: HashMap<PageKey, u64>,
    order: VecDeque<(PageKey, u64)>,
    tick: u64,
}

impl LruClock {
    /// Marks `key` as most recently used.
    pub(crate) fn touch(&mut self, key: PageKey) {
        self.tick += 1;
        let t = self.tick;
        self.stamps.insert(key, t);
        self.order.push_back((key, t));
    }

    /// Removes `key` from live tracking.
    pub(crate) fn remove(&mut self, key: &PageKey) {
        self.stamps.remove(key);
    }

    /// Pops the oldest live key.
    pub(crate) fn pop_oldest(&mut self) -> Option<PageKey> {
        while let Some((key, stamp)) = self.order.pop_front() {
            if self.stamps.get(&key) == Some(&stamp) {
                return Some(key);
            }
        }
        None
    }

    /// Clears all entries.
    pub(crate) fn clear(&mut self) {
        self.stamps.clear();
        self.order.clear();
    }
}

/// LRU-bounded directory of immutable KV page files.
pub struct PageStore {
    dir: PathBuf,
    max_bytes: u64,
    used_bytes: u64,
    entries: HashMap<PageKey, u64>,
    lru: LruClock,
    hits: u64,
    misses: u64,
    saves: u64,
    evictions: u64,
}

impl PageStore {
    /// Opens or creates a persistent page store bounded to `max_bytes`.
    pub fn open(dir: &Path, max_bytes: u64) -> io::Result<Self> {
        Self::open_mode(dir, max_bytes, true)
    }

    /// Opens a page store, clearing existing pages when `!persist`.
    pub fn open_mode(dir: &Path, max_bytes: u64, persist: bool) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let mut found: Vec<(SystemTime, PageKey, u64)> = Vec::new();
        for bucket in fs::read_dir(dir)? {
            let bucket = bucket?;
            if !bucket.file_type()?.is_dir() {
                continue;
            }
            for item in fs::read_dir(bucket.path())? {
                let item = item?;
                let name = item.file_name();
                let Some(stem) = name.to_str().and_then(|s| s.strip_suffix(".kv")) else {
                    continue;
                };
                let Some(key) = PageKey::from_hex(stem) else {
                    continue;
                };
                let meta = item.metadata()?;
                if !meta.is_file() {
                    continue;
                }
                if !persist {
                    let _ = fs::remove_file(item.path());
                    continue;
                }
                let when = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                found.push((when, key, meta.len()));
            }
        }
        found.sort_by_key(|&(when, _, _)| when);
        let mut store = Self {
            dir: dir.to_path_buf(),
            max_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            lru: LruClock::default(),
            hits: 0,
            misses: 0,
            saves: 0,
            evictions: 0,
        };
        for (_, key, len) in found {
            store.record(key, len);
        }
        store.evict_for(0);
        Ok(store)
    }

    /// Whether `key` exists on disk.
    pub fn has(&self, key: PageKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Contiguous leading keys present on disk.
    pub fn prefix(&self, keys: &[PageKey]) -> usize {
        keys.iter().take_while(|k| self.has(**k)).count()
    }

    /// Reads `key` into `dst`. False on miss.
    pub fn load(&mut self, key: PageKey, dst: &mut [u8]) -> io::Result<bool> {
        let Some(&len) = self.entries.get(&key) else {
            self.misses += 1;
            return Ok(false);
        };
        if len != dst.len() as u64 {
            self.drop_key(key);
            self.misses += 1;
            return Ok(false);
        }
        let path = self.page_path(key);
        let Ok(raw) = fs::read(&path) else {
            self.drop_key(key);
            self.misses += 1;
            return Ok(false);
        };
        if raw.len() != dst.len() {
            self.drop_key(key);
            self.misses += 1;
            return Ok(false);
        }
        dst.copy_from_slice(&raw);
        self.lru.touch(key);
        self.hits += 1;
        Ok(true)
    }

    /// Saves `data` under `key` unless already stored.
    pub fn save(&mut self, key: PageKey, data: &[u8]) -> io::Result<bool> {
        let len = data.len() as u64;
        if len == 0 || len > self.max_bytes {
            return Ok(false);
        }
        if self.entries.get(&key) == Some(&len) {
            self.lru.touch(key);
            return Ok(false);
        }
        self.evict_for(len);
        let path = self.page_path(key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        fs::write(&tmp, data)?;
        fs::rename(&tmp, &path)?;
        self.record(key, len);
        self.saves += 1;
        Ok(true)
    }

    /// Current page count, bytes, and hit/miss counters.
    pub fn stats(&self) -> StoreStats {
        StoreStats {
            pages: self.entries.len(),
            bytes: self.used_bytes,
            hits: self.hits,
            misses: self.misses,
            saves: self.saves,
            evictions: self.evictions,
        }
    }

    fn page_path(&self, key: PageKey) -> PathBuf {
        let hex = key.hex();
        self.dir.join(&hex[..2]).join(format!("{hex}.kv"))
    }

    fn record(&mut self, key: PageKey, len: u64) {
        if let Some(prev) = self.entries.insert(key, len) {
            self.used_bytes = self.used_bytes.saturating_sub(prev);
        }
        self.used_bytes += len;
        self.lru.touch(key);
    }

    fn drop_key(&mut self, key: PageKey) {
        if let Some(len) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(len);
            self.lru.remove(&key);
            let _ = fs::remove_file(self.page_path(key));
        }
    }

    fn evict_for(&mut self, need: u64) {
        while self.used_bytes.saturating_add(need) > self.max_bytes {
            let Some(key) = self.lru.pop_oldest() else {
                break;
            };
            self.drop_key(key);
            self.evictions += 1;
        }
    }
}

/// Wire tag for rank-0 disk prefix query.
const TAG_PREFIX: u32 = 3;
/// Wire tag for rank-0 disk load command.
const TAG_LOAD: u32 = 4;
/// Wire tag for rank-0 disk save command.
const TAG_SAVE: u32 = 5;

/// Decoded rank-0 disk cache control message.
pub enum DiskMsg {
    Prefix(Vec<PageKey>),
    Load(Vec<(u32, PageKey)>),
    Save(Vec<(u32, PageKey)>),
}

impl DiskMsg {
    /// Whether this message is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Prefix(k) => k.is_empty(),
            Self::Load(p) | Self::Save(p) => p.is_empty(),
        }
    }

    /// Encodes this control message for follower ranks.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Self::Prefix(keys) => encode_prefix(keys),
            Self::Load(pairs) => encode_pairs(true, pairs),
            Self::Save(pairs) => encode_pairs(false, pairs),
        }
    }
}

/// Computes layout fingerprint and byte size per physical page.
fn layout_seed(
    rank: u32,
    ranks: u32,
    page: usize,
    kept: &[(u32, u32, usize, &str)],
) -> Option<([u64; 2], usize)> {
    let mut words: Vec<u32> = vec![rank, ranks, page as u32];
    let mut total = 0usize;
    for &(index, width, bytes, name) in kept {
        total += bytes;
        words.extend([index, width, bytes as u32]);
        words.extend(name.bytes().map(u32::from));
    }
    if total == 0 {
        return None;
    }
    let key = page_keys(
        [0x6b765f6469736b31, 0x7465737365726131],
        &words,
        words.len(),
    )[0];
    Some((key.0, total))
}

/// Encodes a disk prefix query for follower ranks.
fn encode_prefix(keys: &[PageKey]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + keys.len() * 16);
    out.extend(TAG_PREFIX.to_le_bytes());
    out.extend((keys.len() as u32).to_le_bytes());
    for k in keys {
        out.extend(k.0[0].to_le_bytes());
        out.extend(k.0[1].to_le_bytes());
    }
    out
}

/// Encodes a disk load or save message for follower ranks.
fn encode_pairs(load: bool, pairs: &[(u32, PageKey)]) -> Vec<u8> {
    let tag = if load { TAG_LOAD } else { TAG_SAVE };
    let mut out = Vec::with_capacity(8 + pairs.len() * 20);
    out.extend(tag.to_le_bytes());
    out.extend((pairs.len() as u32).to_le_bytes());
    for &(page, k) in pairs {
        out.extend(page.to_le_bytes());
        out.extend(k.0[0].to_le_bytes());
        out.extend(k.0[1].to_le_bytes());
    }
    out
}

fn take_u32(src: &mut &[u8]) -> Option<u32> {
    let (h, rest) = src.split_first_chunk::<4>()?;
    *src = rest;
    Some(u32::from_le_bytes(*h))
}

fn take_key(src: &mut &[u8]) -> Option<PageKey> {
    let (a, r) = src.split_first_chunk::<8>()?;
    let (b, r) = r.split_first_chunk::<8>()?;
    *src = r;
    Some(PageKey([u64::from_le_bytes(*a), u64::from_le_bytes(*b)]))
}

/// Decodes a disk control message if the tag matches.
pub fn decode_msg(mut bytes: &[u8]) -> Option<DiskMsg> {
    let tag = take_u32(&mut bytes)?;
    if tag == TAG_PREFIX {
        let n = take_u32(&mut bytes)? as usize;
        let keys = (0..n)
            .map(|_| take_key(&mut bytes))
            .collect::<Option<_>>()?;
        return Some(DiskMsg::Prefix(keys));
    }
    if tag == TAG_LOAD || tag == TAG_SAVE {
        let n = take_u32(&mut bytes)? as usize;
        let pairs = (0..n)
            .map(|_| Some((take_u32(&mut bytes)?, take_key(&mut bytes)?)))
            .collect::<Option<_>>()?;
        return Some(if tag == TAG_LOAD {
            DiskMsg::Load(pairs)
        } else {
            DiskMsg::Save(pairs)
        });
    }
    None
}

/// Open multi-tier KV page cache bound to one engine layout.
pub struct DiskKv {
    store: crate::tier::TieredStore,
    pub seed: [u64; 2],
    pub page: usize,
    page_bytes: usize,
    scratch: Vec<u8>,
}

impl DiskKv {
    /// Opens a tiered cache from `cfg` for `rank`/`ranks` and layout.
    pub fn open_config(
        cfg: &crate::spec::KvConfig,
        rank: u32,
        ranks: u32,
        page: usize,
        meta: &[(u32, u32, usize, &str)],
    ) -> Option<io::Result<Self>> {
        if cfg.disabled || cfg.tiers.is_empty() {
            return None;
        }
        let (seed, page_bytes) = layout_seed(rank, ranks, page, meta)?;
        Some(
            crate::tier::TieredStore::open(cfg, rank, ranks).map(|store| Self {
                store,
                seed,
                page,
                page_bytes,
                scratch: Vec::new(),
            }),
        )
    }

    /// Cache statistics snapshot.
    pub fn stats(&self) -> StoreStats {
        self.store.stats()
    }

    /// Contiguous leading keys present on disk.
    pub fn prefix(&self, keys: &[PageKey]) -> usize {
        self.store.prefix(keys)
    }

    /// Disk keys for full pages past `skip` cached rows.
    pub fn missing_keys(&self, tokens: &[u32], skip: usize) -> (usize, Vec<PageKey>) {
        let vram = skip / self.page;
        let keys = page_keys(self.seed, tokens, self.page);
        let rest = if keys.len() > vram {
            keys[vram..].to_vec()
        } else {
            Vec::new()
        };
        (vram, rest)
    }

    /// Physical-page and key pairs to spill for `history`.
    pub fn spill_pairs(&self, pages: &[u32], history: &[u32]) -> Vec<(u32, PageKey)> {
        pages
            .iter()
            .copied()
            .zip(page_keys(self.seed, history, self.page))
            .collect()
    }

    /// Physical-page and key pairs for `hit` pages past `vram`.
    pub fn load_pairs_slice(
        pages: &[u32],
        keys: Vec<PageKey>,
        vram: usize,
        hit: usize,
    ) -> Vec<(u32, PageKey)> {
        pages[vram..vram + hit]
            .iter()
            .copied()
            .zip(keys.into_iter().take(hit))
            .collect()
    }

    /// Runs `msg` against this local store, returning `(count, sync)`.
    pub fn exec<E>(
        &mut self,
        msg: &DiskMsg,
        io_err: impl FnMut(io::Error) -> E,
        mut io: impl FnMut(bool, &[u32], usize, &mut [u8]) -> Result<(), E>,
    ) -> Result<(u32, bool), E> {
        match msg {
            DiskMsg::Prefix(k) => Ok((self.prefix(k) as u32, true)),
            DiskMsg::Load(p) => Ok((
                self.load_pairs(p, io_err, |a, g, b| io(true, a, g, b))? as u32,
                true,
            )),
            DiskMsg::Save(p) => {
                self.save_pairs(p, io_err, |a, g, b| io(false, a, g, b))?;
                Ok((0, false))
            }
        }
    }

    /// Loads `pairs` from disk into `write` in batches.
    pub fn load_pairs<E>(
        &mut self,
        pairs: &[(u32, PageKey)],
        mut io_err: impl FnMut(io::Error) -> E,
        mut write: impl FnMut(&[u32], usize, &mut [u8]) -> Result<(), E>,
    ) -> Result<usize, E> {
        if pairs.is_empty() || self.page_bytes == 0 {
            return Ok(0);
        }
        let batch = ((64 << 20) / self.page_bytes).max(1).min(pairs.len());
        let need = batch * self.page_bytes;
        if self.scratch.len() < need {
            self.scratch.resize(need, 0);
        }
        let mut phys = Vec::with_capacity(batch);
        let mut done = 0usize;
        for chunk in pairs.chunks(batch) {
            phys.clear();
            let mut n = 0usize;
            for &(p, key) in chunk {
                let dst = &mut self.scratch[n * self.page_bytes..(n + 1) * self.page_bytes];
                if !self.store.load(key, dst).map_err(&mut io_err)? {
                    break;
                }
                phys.push(p);
                n += 1;
            }
            if n > 0 {
                write(&phys, self.page, &mut self.scratch[..n * self.page_bytes])?;
                done += n;
            }
            if n < chunk.len() {
                break;
            }
        }
        Ok(done)
    }

    /// Spills missing pages read through `read` in batches.
    pub fn save_pairs<E>(
        &mut self,
        pairs: &[(u32, PageKey)],
        mut io_err: impl FnMut(io::Error) -> E,
        mut read: impl FnMut(&[u32], usize, &mut [u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        if pairs.is_empty() || self.page_bytes == 0 {
            return Ok(());
        }
        let todo: Vec<(u32, PageKey)> = pairs
            .iter()
            .copied()
            .filter(|&(_, key)| self.store.should_save(key))
            .collect();
        if todo.is_empty() {
            return Ok(());
        }
        let batch = ((64 << 20) / self.page_bytes).max(1).min(todo.len());
        let need = batch * self.page_bytes;
        if self.scratch.len() < need {
            self.scratch.resize(need, 0);
        }
        let mut phys = Vec::with_capacity(batch);
        for chunk in todo.chunks(batch) {
            phys.clear();
            phys.extend(chunk.iter().map(|&(p, _)| p));
            let slice = &mut self.scratch[..chunk.len() * self.page_bytes];
            read(&phys, self.page, slice)?;
            for (i, &(_, key)) in chunk.iter().enumerate() {
                let page_buf = &slice[i * self.page_bytes..(i + 1) * self.page_bytes];
                self.store.save_seen(key, page_buf).map_err(&mut io_err)?;
            }
        }
        Ok(())
    }
}
