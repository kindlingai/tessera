//! Host-RAM and multi-tier page stores below the GPU pool.

use crate::spec::{Keep, KvConfig, StoreKind, rank_dir};
use crate::store::{LruClock, PageKey, PageStore, StoreStats};
use std::collections::HashMap;
use std::io;
use std::time::{Duration, Instant};

/// LRU-bounded in-memory KV page store in host RAM.
pub struct RamStore {
    max_bytes: u64,
    used_bytes: u64,
    entries: HashMap<PageKey, Box<[u8]>>,
    lru: LruClock,
    hits: u64,
    misses: u64,
    saves: u64,
    evictions: u64,
}

impl RamStore {
    /// Creates an empty host-RAM store bounded to `max_bytes`.
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            lru: LruClock::default(),
            hits: 0,
            misses: 0,
            saves: 0,
            evictions: 0,
        }
    }

    /// Whether `key` lives in host RAM.
    pub fn has(&self, key: PageKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Copies `key` into `dst`. Returns false on miss.
    pub fn load(&mut self, key: PageKey, dst: &mut [u8]) -> bool {
        let Some(raw) = self.entries.get(&key) else {
            self.misses += 1;
            return false;
        };
        if raw.len() != dst.len() {
            self.drop_key(key);
            self.misses += 1;
            return false;
        }
        dst.copy_from_slice(raw);
        self.lru.touch(key);
        self.hits += 1;
        true
    }

    /// Saves `data` under `key`, returning any evicted entries.
    pub fn save(&mut self, key: PageKey, data: &[u8]) -> Vec<(PageKey, Box<[u8]>)> {
        let len = data.len() as u64;
        if len == 0 || len > self.max_bytes {
            return Vec::new();
        }
        if self
            .entries
            .get(&key)
            .is_some_and(|b| b.len() as u64 == len)
        {
            self.lru.touch(key);
            return Vec::new();
        }
        let evicted = self.evict_for(len);
        let owned: Box<[u8]> = data.into();
        if let Some(prev) = self.entries.insert(key, owned) {
            self.used_bytes = self.used_bytes.saturating_sub(prev.len() as u64);
        }
        self.used_bytes += len;
        self.lru.touch(key);
        self.saves += 1;
        evicted
    }

    /// Drops every cached page (used under memory pressure).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.used_bytes = 0;
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

    fn drop_key(&mut self, key: PageKey) -> Option<Box<[u8]>> {
        let data = self.entries.remove(&key)?;
        self.used_bytes = self.used_bytes.saturating_sub(data.len() as u64);
        self.lru.remove(&key);
        Some(data)
    }

    fn evict_for(&mut self, need: u64) -> Vec<(PageKey, Box<[u8]>)> {
        let mut out = Vec::new();
        while self.used_bytes.saturating_add(need) > self.max_bytes {
            let Some(key) = self.lru.pop_oldest() else {
                break;
            };
            if let Some(data) = self.drop_key(key) {
                self.evictions += 1;
                out.push((key, data));
            }
        }
        out
    }
}

/// Optional spare-RAM front-cache over an SSD tier.
struct SpareCache {
    ram: RamStore,
    floor: u64,
    checked: Option<Instant>,
    room: bool,
}

impl SpareCache {
    fn new(max_bytes: u64, floor: u64) -> Self {
        Self {
            ram: RamStore::new(max_bytes),
            floor,
            checked: None,
            room: false,
        }
    }

    fn has_room(&mut self) -> bool {
        let stale = self
            .checked
            .is_none_or(|t| t.elapsed() >= Duration::from_millis(100));
        if stale {
            self.checked = Some(Instant::now());
            self.room = mem_available().is_some_and(|a| a > self.floor);
            if !self.room {
                self.ram.clear();
            }
        }
        self.room
    }

    fn load(&mut self, key: PageKey, dst: &mut [u8]) -> bool {
        self.ram.load(key, dst)
    }

    fn fill(&mut self, key: PageKey, data: &[u8]) {
        if self.has_room() {
            let _ = self.ram.save(key, data);
        }
    }
}

fn mem_available() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        macos_available()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn macos_available() -> Option<u64> {
    use std::ffi::{CStr, c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old: *mut c_void,
            len: *mut usize,
            new: *const c_void,
            new_len: usize,
        ) -> c_int;
    }

    let query = |name: &CStr| -> Option<u64> {
        let mut raw = [0u8; 8];
        let mut len = raw.len();
        // SAFETY: `sysctlbyname` writes at most `len` bytes into `raw`.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr(),
                raw.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null(),
                0,
            )
        };
        if rc != 0 {
            return None;
        }
        match len {
            4 => Some(u64::from(u32::from_ne_bytes(raw[..4].try_into().ok()?))),
            8 => Some(u64::from_ne_bytes(raw)),
            _ => None,
        }
    };

    let page_bytes = query(c"hw.pagesize")?;
    let keys = [
        c"vm.page_free_count",
        c"vm.page_speculative_count",
        c"vm.page_purgeable_count",
        c"vm.page_pageable_external_count",
    ];
    let mut pages = 0u64;
    for key in keys {
        pages = pages.saturating_add(query(key).unwrap_or(0));
    }
    let bytes = pages.saturating_mul(page_bytes);
    (bytes > 0).then_some(bytes)
}

enum Backend {
    Ram(RamStore),
    Ssd {
        disk: PageStore,
        cache: Option<Box<SpareCache>>,
    },
}

struct Tier {
    keep: Keep,
    backend: Backend,
}

impl Tier {
    fn has(&self, key: PageKey) -> bool {
        match &self.backend {
            Backend::Ram(r) => r.has(key),
            Backend::Ssd { disk, .. } => disk.has(key),
        }
    }

    fn load(&mut self, key: PageKey, dst: &mut [u8]) -> io::Result<bool> {
        match &mut self.backend {
            Backend::Ram(r) => Ok(r.load(key, dst)),
            Backend::Ssd { disk, cache } => {
                if let Some(c) = cache
                    && c.load(key, dst)
                {
                    return Ok(true);
                }
                let hit = disk.load(key, dst)?;
                if hit && let Some(c) = cache {
                    c.fill(key, dst);
                }
                Ok(hit)
            }
        }
    }

    fn save(&mut self, key: PageKey, data: &[u8]) -> io::Result<Vec<(PageKey, Box<[u8]>)>> {
        match &mut self.backend {
            Backend::Ram(r) => Ok(r.save(key, data)),
            Backend::Ssd { disk, cache } => {
                let saved = disk.save(key, data)?;
                if saved && let Some(c) = cache {
                    c.fill(key, data);
                }
                Ok(Vec::new())
            }
        }
    }

    fn stats(&self) -> StoreStats {
        match &self.backend {
            Backend::Ram(r) => r.stats(),
            Backend::Ssd { disk, .. } => disk.stats(),
        }
    }
}

/// Ordered multi-tier KV store with cascading LRU eviction.
pub struct TieredStore {
    tiers: Vec<Tier>,
    system_after: u32,
    seen: HashMap<PageKey, u32>,
}

impl TieredStore {
    /// Opens every tier in `cfg` for `rank`/`ranks`.
    pub fn open(cfg: &KvConfig, rank: u32, ranks: u32) -> io::Result<Self> {
        let mut tiers = Vec::with_capacity(cfg.tiers.len());
        for t in &cfg.tiers {
            let keep = cfg.tier_keep(t);
            let backend = match &t.store {
                StoreKind::Ram { bytes } => Backend::Ram(RamStore::new(*bytes)),
                StoreKind::Ssd {
                    path,
                    bytes,
                    persist,
                } => {
                    let dir = rank_dir(path, rank, ranks);
                    let disk = PageStore::open_mode(&dir, *bytes, *persist)?;
                    let cache = t.spare_ram.map(|f| Box::new(SpareCache::new(*bytes, f)));
                    Backend::Ssd { disk, cache }
                }
                StoreKind::Remote { address, .. } => {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        format!("remote KV store {address} is not implemented yet"),
                    ));
                }
            };
            tiers.push(Tier { keep, backend });
        }
        Ok(Self {
            tiers,
            system_after: cfg.system_after,
            seen: HashMap::new(),
        })
    }

    /// Whether any tier holds `key`.
    pub fn has(&self, key: PageKey) -> bool {
        self.tiers.iter().any(|t| t.has(key))
    }

    /// Contiguous leading keys present in any tier.
    pub fn prefix(&self, keys: &[PageKey]) -> usize {
        keys.iter().take_while(|k| self.has(**k)).count()
    }

    /// Reads `key` from the fastest tier and promotes upward.
    pub fn load(&mut self, key: PageKey, dst: &mut [u8]) -> io::Result<bool> {
        for i in 0..self.tiers.len() {
            if !self.tiers[i].has(key) {
                continue;
            }
            if self.tiers[i].load(key, dst)? {
                for up in (0..i).rev() {
                    if matches!(self.tiers[up].backend, Backend::Ram(_)) {
                        let evicted = self.tiers[up].save(key, dst)?;
                        self.cascade(up + 1, evicted)?;
                    }
                }
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Records a touch and reports whether any tier saves it.
    pub fn should_save(&mut self, key: PageKey) -> bool {
        if self.has(key) {
            return false;
        }
        let count = self.seen.entry(key).or_insert(0);
        *count = count.saturating_add(1);
        let hits = *count;
        self.tiers
            .iter()
            .any(|t| accepts(t.keep, hits, self.system_after))
    }

    /// Saves `data` into the first matching tier and cascades evictions.
    pub fn save_seen(&mut self, key: PageKey, data: &[u8]) -> io::Result<bool> {
        let hits = self.seen.get(&key).copied().unwrap_or(1);
        for i in 0..self.tiers.len() {
            if !accepts(self.tiers[i].keep, hits, self.system_after) {
                continue;
            }
            let evicted = self.tiers[i].save(key, data)?;
            self.cascade(i + 1, evicted)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Saves `data` after recording one touch for `key`.
    pub fn save(&mut self, key: PageKey, data: &[u8]) -> io::Result<bool> {
        if !self.should_save(key) {
            return Ok(false);
        }
        self.save_seen(key, data)
    }

    /// Combined counters across every tier.
    pub fn stats(&self) -> StoreStats {
        let mut total = StoreStats::default();
        for t in &self.tiers {
            let s = t.stats();
            total.pages += s.pages;
            total.bytes += s.bytes;
            total.hits += s.hits;
            total.misses += s.misses;
            total.saves += s.saves;
            total.evictions += s.evictions;
        }
        total
    }

    fn cascade(&mut self, start: usize, mut items: Vec<(PageKey, Box<[u8]>)>) -> io::Result<()> {
        for i in start..self.tiers.len() {
            if items.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for (k, bytes) in items {
                let hits = self.seen.get(&k).copied().unwrap_or(1);
                if accepts(self.tiers[i].keep, hits, self.system_after) {
                    next.extend(self.tiers[i].save(k, &bytes)?);
                }
            }
            items = next;
        }
        Ok(())
    }
}

fn accepts(keep: Keep, seen: u32, system_after: u32) -> bool {
    match keep {
        Keep::All => true,
        Keep::Sessions => seen >= 2,
        Keep::System => seen >= system_after.max(1),
    }
}
