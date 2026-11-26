//! Composable KV store tier definitions matching gr8 syntax.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Default spare-RAM floor (2 GiB) before caching stops.
const DEFAULT_RAM_FLOOR: u64 = 2 << 30;

/// Retention policy controlling which evicted pages a tier keeps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Keep {
    System,
    Sessions,
    #[default]
    All,
}

impl Keep {
    /// Parses `system`, `sessions`, or `all`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "system" => Ok(Self::System),
            "sessions" => Ok(Self::Sessions),
            "all" => Ok(Self::All),
            _ => Err(format!("keep={s}: expected system, sessions or all")),
        }
    }

    /// Canonical keyword name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Sessions => "sessions",
            Self::All => "all",
        }
    }
}

/// Optional cluster-shared lease settings for remote stores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupSpec {
    pub name: String,
    pub cap: Option<u64>,
    pub weight: Option<u32>,
}

/// Named front-cache specification.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheSpec {
    name: String,
    floor: u64,
}

/// Backing medium of one KV store tier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreKind {
    Ram {
        bytes: u64,
    },
    Ssd {
        path: PathBuf,
        bytes: u64,
        persist: bool,
    },
    Remote {
        address: String,
        bytes: u64,
        group: Option<GroupSpec>,
    },
}

impl StoreKind {
    /// Short kind tag matching gr8 naming.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Ram { .. } => "ram",
            Self::Ssd { .. } => "ssd",
            Self::Remote { .. } => "remote",
        }
    }

    /// Capacity limit in bytes per rank.
    pub fn bytes(&self) -> u64 {
        match self {
            Self::Ram { bytes } | Self::Ssd { bytes, .. } | Self::Remote { bytes, .. } => *bytes,
        }
    }
}

/// One ordered KV tier below the GPU pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TierSpec {
    pub name: String,
    pub store: StoreKind,
    pub keep: Option<Keep>,
    pub spare_ram: Option<u64>,
}

/// Parsed multi-tier KV store configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct KvConfig {
    pub tiers: Vec<TierSpec>,
    pub keep: Keep,
    pub system_after: u32,
    pub write_limit: Option<f64>,
    pub disabled: bool,
}

impl Default for KvConfig {
    fn default() -> Self {
        Self {
            tiers: Vec::new(),
            keep: Keep::All,
            system_after: 2,
            write_limit: None,
            disabled: false,
        }
    }
}

impl KvConfig {
    /// Default host memory tier of `bytes` per rank.
    pub fn default_ram(bytes: u64) -> Self {
        Self {
            tiers: vec![TierSpec {
                name: "ram".into(),
                store: StoreKind::Ram { bytes },
                keep: Some(Keep::All),
                spare_ram: None,
            }],
            ..Self::default()
        }
    }

    /// Effective keep policy for `tier`.
    pub fn tier_keep(&self, tier: &TierSpec) -> Keep {
        tier.keep.unwrap_or(self.keep)
    }

    /// Parses one or more `--kv-store` values into a config.
    pub fn parse_all(raw: &[String], keep: Option<Keep>) -> Result<Self, String> {
        let mut caches: BTreeMap<String, CacheSpec> = BTreeMap::new();
        let mut pending: Vec<RawTier> = Vec::new();
        let mut disabled = false;
        for item in raw {
            let trimmed = item.trim();
            if trimmed.is_empty() {
                continue;
            }
            if matches!(trimmed, "none" | "off") {
                disabled = true;
                continue;
            }
            if let Some(file) = trimmed.strip_prefix('@') {
                let text = std::fs::read_to_string(file)
                    .map_err(|e| format!("--kv-store @{file}: {e}"))?;
                parse_yaml(&text, &mut caches, &mut pending)?;
                continue;
            }
            for entry in trimmed.split(';') {
                let entry = entry.trim();
                if entry.is_empty() {
                    continue;
                }
                if matches!(entry, "none" | "off") {
                    disabled = true;
                    continue;
                }
                if entry.starts_with("cache:") {
                    let c = parse_cache(entry)?;
                    caches.insert(c.name.clone(), c);
                } else {
                    pending.push(parse_tier(entry)?);
                }
            }
        }
        if disabled && !pending.is_empty() {
            return Err("--kv-store none/off cannot combine with other stores".into());
        }
        let tiers = resolve_tiers(pending, &caches)?;
        Ok(Self {
            tiers,
            keep: keep.unwrap_or(Keep::All),
            disabled,
            ..Self::default()
        })
    }

    /// Parses a single `--kv-store` string.
    pub fn parse(spec: &str) -> Result<Self, String> {
        Self::parse_all(&[spec.to_string()], None)
    }

    /// Canonical one-line summary for boot logs.
    pub fn describe(&self) -> String {
        if self.disabled || self.tiers.is_empty() {
            return "off".into();
        }
        let parts: Vec<String> = self
            .tiers
            .iter()
            .map(|t| format_tier(t, self.tier_keep(t)))
            .collect();
        parts.join("; ")
    }
}

/// Parses binary sizes `N(M|G|T)` (M=2^20, G=2^30, T=2^40).
fn parse_bytes(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty size".into());
    }
    let (num, unit) = s.split_at(s.len() - 1);
    let mul = match unit {
        "M" => (1u64 << 20) as f64,
        "G" => (1u64 << 30) as f64,
        "T" => (1u64 << 40) as f64,
        _ => return Err(format!("{s}: expected N with M, G or T")),
    };
    let n: f64 = num
        .parse()
        .map_err(|_| format!("{s}: expected N with M, G or T"))?;
    if !n.is_finite() || n <= 0.0 {
        return Err(format!("{s}: size must be positive"));
    }
    Ok((n * mul) as u64)
}

fn format_bytes(bytes: u64) -> String {
    const T: u64 = 1 << 40;
    const G: u64 = 1 << 30;
    const M: u64 = 1 << 20;
    if bytes.is_multiple_of(T) {
        format!("{}T", bytes / T)
    } else if bytes.is_multiple_of(G) {
        format!("{}G", bytes / G)
    } else if bytes >= G {
        format!("{:.1}G", bytes as f64 / G as f64)
    } else {
        format!("{}M", bytes.div_ceil(M).max(1))
    }
}

fn format_tier(t: &TierSpec, keep: Keep) -> String {
    let mut out = match &t.store {
        StoreKind::Ram { bytes } => format!("ram:,size={}", format_bytes(*bytes)),
        StoreKind::Ssd {
            path,
            bytes,
            persist,
        } => {
            let p = if *persist { ",persist=on" } else { "" };
            format!("ssd:{},size={}{p}", path.display(), format_bytes(*bytes))
        }
        StoreKind::Remote {
            address,
            bytes,
            group,
        } => {
            let mut s = format!("remote:{address},size={}", format_bytes(*bytes));
            if let Some(g) = group {
                s.push_str(&format!(",group={}", g.name));
                if let Some(c) = g.cap {
                    s.push_str(&format!(",cap={}", format_bytes(c)));
                }
                if let Some(w) = g.weight {
                    s.push_str(&format!(",weight={w}"));
                }
            }
            s
        }
    };
    out.push_str(&format!(",keep={}", keep.as_str()));
    if let Some(floor) = t.spare_ram {
        out.push_str(&format!(
            ",spare_ram_cache=on,spare_ram_floor={}",
            format_bytes(floor)
        ));
    }
    out
}

struct RawTier {
    store: String,
    bytes: Option<u64>,
    persist: bool,
    keep: Option<Keep>,
    name: Option<String>,
    cache_ref: Option<String>,
    spare_ram: bool,
    spare_floor: Option<u64>,
    group: Option<String>,
    cap: Option<u64>,
    weight: Option<u32>,
}

fn parse_cache(entry: &str) -> Result<CacheSpec, String> {
    let mut parts = entry.split(',');
    let head = parts.next().unwrap_or_default().trim();
    let name = head
        .strip_prefix("cache:")
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| format!("--kv-store {entry}: cache:NAME is required"))?
        .to_string();
    let mut kind: Option<&str> = None;
    let mut floor = DEFAULT_RAM_FLOOR;
    for kv in parts {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| format!("--kv-store {entry}: bad option {kv}"))?;
        match k.trim() {
            "kind" => kind = Some(v.trim()),
            "floor" => floor = parse_bytes(v.trim())?,
            other => return Err(format!("--kv-store {entry}: unknown cache option {other}")),
        }
    }
    if kind != Some("spare_ram") {
        return Err(format!("--kv-store {entry}: cache kind must be spare_ram"));
    }
    Ok(CacheSpec { name, floor })
}

fn parse_on_off(entry: &str, key: &str, val: &str) -> Result<bool, String> {
    match val {
        "on" | "true" => Ok(true),
        "off" | "false" => Ok(false),
        _ => Err(format!("--kv-store {entry}: {key}=on or off")),
    }
}

fn parse_tier(entry: &str) -> Result<RawTier, String> {
    let mut parts = entry.split(',');
    let store = parts.next().unwrap_or_default().trim().to_string();
    let mut t = RawTier {
        store,
        bytes: None,
        persist: false,
        keep: None,
        name: None,
        cache_ref: None,
        spare_ram: false,
        spare_floor: None,
        group: None,
        cap: None,
        weight: None,
    };
    for kv in parts {
        let kv = kv.trim();
        if kv.is_empty() {
            continue;
        }
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| format!("--kv-store {entry}: bad option {kv}"))?;
        let (k, v) = (k.trim(), v.trim());
        match k {
            "size" => t.bytes = Some(parse_bytes(v)?),
            "cap" => t.cap = Some(parse_bytes(v)?),
            "persist" => t.persist = parse_on_off(entry, k, v)?,
            "keep" => t.keep = Some(Keep::parse(v)?),
            "name" if !v.is_empty() => t.name = Some(v.to_string()),
            "cache" if !v.is_empty() => t.cache_ref = Some(v.to_string()),
            "group" if !v.is_empty() => t.group = Some(v.to_string()),
            "weight" => {
                let w: u32 = v
                    .parse()
                    .ok()
                    .filter(|&w| (1..=1_000_000).contains(&w))
                    .ok_or_else(|| format!("--kv-store {entry}: weight=N, 1000 = 1x"))?;
                t.weight = Some(w);
            }
            "spare_ram_cache" => t.spare_ram = parse_on_off(entry, k, v)?,
            "spare_ram_floor" => t.spare_floor = Some(parse_bytes(v)?),
            other => return Err(format!("--kv-store {entry}: unknown option {other}")),
        }
    }
    Ok(t)
}

fn resolve_tiers(
    raw: Vec<RawTier>,
    caches: &BTreeMap<String, CacheSpec>,
) -> Result<Vec<TierSpec>, String> {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        let bytes = r
            .bytes
            .ok_or_else(|| format!("--kv-store {}: size=N(M|G|T) is needed", r.store))?;
        let store = build_store(&r, bytes)?;
        let spare_ram = resolve_cache(&r, &store, caches)?;
        let tag = store.tag();
        let n = counts.entry(tag).or_insert(0);
        *n += 1;
        let name = r.name.unwrap_or_else(|| {
            if *n == 1 {
                tag.into()
            } else {
                format!("{tag}-{n}")
            }
        });
        out.push(TierSpec {
            name,
            store,
            keep: r.keep,
            spare_ram,
        });
    }
    Ok(out)
}

fn build_store(r: &RawTier, bytes: u64) -> Result<StoreKind, String> {
    let is_ssd = r.store.starts_with("ssd:") && r.store.len() > 4;
    let is_remote = r.store.starts_with("remote:") && r.store.len() > 7;
    let is_ram = r.store == "ram:" || r.store == "mem:";
    if !is_ssd && !is_remote && !is_ram {
        return Err(format!(
            "--kv-store {}: expected ram:, mem:, ssd:PATH or remote:HOST:PORT",
            r.store
        ));
    }
    if r.persist && !is_ssd {
        return Err(format!(
            "--kv-store {}: persist=on is for ssd: stores",
            r.store
        ));
    }
    if r.group.is_some() && !is_remote {
        return Err(format!(
            "--kv-store {}: group= is for remote: stores",
            r.store
        ));
    }
    if r.group.is_none() && (r.cap.is_some() || r.weight.is_some()) {
        return Err(format!(
            "--kv-store {}: cap= and weight= need group=",
            r.store
        ));
    }
    if let Some(cap) = r.cap
        && cap < bytes
    {
        return Err(format!("--kv-store {}: cap= is below size=", r.store));
    }
    if is_ram {
        return Ok(StoreKind::Ram { bytes });
    }
    if is_ssd {
        return Ok(StoreKind::Ssd {
            path: PathBuf::from(&r.store[4..]),
            bytes,
            persist: r.persist,
        });
    }
    let group = r.group.clone().map(|name| GroupSpec {
        name,
        cap: r.cap,
        weight: r.weight,
    });
    Ok(StoreKind::Remote {
        address: r.store[7..].to_string(),
        bytes,
        group,
    })
}

fn resolve_cache(
    r: &RawTier,
    store: &StoreKind,
    caches: &BTreeMap<String, CacheSpec>,
) -> Result<Option<u64>, String> {
    let floor = if let Some(ref name) = r.cache_ref {
        let c = caches
            .get(name)
            .ok_or_else(|| format!("--kv-store {}: unknown cache {name}", r.store))?;
        Some(c.floor)
    } else if r.spare_ram {
        Some(r.spare_floor.unwrap_or(DEFAULT_RAM_FLOOR))
    } else {
        None
    };
    if floor.is_some() {
        if matches!(store, StoreKind::Remote { group: Some(_), .. }) {
            return Err(format!(
                "--kv-store {}: spare_ram_cache=on needs a fixed lease (no group=)",
                r.store
            ));
        }
        if matches!(store, StoreKind::Ram { .. }) {
            return Err(format!(
                "--kv-store {}: ram: store cannot attach spare_ram cache",
                r.store
            ));
        }
    }
    Ok(floor)
}

/// Parses gr8 YAML `@FILE` format (`caches:` and `tiers:`).
fn parse_yaml(
    text: &str,
    caches: &mut BTreeMap<String, CacheSpec>,
    out: &mut Vec<RawTier>,
) -> Result<(), String> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Sec {
        None,
        Caches,
        Tiers,
    }
    let mut sec = Sec::None;
    let mut cur: Option<RawTier> = None;
    let mut in_store = false;
    for raw_line in text.lines() {
        let line = strip_comment(raw_line);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if indent == 0 {
            if let Some(t) = cur.take() {
                out.push(t);
            }
            in_store = false;
            match trimmed {
                "caches:" => sec = Sec::Caches,
                "tiers:" => sec = Sec::Tiers,
                other => {
                    let key = other.split(':').next().unwrap_or(other).trim();
                    if matches!(key, "transforms" | "takes" | "runs") {
                        return Err(format!("kv yaml: {key} is not supported yet"));
                    }
                    return Err(format!("kv yaml: unknown top-level key {key}"));
                }
            }
            continue;
        }
        if sec == Sec::Caches {
            parse_yaml_cache(trimmed, caches)?;
            continue;
        }
        if sec != Sec::Tiers {
            return Err(format!("kv yaml: unexpected line {trimmed}"));
        }
        if let Some(rest) = trimmed.strip_prefix('-') {
            if let Some(t) = cur.take() {
                out.push(t);
            }
            in_store = false;
            let mut t = empty_raw();
            let rest = rest.trim();
            if !rest.is_empty() {
                apply_tier_kv(&mut t, rest, &mut in_store)?;
            }
            cur = Some(t);
            continue;
        }
        let t = cur
            .as_mut()
            .ok_or_else(|| format!("kv yaml: expected '-' before {trimmed}"))?;
        if in_store && indent >= 6 {
            apply_store_line(t, trimmed)?;
        } else {
            in_store = false;
            apply_tier_kv(t, trimmed, &mut in_store)?;
        }
    }
    if let Some(t) = cur.take() {
        out.push(t);
    }
    Ok(())
}

fn strip_comment(line: &str) -> &str {
    line.split('#').next().unwrap_or(line)
}

fn empty_raw() -> RawTier {
    RawTier {
        store: String::new(),
        bytes: None,
        persist: false,
        keep: None,
        name: None,
        cache_ref: None,
        spare_ram: false,
        spare_floor: None,
        group: None,
        cap: None,
        weight: None,
    }
}

fn parse_flow_map(s: &str) -> Result<Vec<(String, String)>, String> {
    let inner = s
        .trim()
        .strip_prefix('{')
        .and_then(|x| x.strip_suffix('}'))
        .ok_or_else(|| format!("kv yaml: expected {{...}}, got {s}"))?;
    let mut out = Vec::new();
    for pair in inner.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (k, v) = pair
            .split_once(':')
            .ok_or_else(|| format!("kv yaml: bad entry {pair}"))?;
        out.push((k.trim().to_string(), unquote(v.trim())));
    }
    Ok(out)
}

fn unquote(s: &str) -> String {
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn parse_yaml_cache(line: &str, caches: &mut BTreeMap<String, CacheSpec>) -> Result<(), String> {
    let (name, rest) = line
        .split_once(':')
        .ok_or_else(|| format!("kv yaml: bad cache line {line}"))?;
    let name = name.trim().to_string();
    let pairs = parse_flow_map(rest)?;
    let mut kind = String::new();
    let mut floor = DEFAULT_RAM_FLOOR;
    for (k, v) in pairs {
        match k.as_str() {
            "kind" => kind = v,
            "floor" => floor = parse_bytes(&v)?,
            other => return Err(format!("kv yaml: unknown cache key {other}")),
        }
    }
    if kind != "spare_ram" {
        return Err(format!("kv yaml: cache {name} kind must be spare_ram"));
    }
    caches.insert(name.clone(), CacheSpec { name, floor });
    Ok(())
}

fn apply_tier_kv(t: &mut RawTier, line: &str, in_store: &mut bool) -> Result<(), String> {
    let (k, v) = line
        .split_once(':')
        .ok_or_else(|| format!("kv yaml: bad tier line {line}"))?;
    let (k, v) = (k.trim(), unquote(v.trim()));
    match k {
        "keep" => t.keep = Some(Keep::parse(&v)?),
        "cache" => t.cache_ref = Some(v),
        "name" => t.name = Some(v),
        "store" => *in_store = true,
        "transforms" | "takes" | "runs" => {
            return Err(format!("kv yaml: {k} is not supported yet"));
        }
        other => return Err(format!("kv yaml: unknown tier key {other}")),
    }
    Ok(())
}

fn apply_store_line(t: &mut RawTier, line: &str) -> Result<(), String> {
    let (kind, rest) = line
        .split_once(':')
        .ok_or_else(|| format!("kv yaml: bad store line {line}"))?;
    let kind = kind.trim();
    let pairs = parse_flow_map(rest)?;
    match kind {
        "ram" | "mem" => {
            t.store = format!("{kind}:");
            for (k, v) in pairs {
                match k.as_str() {
                    "size_per_rank" | "size" => t.bytes = Some(parse_bytes(&v)?),
                    other => return Err(format!("kv yaml: unknown ram key {other}")),
                }
            }
        }
        "ssd" => {
            let mut path = String::new();
            for (k, v) in pairs {
                match k.as_str() {
                    "path" => path = v,
                    "size_per_rank" | "size" => t.bytes = Some(parse_bytes(&v)?),
                    "persist" => t.persist = parse_on_off("yaml", "persist", &v)?,
                    other => return Err(format!("kv yaml: unknown ssd key {other}")),
                }
            }
            t.store = format!("ssd:{path}");
        }
        "remote" => {
            let mut addr = String::new();
            for (k, v) in pairs {
                match k.as_str() {
                    "address" => addr = v,
                    "size_per_rank" | "size" => t.bytes = Some(parse_bytes(&v)?),
                    "group" => t.group = Some(v),
                    "cap_per_rank" | "cap" => t.cap = Some(parse_bytes(&v)?),
                    "weight" => {
                        let w: u32 = v
                            .parse()
                            .ok()
                            .filter(|&w| (1..=1_000_000).contains(&w))
                            .ok_or_else(|| format!("kv yaml: bad weight {v}"))?;
                        t.weight = Some(w);
                    }
                    other => return Err(format!("kv yaml: unknown remote key {other}")),
                }
            }
            t.store = format!("remote:{addr}");
        }
        other => return Err(format!("kv yaml: unknown store kind {other}")),
    }
    Ok(())
}

/// Per-rank subdirectory path for an SSD tier.
pub(crate) fn rank_dir(base: &Path, rank: u32, ranks: u32) -> PathBuf {
    if ranks > 1 {
        base.join(format!("r{rank}"))
    } else {
        base.to_path_buf()
    }
}
