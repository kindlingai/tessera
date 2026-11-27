//! Seconds spent per kernel or phase, summed over launches.

use std::collections::BTreeMap;
use std::fmt;
use tessera_engine_comm::{Reader, Writer};
use tessera_gpukit::device::KernelAttrs;
use tessera_lower::Step;

/// Rows past this many print as one "other" row.
const SHOWN: usize = 40;

/// Totals per name, since the last take.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Profile {
    pub spent: BTreeMap<String, Spent>,
}

/// One name's seconds, count, work, bytes and kernel attributes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spent {
    pub seconds: f64,
    pub count: u64,
    /// Multiply-adds, or elements touched. Zero for phases.
    pub work: f64,
    pub bytes: f64,
    /// Compiled kernel resource counts, when reported by the device.
    pub attrs: Option<KernelAttrs>,
}

impl Spent {
    fn sum(&mut self, other: Spent) {
        self.seconds += other.seconds;
        self.count += other.count;
        self.work += other.work;
        self.bytes += other.bytes;
        if let Some(a) = other.attrs {
            self.attrs = Some(match self.attrs {
                Some(cur) if cur.local_bytes >= a.local_bytes => cur,
                _ => a,
            });
        }
    }
}

/// Compact byte count for profile and load telemetry.
pub(crate) fn format_bytes(bytes: u32) -> String {
    if bytes == 0 {
        "0".into()
    } else if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes.is_multiple_of(1024) {
        format!("{}K", bytes / 1024)
    } else {
        format!("{:.1}K", bytes as f64 / 1024.0)
    }
}

/// One-line summary of compiled kernel attributes.
pub(crate) fn format_attrs(a: KernelAttrs) -> String {
    let spill = if a.local_bytes > 0 { " SPILL" } else { "" };
    format!(
        "regs={} smem={} lmem={} ctas/sm={}{spill}",
        a.regs,
        format_bytes(a.shared_bytes),
        format_bytes(a.local_bytes),
        a.ctas_per_sm
    )
}

impl Profile {
    pub fn add(&mut self, name: &str, seconds: f64) {
        self.add_many(name, seconds, 1);
    }

    pub fn add_many(&mut self, name: &str, seconds: f64, count: u64) {
        let one = Spent {
            seconds,
            count,
            ..Spent::default()
        };
        self.spent.entry(name.to_string()).or_default().sum(one);
    }

    /// Adds all rows from `other` into `self`.
    pub fn merge(&mut self, other: Self) {
        for (name, spent) in other.spent {
            self.spent.entry(name).or_default().sum(spent);
        }
    }

    /// A step's row: kernel name, sizes, and provenance.
    pub fn key(step: &Step) -> String {
        let kernel = &step.kernel;
        let mut name = kernel.name().to_string();
        for (i, size) in kernel.sizes().iter().enumerate() {
            name.push(if i == 0 { ' ' } else { 'x' });
            name.push_str(&size.to_string());
        }
        let entry = kernel.entry();
        let parts = [
            (!entry.is_empty()).then_some(entry.as_ref()),
            (!step.rule.is_empty()).then_some(step.rule),
            step.site.as_deref().filter(|s| !s.is_empty()),
        ];
        let mut first = true;
        for part in parts.into_iter().flatten() {
            name.push_str(if first { " [" } else { " | " });
            name.push_str(part);
            first = false;
        }
        if !first {
            name.push(']');
        }
        name
    }

    /// Adds one run of `step`, by kernel name and sizes.
    pub fn step(&mut self, step: &Step, seconds: f64) {
        self.step_with(step, seconds, 1, None);
    }

    /// Adds `count` runs of `step` with optional device kernel attributes.
    pub fn step_with(&mut self, step: &Step, seconds: f64, count: u64, attrs: Option<KernelAttrs>) {
        let kernel = &step.kernel;
        let name = Self::key(step);
        let one = Spent {
            seconds,
            count,
            work: kernel.work(step.write.len) * count as f64,
            bytes: step.bytes() as f64 * count as f64,
            attrs,
        };
        self.spent.entry(name).or_default().sum(one);
    }

    /// Names by seconds, most first.
    pub fn ranked(&self) -> Vec<(&str, Spent)> {
        let mut rows: Vec<_> = self.spent.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        rows.sort_by(|a, b| b.1.seconds.total_cmp(&a.1.seconds));
        rows
    }

    /// Each name, then its seconds, count, work, bytes and attributes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut words = Writer::default();
        for (name, s) in &self.spent {
            words.bytes(name.as_bytes()).f64(s.seconds).u64(s.count);
            words.f64(s.work).f64(s.bytes);
            match s.attrs {
                Some(a) => {
                    words.u32(1).u32(a.regs).u32(a.local_bytes);
                    words.u32(a.shared_bytes).u32(a.ctas_per_sm);
                }
                None => _ = words.u32(0),
            }
        }
        words.0
    }

    /// The profile `to_bytes` wrote. None when empty or malformed.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let (mut words, mut profile) = (Reader(bytes), Self::default());
        while !words.is_empty() {
            let name = String::from_utf8(words.bytes()?.to_vec()).ok()?;
            let (seconds, count) = (words.f64()?, words.u64()?);
            let (work, bytes) = (words.f64()?, words.f64()?);
            let attrs = match words.u32()? {
                0 => None,
                1 => Some(KernelAttrs {
                    regs: words.u32()?,
                    local_bytes: words.u32()?,
                    shared_bytes: words.u32()?,
                    ctas_per_sm: words.u32()?,
                }),
                _ => return None,
            };
            let spent = Spent {
                seconds,
                count,
                work,
                bytes,
                attrs,
            };
            profile.spent.insert(name, spent);
        }
        (!profile.spent.is_empty()).then_some(profile)
    }

    /// A table per `launches` launches.
    pub fn report(&self, launches: u64) -> Report<'_> {
        Report {
            profile: self,
            launches: launches.max(1),
        }
    }
}

/// A profile as text, per launch.
pub struct Report<'a> {
    profile: &'a Profile,
    launches: u64,
}

fn write_rates(f: &mut fmt::Formatter<'_>, work: f64, bytes: f64, seconds: f64) -> fmt::Result {
    for amount in [work, bytes] {
        match amount > 0.0 && seconds > 0.0 {
            true => write!(f, " {:>8.1}", amount / seconds / 1e9)?,
            false => write!(f, " {:>8}", "-")?,
        }
    }
    Ok(())
}

fn write_attrs(f: &mut fmt::Formatter<'_>, attrs: Option<KernelAttrs>) -> fmt::Result {
    match attrs {
        Some(a) => {
            let smem = format_bytes(a.shared_bytes);
            let lmem = format_bytes(a.local_bytes);
            let cta = match a.ctas_per_sm > 0 {
                true => a.ctas_per_sm.to_string(),
                false => "-".into(),
            };
            write!(f, " {:>4} {:>6} {:>6} {:>3}", a.regs, smem, lmem, cta)
        }
        None => write!(f, " {:>4} {:>6} {:>6} {:>3}", "-", "-", "-", "-"),
    }
}

impl Report<'_> {
    fn row(
        &self,
        f: &mut fmt::Formatter<'_>,
        name: &str,
        s: Spent,
        show_attrs: bool,
        spilled: bool,
    ) -> fmt::Result {
        let per = self.launches as f64;
        let ms = s.seconds * 1e3 / per;
        let each = s.seconds * 1e6 / s.count.max(1) as f64;
        let count = s.count as f64 / per;
        write!(f, "{ms:>10.3} {count:>8.1} {each:>10.1}")?;
        write_rates(f, s.work, s.bytes, s.seconds)?;
        if show_attrs {
            write_attrs(f, s.attrs)?;
        }
        let spill = spilled || s.attrs.is_some_and(|a| a.local_bytes > 0);
        match spill {
            true => writeln!(f, "  {name}  SPILL"),
            false => writeln!(f, "  {name}"),
        }
    }
}

impl fmt::Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let show_attrs = self.profile.spent.values().any(|s| s.attrs.is_some());
        match show_attrs {
            true => writeln!(
                f,
                "{:>10} {:>8} {:>10} {:>8} {:>8} {:>4} {:>6} {:>6} {:>3}  name",
                "ms/launch", "count", "us each", "G/s", "GB/s", "regs", "smem", "lmem", "cta"
            )?,
            false => writeln!(
                f,
                "{:>10} {:>8} {:>10} {:>8} {:>8}  name",
                "ms/launch", "count", "us each", "G/s", "GB/s"
            )?,
        }
        let rows = self.profile.ranked();
        let (shown, rest) = rows.split_at(rows.len().min(SHOWN));
        for (name, s) in shown {
            self.row(f, name, *s, show_attrs, false)?;
        }
        let mut other = Spent::default();
        let mut other_spill = false;
        for (_, s) in rest {
            other_spill |= s.attrs.is_some_and(|a| a.local_bytes > 0);
            other.sum(*s);
        }
        other.attrs = None;
        match rest.len() {
            0 => Ok(()),
            n => self.row(
                f,
                &format!("other ({n} rows)"),
                other,
                show_attrs,
                other_spill,
            ),
        }
    }
}

/// One profile per rank, None where a rank sent none.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ranks {
    pub launches: u64,
    pub profiles: Vec<Option<Profile>>,
}

impl Ranks {
    /// Seconds of `name` on each rank that sent a profile.
    fn seconds(&self, name: &str) -> impl Iterator<Item = Option<f64>> {
        let at = |p: &Profile| p.spent.get(name).map_or(0.0, |s| s.seconds);
        self.profiles.iter().map(move |p| p.as_ref().map(at))
    }
}

impl fmt::Display for Ranks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names: Vec<(&str, f64)> = Vec::new();
        for p in self.profiles.iter().flatten() {
            for (name, s) in &p.spent {
                match names.iter_mut().find(|(n, _)| n == name) {
                    Some((_, total)) => *total += s.seconds,
                    None => names.push((name, s.seconds)),
                }
            }
        }
        names.sort_by(|a, b| b.1.total_cmp(&a.1));
        let show_attrs = self
            .profiles
            .iter()
            .flatten()
            .any(|p| p.spent.values().any(|s| s.attrs.is_some()));
        writeln!(f, "ms/launch per rank")?;
        for rank in 0..self.profiles.len() {
            write!(f, "{:>10}", format!("rank {rank}"))?;
        }
        match show_attrs {
            true => writeln!(
                f,
                "{:>9} {:>8} {:>10} {:>8} {:>8} {:>4} {:>6} {:>6} {:>3}  name",
                "max/min", "count", "us each", "G/s", "GB/s", "regs", "smem", "lmem", "cta"
            )?,
            false => writeln!(
                f,
                "{:>9} {:>8} {:>10} {:>8} {:>8}  name",
                "max/min", "count", "us each", "G/s", "GB/s"
            )?,
        }
        let per = self.launches.max(1) as f64;
        for (name, _) in names {
            let (mut least, mut most) = (f64::MAX, 0.0f64);
            for seconds in self.seconds(name) {
                let Some(s) = seconds else {
                    write!(f, "{:>10}", "-")?;
                    continue;
                };
                write!(f, "{:>10.3}", s * 1e3 / per)?;
                (least, most) = (least.min(s), most.max(s));
            }
            match least > 0.0 && least < f64::MAX {
                true => write!(f, "{:>9.2}", most / least)?,
                false => write!(f, "{:>9}", "-")?,
            }
            let mut total = Spent::default();
            let mut present = 0u64;
            for p in self.profiles.iter().flatten() {
                if let Some(&s) = p.spent.get(name) {
                    total.sum(s);
                    present += 1;
                }
            }
            let count = total.count as f64 / (per * present.max(1) as f64);
            let each = total.seconds * 1e6 / total.count.max(1) as f64;
            write!(f, " {count:>8.1} {each:>10.1}")?;
            write_rates(f, total.work, total.bytes, total.seconds)?;
            if show_attrs {
                write_attrs(f, total.attrs)?;
            }
            match total.attrs.is_some_and(|a| a.local_bytes > 0) {
                true => writeln!(f, "  {name}  SPILL")?,
                false => writeln!(f, "  {name}")?,
            }
        }
        Ok(())
    }
}

/// Seconds between two `TS_MARK()` lines, once per block and pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Region {
    pub from: u32,
    pub to: u32,
    pub seconds: Vec<f64>,
}

impl Region {
    /// The middle and the largest of `seconds`.
    pub fn spread(&self) -> (f64, f64) {
        let mut sorted = self.seconds.clone();
        sorted.sort_by(f64::total_cmp);
        let last = sorted.last().copied().unwrap_or(0.0);
        (sorted.get(sorted.len() / 2).copied().unwrap_or(0.0), last)
    }
}

/// A kernel's marked regions, and marks past each block's room.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Marks {
    pub regions: Vec<Region>,
    pub lost: u64,
}
