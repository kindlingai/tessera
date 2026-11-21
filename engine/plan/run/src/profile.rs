//! Seconds spent per kernel or phase, summed over launches.

use std::collections::BTreeMap;
use std::fmt;
use tessera_engine_comm::{Reader, Writer};

/// Seconds and count per name, since the last take.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Profile {
    pub spent: BTreeMap<String, Spent>,
}

/// One name's seconds and how often it ran.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spent {
    pub seconds: f64,
    pub count: u64,
}

impl Profile {
    pub fn add(&mut self, name: &str, seconds: f64) {
        let spent = self.spent.entry(name.to_string()).or_default();
        spent.seconds += seconds;
        spent.count += 1;
    }

    /// Names by seconds, most first.
    pub fn ranked(&self) -> Vec<(&str, Spent)> {
        let mut rows: Vec<_> = self.spent.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        rows.sort_by(|a, b| b.1.seconds.total_cmp(&a.1.seconds));
        rows
    }

    /// Each name, then its seconds and count.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut words = Writer::default();
        for (name, s) in &self.spent {
            words.bytes(name.as_bytes()).f64(s.seconds).u64(s.count);
        }
        words.0
    }

    /// The profile `to_bytes` wrote. None when empty or malformed.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let (mut words, mut profile) = (Reader(bytes), Self::default());
        while !words.is_empty() {
            let name = String::from_utf8(words.bytes()?.to_vec()).ok()?;
            let (seconds, count) = (words.f64()?, words.u64()?);
            profile.spent.insert(name, Spent { seconds, count });
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

impl fmt::Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let per = self.launches as f64;
        writeln!(
            f,
            "{:>10} {:>8} {:>10}  name",
            "ms/launch", "count", "us each"
        )?;
        for (name, s) in self.profile.ranked() {
            let ms = s.seconds * 1e3 / per;
            let each = s.seconds * 1e6 / s.count.max(1) as f64;
            let count = s.count as f64 / per;
            writeln!(f, "{ms:>10.3} {count:>8.1} {each:>10.1}  {name}")?;
        }
        Ok(())
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
        writeln!(f, "ms/launch per rank")?;
        for rank in 0..self.profiles.len() {
            write!(f, "{:>10}", format!("rank {rank}"))?;
        }
        writeln!(f, "{:>9}  name", "max/min")?;
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
                true => writeln!(f, "{:>9.2}  {name}", most / least)?,
                false => writeln!(f, "{:>9}  {name}", "-")?,
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
