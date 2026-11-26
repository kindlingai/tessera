//! Seconds spent per kernel or phase, summed over launches.

use std::collections::BTreeMap;
use std::fmt;
use tessera_engine_comm::{Reader, Writer};
use tessera_lower::Step;

/// Rows past this many print as one "other" row.
const SHOWN: usize = 40;

/// Totals per name, since the last take.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Profile {
    pub spent: BTreeMap<String, Spent>,
}

/// One name's seconds, count, work and bytes moved.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spent {
    pub seconds: f64,
    pub count: u64,
    /// Multiply-adds, or elements touched. Zero for phases.
    pub work: f64,
    pub bytes: f64,
}

impl Spent {
    fn sum(&mut self, other: Spent) {
        self.seconds += other.seconds;
        self.count += other.count;
        self.work += other.work;
        self.bytes += other.bytes;
    }
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

    /// Adds one run of `step`, by kernel name and sizes.
    pub fn step(&mut self, step: &Step, seconds: f64) {
        self.step_many(step, seconds, 1);
    }

    /// Adds `count` runs of `step`, by kernel name and sizes.
    pub fn step_many(&mut self, step: &Step, seconds: f64, count: u64) {
        let kernel = &step.kernel;
        let mut name = kernel.name().to_string();
        for (i, size) in kernel.sizes().iter().enumerate() {
            name.push(if i == 0 { ' ' } else { 'x' });
            name.push_str(&size.to_string());
        }
        let one = Spent {
            seconds,
            count,
            work: kernel.work(step.write.len) * count as f64,
            bytes: step.bytes() as f64 * count as f64,
        };
        self.spent.entry(name).or_default().sum(one);
    }

    /// Names by seconds, most first.
    pub fn ranked(&self) -> Vec<(&str, Spent)> {
        let mut rows: Vec<_> = self.spent.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        rows.sort_by(|a, b| b.1.seconds.total_cmp(&a.1.seconds));
        rows
    }

    /// Each name, then its seconds, count, work and bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut words = Writer::default();
        for (name, s) in &self.spent {
            words.bytes(name.as_bytes()).f64(s.seconds).u64(s.count);
            words.f64(s.work).f64(s.bytes);
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
            let spent = Spent {
                seconds,
                count,
                work,
                bytes,
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

impl Report<'_> {
    fn row(&self, f: &mut fmt::Formatter<'_>, name: &str, s: Spent) -> fmt::Result {
        let per = self.launches as f64;
        let ms = s.seconds * 1e3 / per;
        let each = s.seconds * 1e6 / s.count.max(1) as f64;
        let count = s.count as f64 / per;
        write!(f, "{ms:>10.3} {count:>8.1} {each:>10.1}")?;
        for amount in [s.work, s.bytes] {
            match amount > 0.0 && s.seconds > 0.0 {
                true => write!(f, " {:>8.1}", amount / s.seconds / 1e9)?,
                false => write!(f, " {:>8}", "-")?,
            }
        }
        writeln!(f, "  {name}")
    }
}

impl fmt::Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:>10} {:>8} {:>10} {:>8} {:>8}  name",
            "ms/launch", "count", "us each", "G/s", "GB/s"
        )?;
        let rows = self.profile.ranked();
        let (shown, rest) = rows.split_at(rows.len().min(SHOWN));
        for (name, s) in shown {
            self.row(f, name, *s)?;
        }
        let mut other = Spent::default();
        for (_, s) in rest {
            other.sum(*s);
        }
        match rest.len() {
            0 => Ok(()),
            n => self.row(f, &format!("other ({n} rows)"), other),
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
