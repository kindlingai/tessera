//! Seconds spent per kernel or phase, summed over launches.

use std::collections::BTreeMap;
use std::fmt;

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
