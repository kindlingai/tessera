//! Open pull requests and their tiers.

use crate::overlap::Overlap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    Priority,
    Landing,
    Untagged,
}

impl Tier {
    /// Highest tier first.
    pub const ALL: [Tier; 3] = [Tier::Priority, Tier::Landing, Tier::Untagged];

    pub fn name(self) -> &'static str {
        match self {
            Tier::Priority => "priority",
            Tier::Landing => "landing",
            Tier::Untagged => "untagged",
        }
    }

    /// The label that marks this tier.
    pub fn label(self) -> Option<&'static str> {
        (self != Tier::Untagged).then(|| self.name())
    }

    pub fn named(name: &str) -> Option<Tier> {
        Tier::ALL.into_iter().find(|t| t.name() == name)
    }
}

pub struct Pull {
    pub number: u64,
    pub title: String,
    pub author: String,
    pub draft: bool,
    /// Has the `needs-human-review` label.
    pub waits: bool,
    pub ci: Option<String>,
    pub tier: Tier,
    /// When the tier label was set.
    pub since: String,
    pub setter: String,
    /// The app set the tier label.
    pub trusted: bool,
    /// `None` when the pull request has no overlap report.
    pub overlaps: Option<Vec<Overlap>>,
}

/// One tier's pull requests, first labelled first.
pub fn column(pulls: &[Pull], tier: Tier) -> Vec<&Pull> {
    let mut column: Vec<&Pull> = pulls.iter().filter(|p| p.tier == tier).collect();
    column.sort_by(|a, b| (&a.since, a.number).cmp(&(&b.since, b.number)));
    column
}
