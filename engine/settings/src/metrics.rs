//! Counters and gauges, declared at their call sites, gathered here.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub use scattered_collect::declarative::{gather, scatter};
pub use scattered_collect::{ScatteredSortedReferencedSlice, ScatteredSortedSlice};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Counter,
    Gauge,
}

/// One value, read without a lock. Metrics may tear.
pub struct Metric {
    pub name: &'static str,
    pub help: &'static str,
    pub kind: Kind,
    value: AtomicU64,
}

impl Metric {
    pub const fn new(kind: Kind, name: &'static str, help: &'static str) -> Self {
        let value = AtomicU64::new(0);
        Self {
            name,
            help,
            kind,
            value,
        }
    }

    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Relaxed);
    }

    pub fn sub(&self, n: u64) {
        self.value.fetch_sub(n, Relaxed);
    }

    pub fn set(&self, n: u64) {
        self.value.store(n, Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.value.load(Relaxed)
    }
}

/// A group of metrics. `text` prints every group.
pub struct Registry {
    pub name: &'static str,
    pub metrics: &'static ScatteredSortedReferencedSlice<Metric>,
}

macro_rules! by_name {
    ($($ty:ty),*) => {$(
        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self.name == other.name
            }
        }

        impl Eq for $ty {}

        impl PartialOrd for $ty {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $ty {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.name.cmp(other.name)
            }
        }
    )*};
}

by_name!(Metric, Registry);

gather! {
    #[gather]
    pub static REGISTRIES: ScatteredSortedSlice<Registry>;
}

/// Every registry's metrics as Prometheus text.
pub fn text() -> String {
    let mut out = String::new();
    for registry in REGISTRIES.iter() {
        for m in registry.metrics.iter() {
            let kind = match m.kind {
                Kind::Counter => "counter",
                Kind::Gauge => "gauge",
            };
            let (name, help, value) = (m.name, m.help, m.get());
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}"
            );
        }
    }
    out
}

/// Declares a registry: `declare_registry!(NAME = "name")`.
#[macro_export]
macro_rules! declare_registry {
    ($(#[$doc:meta])* $vis:vis $name:ident = $label:literal) => {
        $crate::metrics::gather! {
            #[gather]
            $(#[$doc])*
            $vis static $name:
                $crate::metrics::ScatteredSortedReferencedSlice<$crate::metrics::Metric>;
        }

        const _: () = {
            $crate::metrics::scatter! {
                #[scatter($crate::metrics::REGISTRIES)]
                const REGISTRY: $crate::metrics::Registry = $crate::metrics::Registry {
                    name: $label,
                    metrics: &$name,
                };
            }
        };
    };
}

/// Adds a metric: `declare_metric!(NAME in REGISTRY = Gauge "name", "Help.")`.
#[macro_export]
macro_rules! declare_metric {
    ($(#[$doc:meta])* $vis:vis $name:ident in $registry:ident = $kind:ident $metric:literal, $help:literal) => {
        $crate::metrics::scatter! {
            #[scatter($registry)]
            $(#[$doc])*
            $vis static $name: $crate::metrics::Metric =
                $crate::metrics::Metric::new($crate::metrics::Kind::$kind, $metric, $help);
        }
    };
}
