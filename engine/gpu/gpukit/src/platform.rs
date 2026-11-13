//! The hardware as asked, plus a table for the rest.

use crate::Limits;
use include_dir::{Dir, include_dir};
use serde::{Deserialize, Serialize};

/// One JSON file per architecture, matched by platform id.
static TABLE: Dir = include_dir!("$CARGO_MANIFEST_DIR/platforms");

/// What a driver cannot say about an architecture.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Row {
    /// Ids this row covers. A trailing star takes any rest.
    pub matches: Vec<String>,
    /// The matrix unit's tile: rows, columns, depth. None: no unit.
    #[serde(default)]
    pub tile: Option<[usize; 3]>,
    /// Whether a launch may overlap the one before it.
    #[serde(default)]
    pub pdl: bool,
    #[serde(default)]
    pub note: String,
}

/// A device as asked, plus its table row.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Platform {
    pub backend: String,
    /// The architecture as the driver names it: sm_121, gfx11_5, apple9.
    pub arch: String,
    pub name: String,
    pub driver: String,
    pub limits: Limits,
    /// Streaming multiprocessors, or compute units.
    pub units: usize,
    /// Bytes a second from clock and bus, or zero.
    pub bandwidth: f64,
    pub integrated: bool,
    pub row: Option<Row>,
}

/// Facts a backend asks its device, before the table.
pub struct Asked {
    pub backend: &'static str,
    pub arch: String,
    pub name: String,
    pub driver: String,
    pub limits: Limits,
    pub units: usize,
    pub bandwidth: f64,
    pub integrated: bool,
}

impl Platform {
    /// `asked`, with its row found and the row's facts applied.
    pub fn new(asked: Asked) -> Self {
        let id = format!("{} {}", asked.backend, asked.arch);
        let row = row_for(&id);
        let mut limits = asked.limits;
        limits.pdl = row.as_ref().is_some_and(|r| r.pdl);
        Self {
            backend: asked.backend.into(),
            arch: asked.arch,
            name: asked.name,
            driver: asked.driver,
            limits,
            units: asked.units,
            bandwidth: asked.bandwidth,
            integrated: asked.integrated,
            row,
        }
    }

    /// The host, whose executor plays a small matrix unit.
    pub fn host(limits: Limits) -> Self {
        Self {
            backend: "host".into(),
            arch: "cpu".into(),
            ..Self::assumed(limits)
        }
    }

    /// A device not asked yet, with `limits` and a unit.
    pub fn assumed(limits: Limits) -> Self {
        let row = Row {
            tile: Some([8, 8, 8]),
            note: "assumed, not asked".into(),
            ..Row::default()
        };
        Self {
            limits,
            row: Some(row),
            ..Self::default()
        }
    }

    /// The backend and architecture, which rows and samples key on.
    pub fn id(&self) -> String {
        format!("{} {}", self.backend, self.arch)
    }

    /// Whether the table knows this architecture.
    pub fn tabled(&self) -> bool {
        self.row.is_some()
    }

    /// The matrix unit's tile, if the table lists one.
    pub fn tile(&self) -> Option<[usize; 3]> {
        self.row.as_ref()?.tile
    }
}

/// The row whose pattern fits `id`, from the embedded table.
fn row_for(id: &str) -> Option<Row> {
    let fits = |pattern: &str| match pattern.strip_suffix('*') {
        Some(head) => id.starts_with(head),
        None => id == pattern,
    };
    TABLE
        .files()
        .filter_map(|f| serde_json::from_slice::<Row>(f.contents()).ok())
        .find(|row| row.matches.iter().any(|p| fits(p)))
}

/// Bytes a second from clock in kHz and bus bits.
pub fn bandwidth(clock_khz: i32, bus_bits: i32) -> f64 {
    // Memory moves on both clock edges.
    2.0 * clock_khz as f64 * 1e3 * bus_bits as f64 / 8.0
}
