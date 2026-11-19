//! The kept rows' totals as Prometheus text.

use crate::http::{Body, Response};
use crate::{Slot, with_model};
use sensai_engine::Totals;
use std::fmt::Write;
use std::sync::{Arc, Mutex};

/// The engine thread's last totals. None without kept rows.
pub type Shared = Arc<Mutex<Option<Totals>>>;

fn text(t: &Totals) -> String {
    let lines = [
        ("vllm:kv_cache_size_tokens", "gauge", t.pool.size as u64),
        ("vllm:kv_active_tokens", "gauge", t.pool.active as u64),
        ("tessera:kv_worst_tokens", "gauge", t.pool.worst as u64),
        ("tessera:kv_live_runs", "gauge", t.pool.live_runs as u64),
        ("tessera:kv_kept_runs", "gauge", t.pool.kept_runs as u64),
        ("tessera:kv_hits_total", "counter", t.hits),
        ("tessera:kv_misses_total", "counter", t.misses),
    ];
    let mut out = String::new();
    for (name, kind, value) in lines {
        let _ = writeln!(out, "# TYPE {name} {kind}\n{name} {value}");
    }
    out
}

pub(crate) fn metrics(slot: &Slot) -> Response {
    let seen = with_model(slot, |m| Ok(m.kv.lock().ok().and_then(|t| *t)));
    match seen {
        Ok(totals) => Response {
            status: 200,
            body: Body::Full {
                content_type: "text/plain; version=0.0.4",
                bytes: totals.map_or_else(String::new, |t| text(&t)).into_bytes(),
            },
        },
        Err(refused) => refused.into(),
    }
}
