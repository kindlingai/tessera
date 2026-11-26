//! The repository's GitHub Actions cache against its limit.

use crate::github::{Client, Install};
use crate::page::{self, NEAR};
use crate::session::Session;
use crate::{Config, Fault, Host};
use std::collections::BTreeMap;
use tessera_html::{El, Part, el, table, text};

/// GitHub's documented default, for accounts that cannot read the limit.
const DEFAULT_GIB: u64 = 10;
const GIB: u64 = 1 << 30;

/// The caches of one ref.
#[derive(Default)]
struct Ref {
    bytes: u64,
    count: u64,
    /// The latest `last_accessed_at` of its caches.
    used: String,
}

pub struct Cache {
    bytes: u64,
    count: u64,
    /// The limit in GiB, when the account can read it.
    limit: Option<u64>,
    refs: BTreeMap<String, Ref>,
}

/// Reads usage, the limit and every cache entry.
pub async fn read<H: Host>(client: &Client<'_, H>, install: &Install) -> Result<Cache, Fault> {
    let base = format!("/repos/{}/actions", client.config.repo);
    let path = format!("{base}/cache/usage");
    let usage = client.app("GET", &path, install, &[]).await?;
    // Accounts without billing get 402 for the limit.
    let limit = format!("{base}/cache/storage-limit");
    let limit = client.app("GET", &limit, install, &[402, 403, 404]).await?;
    let mut cache = Cache {
        bytes: usage["active_caches_size_in_bytes"]
            .as_u64()
            .unwrap_or_default(),
        count: usage["active_caches_count"].as_u64().unwrap_or_default(),
        limit: limit["max_cache_size_gb"].as_u64(),
        refs: BTreeMap::new(),
    };
    let mut page = 1;
    loop {
        let path = format!("{base}/caches?per_page=100&page={page}");
        let found = client.app("GET", &path, install, &[]).await?;
        let entries = found["actions_caches"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if entries.is_empty() {
            return Ok(cache);
        }
        for entry in entries {
            let name = entry["ref"].as_str().unwrap_or_default();
            let at = entry["last_accessed_at"].as_str().unwrap_or_default();
            let one = cache.refs.entry(name.into()).or_default();
            one.bytes += entry["size_in_bytes"].as_u64().unwrap_or_default();
            one.count += 1;
            if *at > *one.used {
                one.used = at.into();
            }
        }
        page += 1;
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / GIB as f64)
}

/// The usage line, marked when near or over the limit.
fn usage(cache: &Cache) -> El {
    let limit = cache.limit.unwrap_or(DEFAULT_GIB);
    let share = cache.bytes * 100 / (limit * GIB);
    let source = if cache.limit.is_some() {
        ""
    } else {
        ", GitHub default"
    };
    let line = format!(
        "{} of {limit} GiB{source} ({share}%), {} caches. Over the limit, \
GitHub evicts the least recently used caches.",
        gib(cache.bytes),
        cache.count,
    );
    let class = if share >= NEAR { "bad" } else { "" };
    text("p", line).attr("class", class)
}

pub(crate) fn render(config: &Config, session: &Session, cache: &Cache) -> String {
    let mut refs: Vec<_> = cache.refs.iter().collect();
    refs.sort_by_key(|(_, r)| std::cmp::Reverse(r.bytes));
    let mut rows = Vec::new();
    for (name, one) in refs {
        let row: Vec<Part> = vec![
            name.as_str().into(),
            one.count.to_string().into(),
            gib(one.bytes).into(),
            one.used.as_str().into(),
        ];
        rows.push(row);
    }
    let head = ["Ref", "Caches", "Size", "Last used"];
    let main = el("main").kid(usage(cache)).kid(table(&head, rows));
    page::frame(config, session, main)
}
