//! Crates and files near their `lint.toml` limits.

use crate::github::{Client, Install};
use crate::page::{self, NEAR};
use crate::session::Session;
use crate::{Config, Fault, Host, unzip};
use serde::Deserialize;
use tessera_html::{Part, el, table, text};

const ARTIFACT: &str = "checks-lint";
const FILE: &str = "lint.json";
/// The tab lists measures at this percent or more.
const WATCH: u64 = 75;

#[derive(Deserialize)]
pub struct Measure {
    rule: String,
    path: String,
    value: u64,
    max: u64,
}

impl Measure {
    fn share(&self) -> u64 {
        self.value * 100 / self.max.max(1)
    }
}

#[derive(Deserialize)]
struct Report {
    #[serde(default)]
    measures: Option<Vec<Measure>>,
}

pub struct Limits {
    /// The commit the report measured.
    sha: String,
    /// `None` when the report predates measures.
    measures: Option<Vec<Measure>>,
}

/// The measures in a lint report.
fn read_report(json: &[u8]) -> Result<Option<Vec<Measure>>, Fault> {
    let report: Report = serde_json::from_slice(json)
        .map_err(|e| Fault::new(502, format!("The lint report is bad JSON: {e}")))?;
    Ok(report.measures)
}

/// The newest lint report of a CI push to main.
// A newer push can cancel a run before its upload.
pub async fn read<H: Host>(client: &Client<'_, H>, install: &Install) -> Result<Limits, Fault> {
    let base = format!("/repos/{}/actions", client.config.repo);
    let mut page = 1;
    loop {
        let path =
            format!("{base}/workflows/ci.yml/runs?branch=main&event=push&per_page=20&page={page}");
        let runs = client.app("GET", &path, install, &[]).await?;
        let runs = runs["workflow_runs"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if runs.is_empty() {
            return Err(Fault::new(404, "No CI push to main has a lint report."));
        }
        for run in runs {
            let path = format!("{base}/runs/{}/artifacts?name={ARTIFACT}", run["id"]);
            let found = client.app("GET", &path, install, &[]).await?;
            let Some(id) = found["artifacts"][0]["id"].as_u64() else {
                continue;
            };
            let path = format!("{base}/artifacts/{id}/zip");
            let zip = client.download(&path, install).await?;
            let Some(json) = unzip::file(&zip, FILE) else {
                return Err(Fault::new(502, format!("Artifact {id} lacks {FILE}.")));
            };
            let sha = run["head_sha"].as_str().unwrap_or_default().into();
            let measures = read_report(&json)?;
            return Ok(Limits { sha, measures });
        }
        page += 1;
    }
}

pub(crate) fn render(config: &Config, session: &Session, limits: &Limits) -> String {
    let short = limits.sha.get(..8).unwrap_or_default();
    let Config { web, repo, .. } = config;
    let commit = text("a", short).attr("href", format!("{web}/{repo}/commit/{}", limits.sha));
    let mut head = el("p").text("Lint report of main at ").kid(commit);
    let Some(measures) = &limits.measures else {
        let main = el("main").kid(head.text(" has no measures. Its stickler is too old."));
        return page::frame(config, session, main);
    };
    let mut near: Vec<&Measure> = measures.iter().filter(|m| m.share() >= WATCH).collect();
    near.sort_by_key(|m| std::cmp::Reverse(m.share()));
    head = head.text(format!(
        ". {} of {} measures are at {WATCH}% of their limit or more.",
        near.len(),
        measures.len()
    ));
    let mut rows = Vec::new();
    for m in near {
        let share = text("span", format!("{}%", m.share()));
        let share = share.attr("class", if m.share() >= NEAR { "bad" } else { "" });
        let row: Vec<Part> = vec![
            m.rule.as_str().into(),
            m.path.as_str().into(),
            format!("{} of {}", m.value, m.max).into(),
            share.into(),
        ];
        rows.push(row);
    }
    let main = el("main")
        .kid(head)
        .kid(table(&["Rule", "Path", "Size", "Share"], rows));
    page::frame(config, session, main)
}
