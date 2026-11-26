//! Tokens through the ranks: a timeline of launches and sums.

use crate::lowering::head;
use crate::stages::Pass;
use serde::Serialize;
use std::error::Error;
use std::fmt::Write;
use tessera_html::{el, json, text};

const SCRIPT: &str = include_str!("pump.js");

/// One launch on one rank, placed in time.
#[derive(Serialize)]
pub struct Event {
    pub rank: usize,
    pub step: usize,
    pub kernel: &'static str,
    pub rule: &'static str,
    pub covers: Vec<u32>,
    pub start: f64,
    pub end: f64,
    pub sum: bool,
    pub bytes: usize,
}

/// One pass: the prompt, or one new token.
#[derive(Serialize)]
pub struct Lap {
    pub label: String,
    pub rows: u64,
    pub context: u64,
    pub start: f64,
    pub end: f64,
    pub events: Vec<Event>,
}

#[derive(Serialize)]
struct Data<'a> {
    model: &'a str,
    ranks: Vec<String>,
    laps: &'a [Lap],
}

/// Ranks run alone until a sum, then wait.
fn lap(pass: &Pass, start: f64) -> Result<Lap, Box<dyn Error>> {
    let sums: Vec<Vec<usize>> = pass
        .ranks
        .iter()
        .map(|r| (0..r.steps.len()).filter(|i| r.steps[*i].sum).collect())
        .collect();
    if sums.iter().any(|s| s.len() != sums[0].len()) {
        return Err(format!("{}: ranks disagree on the number of sums", pass.label).into());
    }
    let mut clocks = vec![start; pass.ranks.len()];
    let mut next = vec![0usize; pass.ranks.len()];
    let mut events = Vec::new();
    for k in 0..=sums[0].len() {
        for (r, rank) in pass.ranks.iter().enumerate() {
            let until = sums[r].get(k).copied().unwrap_or(rank.steps.len());
            for i in next[r]..until {
                let s = &rank.steps[i];
                events.push(Event {
                    rank: r,
                    step: i,
                    kernel: s.kernel,
                    rule: s.rule,
                    covers: s.covers.clone(),
                    start: clocks[r],
                    end: clocks[r] + s.seconds,
                    sum: false,
                    bytes: s.bytes,
                });
                clocks[r] += s.seconds;
            }
            next[r] = until;
        }
        if k == sums[0].len() {
            break;
        }
        let met = clocks.iter().copied().fold(start, f64::max);
        let took = (0..clocks.len())
            .map(|r| pass.ranks[r].steps[sums[r][k]].seconds)
            .fold(0.0, f64::max);
        for (r, rank) in pass.ranks.iter().enumerate() {
            let s = &rank.steps[sums[r][k]];
            events.push(Event {
                rank: r,
                step: sums[r][k],
                kernel: s.kernel,
                rule: s.rule,
                covers: s.covers.clone(),
                start: met,
                end: met + took,
                sum: true,
                bytes: s.bytes,
            });
            clocks[r] = met + took;
            next[r] = sums[r][k] + 1;
        }
    }
    let end = clocks.iter().copied().fold(start, f64::max);
    Ok(Lap {
        label: pass.label.clone(),
        rows: pass.rows,
        context: pass.context,
        start,
        end,
        events,
    })
}

/// Every pass, one after another.
pub fn laps(passes: &[Pass]) -> Result<Vec<Lap>, Box<dyn Error>> {
    let mut out: Vec<Lap> = Vec::new();
    for pass in passes {
        let start = out.last().map_or(0.0, |l| l.end);
        out.push(lap(pass, start)?);
    }
    Ok(out)
}

/// The stepped view as text, one line per rank.
pub fn terminal(laps: &[Lap], names: &[String]) -> String {
    let mut out = String::new();
    for lap in laps {
        let _ = writeln!(
            out,
            "{} ({} rows, context {})",
            lap.label, lap.rows, lap.context
        );
        let mut runs = vec![(0usize, 0.0f64); names.len()];
        for e in &lap.events {
            if !e.sum {
                runs[e.rank].0 += 1;
                runs[e.rank].1 += e.end - e.start;
                continue;
            }
            if e.rank != 0 {
                continue;
            }
            for (r, (n, t)) in runs.iter_mut().enumerate() {
                let _ = writeln!(out, "  {:<8} {n:>3} kernels {:>9.2} µs", names[r], *t * 1e6);
                (*n, *t) = (0, 0.0);
            }
            let _ = writeln!(
                out,
                "  <-> sum of {:.1} KB across the transport, {:.2} µs",
                e.bytes as f64 / 1024.0,
                (e.end - e.start) * 1e6
            );
        }
        for (r, (n, t)) in runs.iter().enumerate() {
            let _ = writeln!(out, "  {:<8} {n:>3} kernels {:>9.2} µs", names[r], t * 1e6);
        }
        let _ = writeln!(out, "  done at {:.2} µs", lap.end * 1e6);
    }
    out
}

/// The animated page: lanes per rank, a cursor.
pub fn page(model: &str, names: &[String], laps: &[Lap]) -> String {
    let data = Data {
        model,
        ranks: names.to_vec(),
        laps,
    };
    let title = format!("{model}: tokens through the ranks");
    let controls = el("p")
        .kid(text("button", "play").attr("id", "play"))
        .kid(text("button", "step").attr("id", "step"))
        .kid(text("button", "restart").attr("id", "restart"))
        .kid(el("select").attr("id", "lap"))
        .kid(text("a", "lowering").attr("href", "lowering.html"));
    let body = el("body")
        .kid(text("h1", title.clone()))
        .kid(controls)
        .kid(el("div").attr("id", "lanes"))
        .kid(el("div").attr("id", "info"))
        .kid(
            el("script")
                .attr("id", "data")
                .attr("type", "application/json")
                .kid(json(&data)),
        )
        .kid(el("script").kid(tessera_html::Part::Raw(SCRIPT.into())));
    el("html").kid(head(&title)).kid(body).page()
}
