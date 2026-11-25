//! Progress on stderr: one line that redraws on a terminal.

use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tessera_engine::progress::{Event, Log, Open, Progress, Report, State};

/// One redrawn line on a terminal, else lines.
pub fn stderr() -> Progress {
    match std::io::stderr().is_terminal() {
        true => Progress::new(Arc::new(Bar::default())),
        false => Progress::new(Arc::new(Log)),
    }
}

/// Redraws the open steps as one line.
#[derive(Default)]
struct Bar {
    open: Open,
    /// When each top step started.
    started: Mutex<BTreeMap<u64, Instant>>,
}

impl Report for Bar {
    fn report(&self, event: &Event) {
        self.open.report(event);
        let Ok(mut started) = self.started.lock() else {
            return;
        };
        let mut out = std::io::stderr().lock();
        let _ = write!(out, "\r\x1b[2K");
        match (event.state, event.parent) {
            (State::Start, None) => drop(started.insert(event.id, Instant::now())),
            (State::Done, None) => {
                let took = started.remove(&event.id).map(|s| s.elapsed().as_secs_f64());
                let took = took.unwrap_or_default();
                let _ = writeln!(out, "{}: {} in {took:.1} s", event.name, event.amount());
            }
            (State::Stopped, None) => drop(started.remove(&event.id)),
            _ => {}
        }
        let _ = write!(out, "{}", line(&self.open.steps()).0);
        let _ = out.flush();
    }
}

/// Top step, then newest children until one has a total.
pub(crate) fn line(steps: &[Event]) -> (String, Option<f64>) {
    let Some(mut at) = steps.iter().find(|s| s.parent.is_none()) else {
        return (String::new(), None);
    };
    let mut names = vec![at.name.to_string()];
    while at.total.is_none()
        && let Some(child) = steps.iter().rev().find(|s| s.parent == Some(at.id))
    {
        at = child;
        names.push(at.name.to_string());
    }
    let mut out = names.join(" > ");
    if at.done > 0 || at.total.is_some() {
        out += &format!(" {}", at.amount());
    }
    if let Some(part) = at.fraction() {
        out += &format!(" {:.0}%", part * 100.0);
    }
    (out, at.fraction())
}

/// MCP progress notifications for one tool call.
pub struct Notify {
    token: Value,
    open: Open,
    told: Mutex<Told>,
}

/// The last notification a `Notify` sent.
#[derive(Default)]
struct Told {
    /// Top steps that ended. Each one adds 1 to `progress`.
    ended: u64,
    progress: f64,
    message: String,
}

impl Notify {
    pub fn new(token: Value) -> Self {
        Self {
            token,
            open: Open::default(),
            told: Mutex::default(),
        }
    }
}

impl Report for Notify {
    fn report(&self, event: &Event) {
        self.open.report(event);
        let Ok(mut told) = self.told.lock() else {
            return;
        };
        let ends = matches!(event.state, State::Done | State::Stopped);
        if ends && event.parent.is_none() {
            told.ended += 1;
        }
        let (mut message, part) = line(&self.open.steps());
        if message.is_empty() {
            message = event.name.to_string();
        }
        let floor = told.ended as f64;
        let mut progress = floor + part.unwrap_or(0.0) * 0.99;
        // Progress must rise with each notification.
        if progress <= told.progress && message != told.message {
            progress = told.progress + (floor + 1.0 - told.progress) * 0.001;
        }
        if progress <= told.progress {
            return;
        }
        let params = json!({"progressToken": self.token, "progress": progress, "message": message});
        let note = json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": params});
        println!("{note}");
        (told.progress, told.message) = (progress, message);
    }
}
