//! Findings as colored text, for the hook and local runs.

use findings::{Level, Report};
use std::fmt::Write;

/// ANSI styles, or plain text off a terminal.
struct Paint(bool);

impl Paint {
    fn wrap(&self, code: &str, text: &str) -> String {
        match self.0 {
            true => format!("\x1b[{code}m{text}\x1b[0m"),
            false => text.to_string(),
        }
    }
}

pub fn render(reports: &[Report], color: bool) -> String {
    let p = Paint(color);
    let mut out = String::new();
    for r in reports {
        let head = p.wrap("1", &r.title);
        let _ = match (&r.skipped, r.findings.is_empty()) {
            (Some(why), _) => writeln!(out, "{head}: {}", p.wrap("33", &format!("skipped, {why}"))),
            (None, true) => writeln!(out, "{head}: {}", p.wrap("32", "clean")),
            (None, false) => writeln!(out, "{head}:"),
        };
        let mut shown = Vec::new();
        for f in &r.findings {
            let (code, word) = if f.level == Level::Fail {
                ("31", "fail")
            } else {
                ("33", "note")
            };
            let at = p.wrap("1", &format!("{}:{}", f.path, f.line));
            let _ = writeln!(
                out,
                "  {at}: {} {}: {}",
                p.wrap(code, word),
                f.rule,
                f.message
            );
            for s in &f.snippets {
                for line in s.text.lines().take(6) {
                    let _ = writeln!(out, "    {}", p.wrap("2", &format!("| {line}")));
                }
            }
            if let Some(fix) = &f.fix {
                let _ = writeln!(out, "    {}", p.wrap("36", &format!("fix: {fix}")));
            }
            if !shown.contains(&&f.rule) {
                shown.push(&f.rule);
            }
        }
        for id in shown {
            if let Some(rule) = r.rules.get(id) {
                let _ = writeln!(
                    out,
                    "  {} {}",
                    p.wrap("36", &format!("{}:", rule.name)),
                    rule.fix
                );
            }
        }
    }
    out
}
