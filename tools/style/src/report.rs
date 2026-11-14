//! Hits as a report, for tessera-check to render.

use crate::rules::Rule;
use crate::tags::Hit;
use findings::{Finding, Level, Report, Snippet};

/// A hit, its file and its line of text.
pub struct Found {
    pub path: String,
    pub hit: Hit,
    pub text: String,
}

pub fn report(found: &[Found], rules: &[Rule]) -> Report {
    let mut r = Report::new("style", "Comment style");
    for rule in rules {
        let fix = Some(rule.guidance())
            .filter(|g| !g.is_empty())
            .unwrap_or(rule.description.clone());
        r.rules.insert(
            rule.id.clone(),
            findings::Rule {
                name: rule.name.clone(),
                fix,
            },
        );
    }
    for f in found {
        r.findings.push(Finding {
            rule: f.hit.rule.clone(),
            level: Level::Fail,
            path: f.path.clone(),
            line: f.hit.line,
            message: format!("matched \"{}\"", f.hit.found),
            snippets: vec![Snippet {
                path: f.path.clone(),
                line: f.hit.line,
                text: f.text.trim().to_string(),
            }],
            fix: None,
        });
    }
    r
}
