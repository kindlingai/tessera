use crate::diff::Hunk;
use crate::rules::Rule;
use crate::{Checker, RULES};
use fancy_regex::Regex;
use std::path::Path;

/// Each of a rule's examples, and whether it should hit.
fn cases(rule: &Rule) -> Vec<(String, bool)> {
    let split = Regex::new(r"(?m)^---[ \t]*$").expect("valid");
    let read = |name: &str| std::fs::read_to_string(rule.dir.join(name)).unwrap_or_default();
    let blocks = |text: String| -> Vec<String> {
        let parts = split.split(&text).filter_map(Result::ok);
        parts
            .map(|b| b.trim_matches('\n').to_string())
            .filter(|b| !b.trim().is_empty())
            .collect()
    };
    let mut out: Vec<(String, bool)> = blocks(read("bad.txt"))
        .into_iter()
        .map(|b| (b, true))
        .collect();
    out.extend(blocks(read("good.txt")).into_iter().map(|b| (b, false)));
    for pair in blocks(read("fixes.txt")) {
        if let Some((bad, good)) = pair.split_once("\n=>\n") {
            out.extend([(bad.to_string(), true), (good.to_string(), false)]);
        }
    }
    out
}

/// Bad examples hit their rule. Good examples do not.
#[test]
fn examples_match() {
    let Some(dir) = settings::read().spacy else {
        eprintln!("TESSERA_SPACY unset: skipped");
        return;
    };
    let checker = Checker::new(&dir).unwrap();
    let mut wrong = Vec::new();
    for rule in &checker.rules {
        for (text, bad) in cases(rule) {
            let found = checker.hunk(Hunk::file("example.txt", &text)).unwrap();
            if found.iter().any(|f| f.hit.rule == rule.id) != bad {
                wrong.push(format!(
                    "{} {}: {}",
                    rule.id,
                    if bad { "missed" } else { "flagged" },
                    text
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn rules_load() {
    let rules = crate::rules::load(Path::new(RULES)).unwrap();
    let ids: Vec<&str> = rules.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["agent-passive", "chained-clause", "hidden-negation"]);
    assert!(
        rules
            .iter()
            .all(|r| !r.name.is_empty() && !r.guidance().is_empty())
    );
}
