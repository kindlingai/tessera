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

/// The rule's example type, else a guess from the text.
fn path(rule: &Rule, text: &str) -> String {
    let code = Regex::new(r"(?m)^\s*(//|/\*|\*\s)|\{\s*$|^\s*\}|;\s*$|^\s*(fn|def|let|const|return|pub|impl|struct)\b|^\s*#\[").expect("valid");
    let markdown =
        Regex::new(r"(?m)^#{1,6}\s|^\s*[-*+]\s|^\s*\|.*\|\s*$|\*\*[^*]+\*\*").expect("valid");
    let guess = match (code.is_match(text), markdown.is_match(text)) {
        (Ok(true), _) => ".rs",
        (_, Ok(true)) => ".md",
        _ => ".txt",
    };
    format!("example{}", rule.examples.as_deref().unwrap_or(guess))
}

/// Bad examples caught, and good examples that fail.
fn score(checker: &Checker, rules: &[&Rule], leads: bool) -> (usize, usize, Vec<String>) {
    let (mut bad, mut caught, mut flagged) = (0, 0, Vec::new());
    for rule in rules {
        for (text, should_hit) in cases(rule) {
            let found = checker.hunk(Hunk::file(&path(rule, &text), &text)).unwrap();
            let mine = |lead_ok: bool| {
                let counts = |l: crate::Level| lead_ok || l == crate::Level::Fail;
                found
                    .iter()
                    .any(|f| f.hit.rule == rule.id && counts(f.hit.level))
            };
            // Leads may sit on good examples, and count on bad.
            let (hit, fails) = (mine(leads), mine(false));
            bad += usize::from(should_hit);
            caught += usize::from(should_hit && hit);
            if fails && !should_hit {
                flagged.push(format!(
                    "{}: {}",
                    rule.id,
                    text.lines().next().unwrap_or_default()
                ));
            }
        }
    }
    (bad, caught, flagged)
}

#[test]
fn detectors_pass_good() {
    let checker = Checker::new(Path::new("/nonexistent")).unwrap();
    let rules: Vec<&Rule> = checker
        .rules
        .iter()
        .filter(|r| !r.detectors.is_empty() && !r.parse)
        .collect();
    let (bad, caught, flagged) = score(&checker, &rules, false);
    eprintln!("detectors: {caught}/{bad} bad caught");
    assert!(flagged.is_empty(), "good examples flagged: {flagged:#?}");
}

/// Tag patterns, leads included, pass every good example.
#[test]
fn tags_pass_good() {
    let Some(dir) = settings::read().spacy else {
        eprintln!("TESSERA_SPACY unset: skipped");
        return;
    };
    let checker = Checker::new(&dir).unwrap();
    let rules: Vec<&Rule> = checker.rules.iter().filter(|r| r.parse).collect();
    let (bad, caught, flagged) = score(&checker, &rules, true);
    eprintln!("tags: {caught}/{bad} bad caught");
    assert!(flagged.is_empty(), "good examples flagged: {flagged:#?}");
    assert!(caught * 10 >= bad * 9, "{caught}/{bad} bad caught");
}

#[test]
fn rules_load() {
    let rules = crate::rules::load(Path::new(RULES)).unwrap();
    assert!(
        rules
            .iter()
            .all(|r| !r.name.is_empty() && !r.guidance().is_empty())
    );
}
