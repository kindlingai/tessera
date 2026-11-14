use crate::markdown::{MARKER, render};
use findings::{Finding, Level, Report, Rule};

fn finding(level: Level) -> Finding {
    Finding {
        rule: "r".into(),
        level,
        path: "a.rs".into(),
        line: 3,
        message: "m".into(),
        snippets: Vec::new(),
        fix: Some("f".into()),
    }
}

#[test]
fn comment_renders() {
    let clean = render(&[Report::new("lint", "Code shape")], None).unwrap();
    assert!(clean.starts_with(MARKER) && clean.contains("None found."));
    let mut r = Report::new("style", "Comment style");
    r.rules.insert(
        "r".into(),
        Rule {
            name: "Rule".into(),
            fix: "Do this.".into(),
        },
    );
    r.findings = vec![finding(Level::Fail), finding(Level::Note)];
    let md = render(&[r], Some("https://x/blob/h")).unwrap();
    assert!(
        md.contains("| Comment style | 1 failed, 1 notes |")
            && md.contains("[`a.rs:3`](https://x/blob/h/a.rs#L3)")
    );
    assert!(md.contains("#### Rule\n\nDo this.") && !md.contains("None found."));
}
