use crate::far::leans;
use crate::windows::Window;

fn win(path: &str) -> Window {
    Window {
        path: path.into(),
        line: 1,
        text: String::new(),
    }
}

#[test]
fn leans_to_movable() {
    let paths = [
        "e/a/src/x.rs",
        "e/a/src/y.rs",
        "e/a/src/new.rs",
        "e/b/src/z.rs",
        "e/b/src/w.rs",
    ];
    let wins: Vec<Window> = paths.iter().map(|p| win(p)).collect();
    let embs = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
    ];
    let crates = ["e/a".to_string(), "e/b".to_string()];
    let added = ["e/a/src/new.rs".to_string()];
    let found = leans(&wins, &embs, &crates, &[true, true], (&added, 0.03));
    assert_eq!(found.len(), 1);
    assert_eq!(
        (found[0].path.as_str(), found[0].other.as_str()),
        ("e/a/src/new.rs", "e/b")
    );
    assert!(leans(&wins, &embs, &crates, &[true, false], (&added, 0.03)).is_empty());
}
