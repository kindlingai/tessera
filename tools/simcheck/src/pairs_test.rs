use crate::pairs::{Allow, near};
use crate::windows::Window;

fn win(path: &str, line: usize) -> Window {
    Window {
        path: path.into(),
        line,
        text: String::new(),
    }
}

/// A unit vector `t` radians around a plane.
fn unit(t: f32) -> Vec<f32> {
    vec![t.cos(), t.sin()]
}

#[test]
fn near_copies_found() {
    let wins = [
        win("a.rs", 1),
        win("a.rs", 8),
        win("b.rs", 1),
        win("c.rs", 1),
        win("a.cu", 1),
    ];
    let embs = [unit(0.0), unit(0.0), unit(0.1), unit(1.0), unit(0.0)];
    let (hits, _) = near(&wins, &embs, 20, 0.99, &Allow::parse("pairs = []").unwrap());
    // a.rs windows overlap. a.cu is another language.
    let found: Vec<_> = hits.iter().map(|h| (h.a, h.b, h.count)).collect();
    assert_eq!(found, [(0, 2, 2)]);
    let ab = Allow::parse(r#"pairs = [["b.rs", "a.rs"], ["c.rs", "a.rs"]]"#).unwrap();
    let (hits, used) = near(&wins, &embs, 20, 0.99, &ab);
    assert!(hits.is_empty());
    // c.rs is far from a.rs, so its pair allows nothing.
    let stale = ab.stale(&used);
    assert_eq!(stale, [&("c.rs".to_string(), "a.rs".to_string())]);
}
