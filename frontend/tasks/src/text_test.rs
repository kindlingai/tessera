use crate::{Spool, Stops};
use std::cell::Cell;

/// Ids 1..=6 split UTF-8, 8 mixes ASCII+UTF-8, 10..=30 model SentencePiece.
fn decode(ids: &[u32]) -> String {
    let mut bytes = Vec::new();
    for id in ids {
        match id {
            1 => bytes.push(0xC3),
            2 => bytes.push(0xA9),
            3 => bytes.push(0xF0),
            4 => bytes.push(0x9F),
            5 => bytes.push(0x98),
            6 => bytes.push(0x80),
            8 => bytes.extend_from_slice(b"a\xc3"),
            9 => {}
            10 => bytes.extend_from_slice(b" hello"),
            20 => bytes.extend_from_slice(b" world"),
            30 => bytes.push(b' '),
            _ => bytes.extend_from_slice(b"ab"),
        }
    }
    let raw = String::from_utf8_lossy(&bytes);
    match ids.first() {
        Some(10 | 20 | 30) => raw.strip_prefix(' ').unwrap_or(&raw).to_string(),
        _ => raw.into_owned(),
    }
}

fn check_spool(ids: &[u32]) -> (Vec<String>, usize) {
    let calls = Cell::new(0usize);
    let mut spool = Spool::default();
    let pieces: Vec<String> = ids
        .iter()
        .map(|&id| {
            spool.push(id, |s| {
                calls.set(calls.get() + s.len());
                decode(s)
            })
        })
        .collect();
    assert_eq!(pieces.concat(), decode(ids));
    (pieces, calls.get())
}

#[test]
fn whole_characters() {
    let (pieces, _) = check_spool(&[7, 1, 2, 7, 3, 4, 5, 6, 7]);
    assert_eq!(
        pieces,
        ["ab", "", "\u{e9}", "ab", "", "", "", "\u{1f600}", "ab"]
    );
    assert_eq!(
        check_spool(&[10, 20, 30, 20]).0,
        ["hello", " world", " ", " world"]
    );
    let mut seq: Vec<u32> = (0..2_000)
        .map(|i| if i % 2 == 0 { 10 } else { 20 })
        .collect();
    seq.extend(std::iter::repeat_n(9, 100));
    seq.extend(std::iter::repeat_n(30, 100));
    seq.push(20);
    assert!(check_spool(&seq).1 < 15_000);
    for n in [16usize, 200] {
        let mut lead: Vec<u32> = std::iter::repeat_n(9, n).collect();
        lead.push(20);
        let (p, v) = check_spool(&lead);
        assert!(p.concat() == " world" && v < n * 20);
    }
    for tail in [2u32, 7, 20] {
        let mut bad: Vec<u32> = std::iter::repeat_n(1, 2_000).collect();
        bad.push(tail);
        assert!(check_spool(&bad).1 < 25_000);
    }
    let mut mix: Vec<u32> = std::iter::repeat_n(8, 500).collect();
    mix.push(2);
    assert!(check_spool(&mix).1 < 10_000);
    let mut stall = vec![1u32];
    stall.extend(std::iter::repeat_n(9, 500));
    stall.push(2);
    let (p, v) = check_spool(&stall);
    assert!(p.concat() == "\u{e9}" && v < 12_000);
}

#[test]
fn stops_hold() {
    let mut held = Stops::new(vec!["END".into()]);
    let got: Vec<_> = ["ab", "cE", "N", "Dx"].map(|p| held.push(p)).into();
    assert_eq!(
        got,
        [("", false), ("ab", false), ("c", false), ("", true)].map(|(t, c)| (t.into(), c))
    );
    let mut tail = Stops::new(vec!["END".into()]);
    assert_eq!(tail.push("abEN"), ("ab".into(), false));
    assert_eq!(tail.rest(), "EN");
    check_stops_tagged();
}

fn check_stops_tagged() {
    let mut stops = Stops::new(vec!["ZZ".into()]);
    assert!(stops.is_buffering());
    assert_eq!(
        stops.push_tagged(1, "preface "),
        (vec![(1, "preface".into())], false)
    );
    assert_eq!(
        stops.push_tagged(2, "<think>é"),
        (vec![(1, " ".into()), (2, "<think>".into())], false)
    );
    assert_eq!(stops.rest_tagged(), vec![(2, "é".into())]);
    // Empty UTF-8 lead token stays ordered right before its continuation.
    let mut utf8 = Stops::new(vec!["ZZ".into()]);
    assert_eq!(
        utf8.push_tagged(1, "<think>"),
        (vec![(1, "<think".into())], false)
    );
    assert_eq!(utf8.push_tagged(2, ""), (vec![], false));
    assert_eq!(utf8.push_tagged(3, "é"), (vec![(1, ">".into())], false));
    assert_eq!(utf8.rest_tagged(), vec![(2, "".into()), (3, "é".into())]);
}
