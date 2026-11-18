use super::image::{parse, path};
use std::path::PathBuf;

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

#[test]
fn flags_and_words() {
    let got = parse(&args(&[
        "--n", "2", "--seed", "7", "-o", "a.png", "big", "pizza",
    ]));
    let got = got.ok().unwrap();
    assert_eq!((got.n, got.seed, got.words.len()), (Some(2), Some(7), 2));
    assert!(parse(&args(&["--n", "x"])).is_err());
    assert!(parse(&args(&["--bad", "1"])).is_err());
}

#[test]
fn output_paths() {
    let out = Some(PathBuf::from("d/out.png"));
    assert_eq!(path(&out, 1, 0, 1), PathBuf::from("d/out.png"));
    assert_eq!(path(&out, 1, 1, 3), PathBuf::from("d/out-2.png"));
    assert_eq!(path(&None, 9, 0, 1), PathBuf::from("image-9.png"));
}
