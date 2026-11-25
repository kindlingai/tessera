use crate::task::{Image, Task, path};
use serde_json::json;
use std::path::PathBuf;

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

/// The `--clip <dir>` pipeline flag.
fn clip(flag: &str) -> Option<bool> {
    (flag == "--clip").then_some(true)
}

#[test]
fn flags_and_words() {
    let given = [
        "--n", "2", "--seed", "7", "--clip", "c", "-o", "a.png", "big", "pizza",
    ];
    let got = Image::parse(&args(&given), &clip).ok().unwrap();
    assert_eq!(
        (got.n, got.seed, got.prompt.as_str()),
        (Some(2), Some(7), "big pizza")
    );
    assert_eq!(got.load_flags(), ["--clip", "c"]);
    assert!(Image::parse(&args(&["--n", "x", "a"]), &clip).is_err());
    assert!(Image::parse(&args(&["--bad", "1", "a"]), &clip).is_err());
    let usage =
        "image [--n <k>] [--seed <s>] [--image-size <px>] [--clip <dir>] [-o <path>] [prompt...]";
    assert_eq!(Image::usage("[--clip <dir>]"), usage);
}

#[test]
fn tool_calls() {
    let schema = Image::schema();
    assert_eq!(schema["required"], json!(["prompt"]));
    assert_eq!(schema["properties"]["n"]["maximum"], json!(8));
    assert_eq!(Image::function()["function"]["name"], json!("image"));
    let words = Image::tool_args(&Image::example()).ok();
    assert_eq!(words, Some(args(&["--n", "2", "a red bicycle"])));
    assert!(Image::from_json(&json!({"prompt": "a", "n": 9})).is_err());
    assert!(Image::from_json(&json!({"prompt": "a", "seed": 1})).is_err());
}

#[test]
fn output_paths() {
    let out = Some(PathBuf::from("d/out.png"));
    assert_eq!(path(&out, 1, 0, 1), PathBuf::from("d/out.png"));
    assert_eq!(path(&out, 1, 1, 3), PathBuf::from("d/out-2.png"));
    assert_eq!(path(&None, 9, 0, 1), PathBuf::from("image-9.png"));
}
