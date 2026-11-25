use super::route::{argv, prompt, schema, tools};
use crate::source::{CATALOG, Origin, catalog, split};
use serde_json::json;

/// Tools come from the entry list, and picks become commands.
#[test]
fn picks_become_commands() {
    let catalog = catalog();
    assert_eq!(catalog.len(), CATALOG.len());
    let tools = tools(&catalog);
    let names: Vec<&str> = tools.iter().map(|t| t.name).collect();
    assert_eq!(names, ["serve", "fetch", "image", "embed"]);
    let branches = schema(&catalog, &tools)["anyOf"].as_array().map(Vec::len);
    assert_eq!(branches, Some(2 * tools.len() + 1));
    assert!(prompt(&catalog, &tools, "hi").ends_with("Request: hi\n"));
    let image = json!({ "tool": "image", "model": "flux2-klein", "prompt": "a fox", "n": 2 });
    let found = |words: &str| Ok(format!("found/{}", words.replace(' ', "-")));
    let args = argv(&image, &catalog, found)
        .ok()
        .flatten()
        .unwrap_or_default();
    assert_eq!(
        args,
        [
            "--recipe",
            "flux2-klein",
            "task",
            "image",
            "--n",
            "2",
            "a fox"
        ]
    );
    let given: Vec<String> = args.iter().map(String::from).collect();
    let origin = split(&given).ok().and_then(|(origin, _)| origin);
    assert!(matches!(origin, Some(Origin::Builtin("flux2-klein"))));
    let hub = json!({ "tool": "serve", "hub": "qwen 27b" });
    let args = argv(&hub, &catalog, found)
        .ok()
        .flatten()
        .unwrap_or_default();
    assert_eq!(args, ["--model", "hf:found/qwen-27b", "serve"]);
    let kind = json!({ "tool": "fetch", "hub": "the image model" });
    let args = argv(&kind, &catalog, found)
        .ok()
        .flatten()
        .unwrap_or_default();
    assert_eq!(args, ["--recipe", "flux2-klein", "fetch"]);
    let reply = argv(&json!({ "tool": "reply" }), &catalog, found);
    assert!(matches!(reply, Ok(None)));
}
