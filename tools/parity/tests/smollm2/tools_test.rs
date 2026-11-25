use serde_json::{from_value, json};
use tessera_bundle::{Message, Overrides, RenderOptions, ResponseParser, Template};

/// The SmolLM2 recipe renders calls and parses them back.
#[test]
fn recipe_round_trips_calls() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/recipes/smollm2"
    );
    let recipe = tessera_recipe::Recipe::read(dir.as_ref()).unwrap();
    let base = tessera_testkit::stage("tokenizer_config.json", b"{}");
    let text = Overrides::new(recipe.overrides()).merged("tokenizer_config.json", &base);
    let text = text.unwrap();
    let call =
        |name| json!({"type": "function", "function": {"name": name, "arguments": {"a": 1}}});
    let calls = json!([call("add"), call("neg")]);
    let messages: Vec<Message> = from_value(json!([{"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": calls}, {"role": "tool", "content": "3"}]))
    .unwrap();
    let tools = [json!({"name": "add"})];
    let options = RenderOptions {
        tools: &tools,
        ..RenderOptions::default()
    };
    let template = Template::from_config(dir.as_ref(), &text).unwrap().unwrap();
    let prompt = template.render_options(&messages, &options).unwrap();
    assert!(prompt.contains(r#"<tools>[{"name": "add"}]</tools>"#));
    assert!(prompt.ends_with("<|im_start|>tool\n3<|im_end|>\n<|im_start|>assistant\n"));
    let reply = prompt.split("<|im_start|>assistant\n").nth(1).unwrap();
    let json = include_str!("../../../../frontend/bundle/src/response/smollm2.json");
    let parser = ResponseParser::fallback(json, &template).unwrap().unwrap();
    assert_eq!(json!(parser.parse(reply, true, false).tool_calls), calls);
}
