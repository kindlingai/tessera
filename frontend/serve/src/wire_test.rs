use crate::mocks::{MockGenerator, json, loaded};
use crate::wire::ResponseFormat;
use sensai_tasks::api::Format;
use serde_json::json;
use std::sync::Arc;

#[test]
fn formats_parse() {
    let schema = json!({"type": "object"});
    let named = json!({"name": "n", "strict": true, "schema": schema});
    let cases = [
        (json!({"type": "text"}), None),
        (json!({"type": "json_object"}), Some(Format::Object)),
        (
            json!({"type": "json_schema", "json_schema": named}),
            Some(Format::Schema(schema.clone())),
        ),
    ];
    for (given, want) in cases {
        let parsed: ResponseFormat = serde_json::from_value(given).unwrap();
        assert_eq!(parsed.format(), want);
    }
}

#[test]
fn bad_schemas_refused() {
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let format = json!({"type": "json_schema", "json_schema": {"schema": true}});
    let body = json!({"prompt": "a", "response_format": format});
    let (status, error) = json(&slot, "/v1/completions", Some(body));
    assert_eq!(
        (status, &error["error"]["param"]),
        (400, &json!("response_format"))
    );
}
