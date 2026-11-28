use crate::mocks::{MockGenerator, call, json, loaded, loaded_with};
use serde_json::{Value, json};
use std::sync::Arc;
use tessera_bundle::ResponseParser;

#[test]
fn completion_whole() {
    let mock = Arc::new(MockGenerator::default());
    let slot = loaded(&mock, false);
    let body =
        json!({"model": "m", "prompt": "abcd", "stop": "x", "min_tokens": 2, "temperature": 0});
    let (status, got) = json(&slot, "/v1/completions", Some(body));
    assert_eq!(status, 200);
    assert_eq!(got["object"], "text_completion");
    assert_eq!(got["choices"][0]["text"], "hello");
    assert_eq!(got["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        got["usage"],
        json!({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 3}})
    );
    let asked = mock.asked.lock().unwrap().clone().unwrap();
    // The room the 16-token context leaves.
    assert_eq!(
        (asked.max_tokens, asked.min_tokens, asked.stop),
        (12, 2, vec!["x".to_string()])
    );
}

#[test]
fn chat_streamed() {
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let body = json!({
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let (status, text) = call(&slot, "/v1/chat/completions", Some(body));
    assert_eq!(status, 200);
    let data: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert_eq!(data.last(), Some(&"[DONE]"));
    let chunks: Vec<Value> = data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let deltas: Vec<&Value> = chunks.iter().map(|c| &c["choices"][0]["delta"]).collect();
    assert_eq!(
        deltas[..3],
        [
            &json!({"role": "assistant", "content": ""}),
            &json!({"content": "he"}),
            &json!({"content": "llo"})
        ]
    );
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
    assert_eq!(chunks[3]["choices"], json!([]));
    assert_eq!(chunks[3]["usage"]["total_tokens"], 9);
    let objects: Vec<&Value> = chunks.iter().map(|c| &c["object"]).collect();
    assert!(objects.iter().all(|o| **o == "chat.completion.chunk"));
}

#[test]
fn refusals() {
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let cases = [
        (json!({"model": "other", "prompt": "a"}), 404, "model"),
        (json!({"prompt": "a", "tools": [{}]}), 400, "tools"),
        (
            json!({"prompt": "a", "presence_penalty": 2.5}),
            400,
            "presence_penalty",
        ),
        (
            json!({"prompt": "a", "temperature": -0.1}),
            400,
            "temperature",
        ),
        (json!({"prompt": "a", "top_p": 0.0}), 400, "top_p"),
        (json!({"prompt": "a", "top_k": -2}), 400, "top_k"),
        (json!({"prompt": "a", "min_p": 1.5}), 400, "min_p"),
        (
            json!({"prompt": "abcd", "max_tokens": 13}),
            400,
            "max_tokens",
        ),
    ];
    for (body, status, param) in cases {
        let (got, error) = json(&slot, "/v1/completions", Some(body));
        assert_eq!((got, &error["error"]["param"]), (status, &json!(param)));
        assert_eq!(error["error"]["code"], status);
    }
}

#[test]
fn unknown_fields() {
    let said = json!({"role": "user", "content": "hi", "bogus": "x", "reasoning": "r"});
    let body = json!({"messages": [said]});
    for (strict, status) in [(false, 200), (true, 400)] {
        let slot = loaded(&Arc::new(MockGenerator::default()), strict);
        let (got, answer) = json(&slot, "/v1/chat/completions", Some(body.clone()));
        assert_eq!(got, status);
        assert_eq!(strict, answer["error"]["param"] == "messages[0].bogus");
    }
}

fn tool_parser() -> ResponseParser {
    let cfg = r#"{"response_template":{"fields":{
      "thinking":{"open":"<|channel>thought\n","close":"<channel|>","content":"text"},
      "tool_calls":{"open_pattern":"<\\|tool_call>call:(?P<name>\\w+)","close":"<tool_call|>",
        "content":"json","content_args":{"string_delims":[["<|\"|>","<|\"|>"]],"unquoted_keys":true},
        "transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}}
    }}}"#;
    ResponseParser::from_config(cfg).unwrap().unwrap()
}

#[test]
fn chat_tool_calls() {
    let mock = Arc::new(MockGenerator {
        pieces: vec![
            "<|channel>thought\ncheck<channel|>".into(),
            "<|tool_call>call:ping{a:<|\"|>o".into(),
            "k<|\"|>}<tool_call|><|tool_call>call:go{}<tool_call|>".into(),
        ],
        ..MockGenerator::default()
    });
    let slot = loaded_with(&mock, true, Some(tool_parser()));
    let tool = json!({"type": "function", "function": {"name": "ping"}});
    let mut body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [tool],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });
    let (status, got) = json(&slot, "/v1/chat/completions", Some(body.clone()));
    assert_eq!(status, 200);
    assert!(mock.asked.lock().unwrap().as_ref().unwrap().raw);
    let c = &got["choices"][0];
    assert_eq!(
        (&c["finish_reason"], &c["message"]["content"]),
        (&json!("tool_calls"), &Value::Null)
    );
    assert_eq!(c["message"]["reasoning_content"], "check");
    assert_eq!(
        got["usage"]["completion_tokens_details"]["reasoning_tokens"],
        1
    );
    assert_eq!(
        c["message"]["tool_calls"][0]["function"],
        json!({"name": "ping", "arguments": "{\"a\":\"ok\"}"})
    );
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let (s_status, sse) = call(&slot, "/v1/chat/completions", Some(body.clone()));
    assert_eq!(s_status, 200);
    assert!(
        sse.contains("\"finish_reason\":\"tool_calls\"") && sse.contains("\"reasoning_tokens\":1")
    );
    let (mut args, mut chunks) = ([String::new(), String::new()], [0usize; 2]);
    for line in sse
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
    {
        let v: Value = serde_json::from_str(line).unwrap();
        for item in v["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let i = item["index"].as_u64().unwrap() as usize;
            if item["function"]["name"].is_null() {
                args[i].push_str(item["function"]["arguments"].as_str().unwrap());
                chunks[i] += 1;
            }
        }
    }
    assert!(
        chunks[0] >= 2 && serde_json::from_str::<Value>(&args[0]).unwrap() == json!({"a": "ok"})
    );
    assert_eq!(serde_json::from_str::<Value>(&args[1]).unwrap(), json!({}));
    body["parallel_tool_calls"] = json!(false);
    let (s2, sse2) = call(&slot, "/v1/chat/completions", Some(body));
    assert_eq!(s2, 200);
    assert!(!sse2.contains("\"go\"") && !sse2.contains("\"index\":1"));
}

#[test]
fn sampling_params_accepted() {
    let slot = loaded(&Arc::new(MockGenerator::default()), true);
    let body = json!({
        "prompt": "a",
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "seed": 42,
        "presence_penalty": 0.5,
        "frequency_penalty": -0.25,
        "repetition_penalty": 1.1,
    });
    assert_eq!(json(&slot, "/v1/completions", Some(body)).0, 200);
}

struct UnguidedMock(&'static str);

impl tessera_tasks::api::Generator for UnguidedMock {
    fn supports_guides(&self) -> bool {
        false
    }

    fn generate(
        &self,
        _prompt: Vec<u32>,
        _params: tessera_tasks::api::Params,
        mut sink: tessera_tasks::api::Sink,
    ) -> Result<(), tessera_tasks::Error> {
        sink(tessera_tasks::api::Step::Done {
            text: self.0.into(),
            finish: tessera_tasks::api::Finish::Stop,
            tokens: 1,
            cached: 0,
        });
        Ok(())
    }
}

fn unguided_slot(text: &'static str, parser: ResponseParser) -> crate::Slot {
    let slot: crate::Slot = Arc::new(std::sync::OnceLock::new());
    slot.set(crate::Model {
        info: tessera_tasks::api::Info {
            name: "m".into(),
            context: 256,
        },
        tokens: Arc::new(crate::mocks::MockTokens),
        chat: Some(Arc::new(crate::mocks::MockChat {
            response: Some(parser),
        })),
        generator: Some(Arc::new(UnguidedMock(text))),
        score: None,
        strict: true,
        asks: None,
        image: None,
        embed: None,
    })
    .ok()
    .unwrap();
    slot
}

#[test]
fn forced_tool_choice() {
    let tool = json!({"type": "function", "function": {"name": "ping"}});
    let force_ping = json!({"type": "function", "function": {"name": "ping"}});
    // Mismatched function name fails on both whole and SSE.
    let bad_slot = unguided_slot("pong{}<tool_call|>", tool_parser());
    let req = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [tool],
        "tool_choice": force_ping,
    });
    let (status, got) = json(&bad_slot, "/v1/chat/completions", Some(req.clone()));
    assert_eq!(status, 500);
    assert!(
        got["error"]["message"]
            .as_str()
            .unwrap()
            .contains("pingpong")
    );
    let mut sse_req = req.clone();
    sse_req["stream"] = json!(true);
    let (_, sse) = call(&bad_slot, "/v1/chat/completions", Some(sse_req));
    assert!(
        !sse.contains("\"tool_calls\"")
            && !sse.contains("\"choices\"")
            && sse.contains("\"error\"")
            && sse.contains("[DONE]")
    );
    // Matching function name succeeds on whole and SSE.
    let ok_slot = unguided_slot("{}<tool_call|>", tool_parser());
    let (status, got) = json(&ok_slot, "/v1/chat/completions", Some(req));
    assert_eq!(status, 200);
    assert_eq!(got["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        got["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "ping"
    );
    // Missing required tool call fails on whole and SSE.
    let miss_slot = unguided_slot("never mind", tool_parser());
    let req_req = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "ping"}}],
        "tool_choice": "required",
    });
    let (status, got) = json(&miss_slot, "/v1/chat/completions", Some(req_req.clone()));
    assert_eq!(status, 500);
    assert!(got["error"]["message"].as_str().is_some());
    let mut miss_sse = req_req;
    miss_sse["stream"] = json!(true);
    let (_, sse) = call(&miss_slot, "/v1/chat/completions", Some(miss_sse));
    assert!(!sse.contains("never mind") && !sse.contains("\"stop\"") && sse.contains("\"error\""));
}
