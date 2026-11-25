use super::mcp::{Servers, answer};
use serde_json::{Value, json};

fn ask(servers: &mut Servers, message: Value) -> Value {
    answer(servers, &message.to_string()).unwrap_or_default()
}

/// The handshake, the tool list and refusals.
#[test]
fn answers_the_protocol() {
    let mut servers = Servers::default();
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"}});
    let reply = ask(&mut servers, init);
    assert_eq!(reply["result"]["protocolVersion"], "2025-06-18");
    let note = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    assert_eq!(answer(&mut servers, &note.to_string()), None);
    let bad = answer(&mut servers, "{").unwrap_or_default();
    assert_eq!(bad["error"]["code"], -32700);

    let list = ask(
        &mut servers,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    );
    let tools = list["result"]["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert_eq!(
        names,
        ["serve", "fetch", "ask", "image", "embed", "tag", "stop"]
    );
    assert!(tools[0]["inputSchema"]["properties"]["recipe"].is_object());

    let call = |name: &str| {
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": name, "arguments": {"handle": "serve-1"}}})
    };
    assert_eq!(ask(&mut servers, call("stop"))["result"]["isError"], true);
    assert_eq!(ask(&mut servers, call("chat"))["error"]["code"], -32602);
}
