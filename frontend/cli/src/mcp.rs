//! `mcp`: commands and tasks as MCP tools over stdio.

use crate::source::{CATALOG, Origin, Stop, split};
use crate::task::{ENTRIES, Entry, describe};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use tessera_tasks::Error;
use tessera_tasks::task::{Chat, Collect, Run, Sink, Task};

/// Protocol versions it answers. The first is the default.
const VERSIONS: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];

/// JSON-RPC error codes.
const PARSE: i64 = -32700;
const METHOD: i64 = -32601;
const PARAMS: i64 = -32602;

/// The arguments that pick the model, as model flags.
const ORIGIN: [(&str, &str); 2] = [("recipe", "--recipe"), ("model", "--model")];

/// Stop a server that the serve tool started.
#[derive(Task)]
#[task(name = "stop")]
pub struct Halt {
    /// The handle that serve returned.
    #[arg(positional)]
    pub handle: String,
}

/// The servers that serve started, by handle.
#[derive(Default)]
pub struct Servers {
    running: BTreeMap<String, Child>,
}

impl Run<&mut Servers> for Halt {
    fn run(self, servers: &mut Servers, sink: &mut dyn Sink) -> Result<(), Error> {
        let unknown = || format!("{}: unknown handle", self.handle);
        let mut child = servers.running.remove(&self.handle).ok_or_else(unknown)?;
        let _ = child.kill();
        let status = child.wait()?;
        sink.record(json!({"handle": self.handle, "status": status.to_string()}));
        Ok(())
    }
}

impl Servers {
    /// Starts `tesserax serve` on a free local port.
    fn start(&mut self, flags: &[String], sink: &mut dyn Sink) -> Result<(), Error> {
        let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
        let local = ["serve", "--host", "127.0.0.1", "--port", &port.to_string()];
        let child = Command::new(std::env::current_exe()?)
            .args(flags)
            .args(local)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .spawn()?;
        let handle = format!("serve-{}", child.id());
        let url = format!("http://127.0.0.1:{port}");
        self.running.insert(handle.clone(), child);
        sink.record(json!({"handle": handle, "url": format!("{url}/v1")}));
        sink.note(&format!("the model is ready when {url}/health answers 200"));
        Ok(())
    }
}

impl Drop for Servers {
    fn drop(&mut self) {
        for child in self.running.values_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Answers JSON-RPC lines until stdin ends.
pub fn run(origin: Option<Origin>, rest: &[String]) -> Result<(), Stop> {
    if origin.is_some() || !rest.is_empty() {
        return Err(Stop::Usage);
    }
    let mut servers = Servers::default();
    let mut out = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|e| Stop::Failed(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(reply) = answer(&mut servers, &line) {
            let wrote = writeln!(out, "{reply}").and_then(|()| out.flush());
            wrote.map_err(|e| Stop::Failed(e.to_string()))?;
        }
    }
    Ok(())
}

/// The reply to one message. Notifications get none.
pub fn answer(servers: &mut Servers, line: &str) -> Option<Value> {
    let message: Value = match serde_json::from_str(line) {
        Ok(message) => message,
        Err(e) => return Some(failure(Value::Null, PARSE, &e.to_string())),
    };
    let id = message.get("id")?.clone();
    let params = message.get("params").cloned().unwrap_or_else(|| json!({}));
    let result = match message["method"].as_str().unwrap_or_default() {
        "initialize" => Ok(initialize(&params)),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tools()})),
        "tools/call" => call(servers, &params),
        method => Err((METHOD, format!("unknown method {method}"))),
    };
    Some(match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, text)) => failure(id, code, &text),
    })
}

fn failure(id: Value, code: i64, text: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": text}})
}

fn initialize(params: &Value) -> Value {
    let asked = params["protocolVersion"].as_str().unwrap_or_default();
    let version = VERSIONS.into_iter().find(|v| *v == asked);
    json!({
        "protocolVersion": version.unwrap_or(VERSIONS[0]),
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "tesserax", "version": env!("CARGO_PKG_VERSION")},
    })
}

/// Entries a single call can run. Chat needs turns.
fn entries() -> impl Iterator<Item = &'static Entry> {
    ENTRIES.iter().filter(|e| e.name != Chat::NAME)
}

/// Every entry, with the model arguments, then `stop`.
pub fn tools() -> Vec<Value> {
    let names: Vec<&str> = CATALOG.iter().map(|(name, _)| *name).collect();
    let recipe = format!(
        "A recipe: a catalog name ({}), a directory or git+<url>#<commit>",
        names.join(", ")
    );
    let about = [
        recipe,
        "A model directory or hf:<owner>/<name>[@<revision>]".into(),
    ];
    let mut out = Vec::new();
    for e in entries() {
        let mut schema = (e.schema)();
        for ((key, _), about) in ORIGIN.iter().zip(&about) {
            schema["properties"][key] = json!({"type": "string", "description": about});
        }
        out.push(json!({"name": e.name, "description": e.about, "inputSchema": schema}));
    }
    out.push(
        json!({"name": Halt::NAME, "description": Halt::ABOUT, "inputSchema": Halt::schema()}),
    );
    out
}

fn call(servers: &mut Servers, params: &Value) -> Result<Value, (i64, String)> {
    let name = params["name"].as_str().unwrap_or_default();
    let mut args = match params.get("arguments") {
        Some(Value::Object(args)) => args.clone(),
        _ => Map::new(),
    };
    let mut flags = Vec::new();
    for (key, flag) in ORIGIN {
        if let Some(value) = args.remove(key) {
            flags.extend([flag.to_string(), value.as_str().unwrap_or_default().into()]);
        }
    }
    let args = Value::Object(args);
    let mut out = Collect::default();
    let ran = if name == Halt::NAME {
        Halt::from_json(&args)
            .map_err(|e| Stop::Failed(e.0))
            .and_then(|halt| {
                halt.run(servers, &mut out)
                    .map_err(|e| Stop::Failed(e.to_string()))
            })
    } else {
        let unknown = || (PARAMS, format!("unknown tool {name}"));
        let entry = entries().find(|e| e.name == name).ok_or_else(unknown)?;
        run_entry(entry, servers, &flags, &args, &mut out)
    };
    Ok(match ran {
        Ok(()) => result(&out),
        Err(stop) => json!({"content": [text(&describe(&stop))], "isError": true}),
    })
}

fn run_entry(
    entry: &Entry,
    servers: &mut Servers,
    flags: &[String],
    args: &Value,
    out: &mut Collect,
) -> Result<(), Stop> {
    let words = (entry.tool_args)(args).map_err(|e| Stop::Failed(e.0))?;
    let (origin, _) = split(flags)?;
    if entry.stays {
        return servers
            .start(flags, out)
            .map_err(|e| Stop::Failed(e.to_string()));
    }
    (entry.call)(origin, &words, out)
}

fn text(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// Output as a tool result, with PNG files inline.
fn result(out: &Collect) -> Value {
    let value = out.value();
    let mut content = vec![text(&value.to_string())];
    for file in out
        .files
        .iter()
        .filter(|f| f.extension().is_some_and(|e| e == "png"))
    {
        if let Ok(bytes) = std::fs::read(file) {
            let data = tessera_serve::base64(&bytes);
            content.push(json!({"type": "image", "data": data, "mimeType": "image/png"}));
        }
    }
    json!({"content": content, "structuredContent": value})
}
