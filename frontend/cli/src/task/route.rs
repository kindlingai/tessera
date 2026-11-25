//! A prompt or chat turn: a reply or a task.

use super::{ENTRIES, Entry, Print, describe, load, model};
use crate::source::{self, Source, Stop, kind};
use serde_json::{Map, Value, json};
use std::io::{BufRead, IsTerminal};
use tessera_bundle::{FunctionCall, Message, ToolCall};
use tessera_recipe::Recipe;
use tessera_serve::{Model, Slot};
use tessera_tasks::api::Format;
use tessera_tasks::task::{Chat, Collect, Sink, Task};

/// Whether this build can hold answers to a JSON schema.
const GUIDED: bool = cfg!(feature = "guided");

/// The most characters of a task's result the model reads.
const RESULT: usize = 2000;

/// The turn that asks for an answer from a result.
const AFTER: &str = "Tell me the result of that task in a sentence or two.";

/// A request for an outside model, and its words.
const HUB_EXAMPLE: (&str, &str) = ("run llama 3.2 1b from Hugging Face", "llama 3.2 1b");

/// Answer a prompt, or run the command the model picks.
#[derive(Task)]
struct Route {
    /// Run the picked command without asking.
    yes: bool,
    /// The most tokens in a reply.
    #[arg(value = "<n>")]
    max_tokens: Option<usize>,
    /// The request, from stdin if none.
    #[arg(positional)]
    request: String,
}

/// Catalog names of `kind`, or every name.
fn names(catalog: &[Recipe], of: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    for r in catalog {
        if of.is_none_or(|k| k == kind(r)) {
            out.push(r.name.clone());
        }
    }
    out
}

/// Entries with an example and a catalog model to run.
pub fn tools(catalog: &[Recipe]) -> Vec<&'static Entry> {
    let mut out = Vec::new();
    for e in &ENTRIES {
        if e.example.is_some() && !names(catalog, e.kind).is_empty() {
            out.push(e);
        }
    }
    out
}

/// JSON for a reply, or a tool and its arguments.
pub fn schema(catalog: &[Recipe], tools: &[&Entry]) -> Value {
    let only = |word: &str| json!({ "enum": [word] });
    let mut branches = vec![json!({
        "type": "object",
        "properties": { "tool": only("reply") },
        "required": ["tool"],
        "additionalProperties": false,
    })];
    for t in tools {
        let words = json!({ "type": "string", "minLength": 2, "maxLength": 60 });
        let models = json!({ "enum": names(catalog, t.kind) });
        for (key, value) in [("model", models), ("hub", words)] {
            branches.push(branch(t, key, value));
        }
    }
    json!({ "anyOf": branches })
}

/// Tool `t`, its model under `key`, and its arguments.
fn branch(t: &Entry, key: &str, model: Value) -> Value {
    let mut schema = (t.schema)();
    let mut props = Map::new();
    props.insert("tool".into(), json!({ "enum": [t.name] }));
    props.insert(key.into(), model);
    if let Value::Object(args) = schema["properties"].take() {
        props.extend(args);
    }
    let mut required = vec![json!("tool"), json!(key)];
    if let Value::Array(args) = schema["required"].take() {
        required.extend(args);
    }
    schema["properties"] = Value::Object(props);
    schema["required"] = Value::Array(required);
    schema
}

/// `t`'s example request, and the pick it makes.
fn example(catalog: &[Recipe], t: &Entry) -> (String, Value) {
    let mut models: Vec<&Recipe> = Vec::new();
    for r in catalog {
        if t.kind.is_none_or(|k| k == kind(r)) {
            models.push(r);
        }
    }
    let asked = t.example.unwrap_or_default();
    // A request by kind names the last model.
    let by_kind = asked.contains("{kind}");
    let recipe = if by_kind {
        models.last()
    } else {
        models.first()
    };
    let Some(recipe) = recipe else {
        return (asked.into(), json!({ "tool": "reply" }));
    };
    let model = recipe.name.clone();
    let mut pick = Map::new();
    pick.insert("tool".into(), t.name.into());
    pick.insert("model".into(), model.clone().into());
    if let Value::Object(args) = (t.example_args)() {
        pick.extend(args);
    }
    let asked = asked.replace("{kind}", kind(recipe));
    (asked.replace("{model}", &model), Value::Object(pick))
}

/// The routing prompt for `request`.
pub fn prompt(catalog: &[Recipe], tools: &[&Entry], request: &str) -> String {
    let mut out = String::from("Turn the request into one tesserax command, as JSON.\nTools:\n");
    out += "- reply: answer in text. Use it for questions, chat and anything else.\n";
    for t in tools {
        let mut args = vec!["model or hub".to_string()];
        if let Value::Object(props) = &(t.schema)()["properties"] {
            args.extend(props.keys().cloned());
        }
        out += &format!("- {} ({}): {}\n", t.name, args.join(", "), t.about);
    }
    out += "Models:\n";
    for r in catalog {
        let about = r.about.as_deref().unwrap_or_default();
        out += &format!("- {} ({}): {about}\n", r.name, kind(r));
    }
    out += "model is a name from this list. For any other model, hub holds the words\n";
    out += "the request uses for it, and tesserax finds it on Hugging Face.\n";
    out += "A request for a kind of model, such as an image model, takes model.\n";
    out += "Request: what is the capital of France?\n{\"tool\":\"reply\"}\n";
    for t in tools {
        let (asked, pick) = example(catalog, t);
        out += &format!("Request: {asked}\n{pick}\n");
    }
    if let Some(serve) = tools.iter().find(|t| t.kind == Some("generate")) {
        let pick = json!({ "tool": serve.name, "hub": HUB_EXAMPLE.1 });
        out += &format!("Request: {}\n{pick}\n", HUB_EXAMPLE.0);
    }
    out + &format!("Request: {request}\n")
}

/// The catalog model that words such as "image model" name.
fn by_kind<'a>(catalog: &'a [Recipe], words: &str) -> Option<&'a str> {
    if let Some(r) = catalog.iter().find(|r| r.name == words.trim()) {
        return Some(&r.name);
    }
    let mut named = Vec::new();
    for word in words.split_whitespace() {
        let word = word.to_lowercase();
        if !["a", "an", "the", "model", "models"].contains(&word.as_str()) {
            named.push(word);
        }
    }
    let [word] = named.as_slice() else {
        return None;
    };
    for r in catalog {
        let k = kind(r);
        if word.starts_with(k) || k.starts_with(word.as_str()) {
            return Some(&r.name);
        }
    }
    None
}

/// Arguments for `pick`, or `None` for a reply.
pub fn argv(
    pick: &Value,
    catalog: &[Recipe],
    find: impl Fn(&str) -> Result<String, Stop>,
) -> Result<Option<Vec<String>>, Stop> {
    let field = |k: &str| pick.get(k).and_then(Value::as_str);
    let bad = || Stop::Failed(format!("the model picked {pick}"));
    let name = field("tool").ok_or_else(bad)?;
    if name == "reply" {
        return Ok(None);
    }
    let tool = ENTRIES
        .iter()
        .find(|t| t.name == name && t.example.is_some());
    let tool = tool.ok_or_else(bad)?;
    let mut out = match (field("model"), field("hub")) {
        (Some(model), _) if catalog.iter().any(|r| r.name == model) => {
            vec!["--recipe".to_string(), model.to_string()]
        }
        (_, Some(hub)) => match by_kind(catalog, hub) {
            Some(name) => vec!["--recipe".to_string(), name.to_string()],
            // `find` turns outside words into a Hub id.
            None => vec!["--model".to_string(), format!("hf:{}", find(hub)?)],
        },
        _ => return Err(bad()),
    };
    out.extend(tool.words());
    let mut args = pick.as_object().cloned().unwrap_or_default();
    for key in ["tool", "model", "hub"] {
        args.remove(key);
    }
    let args = (tool.tool_args)(&Value::Object(args));
    out.extend(args.map_err(|e| Stop::Failed(format!("the model picked {pick}: {e}")))?);
    Ok(Some(out))
}

/// Answers `words`, or runs the task the default model picks.
pub fn run(words: &[String]) -> Result<(), Stop> {
    let route = Route::parse(words, &|_| None).map_err(|_| Stop::Usage)?;
    let mut session = Session::open(route.max_tokens, route.yes)?;
    session.turn(route.request.trim())
}

/// `task chat` without a model flag. Turns may run tasks.
pub fn chat(args: &[String]) -> Result<(), Stop> {
    let chat = Chat::parse(args, &|_| None).map_err(|_| Stop::Usage)?;
    let mut session = Session::open(chat.max_tokens, false)?;
    let mut print = Print::default();
    while let Some(line) = print.turn() {
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        let mark = session.history.len();
        if let Err(e) = session.turn(line) {
            session.history.truncate(mark);
            print.note(&describe(&e));
        }
    }
    Ok(())
}

/// The default model, the chat so far, and settings.
struct Session {
    slot: Slot,
    max_tokens: Option<usize>,
    yes: bool,
    token: Option<String>,
    catalog: Vec<Recipe>,
    history: Vec<Message>,
}

impl Session {
    fn open(max_tokens: Option<usize>, yes: bool) -> Result<Self, Stop> {
        let source = Source::open_or_kind(None, None, Print::default().progress())?;
        let switches: &[String] = if GUIDED { &["--guided".into()] } else { &[] };
        let mut options = source.options(switches)?;
        options.slots = 1;
        Ok(Session {
            slot: load(options),
            max_tokens,
            yes,
            token: source.token.clone(),
            catalog: source::catalog(),
            history: Vec::new(),
        })
    }

    /// One request: a reply, or a task, then a reply.
    fn turn(&mut self, request: &str) -> Result<(), Stop> {
        let pick = match GUIDED {
            true => pick(model(&self.slot), self.max_tokens, request)?,
            false => json!({ "tool": "reply" }),
        };
        let token = self.token.as_deref();
        let args = argv(&pick, &self.catalog, |words| find(words, token))?;
        self.history.push(Message::text("user", request));
        if let Some(args) = args
            && !self.call(&pick, &args)?
        {
            self.history.pop();
            return Ok(());
        }
        self.reply()
    }

    /// Runs the picked task. `false` when the person declines.
    fn call(&mut self, pick: &Value, args: &[String]) -> Result<bool, Stop> {
        let name = pick.get("tool").and_then(Value::as_str).unwrap_or_default();
        let entry = ENTRIES.iter().find(|e| e.name == name);
        let entry = entry.ok_or_else(|| Stop::Failed(format!("the model picked {pick}")))?;
        if !confirm(args, self.yes)? {
            return Ok(false);
        }
        if entry.stays {
            return replace(args).map(|()| true);
        }
        let (origin, words) = source::split(args)?;
        let rest = words.get(entry.words().len()..).unwrap_or_default();
        let mut out = Both::default();
        let result = match (entry.call)(origin, rest, &mut out) {
            Ok(()) => out.collect.value(),
            Err(e) => json!({ "error": describe(&e) }),
        };
        let mut arguments = pick.clone();
        if let Value::Object(fields) = &mut arguments {
            fields.remove("tool");
        }
        // The call and its result join the chat.
        let id = format!("call_{}", self.history.len());
        let call = ToolCall {
            id: Some(id.clone()),
            kind: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments,
            },
        };
        self.history.push(Message {
            role: "assistant".into(),
            tool_calls: Some(vec![call]),
            ..Message::default()
        });
        self.history.push(Message {
            role: "tool".into(),
            content: Some(short(&result)),
            tool_call_id: Some(id),
            name: Some(name.into()),
            ..Message::default()
        });
        self.history.push(Message::text("user", AFTER));
        Ok(true)
    }

    /// Prints the model's answer to the chat so far.
    fn reply(&mut self) -> Result<(), Stop> {
        let model = model(&self.slot);
        let failed = |e: tessera_tasks::Error| Stop::Failed(e.to_string());
        let text = model.render(&self.history).map_err(failed)?;
        let mut print = Print::default();
        let answer = model.answer(&text, self.max_tokens, None, &mut |t| print.text(t));
        let answer = answer.map_err(failed)?;
        self.history.push(Message::text("assistant", answer.trim()));
        Ok(())
    }
}

/// Output for the person, also kept for the model.
#[derive(Default)]
struct Both {
    print: Print,
    collect: Collect,
}

impl Sink for Both {
    fn text(&mut self, piece: &str) {
        self.print.text(piece);
        self.collect.text(piece);
    }

    fn record(&mut self, record: Value) {
        self.print.record(record.clone());
        self.collect.record(record);
    }

    fn file(&mut self, path: &std::path::Path) {
        self.print.file(path);
        self.collect.file(path);
    }

    fn note(&mut self, text: &str) {
        self.print.note(text);
        self.collect.note(text);
    }
}

/// The most items of a list the model reads.
const ITEMS: usize = 8;

/// `value` with long lists cut to [`ITEMS`] and a count.
pub fn brief(value: &Value) -> Value {
    match value {
        Value::Array(items) if items.len() > ITEMS => {
            let mut out: Vec<Value> = items[..ITEMS].iter().map(brief).collect();
            let rest = items.len() - ITEMS;
            out.push(json!(format!("{rest} more, {} in all", items.len())));
            Value::Array(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(brief).collect()),
        Value::Object(fields) => {
            let fields = fields.iter().map(|(k, v)| (k.clone(), brief(v)));
            Value::Object(fields.collect())
        }
        other => other.clone(),
    }
}

/// The result text the model reads, cut at [`RESULT`].
fn short(result: &Value) -> String {
    let text = brief(result).to_string();
    match text.char_indices().nth(RESULT) {
        Some((at, _)) => format!("{}…", &text[..at]),
        None => text,
    }
}

/// The Hub id for `words`: checked as given, or searched.
fn find(words: &str, token: Option<&str>) -> Result<String, Stop> {
    let words = words.trim().trim_start_matches("hf:");
    if words.contains('/') && !words.contains(' ') {
        tessera_recipe::check_hf(words, token)?;
        return Ok(words.into());
    }
    let found = tessera_recipe::search_hf(words, token)?;
    let none = || {
        Stop::Failed(format!(
            "the Hugging Face search for {words:?} found nothing"
        ))
    };
    let id = found.ok_or_else(none)?;
    eprintln!("tesserax: {words:?} is {id} on Hugging Face");
    Ok(id)
}

/// The default model's pick. A bad pick is a reply.
fn pick(model: &Model, max_tokens: Option<usize>, request: &str) -> Result<Value, Stop> {
    let catalog = source::catalog();
    let tools = tools(&catalog);
    let user = Message::text("user", prompt(&catalog, &tools, request));
    let format = Some(Format::Schema(schema(&catalog, &tools)));
    let failed = |e: tessera_tasks::Error| Stop::Failed(e.to_string());
    let text = model.render(&[user]).map_err(failed)?;
    let json = model.answer(&text, max_tokens, format, &mut |_| {});
    let json = json.map_err(failed)?;
    Ok(serde_json::from_str(json.trim()).unwrap_or_else(|_| json!({ "tool": "reply" })))
}

/// Shows `args`, then asks on a terminal unless `yes`.
fn confirm(args: &[String], yes: bool) -> Result<bool, Stop> {
    let quoted = |a: &String| match a.contains(' ') {
        true => format!("{a:?}"),
        false => a.clone(),
    };
    let shown: Vec<String> = args.iter().map(quoted).collect();
    eprintln!("tesserax: the model picked `tesserax {}`", shown.join(" "));
    let stdin = std::io::stdin();
    if yes || !stdin.is_terminal() {
        return Ok(true);
    }
    eprint!("run it? [y/N] ");
    let mut line = String::new();
    stdin
        .lock()
        .read_line(&mut line)
        .map_err(|e| Stop::Failed(e.to_string()))?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

/// Replaces this process with tesserax `args` to free the model.
#[cfg(unix)]
fn replace(args: &[String]) -> Result<(), Stop> {
    use std::os::unix::process::CommandExt;
    let exe = std::env::current_exe().map_err(|e| Stop::Failed(e.to_string()))?;
    Err(Stop::Failed(
        std::process::Command::new(exe)
            .args(args)
            .exec()
            .to_string(),
    ))
}

#[cfg(not(unix))]
fn replace(args: &[String]) -> Result<(), Stop> {
    let exe = std::env::current_exe().map_err(|e| Stop::Failed(e.to_string()))?;
    let status = std::process::Command::new(exe).args(args).status();
    let status = status.map_err(|e| Stop::Failed(e.to_string()))?;
    match status.success() {
        true => Ok(()),
        false => Err(Stop::Failed(format!("the picked command failed: {status}"))),
    }
}
