//! A bare prompt: answered, or turned into a command.

use super::{ENTRIES, Entry, Print, load, model, stop};
use crate::source::{self, Source, Stop, kind};
use serde_json::{Map, Value, json};
use std::io::{BufRead, IsTerminal};
use tessera_bundle::Message;
use tessera_recipe::Recipe;
use tessera_serve::Model;
use tessera_tasks::api::Format;
use tessera_tasks::task::{Ask, Run, Task};

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

/// Answers `words`, or runs the command the default model picks.
pub fn run(words: &[String]) -> Result<(), Stop> {
    let route = Route::parse(words, &|_| None).map_err(|_| Stop::Usage)?;
    let request = route.request.trim();
    let source = Source::open_or_kind(None, None)?;
    let guided = cfg!(feature = "guided");
    let switches: &[String] = if guided { &["--guided".into()] } else { &[] };
    let mut options = source.options(switches)?;
    options.slots = 1;
    let slot = load(options);
    let model = model(&slot);
    let token = source.token.as_deref();
    let find = |words: &str| find(words, token);
    let pick = match guided {
        true => argv(
            &pick(model, route.max_tokens, request)?,
            &source::catalog(),
            find,
        )?,
        false => None,
    };
    let Some(args) = pick else {
        let ask = Ask {
            max_tokens: route.max_tokens,
            prompt: request.into(),
        };
        return ask.run(model, &mut Print::default()).map_err(stop);
    };
    launch(&args, route.yes)
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

/// Runs tesserax with `args`, after a yes on a terminal.
fn launch(args: &[String], yes: bool) -> Result<(), Stop> {
    let quoted = |a: &String| match a.contains(' ') {
        true => format!("{a:?}"),
        false => a.clone(),
    };
    let shown: Vec<String> = args.iter().map(quoted).collect();
    eprintln!("tesserax: the model picked `tesserax {}`", shown.join(" "));
    let stdin = std::io::stdin();
    if !yes && stdin.is_terminal() {
        eprint!("run it? [y/N] ");
        let mut line = String::new();
        stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| Stop::Failed(e.to_string()))?;
        if !matches!(line.trim(), "y" | "Y" | "yes") {
            return Ok(());
        }
    }
    let exe = std::env::current_exe().map_err(|e| Stop::Failed(e.to_string()))?;
    let mut command = std::process::Command::new(exe);
    command.args(args);
    replace(command)
}

/// Replaces this process with `command` to free the routing model.
#[cfg(unix)]
fn replace(mut command: std::process::Command) -> Result<(), Stop> {
    use std::os::unix::process::CommandExt;
    Err(Stop::Failed(command.exec().to_string()))
}

#[cfg(not(unix))]
fn replace(mut command: std::process::Command) -> Result<(), Stop> {
    let status = command.status().map_err(|e| Stop::Failed(e.to_string()))?;
    match status.success() {
        true => Ok(()),
        false => Err(Stop::Failed(format!("the picked command failed: {status}"))),
    }
}
