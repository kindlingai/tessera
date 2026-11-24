//! A bare prompt: answered, or turned into a command.

use super::{Talk, ask, flags, text};
use crate::source::{self, Source, Stop};
use serde_json::{Map, Value, json};
use std::io::{BufRead, IsTerminal};
use tessera_bundle::Message;
use tessera_recipe::Recipe;
use tessera_tasks::api::Format;

/// A tesserax command the model may pick.
pub struct Tool {
    /// The verb words, as typed after the model flags.
    pub words: &'static str,
    /// The tool's kind of model. `None` is any kind.
    pub kind: Option<&'static str>,
    pub about: &'static str,
    /// The argument that holds the request's text.
    pub text: Option<&'static str>,
    /// The flag that sets how many to make.
    pub count: Option<&'static str>,
    /// A request it answers, and that request's text.
    pub example: (&'static str, &'static str),
}

/// The commands on offer, when the catalog holds their kind.
pub const TOOLS: [Tool; 5] = [
    Tool {
        words: "serve",
        kind: Some("generate"),
        about: "start or run a text model as an OpenAI-compatible server",
        text: None,
        count: None,
        example: ("start {model} as a server", ""),
    },
    Tool {
        words: "fetch",
        kind: None,
        about: "download a model and print where it is",
        text: None,
        count: None,
        example: ("download the {kind} model", ""),
    },
    Tool {
        words: "task image",
        kind: Some("image"),
        about: "make PNG pictures from a prompt",
        text: Some("prompt"),
        count: Some("--n"),
        example: ("draw two pictures of a red bicycle", "a red bicycle"),
    },
    Tool {
        words: "task embed",
        kind: Some("embed"),
        about: "print an embedding vector for some text",
        text: Some("text"),
        count: None,
        example: ("embed the text 'tea and toast'", "tea and toast"),
    },
    Tool {
        words: "task tag",
        kind: Some("tag"),
        about: "print part-of-speech tags for some text",
        text: Some("text"),
        count: None,
        example: ("tag the words in 'dogs bark'", "dogs bark"),
    },
];

/// A request for an outside model, and its words.
const HUB_EXAMPLE: (&str, &str) = ("run llama 3.2 1b from Hugging Face", "llama 3.2 1b");

/// A recipe's kind: its serve task, or text generation.
fn kind(r: &Recipe) -> &str {
    r.serve
        .get("task")
        .and_then(|v| v.as_str())
        .unwrap_or("generate")
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

/// The tools with a catalog model of their kind.
pub fn tools(catalog: &[Recipe]) -> Vec<&'static Tool> {
    let mut out = Vec::new();
    for t in &TOOLS {
        if !names(catalog, t.kind).is_empty() {
            out.push(t);
        }
    }
    out
}

/// JSON for a reply, or a tool and its arguments.
pub fn schema(catalog: &[Recipe], tools: &[&Tool]) -> Value {
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

/// Tool `t`, its model under `key`, and other arguments.
fn branch(t: &Tool, key: &str, model: Value) -> Value {
    let mut props = Map::new();
    props.insert("tool".into(), json!({ "enum": [t.words] }));
    props.insert(key.into(), model);
    let mut required = vec!["tool", key];
    if let Some(text) = t.text {
        let text_schema = json!({ "type": "string", "minLength": 1, "maxLength": 500 });
        props.insert(text.into(), text_schema);
        required.push(text);
    }
    if t.count.is_some() {
        let n = json!({ "type": "integer", "minimum": 1, "maximum": 8 });
        props.insert("n".into(), n);
    }
    json!({
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": false,
    })
}

/// `t`'s example request, and the pick it makes.
fn example(catalog: &[Recipe], t: &Tool) -> (String, Value) {
    let mut models: Vec<&Recipe> = Vec::new();
    for r in catalog {
        if t.kind.is_none_or(|k| k == kind(r)) {
            models.push(r);
        }
    }
    // A request by kind names the last model.
    let by_kind = t.example.0.contains("{kind}");
    let recipe = if by_kind {
        models.last()
    } else {
        models.first()
    };
    let Some(recipe) = recipe else {
        return (t.example.0.into(), json!({ "tool": "reply" }));
    };
    let model = recipe.name.clone();
    let mut pick = Map::new();
    pick.insert("tool".into(), t.words.into());
    pick.insert("model".into(), model.clone().into());
    if let Some(text) = t.text {
        pick.insert(text.into(), t.example.1.into());
    }
    if t.count.is_some() {
        pick.insert("n".into(), 2.into());
    }
    let asked = t.example.0.replace("{kind}", kind(recipe));
    (asked.replace("{model}", &model), Value::Object(pick))
}

/// The routing prompt for `request`.
pub fn prompt(catalog: &[Recipe], tools: &[&Tool], request: &str) -> String {
    let mut out = String::from("Turn the request into one tesserax command, as JSON.\nTools:\n");
    out += "- reply: answer in text. Use it for questions, chat and anything else.\n";
    for t in tools {
        let args = [Some("model or hub"), t.text, t.count.map(|_| "n")];
        let args: Vec<&str> = args.into_iter().flatten().collect();
        out += &format!("- {} ({}): {}\n", t.words, args.join(", "), t.about);
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
        let pick = json!({ "tool": serve.words, "hub": HUB_EXAMPLE.1 });
        out += &format!("Request: {}\n{pick}\n", HUB_EXAMPLE.0);
    }
    out + &format!("Request: {request}\n")
}

/// The catalog model that words such as "image model" name.
fn by_kind<'a>(catalog: &'a [Recipe], words: &str) -> Option<&'a str> {
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
    let words = field("tool").ok_or_else(bad)?;
    if words == "reply" {
        return Ok(None);
    }
    let tool = TOOLS.iter().find(|t| t.words == words).ok_or_else(bad)?;
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
    out.extend(words.split(' ').map(String::from));
    let n = pick.get("n").and_then(Value::as_u64);
    if let (Some(flag), Some(n)) = (tool.count, n) {
        out.extend([flag.to_string(), n.to_string()]);
    }
    if let Some(text) = tool.text {
        out.push(field(text).ok_or_else(bad)?.to_string());
    }
    Ok(Some(out))
}

/// Answers `words`, or runs the command the default model picks.
pub fn run(words: &[String]) -> Result<(), Stop> {
    let (yes, words) = match words {
        [flag, rest @ ..] if flag == "--yes" => (true, rest),
        _ => (false, words),
    };
    let (max_tokens, words) = flags(words)?;
    let request = text(words)?;
    let request = request.trim();
    if request.is_empty() {
        return Err(Stop::Usage);
    }
    let source = Source::open_or_default(None)?;
    let guided = cfg!(feature = "guided");
    let switches: &[String] = if guided { &["--guided".into()] } else { &[] };
    let mut options = source.options(switches)?;
    options.slots = 1;
    let talk = Talk::open(options, max_tokens);
    let token = source.token.as_deref();
    let find = |words: &str| find(words, token);
    let pick = match guided {
        true => argv(&pick(&talk, request)?, &source::catalog(), find)?,
        false => None,
    };
    match pick {
        None => ask::run(talk, &[request.to_string()]),
        Some(args) => launch(&args, yes),
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
fn pick(talk: &Talk, request: &str) -> Result<Value, Stop> {
    let catalog = source::catalog();
    let tools = tools(&catalog);
    let user = Message::text("user", prompt(&catalog, &tools, request));
    let format = Format::Schema(schema(&catalog, &tools));
    let json = talk.answer(&talk.render(&[user])?, Some(format), false)?;
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
