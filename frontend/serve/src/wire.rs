//! Request bodies, their checks, and the OpenAI error object.

use crate::Model;
use crate::http::Response;
use sensai_bundle::Message;
use sensai_tasks::api::Format;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

/// Why a request was refused, in OpenAI's error format.
#[derive(Debug)]
pub(crate) struct Refusal {
    pub(crate) status: u16,
    pub(crate) message: String,
    pub(crate) param: Option<String>,
}

impl Refusal {
    pub(crate) fn body(&self) -> Value {
        let kind = match self.status {
            404 => "NotFoundError",
            503 => "ServiceUnavailableError",
            500 => "InternalServerError",
            _ => "BadRequestError",
        };
        json!({"error": {
            "message": self.message,
            "type": kind,
            "param": self.param,
            "code": self.status,
        }})
    }
}

impl From<Refusal> for Response {
    fn from(refused: Refusal) -> Response {
        Response::json(refused.status, &refused.body())
    }
}

pub(crate) fn refuse(status: u16, message: impl Into<String>, param: Option<&str>) -> Refusal {
    Refusal {
        status,
        message: message.into(),
        param: param.map(str::to_string),
    }
}

pub(crate) fn bad(message: impl Into<String>, param: &str) -> Refusal {
    refuse(400, message, Some(param))
}

/// Seconds since 1970, for `created`.
pub(crate) fn now() -> u64 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH);
    since.map_or(0, |d| d.as_secs())
}

/// Whether `value` changes nothing. `None`: an unknown field.
fn neutral(key: &str, value: &Value) -> Option<bool> {
    let numbers: &[f64] = match key {
        "seed" | "user" => return Some(true),
        "temperature" | "presence_penalty" | "frequency_penalty" | "min_p" => &[0.0],
        "top_p" | "n" | "best_of" | "repetition_penalty" => &[1.0],
        "top_k" => &[-1.0, 0.0],
        "logprobs" | "echo" => return Some(value.is_null() || *value == json!(false)),
        "tools" | "functions" => return Some(value.is_null() || *value == json!([])),
        "tool_choice" => {
            return Some(value.is_null() || ["none", "auto"].map(Value::from).contains(value));
        }
        "encoding_format" => return Some(value.is_null() || *value == json!("float")),
        "dimensions" => return Some(value.is_null()),
        _ => return None,
    };
    Some(value.is_null() || value.as_f64().is_some_and(|v| numbers.contains(&v)))
}

/// Fields a body names that every route ignores.
pub(crate) type Bag = Map<String, Value>;

/// A body that bags the fields it does not read.
pub(crate) trait Bagged {
    /// The top bag, and the paths of nested unknown fields.
    fn bags(&self) -> (&Bag, Vec<String>);
}

/// The keys of `bag`, under `path`.
fn paths(path: &str, bag: &Bag) -> impl Iterator<Item = String> {
    bag.keys().map(move |k| format!("{path}.{k}"))
}

/// The body. Unknown fields warn, or refuse if strict.
pub(crate) fn parse<T: DeserializeOwned + Bagged>(
    model: &Model,
    body: &[u8],
) -> Result<T, Refusal> {
    let body: Map<String, Value> =
        serde_json::from_slice(body).map_err(|e| refuse(400, e.to_string(), None))?;
    match body.get("model") {
        None | Some(Value::Null) => {}
        Some(Value::String(m)) if *m == model.info.name => {}
        Some(m) => {
            let m = m.as_str().map_or_else(|| m.to_string(), str::to_string);
            let message = format!("The model `{m}` does not exist.");
            return Err(refuse(404, message, Some("model")));
        }
    }
    let parsed: T = serde_json::from_value(Value::Object(body))
        .map_err(|e| refuse(400, e.to_string(), None))?;
    let (top, nested) = parsed.bags();
    let mut unknown = Vec::new();
    for (key, value) in top {
        match neutral(key, value) {
            _ if key == "model" => {}
            Some(true) => {}
            Some(false) => {
                return Err(bad(
                    format!("`{key}`: only its neutral value is supported"),
                    key,
                ));
            }
            None => unknown.push(key.clone()),
        }
    }
    unknown.extend(nested);
    match (unknown.first(), model.strict) {
        (None, _) => {}
        (Some(first), true) => return Err(bad(format!("`{first}` is not supported"), first)),
        (Some(_), false) => eprintln!("tessera-serve: ignored fields: {}", unknown.join(", ")),
    }
    Ok(parsed)
}

/// One string, or a list of strings.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Strings {
    One(String),
    Many(Vec<String>),
}

impl Strings {
    pub(crate) fn list(self) -> Vec<String> {
        match self {
            Strings::One(s) => vec![s],
            Strings::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct StreamOptions {
    #[serde(default)]
    pub(crate) include_usage: bool,
    #[serde(flatten)]
    extra: Bag,
}

/// The required shape of an answer.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchema },
}

impl ResponseFormat {
    /// The answer's shape. `None` allows any JSON.
    pub(crate) fn format(self) -> Option<Format> {
        match self {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(Format::Object),
            ResponseFormat::JsonSchema { json_schema } => {
                Some(Format::Schema(json_schema.schema.unwrap_or(json!({}))))
            }
        }
    }
}

/// A named schema. Answers always follow it, `strict` or not.
#[derive(Deserialize)]
pub(crate) struct JsonSchema {
    pub(crate) schema: Option<Value>,
    #[serde(rename = "name")]
    _name: Option<String>,
    #[serde(rename = "description")]
    _description: Option<String>,
    #[serde(rename = "strict")]
    _strict: Option<bool>,
    #[serde(flatten)]
    extra: Bag,
}

/// Fields completions and chat share.
#[derive(Deserialize)]
pub(crate) struct Common {
    pub(crate) max_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) stream: bool,
    pub(crate) stream_options: Option<StreamOptions>,
    pub(crate) stop: Option<Strings>,
    pub(crate) response_format: Option<ResponseFormat>,
}

impl Common {
    fn nested(&self) -> Vec<String> {
        let options = self.stream_options.as_ref().map(|o| &o.extra);
        let mut out: Vec<String> =
            options.map_or_else(Vec::new, |bag| paths("stream_options", bag).collect());
        if let Some(ResponseFormat::JsonSchema { json_schema }) = &self.response_format {
            out.extend(paths("response_format.json_schema", &json_schema.extra));
        }
        out
    }
}

/// A prompt as ids, a text or texts.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Prompt {
    Ids(Vec<u32>),
    Text(Strings),
}

#[derive(Deserialize)]
pub(crate) struct Completion {
    pub(crate) prompt: Prompt,
    #[serde(flatten)]
    pub(crate) common: Common,
    #[serde(flatten)]
    extra: Bag,
}

impl Bagged for Completion {
    fn bags(&self) -> (&Bag, Vec<String>) {
        (&self.extra, self.common.nested())
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Part {
    Text {
        text: String,
        #[serde(flatten)]
        extra: Bag,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Deserialize)]
pub(crate) struct Said {
    role: String,
    content: Content,
    /// Earlier thinking, unused by templates for now.
    #[serde(rename = "reasoning", alias = "reasoning_content")]
    _reasoning: Option<String>,
    #[serde(flatten)]
    extra: Bag,
}

impl From<Said> for Message {
    fn from(said: Said) -> Message {
        let content = match said.content {
            Content::Text(text) => text,
            Content::Parts(parts) => parts
                .into_iter()
                .map(|Part::Text { text, .. }| text)
                .collect(),
        };
        Message {
            role: said.role,
            content,
        }
    }
}

#[derive(Deserialize, Default)]
pub(crate) struct Kwargs {
    pub(crate) enable_thinking: Option<bool>,
    #[serde(flatten)]
    extra: Bag,
}

/// Messages and how to lay them out.
#[derive(Deserialize)]
pub(crate) struct Messages {
    pub(crate) messages: Vec<Said>,
    #[serde(default)]
    pub(crate) chat_template_kwargs: Kwargs,
}

impl Messages {
    fn nested(&self) -> Vec<String> {
        let mut out: Vec<String> =
            paths("chat_template_kwargs", &self.chat_template_kwargs.extra).collect();
        for (i, said) in self.messages.iter().enumerate() {
            out.extend(paths(&format!("messages[{i}]"), &said.extra));
            let Content::Parts(parts) = &said.content else {
                continue;
            };
            for (j, Part::Text { extra, .. }) in parts.iter().enumerate() {
                out.extend(paths(&format!("messages[{i}].content[{j}]"), extra));
            }
        }
        out
    }
}

#[derive(Deserialize)]
pub(crate) struct Chat {
    #[serde(flatten)]
    pub(crate) messages: Messages,
    pub(crate) max_completion_tokens: Option<usize>,
    #[serde(flatten)]
    pub(crate) common: Common,
    #[serde(flatten)]
    extra: Bag,
}

impl Bagged for Chat {
    fn bags(&self) -> (&Bag, Vec<String>) {
        let mut nested = self.common.nested();
        nested.extend(self.messages.nested());
        (&self.extra, nested)
    }
}

/// The `/tokenize` body: a prompt or messages.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Tokenize {
    Prompt {
        prompt: String,
        #[serde(flatten)]
        extra: Bag,
    },
    Chat {
        #[serde(flatten)]
        messages: Messages,
        #[serde(flatten)]
        extra: Bag,
    },
}

impl Bagged for Tokenize {
    fn bags(&self) -> (&Bag, Vec<String>) {
        match self {
            Tokenize::Prompt { extra, .. } => (extra, Vec::new()),
            Tokenize::Chat { messages, extra } => (extra, messages.nested()),
        }
    }
}

/// A body of one read field, and a bag.
macro_rules! single {
    ($name:ident, $field:ident: $ty:ty) => {
        #[derive(Deserialize)]
        pub(crate) struct $name {
            pub(crate) $field: $ty,
            #[serde(flatten)]
            extra: Bag,
        }

        impl Bagged for $name {
            fn bags(&self) -> (&Bag, Vec<String>) {
                (&self.extra, Vec::new())
            }
        }
    };
}

/// The `/v1/images/generations` body. `seed` is ours.
#[derive(Deserialize)]
pub(crate) struct Images {
    pub(crate) prompt: String,
    pub(crate) n: Option<usize>,
    pub(crate) size: Option<String>,
    pub(crate) response_format: Option<String>,
    pub(crate) seed: Option<u32>,
    #[serde(flatten)]
    extra: Bag,
}

impl Bagged for Images {
    fn bags(&self) -> (&Bag, Vec<String>) {
        (&self.extra, Vec::new())
    }
}

single!(Detokenize, tokens: Vec<u32>);
single!(Embeddings, input: Strings);
