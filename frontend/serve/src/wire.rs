//! Request bodies, their checks, and the OpenAI error object.

use crate::Model;
use crate::http::Response;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use std::time::{SystemTime, UNIX_EPOCH};
use tessera_bundle::{FunctionCall, Message, ToolCall};
use tessera_engine::{Penalties, Sampling};
use tessera_tasks::api::{Format, ToolChoice as TaskToolChoice};
use tessera_tasks::scopes::{Boost, Sequence, Thinking};

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
        "user" | "prompt_cache_key" | "safety_identifier" => return Some(true),
        "n" | "best_of" => &[1.0],
        "logprobs" | "echo" | "store" => return Some(value.is_null() || *value == json!(false)),
        "metadata" => return Some(value.is_null() || *value == json!({})),
        "service_tier" => {
            return Some(value.is_null() || ["auto", "default"].map(Value::from).contains(value));
        }
        "modalities" => return Some(value.is_null() || *value == json!(["text"])),
        "tools" | "functions" => return Some(value.is_null() || *value == json!([])),
        "tool_choice" => {
            return Some(value.is_null() || ["none", "auto"].map(Value::from).contains(value));
        }
        "encoding_format" => return Some(value.is_null() || *value == json!("float")),
        "dimensions" | "top_logprobs" => return Some(value.is_null()),
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

/// The body as one JSON object.
pub(crate) fn object(body: &[u8]) -> Result<Map<String, Value>, Refusal> {
    serde_json::from_slice(body).map_err(|e| refuse(400, e.to_string(), None))
}

/// The body, for this model only.
pub(crate) fn parse<T: DeserializeOwned + Bagged>(
    model: &Model,
    body: &[u8],
) -> Result<T, Refusal> {
    let body = object(body)?;
    match body.get("model") {
        None | Some(Value::Null) => {}
        Some(Value::String(m)) if *m == model.info.name => {}
        Some(m) => {
            let m = m.as_str().map_or_else(|| m.to_string(), str::to_string);
            let message = format!("The model `{m}` does not exist.");
            return Err(refuse(404, message, Some("model")));
        }
    }
    fields(model, body)
}

/// The body's fields. Unknown fields warn, or refuse if strict.
pub(crate) fn fields<T: DeserializeOwned + Bagged>(
    model: &Model,
    body: Map<String, Value>,
) -> Result<T, Refusal> {
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
    pub(crate) min_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) stream: bool,
    pub(crate) stream_options: Option<StreamOptions>,
    pub(crate) stop: Option<Strings>,
    pub(crate) response_format: Option<ResponseFormat>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<i64>,
    pub(crate) min_p: Option<f32>,
    pub(crate) seed: Option<i64>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) thinking_presence_penalty: Option<f32>,
    pub(crate) thinking_frequency_penalty: Option<f32>,
    pub(crate) think_end_boost: Option<Switch>,
    pub(crate) sequence_penalty: Option<Switch>,
}

/// A spec string, or `false` for off.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Switch {
    Flag(bool),
    Spec(String),
}

impl Common {
    /// Validated sampling parameters for this request.
    pub(crate) fn sampling(&self) -> Result<Sampling, Refusal> {
        let temperature = self.temperature.unwrap_or(0.0);
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(bad("temperature must be non-negative", "temperature"));
        }
        let top_p = self.top_p.unwrap_or(1.0);
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(bad("top_p must be in (0, 1]", "top_p"));
        }
        let top_k = match self.top_k {
            None | Some(-1) | Some(0) => 0,
            Some(k) if k > 0 => k as usize,
            Some(_) => return Err(bad("top_k must be -1 or non-negative", "top_k")),
        };
        let min_p = self.min_p.unwrap_or(0.0);
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err(bad("min_p must be in [0, 1]", "min_p"));
        }
        Ok(Sampling {
            temperature,
            top_p,
            top_k,
            min_p,
            seed: self.seed.map(|s| s as u64),
        })
    }

    /// Validated logit penalties for this request.
    pub(crate) fn penalties(&self) -> Result<Penalties, Refusal> {
        let repetition = self.repetition_penalty.unwrap_or(1.0);
        if !repetition.is_finite() || repetition <= 0.0 {
            return Err(bad(
                "repetition_penalty must be positive",
                "repetition_penalty",
            ));
        }
        Ok(Penalties {
            presence: within(self.presence_penalty, "presence_penalty")?,
            frequency: within(self.frequency_penalty, "frequency_penalty")?,
            repetition,
        })
    }

    /// Validated changes inside a think block.
    pub(crate) fn thinking(&self) -> Result<Thinking, Refusal> {
        const KEY: &str = "think_end_boost";
        let boost = match &self.think_end_boost {
            None | Some(Switch::Flag(false)) => None,
            Some(Switch::Spec(s)) if s.trim() == "off" => None,
            Some(Switch::Spec(s)) => {
                Boost::parse(s).map_err(|e| bad(format!("{KEY}: {e}"), KEY))?
            }
            Some(Switch::Flag(true)) => {
                let why = r#"think_end_boost must be a string like "start=0.5x,per1k=2,max=10", "off" or false"#;
                return Err(bad(why, KEY));
            }
        };
        Ok(Thinking {
            presence: within(self.thinking_presence_penalty, "thinking_presence_penalty")?,
            frequency: within(
                self.thinking_frequency_penalty,
                "thinking_frequency_penalty",
            )?,
            boost,
        })
    }
    /// Validated penalties for copies of earlier runs.
    pub(crate) fn sequence(&self) -> Result<Sequence, Refusal> {
        const KEY: &str = "sequence_penalty";
        match &self.sequence_penalty {
            None | Some(Switch::Flag(false)) => Ok(Sequence::default()),
            Some(Switch::Spec(s)) if s.trim() == "off" => Ok(Sequence::default()),
            Some(Switch::Spec(s)) => {
                Sequence::parse(s).map_err(|e| bad(format!("{KEY}: {e}"), KEY))
            }
            Some(Switch::Flag(true)) => {
                let why = r#"sequence_penalty must be a string like "thinking=1:1.5:8:10,answer=1:1.5:8:10", "off" or false"#;
                Err(bad(why, KEY))
            }
        }
    }

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

/// A penalty in [-2, 2]. Missing is 0.
fn within(v: Option<f32>, key: &str) -> Result<f32, Refusal> {
    match v.unwrap_or(0.0) {
        v if (-2.0..=2.0).contains(&v) => Ok(v),
        _ => Err(bad(format!("{key} must be in [-2, 2]"), key)),
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
pub(crate) struct WireFunction {
    name: String,
    #[serde(default)]
    arguments: Value,
    #[serde(flatten)]
    extra: Bag,
}

#[derive(Deserialize)]
pub(crate) struct WireToolCall {
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    function: WireFunction,
    #[serde(flatten)]
    extra: Bag,
}

#[derive(Deserialize)]
pub(crate) struct Said {
    role: String,
    #[serde(default)]
    content: Option<Content>,
    #[serde(rename = "reasoning", alias = "reasoning_content")]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(flatten)]
    extra: Bag,
}

impl Said {
    pub(crate) fn into_message(self) -> Result<Message, Refusal> {
        let content = self.content.map(|c| match c {
            Content::Text(text) => text,
            Content::Parts(parts) => parts
                .into_iter()
                .map(|Part::Text { text, .. }| text)
                .collect(),
        });
        let tool_calls = match self.tool_calls {
            None => None,
            Some(calls) => {
                let mut out = Vec::with_capacity(calls.len());
                for call in calls {
                    out.push(call.into_call()?);
                }
                Some(out)
            }
        };
        let content = match (content, &tool_calls) {
            (None, None) => Some(String::new()),
            (other, _) => other,
        };
        Ok(Message {
            role: self.role,
            content,
            reasoning: self.reasoning,
            tool_calls,
            tool_call_id: self.tool_call_id,
            name: self.name,
        })
    }
}

impl WireToolCall {
    fn into_call(self) -> Result<ToolCall, Refusal> {
        let kind = self.kind.unwrap_or_else(|| "function".into());
        if kind != "function" {
            return Err(bad("tool call type must be \"function\"", "messages"));
        }
        let arguments = parse_args(self.function.arguments)?;
        Ok(ToolCall {
            id: self.id,
            kind,
            function: FunctionCall {
                name: self.function.name,
                arguments,
            },
        })
    }
}

fn parse_args(value: Value) -> Result<Value, Refusal> {
    match value {
        Value::Null => Ok(json!({})),
        Value::Object(map) => Ok(Value::Object(map)),
        Value::String(s) if s.trim().is_empty() => Ok(json!({})),
        Value::String(s) => match serde_json::from_str::<Value>(&s) {
            Ok(Value::Object(map)) => Ok(Value::Object(map)),
            _ => Err(bad("tool call arguments must be a JSON object", "messages")),
        },
        _ => Err(bad("tool call arguments must be a JSON object", "messages")),
    }
}

#[derive(Deserialize)]
pub(crate) struct ToolFunction {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) description: Option<String>,
    #[serde(default)]
    pub(crate) parameters: Option<Value>,
    #[serde(rename = "strict")]
    _strict: Option<bool>,
    #[serde(flatten)]
    extra: Bag,
}

#[derive(Deserialize)]
pub(crate) struct Tool {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) function: ToolFunction,
    #[serde(flatten)]
    extra: Bag,
}

impl Tool {
    pub(crate) fn to_value(&self) -> Result<Value, Refusal> {
        if self.kind != "function" {
            return Err(bad("tool type must be \"function\"", "tools"));
        }
        if self.function.name.trim().is_empty() {
            return Err(bad("tool function name is empty", "tools"));
        }
        let mut func = Map::new();
        func.insert("name".into(), Value::String(self.function.name.clone()));
        if let Some(desc) = &self.function.description {
            func.insert("description".into(), Value::String(desc.clone()));
        }
        if let Some(params) = &self.function.parameters {
            func.insert("parameters".into(), params.clone());
        }
        Ok(json!({"type": "function", "function": Value::Object(func)}))
    }
}

#[derive(Deserialize)]
pub(crate) struct ToolChoice(Value);

impl ToolChoice {
    /// Resolves whether tools stay enabled and any forced choice.
    pub(crate) fn resolve(&self, tools: &[Tool]) -> Result<(bool, TaskToolChoice), Refusal> {
        if let Some(mode) = self.0.as_str() {
            return match mode {
                "auto" => Ok((true, TaskToolChoice::Auto)),
                "none" => Ok((false, TaskToolChoice::Auto)),
                "required" if tools.is_empty() => Err(bad(
                    "`tool_choice` \"required\" needs at least one tool",
                    "tool_choice",
                )),
                "required" => Ok((true, TaskToolChoice::Required)),
                _ => Err(bad(
                    "`tool_choice` must be \"auto\", \"none\", \"required\", or a function object",
                    "tool_choice",
                )),
            };
        }
        let Some(obj) = self.0.as_object() else {
            return Err(bad(
                "`tool_choice` must be a string or function object",
                "tool_choice",
            ));
        };
        let kind = obj.get("type").and_then(Value::as_str);
        let func = obj.get("function").and_then(Value::as_object);
        let name = func.and_then(|f| f.get("name")).and_then(Value::as_str);
        let extra = obj.keys().any(|k| k != "type" && k != "function")
            || func.is_some_and(|f| f.keys().any(|k| k != "name"));
        if kind != Some("function") || name.is_none_or(|n| n.trim().is_empty()) || extra {
            return Err(bad(
                "`tool_choice` object must be `{\"type\": \"function\", \"function\": {\"name\": ...}}`",
                "tool_choice",
            ));
        }
        let name = name.unwrap();
        if !tools.iter().any(|t| t.function.name == name) {
            return Err(bad(
                format!("tool `{name}` is not in `tools`"),
                "tool_choice",
            ));
        }
        Ok((true, TaskToolChoice::Function(name.to_string())))
    }
}

#[derive(Deserialize, Default)]
pub(crate) struct Kwargs {
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) clear_thinking: Option<bool>,
    pub(crate) reasoning_effort: Option<String>,
    #[serde(flatten)]
    extra: Bag,
}

/// Messages and how to lay them out.
#[derive(Deserialize)]
pub(crate) struct Messages {
    pub(crate) messages: Vec<Said>,
    #[serde(default)]
    pub(crate) tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub(crate) reasoning_effort: Option<String>,
    #[serde(default)]
    pub(crate) chat_template_kwargs: Kwargs,
}

impl Messages {
    fn nested(&self) -> Vec<String> {
        let mut out: Vec<String> =
            paths("chat_template_kwargs", &self.chat_template_kwargs.extra).collect();
        if let Some(tools) = &self.tools {
            for (i, tool) in tools.iter().enumerate() {
                out.extend(paths(&format!("tools[{i}]"), &tool.extra));
                out.extend(paths(&format!("tools[{i}].function"), &tool.function.extra));
            }
        }
        for (i, said) in self.messages.iter().enumerate() {
            out.extend(paths(&format!("messages[{i}]"), &said.extra));
            if let Some(calls) = &said.tool_calls {
                for (j, call) in calls.iter().enumerate() {
                    out.extend(paths(
                        &format!("messages[{i}].tool_calls[{j}]"),
                        &call.extra,
                    ));
                    out.extend(paths(
                        &format!("messages[{i}].tool_calls[{j}].function"),
                        &call.function.extra,
                    ));
                }
            }
            let Some(Content::Parts(parts)) = &said.content else {
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
    #[serde(default)]
    pub(crate) tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub(crate) parallel_tool_calls: Option<bool>,
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
