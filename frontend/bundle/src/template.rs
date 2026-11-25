//! The model's chat template, from `tokenizer_config.json`.

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Serde};
use minijinja::{Environment, ErrorKind, Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::{Deserialize, Serialize};
use serde_json::ser::{Formatter, PrettyFormatter};
use std::error::Error;
use std::io;
use std::path::Path;
use std::sync::Arc;

fn function_kind() -> String {
    "function".into()
}

/// Function name and arguments for one tool call.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// One tool call on an assistant message.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// A plain text message for `role`.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            ..Self::default()
        }
    }
}

/// A message with `reasoning` also as `reasoning_content`.
#[derive(Serialize)]
struct Shown<'a> {
    #[serde(flatten)]
    message: &'a Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
}

/// Extra variables for rendering a chat prompt.
#[derive(Clone, Debug, Default)]
pub struct RenderOptions<'a> {
    pub think: Option<bool>,
    pub reasoning_effort: Option<&'a str>,
    pub tools: &'a [serde_json::Value],
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SpecialToken {
    Text(String),
    Object { content: String },
}

impl SpecialToken {
    fn text(self) -> String {
        match self {
            SpecialToken::Text(s) | SpecialToken::Object { content: s } => s,
        }
    }
}

/// Chat settings from `tokenizer_config.json`.
#[derive(Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
    bos_token: Option<SpecialToken>,
    eos_token: Option<SpecialToken>,
}

/// The time `strftime_now` reads. Tests and replays supply their own.
pub type Clock = Arc<dyn Fn() -> jiff::Zoned + Send + Sync>;

/// Hugging Face's `strftime_now`: the clock's time, formatted.
fn strftime_now(clock: &Clock, format: &str) -> Result<String, minijinja::Error> {
    jiff::fmt::strtime::format(format, &clock())
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))
}

/// Python's `json.dumps` spacing: `", "` between items, `": "` after keys.
struct Spaced;

impl Formatter for Spaced {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

/// Hugging Face's `tojson`: Python `json.dumps`, `ensure_ascii` off by default.
fn tojson(value: &Value, kwargs: Kwargs) -> Result<Value, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    kwargs.assert_all_used()?;
    let mut out = Vec::new();
    let done = match indent {
        Some(n) => {
            let pad = vec![b' '; n];
            let fmt = PrettyFormatter::with_indent(&pad);
            value.serialize(&mut serde_json::Serializer::with_formatter(&mut out, fmt))
        }
        None => value.serialize(&mut serde_json::Serializer::with_formatter(
            &mut out, Spaced,
        )),
    };
    done.map_err(|e| minijinja::Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let text = String::from_utf8(out).unwrap_or_default();
    if !ascii.unwrap_or(false) {
        return Ok(Value::from_safe_string(text));
    }
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            escaped.push(c);
            continue;
        }
        let mut units = [0u16; 2];
        for unit in c.encode_utf16(&mut units) {
            escaped.push_str(&format!("\\u{unit:04x}"));
        }
    }
    Ok(Value::from_safe_string(escaped))
}

/// Removes Hugging Face `generation` training-mask block tags.
fn strip_generation(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(start) = rest.find("{%") {
        let Some(end) = rest[start + 2..].find("%}") else {
            break;
        };
        let tag_end = start + 2 + end + 2;
        let inner =
            rest[start + 2..start + 2 + end].trim_matches(|c: char| c.is_whitespace() || c == '-');
        if matches!(inner, "generation" | "endgeneration") {
            out.push_str(&rest[..start]);
            rest = &rest[tag_end..];
        } else {
            out.push_str(&rest[..tag_end]);
            rest = &rest[tag_end..];
        }
    }
    out.push_str(rest);
    out
}

pub struct Template {
    env: Environment<'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl Template {
    /// Compiles a Jinja template written for Hugging Face.
    pub fn new(source: String) -> Result<Self, minijinja::Error> {
        Self::with_clock(source, Arc::new(jiff::Zoned::now))
    }

    /// Like `new`, with `strftime_now` reading `clock`.
    pub fn with_clock(source: String, clock: Clock) -> Result<Self, minijinja::Error> {
        let mut env = Environment::new();
        // Block tags swallow their line, as Hugging Face renders.
        let syntax = SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()?;
        env.set_syntax(syntax);
        // Templates call Python string methods: split, strip, startswith.
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("strftime_now", move |format: &str| {
            strftime_now(&clock, format)
        });
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_template_owned("chat", strip_generation(&source))?;
        Ok(Self {
            env,
            bos_token: None,
            eos_token: None,
        })
    }

    /// Sets `bos_token` and `eos_token` for template rendering.
    pub fn with_tokens(mut self, bos: Option<String>, eos: Option<String>) -> Self {
        self.bos_token = bos;
        self.eos_token = eos;
        self
    }

    /// The template in `dir`, if its tokenizer config has one.
    pub fn load(dir: &Path) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        let text = std::fs::read_to_string(dir.join("tokenizer_config.json"))?;
        Self::from_config(dir, &text)
    }

    /// The template from `tokenizer_config.json` text and `dir`.
    pub fn from_config(
        dir: &Path,
        text: &str,
    ) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        let config: TokenizerConfig = serde_json::from_str(text)?;
        // Newer repositories keep the template in its own file.
        let source = match config.chat_template {
            Some(source) => Some(source),
            None => std::fs::read_to_string(dir.join("chat_template.jinja")).ok(),
        };
        let bos = config.bos_token.map(SpecialToken::text);
        let eos = config.eos_token.map(SpecialToken::text);
        Ok(match source {
            Some(source) => Some(Self::new(source)?.with_tokens(bos, eos)),
            None => None,
        })
    }

    /// Whether the template source contains `text`.
    pub fn mentions(&self, text: &str) -> bool {
        self.env
            .get_template("chat")
            .is_ok_and(|t| t.source().contains(text))
    }

    /// The prompt for `messages`, ready for the assistant's turn.
    pub fn render(&self, messages: &[Message], think: bool) -> Result<String, minijinja::Error> {
        self.render_with(messages, think, None)
    }

    /// As `render`, with an optional `reasoning_effort` template variable.
    pub fn render_with(
        &self,
        messages: &[Message],
        think: bool,
        reasoning_effort: Option<&str>,
    ) -> Result<String, minijinja::Error> {
        let options = RenderOptions {
            think: (!think).then_some(false),
            reasoning_effort,
            tools: &[],
        };
        self.render_options(messages, &options)
    }

    /// Renders `messages` with thinking, effort and tool definitions.
    pub fn render_options(
        &self,
        messages: &[Message],
        options: &RenderOptions<'_>,
    ) -> Result<String, minijinja::Error> {
        let template = self.env.get_template("chat")?;
        let mut shown = Vec::with_capacity(messages.len());
        for message in messages {
            let reasoning_content = message.reasoning.as_deref();
            shown.push(Shown {
                message,
                reasoning_content,
            });
        }
        let messages = Value::from(Serde(shown));
        let mut ctx = std::collections::BTreeMap::new();
        ctx.insert("messages", messages);
        ctx.insert("add_generation_prompt", Value::from(true));
        if let Some(bos) = &self.bos_token {
            ctx.insert("bos_token", Value::from(bos.as_str()));
        }
        if let Some(eos) = &self.eos_token {
            ctx.insert("eos_token", Value::from(eos.as_str()));
        }
        // Left undefined, `enable_thinking` means the model's default.
        if let Some(think) = options.think {
            ctx.insert("enable_thinking", Value::from(think));
        }
        if let Some(effort) = options.reasoning_effort {
            ctx.insert("reasoning_effort", Value::from(effort));
        }
        if !options.tools.is_empty() {
            ctx.insert("tools", Value::from(Serde(options.tools)));
        }
        template.render(Value::from(ctx))
    }
}
