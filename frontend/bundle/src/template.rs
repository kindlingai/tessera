//! The model's chat template, from `tokenizer_config.json`.

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Serde};
use minijinja::{Environment, ErrorKind, Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::{Deserialize, Serialize};
use serde_json::ser::{Formatter, PrettyFormatter};
use std::borrow::Cow;
use std::error::Error;
use std::io;
use std::path::Path;
use std::sync::Arc;

const SPACES: [u8; 32] = [b' '; 32];
const HEX: &[u8; 16] = b"0123456789abcdef";

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
    /// Null, not absent, when unset: templates test `is none`.
    #[serde(default)]
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
    role: &'a str,
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a [ToolCall]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
}

/// Top-level variables passed to the chat template.
#[derive(Serialize)]
struct RenderCtx<'a> {
    messages: Vec<Shown<'a>>,
    add_generation_prompt: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    bos_token: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eos_token: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    clear_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [serde_json::Value]>,
}

/// Extra variables for rendering a chat prompt.
#[derive(Clone, Debug, Default)]
pub struct RenderOptions<'a> {
    pub think: Option<bool>,
    pub clear_thinking: Option<bool>,
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

/// `value` as JSON text with Python's spacing.
pub(crate) fn spaced(value: &serde_json::Value) -> String {
    let mut out = Vec::with_capacity(128);
    let mut ser = serde_json::Serializer::with_formatter(&mut out, Spaced);
    value.serialize(&mut ser).expect("JSON values serialize");
    String::from_utf8(out).expect("JSON text is UTF-8")
}

/// Escapes non-ASCII characters in `text` as `\uXXXX` JSON escapes.
fn escape_ascii(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len() + 16);
    let mut start = 0;
    for (idx, c) in text.char_indices() {
        if c.is_ascii() {
            continue;
        }
        escaped.push_str(&text[start..idx]);
        let mut units = [0u16; 2];
        for &mut unit in c.encode_utf16(&mut units) {
            let buf = [
                b'\\',
                b'u',
                HEX[((unit >> 12) & 0xf) as usize],
                HEX[((unit >> 8) & 0xf) as usize],
                HEX[((unit >> 4) & 0xf) as usize],
                HEX[(unit & 0xf) as usize],
            ];
            escaped.push_str(std::str::from_utf8(&buf).expect("hex is ASCII"));
        }
        start = idx + c.len_utf8();
    }
    escaped.push_str(&text[start..]);
    escaped
}

/// Hugging Face's `tojson`: Python `json.dumps`, `ensure_ascii` off by default.
fn tojson(value: &Value, kwargs: Kwargs) -> Result<Value, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    kwargs.assert_all_used()?;
    let mut out = Vec::with_capacity(128);
    let done = match indent {
        Some(n) if n <= SPACES.len() => {
            let fmt = PrettyFormatter::with_indent(&SPACES[..n]);
            value.serialize(&mut serde_json::Serializer::with_formatter(&mut out, fmt))
        }
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
    if !ascii.unwrap_or(false) || text.is_ascii() {
        return Ok(Value::from_safe_string(text));
    }
    Ok(Value::from_safe_string(escape_ascii(&text)))
}

/// Removes Hugging Face `generation` training-mask block tags.
fn strip_generation(source: String) -> String {
    if !source.contains("generation") {
        return source;
    }
    let mut out: Option<String> = None;
    let mut copied = 0;
    let mut pos = 0;
    while let Some(rel) = source[pos..].find("{%") {
        let start = pos + rel;
        let Some(end) = source[start + 2..].find("%}") else {
            break;
        };
        let tag_end = start + 2 + end + 2;
        let inner = source[start + 2..start + 2 + end]
            .trim_matches(|c: char| c.is_whitespace() || c == '-');
        if matches!(inner, "generation" | "endgeneration") {
            let buf = out.get_or_insert_with(|| String::with_capacity(source.len()));
            buf.push_str(&source[copied..start]);
            copied = tag_end;
        }
        pos = tag_end;
    }
    match out {
        Some(mut buf) => {
            buf.push_str(&source[copied..]);
            buf
        }
        None => source,
    }
}

pub struct Template {
    env: Environment<'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    tool_prompt: bool,
    has_developer: bool,
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
        let stripped = strip_generation(source);
        let has_developer = stripped.contains("developer");
        env.add_template_owned("chat", stripped)?;
        Ok(Self {
            env,
            bos_token: None,
            eos_token: None,
            tool_prompt: false,
            has_developer,
        })
    }

    /// The end-of-sequence token text, if the config names one.
    pub fn eos_token(&self) -> Option<&str> {
        self.eos_token.as_deref()
    }

    /// Jinja that lists `tools` when the chat template does not.
    pub fn with_tool_prompt(&mut self, prompt: &str) -> Result<(), minijinja::Error> {
        if !self.mentions("tools") {
            self.env.add_template_owned("tools", prompt.to_string())?;
            self.tool_prompt = true;
        }
        Ok(())
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
            ..RenderOptions::default()
        };
        self.render_options(messages, &options)
    }

    /// Renders `messages` with thinking, effort and tool definitions.
    pub fn render_options(
        &self,
        messages: &[Message],
        options: &RenderOptions<'_>,
    ) -> Result<String, minijinja::Error> {
        let messages = self.prepared(messages, options.tools)?;
        match self.render_raw(&messages, options) {
            Err(e) if e.kind() == minijinja::ErrorKind::InvalidOperation => {
                // Some templates join arguments to text: retry with JSON text.
                let Some(texts) = text_arguments(&messages) else {
                    return Err(e);
                };
                self.render_raw(&texts, options)
            }
            other => other,
        }
    }

    fn render_raw(
        &self,
        messages: &[Message],
        options: &RenderOptions<'_>,
    ) -> Result<String, minijinja::Error> {
        let template = self.env.get_template("chat")?;
        let shown: Vec<Shown<'_>> = messages
            .iter()
            .map(|m| Shown {
                role: &m.role,
                content: m.content.as_deref(),
                reasoning: m.reasoning.as_deref(),
                tool_calls: m.tool_calls.as_deref(),
                tool_call_id: m.tool_call_id.as_deref(),
                name: m.name.as_deref(),
                reasoning_content: m.reasoning.as_deref(),
            })
            .collect();
        let ctx = RenderCtx {
            messages: shown,
            add_generation_prompt: true,
            bos_token: self.bos_token.as_deref(),
            eos_token: self.eos_token.as_deref(),
            enable_thinking: options.think,
            clear_thinking: options.clear_thinking,
            reasoning_effort: options.reasoning_effort,
            tools: (!options.tools.is_empty()).then_some(options.tools),
        };
        template.render(Serde(ctx))
    }

    /// `messages` with `developer` normalized and any tool prompt added.
    fn prepared<'m>(
        &self,
        messages: &'m [Message],
        tools: &[serde_json::Value],
    ) -> Result<Cow<'m, [Message]>, minijinja::Error> {
        let remap = !self.has_developer && messages.iter().any(|m| m.role == "developer");
        let add_tools = !tools.is_empty() && self.tool_prompt;
        if !remap && !add_tools {
            return Ok(Cow::Borrowed(messages));
        }
        let mut out = messages.to_vec();
        if remap {
            for m in &mut out {
                if m.role == "developer" {
                    m.role = "system".into();
                }
            }
        }
        if add_tools {
            let tools = Value::from(Serde(tools));
            let text = self
                .env
                .get_template("tools")?
                .render(minijinja::context! { tools })?;
            match out.first_mut() {
                Some(first) if first.role == "system" => {
                    let content = first.content.get_or_insert_default();
                    content.push_str("\n\n");
                    content.push_str(&text);
                }
                _ => out.insert(0, Message::text("system", text)),
            }
        }
        Ok(Cow::Owned(out))
    }
}

/// `messages` with object tool-call arguments as JSON text, if any.
fn text_arguments(messages: &[Message]) -> Option<Vec<Message>> {
    let any = messages
        .iter()
        .flat_map(|m| m.tool_calls.iter().flatten())
        .any(|c| !c.function.arguments.is_string());
    if !any {
        return None;
    }
    let mut out = messages.to_vec();
    for call in out
        .iter_mut()
        .flat_map(|m| m.tool_calls.iter_mut().flatten())
    {
        let args = &mut call.function.arguments;
        if !args.is_string() {
            *args = serde_json::Value::String(args.to_string());
        }
    }
    Some(out)
}
