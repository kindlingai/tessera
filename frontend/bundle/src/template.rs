//! The model's chat template, from `tokenizer_config.json`.

use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{Environment, Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Chat settings from `tokenizer_config.json`.
#[derive(Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
}

/// The time `strftime_now` reads. Tests and replays supply their own.
pub type Clock = Arc<dyn Fn() -> jiff::Zoned + Send + Sync>;

/// Hugging Face's `strftime_now`: the clock's time, formatted.
fn strftime_now(clock: &Clock, format: &str) -> Result<String, minijinja::Error> {
    jiff::fmt::strtime::format(format, &clock())
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))
}

pub struct Template {
    env: Environment<'static>,
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
        env.add_function("strftime_now", move |format: &str| {
            strftime_now(&clock, format)
        });
        env.add_template_owned("chat", source)?;
        Ok(Self { env })
    }

    /// The template in `dir`, if its tokenizer config has one.
    pub fn load(dir: &Path) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        let text = std::fs::read_to_string(dir.join("tokenizer_config.json"))?;
        let config: TokenizerConfig = serde_json::from_str(&text)?;
        // Newer repositories keep the template in its own file.
        let source = match config.chat_template {
            Some(source) => Some(source),
            None => std::fs::read_to_string(dir.join("chat_template.jinja")).ok(),
        };
        Ok(match source {
            Some(source) => Some(Self::new(source)?),
            None => None,
        })
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
        let template = self.env.get_template("chat")?;
        let messages = Value::from(Serde(messages));
        let mut ctx = std::collections::BTreeMap::new();
        ctx.insert("messages", messages);
        ctx.insert("add_generation_prompt", Value::from(true));
        // Left undefined, `enable_thinking` means the model's default.
        if !think {
            ctx.insert("enable_thinking", Value::from(false));
        }
        if let Some(effort) = reasoning_effort {
            ctx.insert("reasoning_effort", Value::from(effort));
        }
        template.render(Value::from(ctx))
    }
}
