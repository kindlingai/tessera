//! The model's chat template, as `tokenizer_config.json` carries it.

use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{Environment, Value, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// What `tokenizer_config.json` says about chat.
#[derive(Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
}

pub struct Template {
    env: Environment<'static>,
}

impl Template {
    /// Compiles a Jinja template written for Hugging Face.
    pub fn new(source: String) -> Result<Self, minijinja::Error> {
        let mut env = Environment::new();
        // Block tags swallow their line, as Hugging Face renders.
        let syntax = SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()?;
        env.set_syntax(syntax);
        // Templates call Python string methods: split, strip, startswith.
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_template_owned("chat", source)?;
        Ok(Self { env })
    }

    /// The template in `dir`, if its tokenizer config has one.
    pub fn load(dir: &Path) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        let text = std::fs::read_to_string(dir.join("tokenizer_config.json"))?;
        let config: TokenizerConfig = serde_json::from_str(&text)?;
        Ok(match config.chat_template {
            Some(source) => Some(Self::new(source)?),
            None => None,
        })
    }

    /// The prompt for `messages`, ready for the assistant's turn.
    pub fn render(&self, messages: &[Message], think: bool) -> Result<String, minijinja::Error> {
        let template = self.env.get_template("chat")?;
        let messages = Value::from(Serde(messages));
        // Left undefined, `enable_thinking` means the model's default.
        match think {
            true => template.render(context! { messages, add_generation_prompt => true }),
            false => template.render(context! {
                messages,
                add_generation_prompt => true,
                enable_thinking => false,
            }),
        }
    }
}
