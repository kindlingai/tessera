//! Built-in `response_template`s for checkpoints that ship none.

use super::ResponseParser;
use crate::template::Template;
use std::error::Error;

struct Format {
    name: &'static str,
    json: &'static str,
    /// Text the chat template must hold besides the opener.
    also: &'static [&'static str],
}

macro_rules! formats {
    ($($name:literal $(also $($also:literal),+)?;)*) => {
        &[$(Format {
            name: $name,
            json: include_str!(concat!($name, ".json")),
            also: &[$($($also),+)?],
        }),*]
    };
}

/// Most specific first: a longer marker set wins.
const FORMATS: &[Format] = formats! {
    "gemma4";
    "deepseek";
    "granite";
    "qwen35" also "<function=";
    "glm53" also "<arg_key>";
    "smollm2" also "<tool_call>[";
    "qwen3";
};

/// The first format `chat` writes, and its name.
pub(super) fn pick(
    chat: &Template,
) -> Result<Option<(&'static str, ResponseParser)>, Box<dyn Error + Send + Sync>> {
    for format in FORMATS {
        let parser = ResponseParser::compile(serde_json::from_str(format.json)?)?;
        let opener = parser
            .tool_calls
            .as_ref()
            .and_then(|t| t.open.triggers().first().copied());
        let Some(opener) = opener else { continue };
        if chat.mentions(opener.trim()) && format.also.iter().all(|s| chat.mentions(s)) {
            return Ok(Some((format.name, parser)));
        }
    }
    Ok(None)
}
