//! Built-in `response_template`s, and the block markers of any.

use super::{BlockSpec, Body, Opener, ResponseParser};
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

/// Something for each marker that opens or closes a block.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Markers<T> {
    pub think_open: T,
    pub think_close: T,
    pub call_open: T,
    pub call_close: T,
}

impl ResponseParser {
    /// Each marker's texts. Pattern openers give their fixed start.
    pub fn markers(&self) -> Markers<Vec<&str>> {
        fn open(b: &Option<BlockSpec>) -> Vec<&str> {
            b.as_ref().map_or(Vec::new(), |b| b.open.triggers())
        }
        fn close(b: &Option<BlockSpec>) -> Vec<&str> {
            b.as_ref()
                .map_or(Vec::new(), |b| b.close.iter().map(String::as_str).collect())
        }
        Markers {
            think_open: open(&self.thinking),
            think_close: close(&self.thinking),
            call_open: open(&self.tool_calls),
            call_close: close(&self.tool_calls),
        }
    }

    /// Prefix text that opens a tool call, optionally naming `name`.
    pub fn call_prefix(&self, name: Option<&str>) -> Option<String> {
        let block = self.tool_calls.as_ref()?;
        let outer = block.outer.as_ref().map_or("", |[o, _]| o.as_str());
        let Some(name) = name else {
            let trigger = block.open.triggers().into_iter().next()?;
            return Some(format!("{outer}{trigger}"));
        };
        if let Some(tmpl) = &block.call_prefix {
            return Some(format!("{outer}{}", tmpl.replace("{name}", name)));
        }
        match &block.open {
            Opener::Pattern { prefix, regex } if !prefix.is_empty() => {
                if regex.as_str().contains("<function=(?P<name>") {
                    Some(format!("{outer}{prefix}\n<function={name}>\n"))
                } else {
                    Some(format!("{outer}{prefix}{name}"))
                }
            }
            Opener::Literal(opens) if matches!(block.body, Body::Json(_)) => {
                let open = opens.first()?;
                let quoted = serde_json::to_string(name).ok()?;
                Some(format!(
                    "{outer}{open}{{\"name\": {quoted}, \"arguments\": "
                ))
            }
            _ => None,
        }
    }
}
