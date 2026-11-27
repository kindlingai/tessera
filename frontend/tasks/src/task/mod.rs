//! Tasks defined once for library, CLI and tool calls.

mod args;
mod image;
mod talk;
mod text;

pub use args::{Arg, Argv, Cli, Field, Given, Json, Role, Usage};
pub use image::{Image, path};
pub use serde_json::Value;
pub use talk::{Ask, Chat};
pub use tessera_task_derive::Task;
pub use text::{Embed, Tag};

use crate::Error;
use serde_json::json;
use std::path::{Path, PathBuf};
use tessera_engine::progress::Progress;

/// A task's arguments, from flags or from a tool call.
pub trait Task: Sized {
    const NAME: &'static str;
    /// The catalog's serve task for it. `None` is any.
    const KIND: Option<&'static str>;
    const ABOUT: &'static str;
    /// A request it answers, for the router's examples.
    const EXAMPLE: Option<&'static str>;
    const FIELDS: &'static [Field];

    /// Reads flags and words. `extra` says if flags take values.
    fn parse(args: &[String], extra: &dyn Fn(&str) -> Option<bool>) -> Result<Self, Usage>;
    /// Reads a tool call's arguments.
    fn from_json(value: &Value) -> Result<Self, Usage>;
    fn push(&self, out: &mut Argv);

    /// The arguments as command-line words.
    fn args(&self) -> Vec<String> {
        let mut out = Argv::new(false);
        self.push(&mut out);
        out.finish()
    }

    /// Flags for the model load.
    fn load_flags(&self) -> Vec<String> {
        let mut out = Argv::new(true);
        self.push(&mut out);
        out.finish()
    }

    /// The usage line, with `extra` for the extra flags.
    fn usage(extra: &str) -> String {
        args::usage_line(Self::NAME, Self::FIELDS, extra)
    }

    /// The JSON Schema of a tool call's arguments.
    fn schema() -> Value {
        args::schema(Self::FIELDS)
    }

    /// The arguments for the example request.
    fn example() -> Value {
        args::example(Self::FIELDS)
    }

    /// The OpenAI function definition.
    fn function() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": Self::NAME,
                "description": Self::ABOUT,
                "parameters": Self::schema(),
            },
        })
    }

    /// A tool call's arguments as command-line words.
    fn tool_args(value: &Value) -> Result<Vec<String>, Usage> {
        Ok(Self::from_json(value)?.args())
    }
}

/// A task's work on `On`, such as a loaded model.
pub trait Run<On>: Task {
    fn run(self, on: On, sink: &mut dyn Sink) -> Result<(), Error>;
}

/// Where a task's output goes.
pub trait Sink {
    /// Answer text, as it comes.
    fn text(&mut self, piece: &str);
    /// One result, such as a vector or a tag.
    fn record(&mut self, record: Value);
    /// A file the task wrote.
    fn file(&mut self, path: &Path);
    /// A message for the person, not a result.
    fn note(&mut self, text: &str);
    /// The next chat turn. `None` ends the chat.
    fn turn(&mut self) -> Option<String> {
        None
    }
    /// Where long steps, such as downloads and loads, report.
    fn progress(&self) -> Progress {
        Progress::default()
    }
}

/// Output kept for a return value or a tool result.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Collect {
    pub text: String,
    pub records: Vec<Value>,
    pub files: Vec<PathBuf>,
    pub notes: Vec<String>,
}

impl Sink for Collect {
    fn text(&mut self, piece: &str) {
        self.text += piece;
    }

    fn record(&mut self, record: Value) {
        self.records.push(record);
    }

    fn file(&mut self, path: &Path) {
        self.files.push(path.into());
    }

    fn note(&mut self, text: &str) {
        self.notes.push(text.into());
    }
}

impl Collect {
    /// The output as a tool result, without empty parts.
    pub fn value(&self) -> Value {
        let files: Vec<String> = self.files.iter().map(|p| p.display().to_string()).collect();
        let mut out = serde_json::Map::new();
        let parts = [
            ("text", json!(self.text), self.text.is_empty()),
            ("records", json!(self.records), self.records.is_empty()),
            ("files", json!(files), files.is_empty()),
            ("notes", json!(self.notes), self.notes.is_empty()),
        ];
        for (key, value, empty) in parts {
            if !empty {
                out.insert(key.into(), value);
            }
        }
        Value::Object(out)
    }
}
