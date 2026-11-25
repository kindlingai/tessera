//! The runtime for `#[derive(Task)]` code: flags, tool calls, schemas.

use serde_json::{Map, Value, json};
use std::io::Read;
use std::path::PathBuf;

/// Bad flags, words or tool-call arguments.
#[derive(Debug)]
pub struct Usage(pub String);

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Usage {}

fn usage(text: String) -> Usage {
    Usage(text)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Role {
    Flag,
    /// The words after the flags, or stdin.
    Words,
    /// Flags the run target owns, kept as given.
    Extra,
}

/// One field of a task, from the derive.
#[derive(Debug)]
pub struct Field {
    /// The JSON key: the field's name.
    pub key: &'static str,
    pub flag: &'static str,
    /// The value's placeholder in usage text.
    pub value: &'static str,
    pub about: &'static str,
    /// The JSON Schema type.
    pub kind: &'static str,
    pub role: Role,
    /// Whether the flag takes a value.
    pub takes: bool,
    /// Whether tool calls see it.
    pub tool: bool,
    /// Whether the model load takes the flag too.
    pub load: bool,
    /// Bounds on a number, or on the length of text.
    pub min: Option<i64>,
    pub max: Option<i64>,
    pub example: Option<&'static str>,
}

/// A value that a flag or a tool call gives.
pub trait Arg: Sized {
    /// The JSON Schema type.
    const KIND: &'static str;
    fn parse(text: &str) -> Option<Self>;
    fn from_json(value: &Value) -> Option<Self>;
    fn text(&self) -> String;
}

macro_rules! args {
    ($($t:ty: $kind:literal, $json:expr, $text:expr;)*) => {$(
        impl Arg for $t {
            const KIND: &'static str = $kind;
            fn parse(text: &str) -> Option<Self> {
                text.parse().ok()
            }
            fn from_json(value: &Value) -> Option<Self> {
                $json(value)
            }
            fn text(&self) -> String {
                $text(self)
            }
        }
    )*};
}

fn count<T: TryFrom<u64>>(value: &Value) -> Option<T> {
    value.as_u64()?.try_into().ok()
}

fn string<T: for<'a> From<&'a str>>(value: &Value) -> Option<T> {
    value.as_str().map(T::from)
}

args! {
    usize: "integer", count, ToString::to_string;
    u32: "integer", count, ToString::to_string;
    String: "string", string, String::clone;
    PathBuf: "string", string, |p: &PathBuf| p.display().to_string();
}

/// Flag fields: `Option` takes a value, `bool` is a switch.
pub trait Given: Sized {
    const KIND: &'static str;
    const TAKES: bool;
    /// From the flag's value, or `None` when it is absent.
    fn from_flag(given: Option<Option<&str>>) -> Option<Self>;
    fn from_json(value: Option<&Value>) -> Option<Self>;
    fn push(&self, flag: &str, out: &mut Vec<String>);
}

impl<T: Arg> Given for Option<T> {
    const KIND: &'static str = T::KIND;
    const TAKES: bool = true;
    fn from_flag(given: Option<Option<&str>>) -> Option<Self> {
        match given {
            None => Some(None),
            Some(value) => T::parse(value?).map(Some),
        }
    }
    fn from_json(value: Option<&Value>) -> Option<Self> {
        match value {
            None | Some(Value::Null) => Some(None),
            Some(v) => T::from_json(v).map(Some),
        }
    }
    fn push(&self, flag: &str, out: &mut Vec<String>) {
        if let Some(value) = self {
            out.extend([flag.to_string(), value.text()]);
        }
    }
}

impl Given for bool {
    const KIND: &'static str = "boolean";
    const TAKES: bool = false;
    fn from_flag(given: Option<Option<&str>>) -> Option<Self> {
        Some(given.is_some())
    }
    fn from_json(value: Option<&Value>) -> Option<Self> {
        match value {
            None | Some(Value::Null) => Some(false),
            Some(v) => v.as_bool(),
        }
    }
    fn push(&self, flag: &str, out: &mut Vec<String>) {
        if *self {
            out.push(flag.into());
        }
    }
}

/// Command-line arguments, split by a task's fields.
pub struct Cli {
    given: Vec<(&'static str, Option<String>)>,
    words: Vec<String>,
    extra: Vec<String>,
}

impl Cli {
    /// Splits `args`. `extra` says if extra flags take values.
    pub fn split(
        args: &[String],
        fields: &[Field],
        extra: &dyn Fn(&str) -> Option<bool>,
    ) -> Result<Cli, Usage> {
        let has = |role| fields.iter().any(|f| f.role == role);
        let needs = |flag: &str| usage(format!("{flag} needs a value"));
        let mut cli = Cli {
            given: Vec::new(),
            words: Vec::new(),
            extra: Vec::new(),
        };
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            if arg == "--" {
                cli.words.extend(rest.by_ref().cloned());
                break;
            }
            if !arg.starts_with('-') {
                cli.words.push(arg.clone());
                continue;
            }
            let own = fields
                .iter()
                .find(|f| f.role == Role::Flag && f.flag == arg);
            if let Some(f) = own {
                let value = match f.takes {
                    true => Some(rest.next().ok_or_else(|| needs(arg))?.clone()),
                    false => None,
                };
                cli.given.push((f.flag, value));
                continue;
            }
            let takes = if has(Role::Extra) { extra(arg) } else { None };
            let takes = takes.ok_or_else(|| usage(format!("unknown flag {arg}")))?;
            cli.extra.push(arg.clone());
            if takes {
                cli.extra
                    .push(rest.next().ok_or_else(|| needs(arg))?.clone());
            }
        }
        if !cli.words.is_empty() && !has(Role::Words) {
            return Err(usage(format!("unexpected words: {}", cli.words.join(" "))));
        }
        Ok(cli)
    }

    /// The last value of `flag`.
    pub fn take<T: Given>(&self, flag: &str) -> Result<T, Usage> {
        let given = self.given.iter().rev().find(|(f, _)| *f == flag);
        let given = given.map(|(_, value)| value.as_deref());
        T::from_flag(given).ok_or_else(|| usage(format!("{flag} takes a {}", T::KIND)))
    }

    /// The words, or stdin when there are none.
    pub fn words<T: Arg>(&self) -> Result<T, Usage> {
        let mut text = self.words.join(" ");
        if self.words.is_empty() {
            let read = std::io::stdin().read_to_string(&mut text);
            read.map_err(|e| usage(e.to_string()))?;
        }
        if text.trim().is_empty() {
            return Err(usage("no words, on the command line or stdin".into()));
        }
        T::parse(&text).ok_or_else(|| usage(format!("bad words: {text}")))
    }

    pub fn extra(&self) -> Vec<String> {
        self.extra.clone()
    }
}

/// A tool call's arguments, checked against a task's fields.
pub struct Json<'a> {
    map: &'a Map<String, Value>,
}

impl<'a> Json<'a> {
    pub fn new(value: &'a Value, fields: &[Field]) -> Result<Json<'a>, Usage> {
        let map = value.as_object();
        let map = map.ok_or_else(|| usage(format!("arguments are not an object: {value}")))?;
        for (key, value) in map {
            let field = fields.iter().find(|f| f.tool && f.key == key);
            let field = field.ok_or_else(|| usage(format!("unknown argument {key}")))?;
            let size = match value {
                Value::String(s) => Some(s.chars().count() as i64),
                other => other.as_i64(),
            };
            let low = field.min.zip(size).is_some_and(|(m, s)| s < m);
            let high = field.max.zip(size).is_some_and(|(m, s)| s > m);
            if low || high {
                return Err(usage(format!("{key} is out of bounds: {value}")));
            }
        }
        Ok(Json { map })
    }

    /// An optional argument.
    pub fn take<T: Given>(&self, key: &str) -> Result<T, Usage> {
        let bad = || usage(format!("{key} is not a {}", T::KIND));
        T::from_json(self.map.get(key)).ok_or_else(bad)
    }

    /// A required argument.
    pub fn need<T: Arg>(&self, key: &str) -> Result<T, Usage> {
        let bad = || usage(format!("{key} is missing or not a {}", T::KIND));
        self.map.get(key).and_then(T::from_json).ok_or_else(bad)
    }
}

/// Fields back to command-line words, or to load flags.
pub struct Argv {
    load: bool,
    flags: Vec<String>,
    words: Option<String>,
}

impl Argv {
    /// With `load`, only flags for the model load.
    pub fn new(load: bool) -> Argv {
        Argv {
            load,
            flags: Vec::new(),
            words: None,
        }
    }

    pub fn flag<T: Given>(&mut self, flag: &str, value: &T, load: bool) {
        if load || !self.load {
            value.push(flag, &mut self.flags);
        }
    }

    pub fn words<T: Arg>(&mut self, value: &T) {
        if !self.load {
            self.words = Some(value.text());
        }
    }

    pub fn extra(&mut self, flags: &[String]) {
        self.flags.extend_from_slice(flags);
    }

    /// The flags, then the words, after `--` if flag-like.
    pub fn finish(self) -> Vec<String> {
        let mut out = self.flags;
        if let Some(words) = self.words {
            if words.starts_with('-') {
                out.push("--".into());
            }
            out.push(words);
        }
        out
    }
}

/// `name` and its flags, `extra` where the extra flags go.
pub fn usage_line(name: &str, fields: &[Field], extra: &str) -> String {
    let (mut out, mut words) = (name.to_string(), String::new());
    for f in fields {
        let part = match (f.role, f.takes) {
            (Role::Flag, true) => format!("[{} {}]", f.flag, f.value),
            (Role::Flag, false) => format!("[{}]", f.flag),
            (Role::Extra, _) => extra.to_string(),
            (Role::Words, _) => {
                words = format!(" [{}...]", f.key);
                continue;
            }
        };
        if !part.is_empty() {
            out = out + " " + &part;
        }
    }
    out + &words
}

/// The JSON Schema of the fields tool calls see.
pub fn schema(fields: &[Field]) -> Value {
    let (mut properties, mut required) = (Map::new(), Vec::new());
    for f in fields.iter().filter(|f| f.tool) {
        let mut property = Map::new();
        property.insert("type".into(), f.kind.into());
        property.insert("description".into(), f.about.into());
        let (low, high) = match f.kind {
            "string" => ("minLength", "maxLength"),
            _ => ("minimum", "maximum"),
        };
        for (key, bound) in [(low, f.min), (high, f.max)] {
            if let Some(bound) = bound {
                property.insert(key.into(), bound.into());
            }
        }
        properties.insert(f.key.into(), Value::Object(property));
        if f.role == Role::Words {
            required.push(f.key);
        }
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

/// The fields' example values, as tool-call arguments.
pub fn example(fields: &[Field]) -> Value {
    let mut out = Map::new();
    for f in fields {
        let Some(text) = f.example else {
            continue;
        };
        let value = match (f.kind, text.parse::<i64>()) {
            ("integer", Ok(n)) => n.into(),
            _ => text.into(),
        };
        out.insert(f.key.into(), value);
    }
    Value::Object(out)
}
