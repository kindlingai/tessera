//! `tesserax --cli-json`: commands, flags and words from usage text.

use crate::task::ENTRIES;
use serde_json::{Map, Value, json};
use tessera_serve::ranks::FLAGS as RANKS;
use tessera_serve::start::flags;

/// The usage of a command that takes the serve flags.
const SERVE_FLAGS: &str = "[flags]";

/// One command: flags, words and subcommands.
#[derive(Default)]
struct Command {
    flags: Vec<(String, bool)>,
    words: bool,
    commands: Map<String, Value>,
}

impl Command {
    /// Adds the flags and words in `usage`.
    fn read(&mut self, usage: &str) {
        let tokens: Vec<&str> = usage.split_whitespace().collect();
        let (mut value, mut depth) = (false, 0);
        for (n, token) in tokens.iter().enumerate() {
            // A placeholder such as `<model and its flags>` spans tokens.
            if depth > 0 || std::mem::take(&mut value) {
                depth += token.matches('<').count() as i64 - token.matches('>').count() as i64;
                continue;
            }
            let bare = token.trim_start_matches(['[', '(', '|']);
            let name_len = bare
                .char_indices()
                .find(|(i, c)| *i > 0 && !(c.is_ascii_alphanumeric() || *c == '-'))
                .map_or(bare.len(), |(i, _)| i);
            let flag = bare.starts_with('-')
                && bare[1..].starts_with(|c: char| c.is_alphabetic() || c == '-');
            if flag {
                let next = tokens.get(n + 1);
                let opens = |t: &&str| !t.starts_with(['-', '[', '|']);
                let takes = name_len == bare.len() && next.is_some_and(opens);
                let name = bare[..name_len].to_string();
                if !self.flags.iter().any(|(f, _)| *f == name) {
                    self.flags.push((name, takes));
                }
                value = takes;
                continue;
            }
            if bare.starts_with('<') || bare.contains("...") {
                self.words = true;
            }
        }
    }

    fn json(self) -> Value {
        let mut flags = Vec::new();
        for (flag, takes) in self.flags {
            flags.push(json!({ "flag": flag, "takes": takes }));
        }
        json!({ "flags": flags, "words": self.words, "commands": self.commands })
    }
}

/// A USAGE line up to its first wide gap.
fn usage_part(line: &str) -> &str {
    line.trim_start().split("  ").next().unwrap_or_default()
}

/// The whole command tree as JSON.
pub fn json() -> Value {
    let mut lines = crate::USAGE.lines();
    let mut top = Command::default();
    top.read(lines.next().unwrap_or_default());
    top.words = true;
    let mut groups: Map<String, Value> = Map::new();
    for e in &ENTRIES {
        let mut command = Command::default();
        let usage = (e.usage)();
        command.read(&usage);
        if usage.contains(SERVE_FLAGS) {
            command.read(&flags());
            command.read(RANKS);
        }
        let into = match e.group {
            "" => &mut top.commands,
            group => {
                let slot = groups
                    .entry(group)
                    .or_insert_with(|| Command::default().json());
                slot["commands"]
                    .as_object_mut()
                    .expect("commands is an object")
            }
        };
        into.insert(e.name.into(), command.json());
    }
    for line in lines.filter(|l| l.starts_with("  ") && !l.starts_with("   ")) {
        let usage = usage_part(line);
        let Some(word) = usage.split_whitespace().next() else {
            continue;
        };
        if !word.starts_with(|c: char| c.is_ascii_lowercase()) {
            top.read(usage);
            continue;
        }
        if groups.contains_key(word) || top.commands.contains_key(word) {
            continue;
        }
        let mut command = Command::default();
        command.read(usage.trim_start_matches(word));
        top.commands.insert(word.into(), command.json());
    }
    top.commands.extend(groups);
    top.json()
}
