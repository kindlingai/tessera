//! Tasks and pipelines the server loads, one file each.

use crate::start::Options;
use crate::{Model, Slot};
use std::sync::{Arc, Mutex};
use tessera_engine::Settings;
use tessera_engine::progress::Step;
use tessera_engine::turns::{Seated, Turns};
use tessera_tasks::api::{GenerateImage, Info, Tokens};

/// Declares each entry's module and lists its `ENTRY`.
macro_rules! entries {
    ($($module:ident),*; $($more:path),*) => {
        $(mod $module;)*
        /// Every entry, in the order the usage text lists flags.
        pub static ENTRIES: &[&Entry] = &[$(&$module::ENTRY,)* $(&$more),*];
    };
}

entries!(generate, draft, mtp, embed, klein, sprite; mtp::DUAL);

mod intake;
#[cfg(test)]
mod intake_test;
#[cfg(test)]
mod task_test;

/// One flag an entry owns.
pub struct Flag {
    pub name: &'static str,
    /// The value's placeholder. Empty for a switch.
    pub value: &'static str,
    /// Whether a value is good.
    pub check: fn(&str) -> bool,
}

impl Flag {
    /// A flag with no value.
    pub const fn switch(name: &'static str) -> Flag {
        Flag::text(name, "")
    }

    /// A flag with any text as its value.
    pub const fn text(name: &'static str, value: &'static str) -> Flag {
        let check = |_: &str| true;
        Flag { name, value, check }
    }

    /// A flag whose value is a count.
    pub const fn count(name: &'static str, value: &'static str) -> Flag {
        let check = |v: &str| v.parse::<usize>().is_ok();
        Flag { name, value, check }
    }

    /// The usage text for this flag.
    pub fn usage(&self) -> String {
        match self.value {
            "" => format!("[{}]", self.name),
            value => format!("[{} {value}]", self.name),
        }
    }
}

/// When an entry runs among those of its task.
#[derive(Clone, Copy)]
pub enum Pick {
    /// Unless a `Given` entry runs.
    Plain,
    /// When all its flags are given. Most flags wins.
    Given(&'static [&'static str]),
    /// When the model's `_class_name` is this.
    Class(&'static str),
}

/// A task, or one way to run it.
pub struct Entry {
    /// The `--task` value.
    pub task: &'static str,
    pub pick: Pick,
    pub flags: &'static [Flag],
    /// Fills the slot, then serves until the process ends.
    pub open: fn(&Load, Settings) -> Result<(), String>,
}

/// The options, model name and slot an entry loads into.
pub struct Load<'a> {
    pub options: &'a Options,
    pub info: Info,
    pub slot: &'a Slot,
    /// The load's step, done when the slot fills.
    pub step: Mutex<Option<Step>>,
    /// Counts requests that ask for a profile, under `TESSERA_PROFILE=dynamic`.
    pub asks: Option<tessera_engine::Asks>,
    /// Where models take turns on the device.
    pub turns: &'a Arc<Turns>,
}

impl Load<'_> {
    /// The value of a flag an entry owns.
    pub fn given(&self, flag: &str) -> Option<&str> {
        self.options.given.get(flag).map(String::as_str)
    }

    /// Whether a switch is given.
    pub fn on(&self, flag: &str) -> bool {
        self.given(flag).is_some()
    }

    /// A count flag's value, else `default`.
    pub fn count(&self, flag: &str, default: usize) -> usize {
        let value = self.given(flag).and_then(|v| v.parse().ok());
        value.unwrap_or(default)
    }

    /// A seat for this model, sized by its files.
    pub fn seat(&self) -> Seated {
        let files = std::fs::read_dir(&self.options.model).into_iter().flatten();
        let sizes = files.filter_map(|f| f.ok()?.metadata().ok());
        let bytes = sizes.filter(|m| m.is_file()).map(|m| m.len()).sum();
        (self.turns).seat(&self.info.name, self.options.priority, bytes)
    }

    /// Fills the slot. Traits `with` leaves empty answer 404.
    pub fn fill(&self, tokens: Arc<dyn Tokens>, with: impl FnOnce(&mut Model)) {
        let mut model = Model {
            info: self.info.clone(),
            tokens,
            chat: None,
            generator: None,
            embed: None,
            image: None,
            score: None,
            strict: self.options.strict,
            asks: self.asks.clone(),
        };
        with(&mut model);
        let _ = self.slot.set(model);
        if let Some(step) = self.step.lock().ok().and_then(|mut s| s.take()) {
            step.finish();
        }
    }

    /// Fills the slot with one image model.
    pub fn picture<P: GenerateImage + Tokens + 'static>(&self, p: P) {
        let p = Arc::new(p);
        self.fill(p.clone(), |m| m.image = Some(p));
    }
}

/// The task names, in order, once each.
pub fn names() -> Vec<&'static str> {
    let mut names = Vec::new();
    for entry in ENTRIES {
        if !names.contains(&entry.task) {
            names.push(entry.task);
        }
    }
    names
}

/// The flag `name` that some entry owns.
pub fn owned(name: &str) -> Option<&'static Flag> {
    ENTRIES
        .iter()
        .flat_map(|e| e.flags)
        .find(|f| f.name == name)
}

/// The flags of `task`'s entries.
pub fn flags_of(task: &str) -> Vec<&'static Flag> {
    let mut flags = Vec::new();
    for entry in ENTRIES.iter().filter(|e| e.task == task) {
        flags.extend(entry.flags);
    }
    flags
}

/// Usage text for `--task` and every owned flag.
pub(crate) fn usage() -> String {
    let mut out = format!("[--task {}]", names().join("|"));
    let mut seen = Vec::new();
    for flag in ENTRIES.iter().flat_map(|e| e.flags) {
        if !seen.contains(&flag.name) {
            seen.push(flag.name);
            out += " ";
            out += &flag.usage();
        }
    }
    out
}

/// The pipeline class: the recipe's, else `model_index.json`'s.
fn class(options: &Options) -> Result<String, String> {
    const FILE: &str = "model_index.json";
    let index = options.model.join(FILE);
    let path = index.is_file().then_some(index.as_path());
    let class = (options.config).read(FILE, path, |k| k.need::<String>(&["_class_name"]));
    let task = options.task;
    class.map_err(|e| format!("--task {task} needs a class in {FILE} or the recipe:\n{e}"))
}

/// The entry the options pick.
pub(crate) fn pick(options: &Options) -> Result<&'static Entry, String> {
    let mut plain = None;
    let mut given: Option<(&[&str], &'static Entry)> = None;
    let mut tie = None;
    let mut classes = Vec::new();
    for entry in ENTRIES.iter().filter(|e| e.task == options.task) {
        match (entry.pick, given) {
            (Pick::Given(flags), _) if !flags.iter().all(|f| options.given.contains_key(f)) => {}
            (Pick::Given(flags), Some((best, _))) if best.len() > flags.len() => {}
            (Pick::Given(flags), Some((best, _))) if best.len() == flags.len() => {
                tie = Some((best[0], flags[0]));
            }
            (Pick::Given(flags), _) => {
                given = Some((flags, *entry));
                tie = None;
            }
            (Pick::Plain, _) => plain = Some(*entry),
            (Pick::Class(class), _) => classes.push((class, *entry)),
        }
    }
    if let Some((first, flag)) = tie {
        return Err(format!("{flag} cannot combine with {first}"));
    }
    if let Some((_, entry)) = given {
        return Ok(entry);
    }
    if let Some(entry) = plain {
        return Ok(entry);
    }
    let class = class(options)?;
    if let Some((_, entry)) = classes.iter().find(|(c, _)| *c == class) {
        return Ok(entry);
    }
    let runs: Vec<&str> = classes.iter().map(|(c, _)| *c).collect();
    let task = options.task;
    let runs = runs.join(", ");
    Err(format!(
        "{class} is not a --task {task} pipeline. This server runs: {runs}"
    ))
}

#[cfg(test)]
pub(crate) use generate::warm_rows;
