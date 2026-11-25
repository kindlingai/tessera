//! Commands and tasks: one list for CLI and router.

mod route;
#[cfg(test)]
mod route_test;

pub use route::run as route;

use crate::fetch::Fetch;
use crate::serve::Serve;
use crate::source::{Origin, Source, Stop};
use std::io::{BufRead, IsTerminal, Write};
use std::path::Path;
use std::time::Duration;
use tessera_engine::progress::Progress;
use tessera_serve::start::{self, Options};
use tessera_serve::{Model, Slot};
use tessera_tasks::task::{Ask, Chat, Embed, Image, Run, Sink, Tag, Task, Usage, Value};
use tessera_tasks::{Error, Tagger};

/// The word before a task's name.
const TASK: &str = "task";

/// Runs an entry on an origin and words.
pub type Call = fn(Option<Origin>, &[String], &mut dyn Sink) -> Result<(), Stop>;

/// One command or task, for the CLI and the router.
pub struct Entry {
    /// [`TASK`] for a task, empty for a command.
    pub group: &'static str,
    pub name: &'static str,
    /// The catalog kind it runs on. `None` is any.
    pub kind: Option<&'static str>,
    pub about: &'static str,
    /// A request it answers. Only these go to the router.
    pub example: Option<&'static str>,
    pub schema: fn() -> Value,
    /// The arguments for `example`.
    pub example_args: fn() -> Value,
    /// A tool call's arguments as command-line words.
    pub tool_args: fn(&Value) -> Result<Vec<String>, Usage>,
    pub usage: fn() -> String,
    /// Whether the arguments parse. Commands read no stdin.
    pub accepts: fn(&[String]) -> bool,
    /// Runs with output to the sink.
    pub call: Call,
    /// Whether it keeps running, as a server does.
    pub stays: bool,
}

impl Entry {
    /// The command words, after the model flags.
    pub fn words(&self) -> Vec<String> {
        let words = [self.group, self.name]
            .into_iter()
            .filter(|w| !w.is_empty());
        words.map(String::from).collect()
    }
}

const fn entry<T: Task, O: Target<T>>(group: &'static str, stays: bool) -> Entry {
    Entry {
        group,
        name: T::NAME,
        kind: T::KIND,
        about: T::ABOUT,
        example: T::EXAMPLE,
        schema: T::schema,
        example_args: T::example,
        tool_args: T::tool_args,
        usage: usage::<T, O>,
        accepts: accepts::<T, O>,
        call: call_entry::<T, O>,
        stays,
    }
}

/// Commands first, then tasks. The router keeps this order.
pub static ENTRIES: [Entry; 7] = [
    entry::<Serve, OnSource>("", true),
    entry::<Fetch, OnSource>("", false),
    entry::<Ask, OnModel>(TASK, false),
    entry::<Chat, OnModel>(TASK, false),
    entry::<Image, OnModel>(TASK, false),
    entry::<Embed, OnModel>(TASK, false),
    entry::<Tag, OnTagger>(TASK, false),
];

/// The entry that `group` and `name` name.
pub fn find(group: &str, name: &str) -> Option<&'static Entry> {
    ENTRIES.iter().find(|e| e.group == group && e.name == name)
}

/// A task's run target, and how the CLI opens one.
trait Target<T: Task> {
    /// Whether extra `flag` takes a value. `None` refuses the flag.
    fn takes(flag: &str) -> Option<bool>;
    /// The extra flags' usage.
    fn extra() -> String;
    fn run(task: T, source: Source, sink: &mut dyn Sink) -> Result<(), Error>;
}

/// A model the serve loader fills.
struct OnModel;

impl<T: Task + for<'a> Run<&'a Model>> Target<T> for OnModel {
    fn takes(flag: &str) -> Option<bool> {
        let owned = tessera_serve::task::flags_of(T::KIND?);
        owned
            .iter()
            .find(|f| f.name == flag)
            .map(|f| !f.value.is_empty())
    }

    fn extra() -> String {
        let mut out = Vec::new();
        for flag in tessera_serve::task::flags_of(T::KIND.unwrap_or_default()) {
            if !T::FIELDS.iter().any(|f| f.flag == flag.name) {
                out.push(flag.usage());
            }
        }
        out.join(" ")
    }

    fn run(task: T, source: Source, sink: &mut dyn Sink) -> Result<(), Error> {
        let mut options = source.options(&task.load_flags())?;
        if let Some(kind) = T::KIND {
            options.flag("--task", kind.into()).ok_or(Stop::Usage)?;
        }
        options.slots = 1;
        let slot = load(options);
        task.run(model(&slot), sink)
    }
}

/// A tagger opened from the model's files.
struct OnTagger;

impl<T: Task + for<'a> Run<&'a mut Tagger>> Target<T> for OnTagger {
    fn takes(_: &str) -> Option<bool> {
        None
    }

    fn extra() -> String {
        String::new()
    }

    fn run(task: T, source: Source, sink: &mut dyn Sink) -> Result<(), Error> {
        let options = source.options(&[])?;
        let settings = tessera_frontend_settings::read();
        let mut tagger = Tagger::open(&options.model, &settings)?;
        task.run(&mut tagger, sink)
    }
}

/// The opened recipe or model, for commands.
struct OnSource;

impl<T: Task + Run<Source>> Target<T> for OnSource {
    fn takes(flag: &str) -> Option<bool> {
        Some(!Options::default().switch(flag))
    }

    fn extra() -> String {
        "[flags]".into()
    }

    fn run(task: T, source: Source, sink: &mut dyn Sink) -> Result<(), Error> {
        task.run(source, sink)
    }
}

fn usage<T: Task, O: Target<T>>() -> String {
    T::usage(&O::extra())
}

fn accepts<T: Task, O: Target<T>>(args: &[String]) -> bool {
    T::parse(args, &O::takes).is_ok()
}

fn call_entry<T: Task, O: Target<T>>(
    origin: Option<Origin>,
    args: &[String],
    sink: &mut dyn Sink,
) -> Result<(), Stop> {
    let task = T::parse(args, &O::takes);
    let task = task.map_err(|e| Stop::Tasks(Some(format!("{}: {e}", T::NAME))))?;
    let source = Source::open_or_kind(origin, T::KIND, sink.progress())?;
    O::run(task, source, sink).map_err(stop)
}

/// A task's error as a `Stop`.
fn stop(e: Error) -> Stop {
    let e = match e.downcast::<Stop>() {
        Ok(stop) => return *stop,
        Err(e) => e,
    };
    match e.downcast::<Usage>() {
        Ok(usage) => Stop::Tasks(Some(usage.0)),
        Err(e) => Stop::Failed(e.to_string()),
    }
}

/// A stop as words for the person or the model.
pub fn describe(stop: &Stop) -> String {
    match stop {
        Stop::Failed(why) | Stop::Tasks(Some(why)) => why.clone(),
        _ => "the task refused its arguments".into(),
    }
}

/// Every command and task: its usage and what it does.
pub fn list() -> String {
    let mut out = String::from("commands and tasks:");
    for e in &ENTRIES {
        let mut words = e.words();
        words.pop();
        words.push((e.usage)());
        let usage = words.join(" ");
        match usage.len() < 36 {
            true => out += &format!("\n  {usage:<36}{}", e.about),
            false => out += &format!("\n  {usage}\n  {:<36}{}", "", e.about),
        }
    }
    out
}

pub fn run(origin: Option<Origin>, rest: &[String]) -> Result<(), Stop> {
    let Some((name, args)) = rest.split_first() else {
        return Err(Stop::Tasks(None));
    };
    let unknown = || Stop::Tasks(Some(format!("unknown task {name}")));
    let entry = find(TASK, name).ok_or_else(unknown)?;
    // Chat without a model flag may run tasks each turn.
    if origin.is_none() && name == Chat::NAME {
        return route::chat(args);
    }
    (entry.call)(origin, args, &mut Print::default())
}

/// Starts the load and waits for the model.
pub(crate) fn load(options: Options) -> Slot {
    let slot = start::spawn(options);
    while slot.get().is_none() {
        std::thread::sleep(Duration::from_millis(20));
    }
    slot
}

fn model(slot: &Slot) -> &Model {
    slot.get().expect("the model loaded")
}

/// Prints text as it comes, and results a line each.
#[derive(Default)]
pub struct Print {
    /// Text ended without a newline.
    open: bool,
}

impl Print {
    fn end_line(&mut self) {
        if std::mem::take(&mut self.open) {
            println!();
        }
    }
}

impl Sink for Print {
    fn text(&mut self, piece: &str) {
        let mut out = std::io::stdout();
        let _ = write!(out, "{piece}");
        let _ = out.flush();
        if !piece.is_empty() {
            self.open = !piece.ends_with('\n');
        }
    }

    fn record(&mut self, record: Value) {
        self.end_line();
        println!("{record}");
    }

    fn file(&mut self, path: &Path) {
        self.end_line();
        println!("{}", path.display());
    }

    fn note(&mut self, text: &str) {
        self.end_line();
        eprintln!("tesserax: {text}");
    }

    fn turn(&mut self) -> Option<String> {
        self.end_line();
        let stdin = std::io::stdin();
        if stdin.is_terminal() {
            eprint!("> ");
            let _ = std::io::stderr().flush();
        }
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(n) if n > 0 => Some(line),
            _ => None,
        }
    }

    fn progress(&self) -> Progress {
        crate::progress::stderr()
    }
}

impl Drop for Print {
    fn drop(&mut self) {
        self.end_line();
    }
}
