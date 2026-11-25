//! `serve`: answer OpenAI-shaped requests.

use crate::source::Source;
use tessera_tasks::Error;
use tessera_tasks::task::{Run, Sink, Task};

/// Start or run a text model as an OpenAI-compatible server.
#[derive(Task)]
#[task(kind = "generate", example = "start {model} as a server")]
pub struct Serve {
    /// Flags for the server and the model load.
    #[arg(extra)]
    pub flags: Vec<String>,
}

impl Run<Source> for Serve {
    fn run(self, source: Source, _: &mut dyn Sink) -> Result<(), Error> {
        let options = source.options(&self.flags)?;
        Ok(tessera_serve::start::run(options)?)
    }
}
