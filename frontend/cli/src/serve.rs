//! `serve`: answer OpenAI-shaped requests.

use crate::source::{Source, Stop};

pub fn run(source: Source, rest: &[String]) -> Result<(), Stop> {
    let options = source.options(rest)?;
    sensai_serve::start::run(options).map_err(Stop::Failed)
}
