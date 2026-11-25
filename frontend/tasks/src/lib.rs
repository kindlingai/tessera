//! Text in, answers out: tasks adapt text to a model.

// The derive names this crate by path, even inside it.
extern crate self as tessera_tasks;

pub mod api;
pub mod embed;
#[cfg(test)]
mod embed_test;
pub mod features;
#[cfg(test)]
mod features_test;
pub mod flow;
#[cfg(test)]
mod flow_test;
pub mod generate;
#[cfg(test)]
mod generate_test;
pub mod guided;
pub mod klein;
#[cfg(test)]
mod klein_test;
pub mod paint;
pub mod pictures;
#[cfg(test)]
mod pictures_test;
pub mod png;
pub mod scopes;
#[cfg(test)]
mod scopes_test;
pub mod tag;
pub mod task;
#[cfg(test)]
mod task_test;
mod text;
#[cfg(test)]
mod text_test;
pub mod worker;

pub use embed::{Embedder, Threaded};
pub use generate::{Generate, Job, Queued};
pub use guided::{Guides, guides};
pub use tag::Tagger;
pub(crate) use text::{Spool, Stops};
pub use worker::Worker;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
