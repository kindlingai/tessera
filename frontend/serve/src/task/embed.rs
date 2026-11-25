//! Text vectors from an embedding model.

use super::{Entry, Load, Pick};
use std::sync::Arc;
use tessera_engine::Settings;
use tessera_tasks::Threaded;
use tessera_tasks::api::Embed;

pub static ENTRY: Entry = Entry {
    task: "embed",
    pick: Pick::Plain,
    flags: &[],
    open,
};

fn open(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    let e = Threaded::open(o.model.clone(), settings, o.config.clone(), o.progress.clone());
    let e = Arc::new(e.map_err(|e| e.to_string())?);
    load.fill(e.clone(), |m| m.embed = Some(e as Arc<dyn Embed>));
    Ok(())
}
