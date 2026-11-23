//! 16x16 sprites by masked diffusion, with a CLIP text tower.

use super::{Entry, Flag, Load, Pick};
use tessera_engine::Settings;
use tessera_tasks::Worker;
use tessera_tasks::paint::Painter;

pub static ENTRY: Entry = Entry {
    task: "image",
    pick: Pick::Class("TesseraSmallPipeline"),
    flags: &[Flag::text("--clip", "<dir>")],
    open,
};

fn open(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    let clip = load.given("--clip").map(Into::into);
    let (dir, module, config) = (o.model.clone(), o.module.clone(), o.config.clone());
    let painter = Worker::<Painter>::open(dir, clip, module, settings, o.slots, config);
    load.picture(painter.map_err(|e| e.to_string())?);
    Ok(())
}
