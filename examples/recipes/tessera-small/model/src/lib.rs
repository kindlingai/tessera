//! tessera-small's graph and tables, built inside the recipe's sandbox.

use tessera_guest::{Build, Table};
use tessera_models::sprite::{self, Config};

fn config(b: &Build) -> Result<Config, String> {
    Config::upstream(&b.config.to_string(), b.seqs)
}

fn program(b: &Build) -> Result<String, String> {
    Ok(sprite::program(&config(b)?).to_json())
}

fn constants(b: &Build) -> Result<Vec<Table>, String> {
    let tables = sprite::constants(&config(b)?).into_iter();
    Ok(tables.map(|t| (t.path, t.data)).collect())
}

tessera_guest::exports! {
    graph: program,
    tables: constants,
}
