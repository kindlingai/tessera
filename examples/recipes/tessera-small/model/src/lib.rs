//! tessera-small's graph and tables, built inside the recipe's sandbox.

use sensai_guest::{encode, input, output};
use sensai_models::sprite::{self, Config};
use serde::Deserialize;

/// Inputs the host passes to `graph` and `tables`.
#[derive(Deserialize)]
struct Build {
    config: serde_json::Value,
    seqs: u64,
}

fn config(at: i32, len: i32) -> Result<Config, String> {
    // SAFETY: the host passes back what `alloc` returned.
    let bytes = unsafe { input(at, len) };
    let b: Build = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    Config::upstream(&b.config.to_string(), b.seqs)
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: i32) -> i32 {
    sensai_guest::alloc(len)
}

#[unsafe(no_mangle)]
pub extern "C" fn manifest() -> i64 {
    let manifest = serde_json::json!({"abi": sensai_guest::ABI, "graph": true});
    output(manifest.to_string().into_bytes())
}

/// The graph as JSON, else why the config is refused.
#[unsafe(no_mangle)]
pub extern "C" fn graph(at: i32, len: i32) -> i64 {
    let text = config(at, len).map_or_else(|e| e, |c| sprite::program(&c).to_json());
    output(text.into_bytes())
}

/// The `const.` tables, or nothing if the config is refused.
#[unsafe(no_mangle)]
pub extern "C" fn tables(at: i32, len: i32) -> i64 {
    let Ok(c) = config(at, len) else {
        return output(Vec::new());
    };
    let tables: Vec<_> = (sprite::constants(&c).into_iter())
        .map(|t| (t.path, t.data))
        .collect();
    output(encode(&tables))
}
