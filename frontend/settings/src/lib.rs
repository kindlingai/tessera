//! Settings read once from the environment, passed down.

use sensai_engine::{Backend, Memory, Pick, Settings};
use serde::Deserialize;
use std::path::PathBuf;

/// The `TESSERA_` variables.
#[derive(Deserialize)]
struct Env {
    backend: Option<String>,
    lanes: Option<usize>,
    costs: Option<PathBuf>,
    check: Option<PathBuf>,
    names: Option<PathBuf>,
    recipes: Option<PathBuf>,
    emulate: Option<Backend>,
    verbs: Option<String>,
    /// Bytes, maybe suffixed K, M, G or T, or `off`.
    memory: Option<String>,
    /// Launches per profile report.
    profile: Option<u64>,
    kernels: Option<PathBuf>,
}

/// Reads the `TESSERA_` variables. Bad values panic.
#[allow(clippy::disallowed_methods, reason = "the one sanctioned reader")]
pub fn read() -> Settings {
    let env: Env = envy::prefixed("TESSERA_")
        .from_env()
        .unwrap_or_else(|e| panic!("TESSERA_ environment: {e}"));
    let backend = match env.backend.as_deref() {
        Some("host") => Pick::Host,
        Some("cpu") => Pick::Cpu,
        _ => sensai_engine::gpu().map_or(Pick::Host, Pick::Gpu),
    };
    let cache = |dir: &str| Some(std::env::home_dir()?.join(".cache/tessera").join(dir));
    let costs = env.costs.or_else(|| cache("costs"));
    let check = env.check.or_else(|| cache("check"));
    let recipes = env.recipes.or_else(|| cache("recipes"));
    let var = |k: &str| std::env::var_os(k).map(PathBuf::from);
    let hub = (var("HF_HUB_CACHE"))
        .or_else(|| var("HF_HOME").map(|h| h.join("hub")))
        .or_else(|| Some(std::env::home_dir()?.join(".cache/huggingface/hub")));
    Settings {
        backend,
        host_lanes: env.lanes.unwrap_or(32),
        costs,
        check,
        names: env.names,
        recipes,
        emulate: env.emulate,
        verbs: env.verbs,
        memory: env.memory.map_or(Memory::Device, |m| memory(&m)),
        hub,
        hf_token: token(&var, std::env::home_dir()),
        profile: env.profile.filter(|n| *n > 0),
        kernels: env.kernels,
    }
}

/// The proxy for `host`, or `None` to fetch directly.
#[allow(clippy::disallowed_methods, reason = "the one sanctioned reader")]
pub fn proxy(host: &str) -> Option<String> {
    let vars = |up: &str, low: &str| {
        std::env::var(up)
            .or_else(|_| std::env::var(low))
            .ok()
            .filter(|v| !v.trim().is_empty())
    };
    let (https, http, no) = (
        vars("HTTPS_PROXY", "https_proxy"),
        vars("HTTP_PROXY", "http_proxy"),
        vars("NO_PROXY", "no_proxy"),
    );
    select(&https, &http, &no, host).map(str::to_owned)
}

/// One of `https` or `http`, unless `no_proxy` exempts `host`.
fn select<'a>(
    https: &'a Option<String>,
    http: &'a Option<String>,
    no_proxy: &Option<String>,
    host: &str,
) -> Option<&'a str> {
    for name in no_proxy.as_deref().unwrap_or_default().split(',') {
        let name = name.trim().trim_start_matches('.');
        if name == "*" || host == name || host.ends_with(&format!(".{name}")) {
            return None;
        }
    }
    https.as_deref().or(http.as_deref())
}

/// `HF_TOKEN`, else the token file the Hugging Face tools write.
fn token(var: &dyn Fn(&str) -> Option<PathBuf>, home: Option<PathBuf>) -> Option<String> {
    let text = match var("HF_TOKEN") {
        Some(t) => t.to_string_lossy().into_owned(),
        None => {
            let home = var("HF_HOME").or_else(|| Some(home?.join(".cache/huggingface")));
            std::fs::read_to_string(home?.join("token")).ok()?
        }
    };
    Some(text.trim().to_string()).filter(|t| !t.is_empty())
}

/// A `TESSERA_MEMORY` value. Bad values panic.
fn memory(value: &str) -> Memory {
    if value == "off" {
        return Memory::Off;
    }
    let units = [("K", 10), ("M", 20), ("G", 30), ("T", 40)];
    let (digits, shift) = units
        .iter()
        .find_map(|(u, shift)| Some((value.strip_suffix(u)?, *shift)))
        .unwrap_or((value, 0));
    match digits.parse::<u64>() {
        Ok(n) => Memory::Bytes(n << shift),
        Err(e) => panic!("TESSERA_MEMORY={value}: {e}"),
    }
}

#[cfg(test)]
mod lib_test;
