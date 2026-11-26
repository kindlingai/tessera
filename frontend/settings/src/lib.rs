//! Settings read once from the environment, passed down.

use serde::Deserialize;
use std::path::PathBuf;
use tessera_engine::{Asks, Attic, Backend, Memory, Pick, Reads, Settings};

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
    /// `true` lets weights page or load by stage.
    allow_weight_streaming: Option<bool>,
    /// `cached` or `direct`.
    reads: Option<String>,
    /// Launches per profile report, or `dynamic`.
    profile: Option<String>,
    kernels: Option<PathBuf>,
    /// Compiled image cache directory, or `off`.
    kernel_cache: Option<String>,
    /// Extra NVRTC or hiprtc flags, whitespace separated.
    nvrtc: Option<String>,
    dump_sites: Option<PathBuf>,
    device: Option<usize>,
    /// Display-carveout lender: `1`/`on`, `off`, or socket path.
    attic: Option<String>,
    /// Key file for the display-carveout lender.
    attic_key_file: Option<PathBuf>,
}

/// Reads `TESSERA_` variables. Bad values panic, empty ones are unset.
#[allow(clippy::disallowed_methods, reason = "the one sanctioned reader")]
pub fn read() -> Settings {
    let set = std::env::vars().filter(|(_, v)| !v.is_empty());
    let env: Env = envy::prefixed("TESSERA_")
        .from_iter(set)
        .unwrap_or_else(|e| panic!("TESSERA_ environment: {e}"));
    if let Some(device) = env.device {
        tessera_engine::select(device);
    }
    let backend = match env.backend.as_deref() {
        Some("host") => Pick::Host,
        Some("cpu") => Pick::Cpu,
        _ => tessera_engine::gpu().map_or(Pick::Host, Pick::Gpu),
    };
    let cache = |dir: &str| Some(std::env::home_dir()?.join(".cache/tessera").join(dir));
    let costs = env.costs.or_else(|| cache("costs"));
    let check = env.check.or_else(|| cache("check"));
    let recipes = env.recipes.or_else(|| cache("recipes"));
    let kernel_cache = match env.kernel_cache.as_deref() {
        Some("off") => None,
        Some(path) => Some(PathBuf::from(path)),
        None => cache("kernels"),
    };
    let nvrtc = env
        .nvrtc
        .as_deref()
        .map(|s| s.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();
    let (profile, asks) = match env.profile.as_deref() {
        None => (None, None),
        Some("dynamic") => (None, Some(Asks::default())),
        Some(n) => match n.parse::<u64>() {
            Ok(n) => ((n > 0).then_some(n), None),
            Err(e) => panic!("TESSERA_PROFILE={n}: want a count or dynamic: {e}"),
        },
    };
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
        weight_streaming: env.allow_weight_streaming.unwrap_or(false),
        reads: env.reads.map_or(Reads::Cached, |r| reads(&r)),
        hub,
        hf_token: token(&var, std::env::home_dir()),
        profile,
        asks,
        kernels: env.kernels,
        kernel_cache,
        nvrtc,
        dump_sites: env.dump_sites,
        device: env.device,
        attic: parse_attic(env.attic.as_deref(), env.attic_key_file, &var),
    }
}

/// The Hugging Face host.
pub const HUB: &str = "huggingface.co";

/// The hub URL of `file` in `repo` at `revision`.
pub fn resolve(repo: &str, revision: &str, file: &str) -> String {
    format!("https://{HUB}/{repo}/resolve/{revision}/{file}")
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

/// A `TESSERA_READS` value. Bad values panic.
fn reads(value: &str) -> Reads {
    match value {
        "cached" => Reads::Cached,
        "direct" => Reads::Direct,
        _ => panic!("TESSERA_READS={value}: want cached or direct"),
    }
}

/// Default `spark-attic` settings, honoring `SPARK_ATTIC_*` overrides.
#[allow(clippy::disallowed_methods, reason = "the one sanctioned reader")]
pub fn attic_default() -> Attic {
    let var = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    Attic {
        socket: var("SPARK_ATTIC_SOCK").unwrap_or_else(|| PathBuf::from(Attic::DEFAULT_SOCKET)),
        key_file: var("SPARK_ATTIC_KEY_FILE")
            .unwrap_or_else(|| PathBuf::from(Attic::DEFAULT_KEY_FILE)),
    }
}

/// Parses `TESSERA_ATTIC` and `TESSERA_ATTIC_KEY_FILE`.
fn parse_attic(
    value: Option<&str>,
    key_file: Option<PathBuf>,
    var: &dyn Fn(&str) -> Option<PathBuf>,
) -> Option<Attic> {
    let raw = value?;
    let socket = match raw {
        "0" | "off" | "false" => return None,
        "1" | "on" | "true" => {
            var("SPARK_ATTIC_SOCK").unwrap_or_else(|| PathBuf::from(Attic::DEFAULT_SOCKET))
        }
        other => PathBuf::from(other),
    };
    let key_file = key_file
        .or_else(|| var("SPARK_ATTIC_KEY_FILE"))
        .unwrap_or_else(|| PathBuf::from(Attic::DEFAULT_KEY_FILE));
    Some(Attic { socket, key_file })
}

#[cfg(test)]
mod lib_test;
