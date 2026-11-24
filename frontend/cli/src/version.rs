//! `tesserax --version`: version, commit, platform and GPU features.

/// `TESSERA_COMMIT` at build time wins over the checkout's commit.
pub const COMMIT: Option<&str> = match option_env!("TESSERA_COMMIT") {
    Some(commit) => Some(commit),
    None => option_env!("TESSERA_GIT_COMMIT"),
};

/// The `<arch>-<os>` pair that `teleport` compares with `uname -sm`.
pub fn platform() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

/// The folder name a teleported build lives in.
pub fn id() -> String {
    match COMMIT {
        Some(commit) => commit.chars().take(12).collect(),
        None => format!("v{}", env!("CARGO_PKG_VERSION")),
    }
}

pub fn line() -> String {
    let mut features = Vec::new();
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "rocm") {
        features.push("rocm");
    }
    if features.is_empty() {
        features.push("-");
    }
    format!(
        "tesserax {} commit={} platform={} features={}",
        env!("CARGO_PKG_VERSION"),
        COMMIT.unwrap_or("unknown"),
        platform(),
        features.join(","),
    )
}
