//! `teleport`: copies a tesserax build to other machines over ssh.

use crate::source::Stop;
use crate::version;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const SSH: [&str; 4] = ["-o", "ConnectTimeout=15", "-o", "BatchMode=yes"];
const ROOT: &str = "$HOME/.cache/tessera/bin";

/// `uname -sm`, platform and release asset of each machine kind.
pub const PLATFORMS: [(&str, &str, &str); 3] = [
    ("Darwin arm64", "aarch64-macos", "tesserax-darwin-arm64"),
    ("Linux aarch64", "aarch64-linux", "tesserax-linux-arm64"),
    ("Linux x86_64", "x86_64-linux", "tesserax-linux-x64"),
];

struct Args {
    from: Option<PathBuf>,
    root: String,
    hosts: Vec<String>,
}

/// A host after its probe: platform, asset and install path.
struct Host {
    platform: &'static str,
    asset: &'static str,
    dest: String,
}

/// A build source: a local file or a release URL.
enum Build {
    File(PathBuf),
    Url(String),
}

fn parse(rest: &[String]) -> Result<Args, Stop> {
    let mut args = Args {
        from: None,
        root: ROOT.into(),
        hosts: Vec::new(),
    };
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--from" => args.from = Some(it.next().ok_or(Stop::Usage)?.into()),
            "--root" => args.root = it.next().ok_or(Stop::Usage)?.clone(),
            _ if arg.starts_with('-') => return Err(Stop::Usage),
            _ => args.hosts.push(arg.clone()),
        }
    }
    if args.hosts.is_empty() {
        return Err(Stop::Usage);
    }
    Ok(args)
}

pub fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

/// Runs a sh script on `host`, or here for `local`.
fn shell(host: &str, script: &str) -> Result<String, String> {
    let mut cmd = Command::new(if host == "local" { "sh" } else { "ssh" });
    if host == "local" {
        cmd.arg("-c").arg(script);
    } else {
        cmd.args(SSH)
            .arg(host)
            .arg(format!("sh -c {}", quote(script)));
    }
    let out = cmd.output().map_err(|e| format!("{host}: {e}"))?;
    if out.status.success() {
        return Ok(String::from_utf8_lossy(&out.stdout).trim().to_string());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    let last = err.trim().lines().last().unwrap_or("failed").to_string();
    Err(format!("{host}: {last}"))
}

/// A file's SHA-256 on `host`, empty when missing.
fn hash(host: &str, path: &str) -> Result<String, String> {
    let f = quote(path);
    let sum = format!("(sha256sum {f} 2>/dev/null || shasum -a 256 {f} 2>/dev/null)");
    shell(host, &format!("{sum} | cut -c1-64"))
}

pub fn platform(uname: &str) -> Option<(&'static str, &'static str)> {
    let found = PLATFORMS.iter().find(|p| p.0 == uname.trim())?;
    Some((found.1, found.2))
}

fn probe(host: &str, root: &str) -> Result<Host, String> {
    let out = shell(host, &format!("uname -sm; echo \"{root}\""))?;
    let (uname, root) = out.split_once('\n').ok_or(format!("{host}: no uname"))?;
    let (platform, asset) = platform(uname).ok_or(format!("{host}: no build for {uname}"))?;
    let dest = format!("{}/{}/tesserax", root.trim(), version::id());
    Ok(Host {
        platform,
        asset,
        dest,
    })
}

/// The release asset URL and its sha256 from `SHA256SUMS`.
fn release(asset: &str) -> Result<(Build, String), String> {
    let tag = format!("v{}", env!("CARGO_PKG_VERSION"));
    let base = format!("{}/releases/download/{tag}", env!("CARGO_PKG_REPOSITORY"));
    let sums = Command::new("curl")
        .args(["-fsSL", "--max-time", "60", &format!("{base}/SHA256SUMS")])
        .output()
        .map_err(|e| e.to_string())?;
    for line in String::from_utf8_lossy(&sums.stdout).lines() {
        if let [sum, name] = line.split_whitespace().collect::<Vec<_>>()[..]
            && name == asset
        {
            return Ok((Build::Url(format!("{base}/{asset}")), sum.to_string()));
        }
    }
    Err(format!("no {asset} in release {tag}: pass --from <dir>"))
}

/// This binary, a `--from` file, or a release asset.
fn binary(platform: &str, asset: &str, from: Option<&Path>) -> Result<(Build, String), String> {
    let path = if platform == version::platform() {
        std::env::current_exe().map_err(|e| e.to_string())?
    } else {
        match from.map(|d| d.join(asset)).filter(|p| p.is_file()) {
            Some(path) => path,
            None => return release(asset),
        }
    };
    let sum = hash("local", &path.to_string_lossy())?;
    Ok((Build::File(path), sum))
}

/// Puts a build at `part` on `host`.
fn upload(host: &str, build: &Build, part: &str) -> Result<(), String> {
    let local = match build {
        Build::Url(url) => {
            let get = format!(
                "curl -fsSL --max-time 900 -o {} {}",
                quote(part),
                quote(url)
            );
            return shell(host, &get).map(drop);
        }
        Build::File(local) => local,
    };
    let mut copy = Command::new(if host == "local" { "cp" } else { "scp" });
    if host != "local" {
        copy.arg("-q").args(SSH);
    }
    let to = if host == "local" {
        part.to_string()
    } else {
        format!("{host}:{part}")
    };
    if !copy.arg(local).arg(to).status().is_ok_and(|s| s.success()) {
        return Err(format!("{host}: copy failed"));
    }
    Ok(())
}

/// Installs a build at `dest` unless already there.
fn place(host: &str, build: &Build, dest: &str, want: &str) -> Result<&'static str, String> {
    if hash(host, dest)? == want {
        return Ok("present");
    }
    let part = format!("{dest}.part");
    let (d, p) = (quote(dest), quote(&part));
    shell(host, &format!("mkdir -p \"$(dirname {d})\" && rm -f {p}"))?;
    upload(host, build, &part)?;
    if hash(host, &part)? != want {
        return Err(format!("{host}: sha256 differs after copy"));
    }
    shell(host, &format!("chmod 755 {p} && mv -f {p} {d}"))?;
    Ok("copied")
}

/// Installs one probed host and checks its `--version` line.
fn send(name: &str, host: &Host, build: &Build, want: &str) -> Result<String, String> {
    let state = place(name, build, &host.dest, want)?;
    let line = shell(name, &format!("{} --version", quote(&host.dest)))?;
    if version::COMMIT.is_some_and(|c| !line.contains(c)) {
        return Err(format!("{name}: commit differs: {line}"));
    }
    let (platform, dest) = (host.platform, &host.dest);
    Ok(format!("{name}  {platform}  {state}  {dest}  {line}"))
}

fn each<I: Sync, T: Send>(items: &[I], f: impl Fn(&I) -> T + Sync) -> Vec<T> {
    std::thread::scope(|s| {
        let jobs: Vec<_> = items.iter().map(|i| s.spawn(|| f(i))).collect();
        let done = jobs.into_iter().map(|j| j.join());
        done.map(|r| r.expect("teleport thread")).collect()
    })
}

pub fn run(rest: &[String]) -> Result<(), Stop> {
    let args = parse(rest)?;
    let probes = each(&args.hosts, |h| probe(h, &args.root));
    let mut builds = BTreeMap::new();
    for host in probes.iter().flatten() {
        if !builds.contains_key(host.asset) {
            let found = binary(host.platform, host.asset, args.from.as_deref());
            builds.insert(host.asset, found);
        }
    }
    let pairs: Vec<_> = args.hosts.iter().zip(&probes).collect();
    let lines = each(&pairs, |(name, probed)| {
        let host = probed.as_ref().map_err(String::clone)?;
        let (build, want) = builds[host.asset].as_ref().map_err(String::clone)?;
        send(name, host, build, want)
    });
    let mut failed = 0;
    for line in lines {
        match line {
            Ok(text) => println!("{text}"),
            Err(e) => {
                failed += 1;
                println!("error {e}");
            }
        }
    }
    if failed > 0 {
        let total = args.hosts.len();
        return Err(Stop::Failed(format!("{failed} of {total} hosts failed")));
    }
    Ok(())
}
