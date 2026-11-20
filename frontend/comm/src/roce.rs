//! RoCE ports by device name, with GID indexes from sysfs.

use std::fs;
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use tessera_engine::comm::{Error, Fabric, Port};

const SYSFS: &str = "/sys/class/infiniband";

/// `device` or `device:gid`. Without a GID, the first IPv4 one.
pub fn port(spec: &str) -> Result<Port, Error> {
    let (device, gid) = match spec.split_once(':') {
        Some((device, gid)) => {
            let gid = gid
                .parse()
                .map_err(|_| Error::Fit(format!("GID index `{gid}`")))?;
            (device, gid)
        }
        None => (spec, pick(spec, &gids(Path::new(SYSFS), spec)?)?),
    };
    Ok(Port {
        device: device.to_string(),
        gid,
    })
}

/// Comma-separated specs, in lane order.
pub fn ports(specs: &str) -> Result<Vec<Port>, Error> {
    specs.split(',').map(port).collect()
}

/// Each port as `device:gid address`, for the log.
pub fn describe(fabric: &Fabric) -> String {
    let mut out = Vec::new();
    for p in fabric.ports() {
        let path = Path::new(SYSFS).join(&p.device).join("ports/1/gids");
        let addr = fs::read_to_string(path.join(p.gid.to_string())).ok();
        match addr.as_deref().and_then(ipv4) {
            Some(addr) => out.push(format!("{}:{} {addr}", p.device, p.gid)),
            None => out.push(format!("{}:{}", p.device, p.gid)),
        }
    }
    out.join(", ")
}

/// The first GID. Lists the others, if any.
fn pick(device: &str, found: &[(u32, Ipv4Addr)]) -> Result<u32, Error> {
    let Some(&(first, _)) = found.first() else {
        return Err(Error::Fit(format!("{device} lacks a RoCE v2 IPv4 GID")));
    };
    if found.len() > 1 {
        let all: Vec<String> = found.iter().map(|(i, a)| format!("{i} ({a})")).collect();
        eprintln!(
            "{device}: RoCE v2 IPv4 GIDs {}. Using {first}. Pick with {device}:GID",
            all.join(", ")
        );
    }
    Ok(first)
}

/// An IPv4 address from a GID in sysfs form.
fn ipv4(gid: &str) -> Option<Ipv4Addr> {
    let low = gid.trim().strip_prefix("0000:0000:0000:0000:0000:ffff:")?;
    let (hi, lo) = low.split_once(':')?;
    let word = u32::from_str_radix(hi, 16).ok()? << 16 | u32::from_str_radix(lo, 16).ok()?;
    Some(Ipv4Addr::from(word))
}

/// The device's RoCE v2 IPv4 GIDs under `root`, by index.
pub(crate) fn gids(root: &Path, device: &str) -> Result<Vec<(u32, Ipv4Addr)>, Error> {
    let base = root.join(device);
    let names = |dir: PathBuf| -> Result<Vec<String>, Error> {
        let mut out = Vec::new();
        for entry in fs::read_dir(dir)? {
            out.push(entry?.file_name().to_string_lossy().into_owned());
        }
        Ok(out)
    };
    let netdevs = names(base.join("device/net"))?;
    let port = base.join("ports/1");
    let mut indexes: Vec<u32> = Vec::new();
    for name in names(port.join("gids"))? {
        indexes.extend(name.parse::<u32>().ok());
    }
    indexes.sort_unstable();
    let text = |path: PathBuf| fs::read_to_string(path).unwrap_or_default();
    let mut found = Vec::new();
    for i in indexes {
        let kind = text(port.join(format!("gid_attrs/types/{i}")));
        let ndev = text(port.join(format!("gid_attrs/ndevs/{i}")));
        let Some(addr) = ipv4(&text(port.join(format!("gids/{i}")))) else {
            continue;
        };
        if kind.trim() == "RoCE v2" && netdevs.contains(&ndev.trim().to_string()) {
            found.push((i, addr));
        }
    }
    Ok(found)
}
