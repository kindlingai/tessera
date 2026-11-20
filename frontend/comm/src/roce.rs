//! RoCE ports by device name, with GID indexes from sysfs.

use sensai_engine::comm::{Error, Port};
use std::fs;
use std::path::PathBuf;

/// `device` or `device:gid`. Without a GID, the IPv4 one.
pub fn port(spec: &str) -> Result<Port, Error> {
    let (device, gid) = match spec.split_once(':') {
        Some((device, gid)) => {
            let gid = gid
                .parse()
                .map_err(|_| Error::Fit(format!("GID index `{gid}`")))?;
            (device, gid)
        }
        None => (spec, ipv4_gid(spec)?),
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

/// The device's RoCE v2 GID for an IPv4 address.
fn ipv4_gid(device: &str) -> Result<u32, Error> {
    let base = PathBuf::from("/sys/class/infiniband").join(device);
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
    for i in indexes {
        let gid = text(port.join(format!("gids/{i}")));
        let kind = text(port.join(format!("gid_attrs/types/{i}")));
        let ndev = text(port.join(format!("gid_attrs/ndevs/{i}")));
        let v4 = gid.starts_with("0000:0000:0000:0000:0000:ffff:");
        if kind.trim() == "RoCE v2" && netdevs.contains(&ndev.trim().to_string()) && v4 {
            return Ok(i);
        }
    }
    Err(Error::Fit(format!("{device} lacks a RoCE v2 IPv4 GID")))
}
