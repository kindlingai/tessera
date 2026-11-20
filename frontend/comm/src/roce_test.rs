use crate::roce::gids;
use crate::{port, ports};
use std::net::Ipv4Addr;

#[test]
fn gid_specs() {
    let p = port("mlx5_0:3").unwrap();
    assert_eq!((p.device.as_str(), p.gid), ("mlx5_0", 3));
    let two: Vec<u32> = ports("a:1,b:2").unwrap().iter().map(|p| p.gid).collect();
    assert_eq!(two, [1, 2]);
    assert!(port("a:x").is_err());
    assert!(port("no-such-device").is_err());
}

#[test]
fn ipv4_gids() {
    let root = tessera_testkit::scratch(&format!("sysfs-{}", std::process::id()));
    let put = |path: String, text: &str| {
        let path = root.join("roce0").join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    put("device/net/eth0/up".into(), "");
    for (i, low, kind) in [
        (2, "0a68:00b9", "RoCE v1"),
        (5, "c0a8:0105", "RoCE v2"),
        (4, "0a68:00b9", "RoCE v2"),
    ] {
        let gid = format!("0000:0000:0000:0000:0000:ffff:{low}");
        put(format!("ports/1/gids/{i}"), &gid);
        put(format!("ports/1/gid_attrs/types/{i}"), kind);
        put(format!("ports/1/gid_attrs/ndevs/{i}"), "eth0");
    }
    let want =
        [(4u32, [10, 104, 0, 185]), (5, [192, 168, 1, 5])].map(|(i, a)| (i, Ipv4Addr::from(a)));
    assert_eq!(gids(&root, "roce0").unwrap(), want);
}
