use crate::Error;
use crate::fabric::{Channel, Fabric, Lane, Peer, Port, parts};

fn ports(names: &[&str]) -> Vec<Port> {
    let port = |n: &&str| Port {
        device: n.to_string(),
        gid: 5,
    };
    names.iter().map(port).collect()
}

fn channel(peer: u32, ports: Vec<usize>, meets: usize) -> Channel {
    Channel { peer, ports, meets }
}

#[test]
fn mesh_wiring() {
    let fabric = Fabric::Mesh(ports(&["a", "b"]));
    let w = fabric.wiring(1, 3).unwrap();
    let want = [
        channel(0, vec![0, 1], 1),
        channel(1, vec![], 1),
        channel(2, vec![0, 1], 1),
    ];
    assert_eq!(
        (w.channels, w.sends, w.relay),
        (want.to_vec(), vec![2, 0], None)
    );
    assert_eq!(fabric.ports().len(), 2);
    assert!(matches!(fabric.wiring(0, 1), Err(Error::Fit(_))));
    assert!(matches!(
        Fabric::Mesh(vec![]).wiring(0, 2),
        Err(Error::Fit(_))
    ));
}

#[test]
fn ring_wiring() {
    let fabric = Fabric::Ring {
        prev: ports(&["p0", "p1"]),
        next: ports(&["n0", "n1"]),
    };
    let w = fabric.wiring(3, 4).unwrap();
    let want = [channel(2, vec![0, 1], 1), channel(0, vec![2, 3], 0)];
    assert_eq!(
        (w.channels, w.sends, w.relay),
        (want.to_vec(), vec![1, 0], Some((0, 1)))
    );
    assert_eq!(fabric.wiring(0, 3).unwrap().relay, None);
    assert_eq!(fabric.wiring(0, 2).unwrap().sends, [1]);
    assert!(matches!(fabric.wiring(0, 5), Err(Error::Fit(_))));
    let names: Vec<String> = fabric.ports().into_iter().map(|p| p.device).collect();
    assert_eq!(names, ["p0", "p1", "n0", "n1"]);
}

#[test]
fn lanes_split_in_sixteens() {
    assert_eq!(parts(100, 2).collect::<Vec<_>>(), [(0, 64), (64, 36)]);
    assert_eq!(parts(16, 2).collect::<Vec<_>>(), [(0, 16), (16, 0)]);
    assert_eq!(parts(0, 3).collect::<Vec<_>>(), [(0, 0), (0, 0), (0, 0)]);
}

#[test]
fn peer_round_trip() {
    let lane = |qpn| Lane {
        qpn,
        mtu: 5,
        rkeys: vec![7, 9, 11],
        gid: [qpn as u8; 16],
    };
    let peer = Peer {
        regions: vec![1 << 40, 1 << 41],
        channels: vec![vec![], vec![lane(1), lane(2)], vec![lane(3)]],
    };
    let bytes = peer.to_bytes();
    assert_eq!(Peer::from_bytes(&bytes), Ok(peer));
    assert!(Peer::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    assert!(Peer::from_bytes(&[bytes.clone(), vec![0]].concat()).is_err());
    let old = [&[0u8; 4][..], &bytes[4..]].concat();
    assert!(Peer::from_bytes(&old).unwrap_err().contains("version 0"));
}
