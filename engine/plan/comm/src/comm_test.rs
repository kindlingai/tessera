use crate::socket_link::SocketLink;
use crate::{
    Blocking, Error, Link, Memory, Put, Shm, Sockets, Span, Transport, exchange_regions, summed,
};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::net::TcpStream;
use std::time::Duration;
use tessera_testkit::{TCP_TIMEOUT, tcp_pair};

#[test]
fn rank_order() {
    let parts = [[1.0f32, 2.0], [10.0, 20.0], [100.0, 200.0]];
    assert_eq!(summed(parts.iter().map(|p| &p[..]), |x| x), [111.0, 222.0]);
}

/// A sized scratch file, open for reading and writing.
fn sized(name: &str, len: usize) -> File {
    let path = tessera_testkit::scratch(&format!("{name}-{}", std::process::id()));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap();
    file.set_len(len as u64).unwrap();
    file
}

#[test]
fn handles_must_fit() {
    // Each rank pads 224 bytes to one 64 KiB block.
    assert_eq!(Shm::room(3, 8), 3 << 16);
    let short = sized("short", Shm::room(2, 8) - 1);
    assert!(matches!(Shm::over(0, 2, short, 8), Err(Error::Fit(_))));
    let late = sized("late", Shm::room(2, 8));
    assert!(matches!(Shm::over(2, 2, late, 8), Err(Error::Fit(_))));
    let own = Sockets::over(0, vec![None, None]);
    assert!(matches!(own, Err(Error::Fit(_))));
}

/// Three ranks on threads, each sending its rank's byte, twice.
fn rounds<T: Transport + Send>(ranks: Vec<T>) {
    let seen: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
        let runs: Vec<_> = (ranks.into_iter().enumerate())
            .map(|(r, mut t)| {
                s.spawn(move || {
                    assert_eq!((t.rank(), t.ranks()), (r as u32, 3));
                    let first = t.exchange(&[r as u8; 3]).unwrap();
                    // A second round reuses the handles.
                    let second = t.exchange(&[r as u8 + 10]).unwrap();
                    [first, second].concat()
                })
            })
            .collect();
        runs.into_iter().map(|t| t.join().unwrap()).collect()
    });
    let want = vec![
        vec![0, 0, 0],
        vec![1, 1, 1],
        vec![2, 2, 2],
        vec![10],
        vec![11],
        vec![12],
    ];
    for rank in &seen {
        assert_eq!(*rank, want);
    }
}

/// Every pair of three ranks connected on the loopback.
fn connected() -> Vec<Vec<Option<TcpStream>>> {
    let mut peers: Vec<Vec<Option<TcpStream>>> = (0..3).map(|_| vec![None, None, None]).collect();
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        let (caller, answer) = tcp_pair().unwrap();
        peers[a][b] = Some(caller);
        peers[b][a] = Some(answer);
    }
    peers
}

#[test]
fn sockets_exchange() {
    let ranks = (connected().into_iter().enumerate())
        .map(|(r, peers)| Sockets::over(r as u32, peers).unwrap())
        .collect();
    rounds(ranks);
}

#[test]
fn memory_exchange() {
    let wait = std::time::Duration::from_secs(10);
    let regions = exchange_regions(3, 16);
    for links in [Memory::ranks(3, &regions), Memory::deferred(3, &regions)] {
        rounds(
            links
                .into_iter()
                .map(|l| Blocking::new(l, 16, wait))
                .collect(),
        );
    }
}

#[test]
fn socket_link_exchange() {
    let regions = exchange_regions(3, 64);
    let link = |(r, peers)| SocketLink::over(r as u32, peers, &regions).unwrap();
    let links = connected().into_iter().enumerate().map(link);
    rounds(links.map(|l| Blocking::new(l, 64, TCP_TIMEOUT)).collect());
}

#[test]
fn socket_link_bad_frames() {
    let header = |len: u64| [0u64, 0, len].map(u64::to_le_bytes).concat();
    for (bytes, close) in [
        (header(65), false),
        ([header(8), vec![1; 4]].concat(), true),
    ] {
        let mut pair = connected();
        let mut raw = pair[1][0].take().unwrap();
        let mut link = SocketLink::over(0, vec![None, pair[0][1].take()], &[64]).unwrap();
        raw.write_all(&bytes).unwrap();
        if close {
            drop(raw);
        }
        let failed = (0..2_000).find_map(|_| {
            std::thread::sleep(Duration::from_millis(1));
            link.poll(&mut Vec::new()).err()
        });
        assert!(matches!(failed, Some(Error::Link(_))), "{failed:?}");
    }
}

#[test]
fn socket_link_reset_drains() {
    let mut pair = connected();
    let links: Vec<_> = (0..2)
        .map(|r| {
            let peers = vec![pair[r][0].take(), pair[r][1].take()];
            SocketLink::over(r as u32, peers, &[4096, 16]).unwrap()
        })
        .collect();
    let landed: Vec<Vec<u8>> = std::thread::scope(|s| {
        let runs: Vec<_> = (links.into_iter().enumerate())
            .map(|(r, mut link)| {
                s.spawn(move || {
                    let mine = Span::new(0, 0, 4096);
                    link.write(mine, &[r as u8 + 1; 4096]).unwrap();
                    let put = Put {
                        to: 1 - r as u32,
                        from: mine,
                        into: mine,
                        flag: Span::new(1, 0, 16),
                        words: [1, 1],
                        tag: 7,
                    };
                    for _ in 0..8 {
                        link.put(put).unwrap();
                    }
                    link.reset().unwrap();
                    link.read(mine).unwrap()
                })
            })
            .collect();
        runs.into_iter().map(|t| t.join().unwrap()).collect()
    });
    assert_eq!(landed, [vec![2; 4096], vec![1; 4096]]);
    let (a, b) = tcp_pair().unwrap();
    let mut lone = SocketLink::over(0, vec![None, Some(a)], &[64]).unwrap();
    drop(b);
    assert!(matches!(lone.reset(), Err(Error::Link(_))));
}

#[test]
fn shm_exchange() {
    let file = sized("shm", Shm::room(3, 16));
    let ranks = (0..3)
        .map(|r| Shm::over(r, 3, file.try_clone().unwrap(), 16).unwrap())
        .collect();
    rounds(ranks);
}
