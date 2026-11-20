use crate::link::exchange_regions;
use crate::{Blocking, Error, Memory, Shm, Sockets, Transport, summed};
use std::fs::{File, OpenOptions};
use std::net::{TcpListener, TcpStream};

#[test]
fn rank_order() {
    let parts = [[1.0f32, 2.0], [10.0, 20.0], [100.0, 200.0]];
    assert_eq!(summed(parts.iter().map(|p| &p[..])), [111.0, 222.0]);
}

/// A sized scratch file, open for reading and writing.
fn sized(name: &str, len: usize) -> File {
    let path = sensai_testkit::scratch(&format!("{name}-{}", std::process::id()));
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
    assert_eq!(Shm::room(3, 8), 3 * (32 + 96 + 96));
    let short = sized("short", Shm::room(2, 8) - 1);
    assert!(matches!(Shm::over(0, 2, short), Err(Error::Fit(_))));
    let late = sized("late", Shm::room(2, 8));
    assert!(matches!(Shm::over(2, 2, late), Err(Error::Fit(_))));
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
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let caller = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (answer, _) = listener.accept().unwrap();
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
    let links = Memory::ranks(3, &exchange_regions(3, 16));
    let wait = std::time::Duration::from_secs(10);
    rounds(
        links
            .into_iter()
            .map(|l| Blocking::new(l, 16, wait))
            .collect(),
    );
}

#[test]
fn shm_exchange() {
    let file = sized("shm", Shm::room(3, 16));
    let ranks = (0..3)
        .map(|r| Shm::over(r, 3, file.try_clone().unwrap()).unwrap())
        .collect();
    rounds(ranks);
}
