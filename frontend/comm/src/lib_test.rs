use crate::{shm, sockets_within, transient};
use std::io::{self, ErrorKind};
use std::sync::atomic::AtomicBool;
use std::time::Duration;
use tessera_engine::comm::{Error, Shm, Transport};
use tessera_testkit::{tcp_bind, tcp_connect};

/// Rank 1 skips a stale file for rank 0's.
#[test]
fn shm_joins() {
    let path = tessera_testkit::scratch(&format!("join-{}", std::process::id()));
    let stale = std::fs::File::create(&path).unwrap();
    stale.set_len(Shm::room(2, 16) as u64 + 2).unwrap();
    std::thread::scope(|s| {
        let late = s.spawn(|| shm(&path, 1, 2, 16).unwrap());
        std::thread::sleep(Duration::from_millis(20));
        let mut first = shm(&path, 0, 2, 16).unwrap();
        let mut late = late.join().unwrap();
        let a = s.spawn(move || first.exchange(&[7]).unwrap());
        assert_eq!(late.exchange(&[8]).unwrap(), [vec![7], vec![8]]);
        assert_eq!(a.join().unwrap(), [vec![7], vec![8]]);
    });
    assert!(!path.exists());
}

/// Handshake rejects sandbox errors, aborts and stalled peers.
#[test]
fn sockets_fast_fails() {
    for kind in [ErrorKind::ConnectionRefused, ErrorKind::Interrupted] {
        assert!(transient(&io::Error::from(kind)));
    }
    for kind in [
        ErrorKind::PermissionDenied,
        ErrorKind::AddrNotAvailable,
        ErrorKind::NetworkUnreachable,
        ErrorKind::HostUnreachable,
        ErrorKind::TimedOut,
    ] {
        assert!(!transient(&io::Error::from(kind)));
    }
    let short = Duration::from_millis(30);
    let listener = tcp_bind().unwrap();
    let addr = listener.local_addr().unwrap();
    let addrs = [addr, addr];
    let abort = AtomicBool::new(true);
    let aborted = sockets_within(
        0,
        listener.try_clone().unwrap(),
        &addrs,
        short,
        Some(&abort),
    );
    assert!(matches!(aborted, Err(Error::Io(e)) if e.kind() == ErrorKind::ConnectionAborted));
    let timed = sockets_within(0, listener.try_clone().unwrap(), &addrs, short, None);
    assert!(matches!(timed, Err(Error::Io(e)) if e.kind() == ErrorKind::TimedOut));
    // A silent peer must not hang the 4-byte rank read.
    let _silent = tcp_connect(addr).unwrap();
    let stalled = sockets_within(0, listener, &addrs, short, None);
    assert!(matches!(
        stalled,
        Err(Error::Io(e)) if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock)
    ));
}
