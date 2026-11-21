use crate::shm;
use tessera_engine::comm::{Shm, Transport};

/// Rank 1 skips a stale file for rank 0's.
#[test]
fn shm_joins() {
    let path = tessera_testkit::scratch(&format!("join-{}", std::process::id()));
    let stale = std::fs::File::create(&path).unwrap();
    stale.set_len(Shm::room(2, 16) as u64 + 2).unwrap();
    std::thread::scope(|s| {
        let late = s.spawn(|| shm(&path, 1, 2, 16).unwrap());
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut first = shm(&path, 0, 2, 16).unwrap();
        let mut late = late.join().unwrap();
        let a = s.spawn(move || first.exchange(&[7]).unwrap());
        assert_eq!(late.exchange(&[8]).unwrap(), [vec![7], vec![8]]);
        assert_eq!(a.join().unwrap(), [vec![7], vec![8]]);
    });
    assert!(!path.exists());
}
