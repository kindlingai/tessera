use crate::{Error, SharedPool};
use std::fs::File;
use std::time::Duration;

const WAIT: Duration = Duration::from_millis(20);

#[test]
fn readers_wait() {
    let room = SharedPool::HEAD + (1 << 16);
    let path = tessera_testkit::scratch(&format!("pool-{}", std::process::id()));
    let file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap();
    file.set_len(room as u64).unwrap();
    let other = file.try_clone().unwrap();
    let mut writer = SharedPool::over(file, 100, None, WAIT).unwrap();
    let strange = SharedPool::over(other.try_clone().unwrap(), 100, Some(5), WAIT);
    assert!(matches!(strange, Err(Error::Fit(_))));
    let mut reader = SharedPool::over(other, 100, Some(0), WAIT).unwrap();
    assert!(writer.writes() && !reader.writes());
    assert!(matches!(reader.pass(), Err(Error::Link(_))));
    writer.pass().unwrap();
    reader.pass().unwrap();
    unsafe { *writer.mapping().0.add(SharedPool::HEAD) = 7 };
    assert_eq!(unsafe { *reader.mapping().0.add(SharedPool::HEAD) }, 7);
    #[cfg(target_os = "linux")]
    {
        let (_lead, offer) = SharedPool::lead(100, WAIT).unwrap();
        assert!(SharedPool::follow(&offer, 100, WAIT).is_ok());
        let other = format!("{:016x}{}", 1, &offer[16..]);
        assert!(SharedPool::follow(&other, 100, WAIT).is_err());
    }
}
