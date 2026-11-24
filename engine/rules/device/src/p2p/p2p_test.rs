use crate::p2p::{CROSSOVER, SLOT_BYTES, Schedule, ack_offset, area_bytes, pick};
use crate::{Backend, Options, flatten};

#[test]
fn kernel_flattens() {
    let source = include_str!("p2p.kernel");
    for backend in [Backend::Cuda, Backend::Rocm] {
        let options = Options {
            backend,
            remote_kv: false,
            marks: false,
        };
        let flat = flatten("p2p.kernel", source, &|_| None, &options).unwrap();
        assert!(flat.text.contains("tessera_p2p_sum"));
    }
}

#[test]
fn layout_and_schedule() {
    assert_eq!(ack_offset(4), 8 * SLOT_BYTES);
    assert_eq!(area_bytes(4), 8 * SLOT_BYTES + 4096);
    assert_eq!(pick(12 << 10, 4, Schedule::Auto), (0, 1));
    assert_eq!(pick(CROSSOVER, 4, Schedule::Auto), (1, 4));
    assert_eq!(pick(CROSSOVER, 3, Schedule::Auto), (1, 4));
    assert_eq!(pick(CROSSOVER, 2, Schedule::Auto), (0, 4));
    assert_eq!(pick(1 << 20, 4, Schedule::Fanout), (0, 4));
    assert_eq!(pick(1024, 4, Schedule::Rounds), (1, 1));
}
