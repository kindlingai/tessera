use crate::p2p::{
    CROSS, CROSSOVER, LARGE_GRID, SLOT_BYTES, Schedule, ack_offset, area_bytes, pick, source,
};
use tessera_device::{Backend, Options, flatten};

#[test]
fn p2p_kernel_flattens() {
    let source = source();
    for backend in [Backend::Cuda, Backend::Rocm] {
        let options = Options::plain(backend);
        let flat = flatten("p2p.kernel", &source, &|_| None, &options).unwrap();
        assert!(flat.text.contains("tessera_p2p_sum"));
        assert!(flat.text.contains("tessera_p2p_sum_bf16"));
        assert!(flat.text.contains("tessera_p2p_sum_fp16"));
        assert!(flat.text.contains("tessera_p2p_sum_fp8"));
        assert!(flat.text.contains("tessera_p2p_rs_fp8"));
        assert!(flat.text.contains("tessera_p2p_gather"));
        assert!(flat.text.contains("sum_range"));
        assert!(flat.text.contains("add_pair"));
        assert!(flat.text.contains("seq_ptr"));
        assert!(flat.text.contains("timeout_ticks"));
    }
}

#[test]
fn p2p_layout_and_schedule() {
    assert_eq!(ack_offset(4), 8 * SLOT_BYTES);
    assert_eq!(area_bytes(4), 8 * SLOT_BYTES + 4096);
    assert_eq!(pick(12 << 10, 4, Schedule::Auto, CROSS), (0, 1));
    assert_eq!(
        pick(12 << 10, 4, Schedule::Auto, [0, 8 << 10]),
        (3, LARGE_GRID)
    );
    assert_eq!(pick(CROSSOVER, 4, Schedule::Auto, CROSS), (2, LARGE_GRID));
    assert_eq!(pick(CROSSOVER, 8, Schedule::Auto, CROSS), (2, LARGE_GRID));
    assert_eq!(pick(CROSSOVER, 3, Schedule::Auto, CROSS), (1, LARGE_GRID));
    assert_eq!(pick(CROSSOVER, 6, Schedule::Auto, CROSS), (1, LARGE_GRID));
    assert_eq!(pick(CROSSOVER, 2, Schedule::Auto, CROSS), (0, LARGE_GRID));
    assert_eq!(pick(1 << 20, 4, Schedule::Auto, CROSS), (3, LARGE_GRID));
    assert_eq!(pick(1 << 20, 8, Schedule::Auto, CROSS), (2, LARGE_GRID));
    assert_eq!(pick(1 << 20, 4, Schedule::Fanout, CROSS), (0, LARGE_GRID));
    assert_eq!(pick(128 << 10, 4, Schedule::Fanout, CROSS), (0, LARGE_GRID));
    assert_eq!(pick(1024, 4, Schedule::Rounds, CROSS), (1, 1));
    assert_eq!(pick(1024, 4, Schedule::Butterfly, CROSS), (2, 1));
    assert_eq!(pick(1024, 8, Schedule::Butterfly, CROSS), (2, 1));
    assert_eq!(pick(1024, 4, Schedule::Scatter, CROSS), (3, 1));
    assert_eq!(pick(1024, 8, Schedule::Scatter, CROSS), (2, 1));
    assert_eq!(pick(1024, 3, Schedule::Butterfly, CROSS), (1, 1));
    assert_eq!(pick(1024, 2, Schedule::Butterfly, CROSS), (0, 1));
}
