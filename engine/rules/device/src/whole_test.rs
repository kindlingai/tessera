use crate::whole::{format_summary, prepare_source};
use tessera_gpukit::device::KernelStats;
use tessera_run::marks;

#[test]
fn prepares_resident_sources() {
    let plain = "extern \"C\" __global__ void step() {}";
    let marked = "extern \"C\" __global__ void step() { TS_MARK(); }";
    assert_eq!(prepare_source(plain, None, false), (None, false));
    assert_eq!(prepare_source(plain, None, true), (None, false));
    let (off, on) = prepare_source(marked, None, false);
    assert!(!on && off.unwrap().starts_with(marks::OFF));
    let (active, on) = prepare_source(marked, None, true);
    assert!(on && active.unwrap().starts_with(marks::ON));
    let (edited, on) = prepare_source(plain, Some(marked), true);
    assert!(on && edited.unwrap().starts_with(marks::ON));
    let (edited_off, on) = prepare_source(plain, Some(plain), false);
    assert!(!on && edited_off.unwrap().starts_with(marks::OFF));
}

#[test]
fn formats_telemetry_lines() {
    let stats = KernelStats {
        regs: 112,
        shared_bytes: 16384,
        local_bytes: 64,
        max_threads: 512,
    };
    let with = format_summary("whole", 20, 256, 2, Some(stats));
    assert!(with.contains("grid=20 threads=256 blocks/sm=2"));
    assert!(with.contains("regs=112 shared_bytes=16384 local_bytes=64"));
    let without = format_summary("whole", 20, 256, 2, None);
    assert!(!without.contains("regs="));
}
