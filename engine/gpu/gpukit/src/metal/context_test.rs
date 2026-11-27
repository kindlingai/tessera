use super::Context;
use super::context::apple_specs;

const SRC: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void pc_k0(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 0.0f; }
kernel void pc_k1(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 1.0f; }
kernel void pc_k2(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 2.0f; }
kernel void pc_k3(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 3.0f; }
";

/// Concurrent compiles share one set of pipelines.
#[test]
fn compile_race_free() {
    if Context::new().is_err() {
        eprintln!("skip: no Metal device");
        return;
    }
    let entries = ["pc_k0", "pc_k1", "pc_k2", "pc_k3"];
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(move || {
                let ctx = Context::new().expect("device");
                for _ in 0..4 {
                    for e in entries {
                        ctx.compile_kernel(SRC, e).expect("pipeline compile");
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker panicked: a pipeline race?");
    }
}

#[test]
fn apple_specs_match_tiers() {
    assert_eq!(apple_specs("Apple M4 Pro"), (20, 273e9));
    assert_eq!(apple_specs("Apple M4 Max"), (40, 546e9));
    assert_eq!(apple_specs("Apple M4"), (10, 120e9));
    assert_eq!(apple_specs("Apple M3 Pro"), (14, 150e9));
    assert_eq!(apple_specs("Unknown GPU"), (0, 0.0));
}
