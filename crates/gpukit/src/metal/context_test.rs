use super::{CacheConfig, Context};

const SRC: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void pc_k0(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 0.0f; }
kernel void pc_k1(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 1.0f; }
kernel void pc_k2(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 2.0f; }
kernel void pc_k3(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 3.0f; }
";

fn test_config() -> CacheConfig {
    CacheConfig {
        enabled: true,
        dir: Some(std::env::temp_dir().join("gpukit-test-pipeline-cache")),
        namespace: "gpukit-test",
        key: 0x67706b74,
        verbose: false,
    }
}

/// Hammer the process-global pipeline archive from many threads at once.
/// Same entry across threads exercises the dedup fast path; different entries
/// race concurrent `addComputePipelineFunctions`. Unsynchronized, this races
/// the non-thread-safe `MTLBinaryArchive` and SIGSEGVs.
#[test]
fn concurrent_compile_is_race_free() {
    if Context::new(test_config()).is_err() {
        eprintln!("skip: no Metal device");
        return;
    }
    let entries = ["pc_k0", "pc_k1", "pc_k2", "pc_k3"];
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(move || {
                let ctx = Context::new(test_config()).expect("device");
                for _ in 0..4 {
                    for e in entries {
                        ctx.compile_kernel(SRC, e).expect("pipeline compile");
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker panicked — pipeline cache race?");
    }
}
