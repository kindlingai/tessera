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

/// Pinned Metal memory is directly writable by host and GPU.
#[test]
fn pin_round_trip() {
    use crate::device::{Device, DeviceBuffer};
    use crate::launch::{KernelArgs, Launch};
    use tessera_mem::Backing;

    let Ok(ctx) = Context::new() else {
        eprintln!("skip: no Metal device");
        return;
    };
    let mut buf = DeviceBuffer::pinned(&ctx, 16).expect("pin");
    let mapped = buf.mapped_mut().expect("host view");
    for (i, chunk) in mapped.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *chunk = (i as f32).to_le_bytes();
    }
    let src = "\
#include <metal_stdlib>
using namespace metal;
kernel void inc(device float* v [[buffer(0)]], uint i [[thread_position_in_grid]]) { v[i] += 1.0f; }
";
    let kernel = ctx.compile_kernel(src, "inc").expect("compile");
    let mut args = KernelArgs::new();
    args.device_ptr(buf.device_ptr());
    ctx.launch(&kernel, (1, 1, 1), (4, 1, 1), 0, &mut args)
        .expect("launch");
    ctx.finish().expect("finish");
    let mut bytes = [0u8; 16];
    buf.read_bytes(&mut bytes).expect("read");
    let out: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    let _ = ctx.unified();
}

/// Concurrent compiles share one set of pipelines.
#[test]
fn compile_race_free() {
    let Ok(ctx) = Context::new() else {
        eprintln!("skip: no Metal device");
        return;
    };
    let a = ctx.compile_library(SRC).expect("first library compile");
    let b = ctx.compile_library(SRC).expect("cached library compile");
    assert!(std::ptr::eq(
        objc2::rc::Retained::as_ptr(&a),
        objc2::rc::Retained::as_ptr(&b)
    ));
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

const CHAIN_SRC: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void add_one(
    device float* x [[buffer(0)]],
    constant float& step [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i == 0u) x[0] += step;
}
";

/// Captured graphs replay across chunked buffers with barriers.
#[test]
fn graph_replay_chunked() {
    use crate::device::Device;
    use crate::launch::{KernelArgs, Launch};

    let Ok(ctx) = Context::new() else {
        eprintln!("skip: no Metal device");
        return;
    };
    let kernel = ctx.compile_kernel(CHAIN_SRC, "add_one").expect("compile");
    let ptr = ctx.alloc(16).expect("alloc");
    let steps = 700usize;
    let graph = ctx
        .capture(|| -> Result<(), crate::Error> {
            assert!(ctx.capturing());
            ctx.set_concurrent(true);
            for _ in 0..steps {
                let mut args = KernelArgs::new();
                args.device_ptr(ptr).bytes(&1.0f32.to_le_bytes());
                ctx.launch(&kernel, (1, 1, 1), (1, 1, 1), 0, &mut args)?;
                ctx.barrier()?;
            }
            ctx.set_concurrent(false);
            Ok(())
        })
        .expect("capture");
    assert!(!ctx.capturing());

    for round in 1..=3 {
        let init = (round as f32) * 10.0;
        ctx.upload(ptr, &init.to_le_bytes()).expect("upload");
        ctx.replay(&graph).expect("replay");
        if round == 3 {
            drop(graph.clone());
        }
        ctx.finish().expect("finish");
        let mut raw = [0u8; 4];
        ctx.download(ptr, &mut raw).expect("download");
        let got = f32::from_le_bytes(raw);
        let want = init + steps as f32;
        assert_eq!(got, want, "round {round}: got {got}, want {want}");
    }
    // Worker teardown without explicit finish still commits the tail.
    ctx.upload(ptr, &0.0f32.to_le_bytes()).expect("reset");
    let (w_ctx, w_graph) = (ctx.clone(), graph.clone());
    std::thread::spawn(move || w_ctx.replay(&w_graph).expect("worker replay"))
        .join()
        .expect("worker join");
    let mut raw = [0u8; 4];
    ctx.download(ptr, &mut raw).expect("download after drop");
    assert_eq!(f32::from_le_bytes(raw), steps as f32);
    ctx.free(ptr);
}

/// Capture guards, mode flips, and mixed replay paths work.
#[test]
fn graph_capture_guards() {
    use crate::device::{Device, DeviceBuffer};
    use crate::launch::{KernelArgs, Launch};
    use tessera_mem::Backing;

    let Ok(ctx) = Context::new() else {
        eprintln!("skip: no Metal device");
        return;
    };
    let kernel = ctx.compile_kernel(CHAIN_SRC, "add_one").expect("compile");
    let ptr = ctx.alloc(16).expect("alloc");
    let mut pinned = DeviceBuffer::pinned(&ctx, 16).expect("pin");
    pinned.upload(0, &[7u8; 4]).expect("init pinned");
    let empty = ctx
        .capture(|| -> Result<(), crate::Error> { Ok(()) })
        .expect("empty");
    ctx.replay(&empty).expect("replay empty");
    ctx.finish().expect("finish empty");

    let g1 = ctx
        .capture(|| -> Result<(), crate::Error> {
            assert!(
                ctx.capture(|| -> Result<(), crate::Error> { Ok(()) })
                    .is_err()
            );
            assert!(ctx.replay(&empty).is_err());
            assert!(ctx.finish().is_err());
            assert!(ctx.upload(ptr, &[0u8; 4]).is_err());
            let mut raw = [0u8; 4];
            assert!(ctx.download(ptr, &mut raw).is_err());
            assert!(ctx.zero(ptr, 4).is_err());
            assert!(ctx.copy_d2d(ptr, ptr, 4).is_err());
            assert!(ctx.mapped(ptr, 4).is_none());
            assert!(pinned.mapped().is_none());
            assert!(pinned.mapped_mut().is_none());
            assert!(pinned.upload(0, &[9u8; 4]).is_err());
            assert!(pinned.download(0, &mut raw).is_err());

            ctx.set_concurrent(true);
            let mut a1 = KernelArgs::new();
            a1.device_ptr(ptr).bytes(&2.0f32.to_le_bytes());
            ctx.launch(&kernel, (1, 1, 1), (1, 1, 1), 0, &mut a1)?;
            // Mode flip preserves the fence between launches.
            ctx.set_concurrent(false);
            ctx.set_concurrent(true);
            let mut a2 = KernelArgs::new();
            a2.device_ptr(ptr).bytes(&3.0f32.to_le_bytes());
            ctx.launch(&kernel, (1, 1, 1), (1, 1, 1), 0, &mut a2)?;
            ctx.set_concurrent(false);
            Ok(())
        })
        .expect("capture g1");
    let mut check = [0u8; 4];
    pinned.download(0, &mut check).expect("read pinned");
    assert_eq!(check, [7u8; 4]);

    // Mixed direct launch and back-to-back graph replays.
    ctx.upload(ptr, &10.0f32.to_le_bytes()).expect("upload");
    let mut direct = KernelArgs::new();
    direct.device_ptr(ptr).bytes(&1.0f32.to_le_bytes());
    ctx.launch(&kernel, (1, 1, 1), (1, 1, 1), 0, &mut direct)
        .expect("direct 1");
    ctx.replay(&g1).expect("replay 1");
    ctx.replay(&g1).expect("replay 2");
    ctx.launch(&kernel, (1, 1, 1), (1, 1, 1), 0, &mut direct)
        .expect("direct 2");
    ctx.finish().expect("finish mixed");

    let mut raw = [0u8; 4];
    ctx.download(ptr, &mut raw).expect("download");
    assert_eq!(f32::from_le_bytes(raw), 22.0);
    ctx.free(ptr);
    drop(pinned);
    drop(ctx);

    // Dropped context IDs never match fresh contexts.
    for _ in 0..8 {
        let fresh = Context::new().expect("fresh context");
        assert!(fresh.replay(&g1).is_err());
    }
}
