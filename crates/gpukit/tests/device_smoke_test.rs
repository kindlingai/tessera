//! Runtime compile, launch and readback. Skips without a device.

#![cfg(any(feature = "cuda", feature = "rocm"))]

use gpukit::device::{BufferPool, Device};
use gpukit::launch::{KernelArgs, Launch, launch_1d};

const SMOKE_CU: &str = include_str!("smoke.cu");

/// saxpy over 1024 elements on `D`, checked on the host.
fn saxpy<D: Device + Launch<Kernel = <D as Device>::Kernel>>(backend: &str) {
    let ctx = match D::open() {
        Ok(ctx) => ctx,
        Err(e) => return eprintln!("skipping {backend} smoke: {e}"),
    };
    let kernel = ctx.compile(SMOKE_CU, "saxpy").expect("runtime compile");
    let n = 1024usize;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let mut pool = BufferPool::<D>::new();
    let dx = pool.allocate(&ctx, n * 4).expect("alloc x");
    let dy = pool.allocate(&ctx, n * 4).expect("alloc y");
    dx.write_f32(&x).expect("upload x");
    dy.write_f32(&vec![1.0f32; n]).expect("upload y");
    let mut args = KernelArgs::new();
    args.device_ptr(dy.device_ptr())
        .device_ptr(dx.device_ptr())
        .f32(2.0)
        .u32(n as u32);
    launch_1d(&ctx, &kernel, n, &mut args).expect("launch saxpy");
    Device::synchronize(&ctx).expect("synchronize");
    let mut got = vec![0.0f32; n];
    dy.read_f32(&mut got).expect("download y");
    for (i, v) in got.iter().enumerate() {
        let want = 2.0 * x[i] + 1.0;
        assert!(
            (v - want).abs() < 1e-6,
            "{backend} saxpy[{i}] = {v}, want {want}"
        );
    }
    pool.release(dx);
    pool.release(dy);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_launches_kernel() {
    saxpy::<gpukit::cuda::Context>("cuda");
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_launches_kernel() {
    saxpy::<gpukit::rocm::Context>("rocm");
}
