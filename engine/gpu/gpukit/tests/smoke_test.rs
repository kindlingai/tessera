//! Runtime compile, launch and readback. Skips without a device.

#![cfg(any(feature = "cuda", feature = "rocm"))]

const SMOKE_CU: &str = include_str!("smoke.cu");

macro_rules! smoke {
    ($bk:ident) => {
        mod $bk {
            use tessera_gpukit::$bk::{BufferPool, Context, ContextConfig, KernelArgs, launch_1d};

            #[test]
            fn launches_kernel() {
                let ctx = match Context::new(ContextConfig::default()) {
                    Ok(ctx) => ctx,
                    Err(e) => return eprintln!("skipping {}: no device: {e}", stringify!($bk)),
                };
                eprintln!("device: {}", ctx.name());
                let kernel =
                    (ctx.compile_kernel(super::SMOKE_CU, "saxpy")).expect("runtime compile");

                let n = 1024usize;
                let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
                let mut pool = BufferPool::new();
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
                ctx.synchronize().expect("synchronize");

                let mut got = vec![0.0f32; n];
                dy.read_f32(&mut got).expect("download y");
                for (i, v) in got.iter().enumerate() {
                    let want = 2.0 * x[i] + 1.0;
                    assert!(
                        (v - want).abs() < 1e-6,
                        "saxpy[{i}] = {v}, want {want} (FFI or launch is wrong)"
                    );
                }
                eprintln!("saxpy OK over {n} elements");

                pool.release(dx);
                pool.release(dy);
            }
        }
    };
}

#[cfg(feature = "cuda")]
smoke!(cuda);

#[cfg(feature = "cuda")]
#[test]
fn dynamic_shared() {
    use tessera_gpukit::cuda::{BufferPool, Context, ContextConfig, KernelArgs};
    use tessera_gpukit::launch::Launch;

    let ctx = match Context::new(ContextConfig::default()) {
        Ok(ctx) => ctx,
        Err(e) => return eprintln!("skipping cuda dynamic_shared: {e}"),
    };
    let kernel = ctx
        .compile_kernel(SMOKE_CU, "high_shared")
        .expect("compile high_shared");
    let shared = 64 * 1024u32;
    let blocks = ctx.resident(&kernel, 128, shared).expect("resident");
    assert!(blocks >= 1, "expected at least one resident block");

    let mut pool = BufferPool::new();
    let out = pool.allocate(&ctx, 4).expect("alloc out");
    let mut args = KernelArgs::new();
    args.device_ptr(out.device_ptr()).u32(shared / 4 - 1);
    ctx.launch(&kernel, (1, 1, 1), (128, 1, 1), shared, &mut args)
        .expect("launch high_shared");
    ctx.synchronize().expect("synchronize");
    let mut got = [0.0f32];
    out.read_f32(&mut got).expect("download out");
    assert_eq!(got[0], 42.0);
    pool.release(out);
}

#[cfg(feature = "rocm")]
smoke!(rocm);
