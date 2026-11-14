//! Runtime compile, launch and readback. Skips without a device.

#![cfg(any(feature = "cuda", feature = "rocm"))]

const SMOKE_CU: &str = include_str!("smoke.cu");

macro_rules! smoke {
    ($bk:ident) => {
        mod $bk {
            use gpukit::$bk::{BufferPool, Context, ContextConfig, KernelArgs, launch_1d};

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

#[cfg(feature = "rocm")]
smoke!(rocm);
