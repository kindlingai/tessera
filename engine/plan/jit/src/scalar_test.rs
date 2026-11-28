use crate::{Arg, Buf, Ctx, compile, failure};
use tessera_codegen::Precision;
use tessera_codegen::form::{BF16_FORM, F32_FORM, U32_FORM};
use tessera_codegen::prelude::*;
use tessera_host::exec::{self, ArgVal, Grid};

#[kernel]
fn narrow_ops(
    x: tessera_codegen::prelude::Buf<Bf16>,
    r: tessera_codegen::prelude::Buf<F32>,
    mut ob: BufMut<Bf16>,
    mut ou: BufMut<U32>,
    n: u32,
) {
    if gid() >= n {
        return;
    }
    let i = gid();
    let mut t = [0.0f32; 2];
    t[0] = r[(i * 4) as u64];
    let mut v = x.load4((i * 4) as u64);
    v.x = clamp(max(v.x, t[0]), -4.0f32, 4.0f32);
    ob.store4((i * 4) as u64, v);
    ou[i as u64] = bf16_pair(v.y, v.z) ^ r.word(i * 16) ^ r.half(i * 16 + 2);
}

#[test]
fn scalar_inline_matches() {
    let body = narrow_ops_ir();
    let f = f32::from_bits;
    let edge = [
        -0.0f32,
        f(0x3f80_8000),
        f(0x3f81_8000),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f(0x0000_8000),
        f32::MAX,
    ];
    let mut xb = vec![0u8; 16];
    for (i, &v) in edge.iter().enumerate() {
        tessera_host::lockstep::st_bf16(&mut xb, i, v);
    }
    let rb: Vec<u8> = edge.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut d0 = [xb.clone(), rb.clone(), vec![0; 16], vec![0; 8]];
    let mut d1 = [xb, rb, vec![0; 16], vec![0; 8]];
    let av = |at, form| ArgVal::Buf { at, skip: 0, form };
    let args = [
        av(0, BF16_FORM),
        av(1, F32_FORM),
        av(2, BF16_FORM),
        av(3, U32_FORM),
        ArgVal::U32(2),
    ];
    let mut s0: Vec<&mut [u8]> = d0.iter_mut().map(Vec::as_mut_slice).collect();
    let grid = Grid {
        groups: 1,
        threads: 4,
    };
    exec::run(&body, &args, &mut s0, grid, LANES, Precision::EXACT);
    let mut bufs = d1.each_mut().map(|b| Buf {
        ptr: b.as_mut_ptr(),
        len: b.len(),
    });
    let ja = [0, 1, 2, 3].map(|buf| Arg {
        buf,
        ..Arg::default()
    });
    let ja = [
        ja[0],
        ja[1],
        ja[2],
        ja[3],
        Arg {
            bits: 2,
            ..Arg::default()
        },
    ];
    let ctx = Ctx {
        bufs: bufs.as_ptr(),
        args: ja.as_ptr(),
        threads: 4,
        lanes: LANES,
        precision: &Precision::EXACT,
    };
    let entry = compile(&body).unwrap();
    // SAFETY: compiled for this body's parameters.
    unsafe { entry(&ctx, 0, 1) };
    assert_eq!((failure(), &d1[2], &d1[3]), (None, &d0[2], &d0[3]));
    bufs[0].len = 4;
    // SAFETY: short buffer exercises bounds-check failure path.
    unsafe {
        entry(
            &Ctx {
                bufs: bufs.as_ptr(),
                ..ctx
            },
            0,
            1,
        )
    };
    assert!(failure().is_some());
}
