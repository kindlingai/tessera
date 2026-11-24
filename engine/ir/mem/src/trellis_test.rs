use crate::block_test::same_part;
use crate::form::Form;
use crate::trellis::{Codebook, INPUT_SIDE, TRELLIS_K2, TRELLIS_K2_ROT, form_for, packed, rotated};
use tessera_graph::F16;

/// Each side's cuts, and a set's, decode as the whole.
#[test]
fn trellis_shares() {
    // Noise is valid codes under any header.
    let mut state = 7u32;
    let mut noise = |n: usize| -> Vec<u8> {
        let byte = |_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        };
        (0..n).map(byte).collect()
    };
    let outputs = packed(32, 256, 0, &noise(32 * 256 * 2 / 8));
    same_part(&TRELLIS_K2, &outputs, [1, 256, 32], [128, 128]);
    let inputs = packed(256, 128, INPUT_SIDE, &noise(128 * 256 * 2 / 8));
    same_part(&TRELLIS_K2, &inputs, [128, 256, 1], [128, 128]);
    assert!(TRELLIS_K2.share(&inputs, [1, 128, 256], [0, 64]).is_none());
    assert!(!TRELLIS_K2.shares(&[128, 256], 0, 64));
    // Across the stored side: tiles from each entry.
    let rows = packed(256, 128, 1, &noise(256 * 128 * 2 / 8));
    same_part(&TRELLIS_K2, &rows, [128, 256, 1], [128, 128]);
    let cols = packed(128, 256, INPUT_SIDE, &noise(128 * 256 * 2 / 8));
    same_part(&TRELLIS_K2, &cols, [1, 256, 128], [128, 128]);
    // A set: matrices behind one header, cut whole.
    let (a, b) = (noise(128 * 256 * 2 / 8), noise(128 * 256 * 2 / 8));
    let set = packed(256, 128, INPUT_SIDE, &[a, b.clone()].concat());
    let second = packed(256, 128, INPUT_SIDE, &b);
    for i in (0..128 * 256).step_by(97) {
        let got = TRELLIS_K2.unpack(&set, 128 * 256 + i);
        assert_eq!(got, TRELLIS_K2.unpack(&second, i), "element {i}");
    }
    same_part(&TRELLIS_K2, &set, [1, 2, 128 * 256], [1, 1]);
    same_part(&TRELLIS_K2, &set, [2 * 128, 256, 1], [128, 128]);
    let out_set = packed(
        128,
        256,
        1,
        &[noise(128 * 256 * 2 / 8), noise(128 * 256 * 2 / 8)].concat(),
    );
    same_part(&TRELLIS_K2, &out_set, [2, 256, 128], [128, 128]);
}

/// A rotated file splits into codes, `suh` and `svh`.
#[test]
fn rotated_trellis_splits() {
    let codes = packed(128, 128, 1, &vec![0x5A; 128 * 128 * 2 / 8]);
    // Values f16 holds exactly, so the halves round trip.
    let suh: Vec<f32> = (0..128).map(|i| 1.0 + i as f32 / 64.0).collect();
    let svh: Vec<f32> = (0..128).map(|i| 0.5 - i as f32 / 512.0).collect();
    let half = |v: &[f32]| {
        v.iter()
            .flat_map(|x| F16::from(*x).0.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let file = [codes.clone(), half(&suh), half(&svh)].concat();
    let want = rotated(&TRELLIS_K2.unpack_all(&codes), 128, 128, &suh, &svh);
    assert_eq!(TRELLIS_K2_ROT.unpack_all(&file), want);
    let floats = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let parts = TRELLIS_K2_ROT.split(&file).unwrap();
    assert_eq!(parts, [codes, floats(&suh), floats(&svh)]);
    assert!(TRELLIS_K2.split(&file).is_none());
}

/// mul1 weights match a decode written from the format alone.
#[test]
fn mul1_values() {
    let want = [
        (0, -3.5),
        (1, 0.625),
        (0x1234, -0.5625),
        (0x8000, -0.875),
        (0xBEEF, 0.15625),
        (0xFFFF, -0.625),
        (0x7A5C, -0.9375),
        (0x0F0F, 0.6875),
    ];
    for (state, v) in want {
        assert_eq!(Codebook::Mul1.weight(state, false), v, "state {state:#x}");
    }
}

/// pmcgx pairs match the exlr8 loader's table.
#[test]
fn pmcgx_values() {
    let want = [
        (0, [1.875, 1.875]),
        (1, [-0.203_125, 0.3125]),
        (74565, [1.625, -2.25]),
        (524287, [-2.5, 0.005_859_375]),
        (782069, [-1.25, 2.25]),
    ];
    for (state, pair) in want {
        let got = [false, true].map(|odd| Codebook::Pmcgx.weight(state, odd));
        assert_eq!(got, pair, "state {state}");
    }
}

/// A tile of bytes `37j+11` decodes as exlr8's loader does.
#[test]
fn pmcgx_tiles() {
    let at = [
        (0, 0),
        (1, 0),
        (0, 1),
        (8, 0),
        (0, 8),
        (9, 9),
        (15, 15),
        (7, 12),
    ];
    let want: [(u32, [f32; 8]); 3] = [
        (
            2,
            [0.171_875, -2.25, -0.875, -2.25, -2.0, -1.375, -0.3125, 0.5],
        ),
        (
            3,
            [2.0, -1.0, 0.9375, 0.109_375, -0.156_25, -1.0, -1.25, 0.75],
        ),
        (
            4,
            [-0.625, -2.5, 0.085_937_5, -1.5, -1.625, 0.6875, 1.625, 0.75],
        ),
    ];
    for (k, values) in want {
        let tile: Vec<u8> = (0..32 * k as usize).map(|j| (37 * j + 11) as u8).collect();
        let form = form_for(k, Codebook::Pmcgx).unwrap();
        let t = form.unpack_all(&packed(16, 16, 1, &tile));
        let got = at.map(|(row, col)| t[col * 16 + row]);
        assert_eq!(got, values, "K{k}");
    }
}
