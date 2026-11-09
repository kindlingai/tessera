use super::fp4::*;

/// E2M1 codes and their magnitudes.
const E2M1: [(u8, f32); 8] = [
    (0x0, 0.0),
    (0x1, 0.5),
    (0x2, 1.0),
    (0x3, 1.5),
    (0x4, 2.0),
    (0x5, 3.0),
    (0x6, 4.0),
    (0x7, 6.0),
];

/// E4M3 codes and their exact values, subnormals included.
const E4M3: [(u8, f32); 10] = [
    (0x00, 0.0),
    (0x01, 1.0 / 512.0),
    (0x08, 1.0 / 64.0),
    (0x28, 0.25),
    (0x30, 0.5),
    (0x38, 1.0),
    (0x3C, 1.5),
    (0x40, 2.0),
    (0x50, 8.0),
    (0x7E, 448.0),
];

fn e2m1_case(&(bits, val): &(u8, f32)) {
    assert_eq!(e2m1_to_f32(bits), val, "bits={bits}");
    assert_eq!(e2m1_from_f32(val), bits);
    assert_eq!(e2m1_from_f32(-val), bits | 0x8);
}

fn e4m3_case(&(bits, val): &(u8, f32)) {
    assert_eq!(e4m3_from_f32(val), bits, "encode {val}");
    assert_eq!(e4m3_to_f32(bits), val, "decode 0x{bits:02X}");
}

/// Re-encoding a decoded nonzero code keeps its value.
fn e4m3_stable(bits: u8) {
    let decoded = e4m3_to_f32(bits);
    let again = e4m3_to_f32(e4m3_from_f32(decoded));
    let close = (again - decoded).abs() < 0.02 * decoded.abs().max(1e-3);
    assert!(close, "0x{bits:02X}: {decoded} came back as {again}");
}

#[test]
fn codes_round_trip() {
    E2M1.iter().for_each(e2m1_case);
    E4M3.iter().for_each(e4m3_case);
    (0x01..0x7F).for_each(e4m3_stable);
    assert!(e4m3_to_f32(0x7F).is_nan());
}
