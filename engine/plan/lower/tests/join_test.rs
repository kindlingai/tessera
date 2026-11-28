//! Verifies 64-bit element offset widening in `Join`.

use tessera_lower::form::F32_FORM;
use tessera_lower::kernels::Join;
use tessera_lower::{Dialect, Extent, Kernel, Source};

fn make_join(left: usize, right: usize, rows: usize) -> Join {
    Join {
        left,
        right,
        rows: Extent::Fixed(rows),
        a_form: F32_FORM,
        b_form: F32_FORM,
        out_form: F32_FORM,
    }
}

fn render_metal(left: usize, right: usize, rows: usize) -> String {
    let join = make_join(left, right, rows);
    let Source::Made(body) = join.source() else {
        panic!("join is generated");
    };
    body.text().render(Dialect::Metal)
}

#[test]
fn offsets_widen() {
    // 32,769 rows x 131,072 cols exceeds u32::MAX output elements.
    let wide = render_metal(65_536, 65_536, 32_769);
    assert!(
        wide.contains("(u64)gid * 4"),
        "expected widened gid * 4:\n{wide}"
    );
    assert!(
        wide.contains("(u64)row * (u64)left"),
        "expected widened row * left:\n{wide}"
    );
    assert!(
        wide.contains("(u64)row * (u64)right"),
        "expected widened row * right:\n{wide}"
    );
    assert!(
        !wide.contains("(u64)(gid * 4)") && !wide.contains("(row * left)"),
        "found u32-wrapping offset:\n{wide}"
    );

    let scalar = render_metal(65_535, 65_535, 32_768);
    assert!(
        scalar.contains("(u64)left + (u64)right"),
        "expected widened scalar width:\n{scalar}"
    );
    assert!(
        scalar.contains("row * (u64)left") && scalar.contains("row * (u64)right"),
        "expected widened scalar row offsets:\n{scalar}"
    );
    assert!(
        !scalar.contains("(row * left)") && !scalar.contains("(row * right)"),
        "found u32-wrapping scalar offset:\n{scalar}"
    );

    // left + right == 2^32 wraps u32 if summed before / 4.
    let huge = render_metal(2_147_483_648, 2_147_483_648, 1);
    assert!(
        huge.contains("left4 + (right / 4)"),
        "expected pre-divided width4:\n{huge}"
    );
    assert!(
        !huge.contains("(left + right)"),
        "found u32-wrapping row width:\n{huge}"
    );
}

#[test]
fn bounds_check() {
    assert!(make_join(0, 0, 2).fits());
    assert!(make_join(65_536, 65_536, 32_769).fits());
    assert!(make_join(2_147_483_648, 2_147_483_648, 1).fits());
    assert!(make_join(65_535, 65_535, 32_768).fits());
    assert!(!make_join(65_535, 65_535, 32_769).fits());
    assert!(!make_join(4_294_967_296, 4_294_967_296, 1).fits());
    assert!(!make_join(65_536, 65_536, 131_072).fits());
}
