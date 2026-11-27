//! Verifies 64-bit element offset widening in `Join`.

use tessera_lower::form::F32_FORM;
use tessera_lower::kernels::Join;
use tessera_lower::{Dialect, Extent, Kernel, Source};

fn render_metal(left: usize, right: usize, rows: usize) -> String {
    let join = Join {
        left,
        right,
        rows: Extent::Fixed(rows),
        a_form: F32_FORM,
        b_form: F32_FORM,
        out_form: F32_FORM,
    };
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

    let scalar = render_metal(65_535, 65_535, 32_769);
    assert!(
        scalar.contains("(u64)row * (u64)left") && scalar.contains("(u64)row * (u64)right"),
        "expected widened scalar row offsets:\n{scalar}"
    );
    assert!(
        !scalar.contains("(row * left)") && !scalar.contains("(row * right)"),
        "found u32-wrapping scalar offset:\n{scalar}"
    );
}
