use crate::{FastStacked, stacked_needs, stacked_source, stacked_source_for};
use tessera_lower::Source;
use tessera_lower::form::BF16_FORM;

fn stacked_kernel(rows: usize, n: usize, k: usize, wide: bool) -> FastStacked {
    FastStacked {
        len: 16 * rows * n,
        batches: 16,
        rows,
        n,
        k,
        block: n * k,
        first: 0,
        step: if wide { n } else { k },
        x_stride: [rows * k, k],
        o_stride: [rows * n, n],
        wide,
        form: BF16_FORM,
    }
}

#[test]
fn stacked_64_picks_prefill() {
    let lanes = stacked_kernel(2048, 256, 512, false);
    assert!(lanes.lanes_64());
    assert!(lanes.tiled_64());
    let Source::Fixed { entry, .. } = stacked_source_for(&lanes) else {
        panic!("expected fixed source");
    };
    assert_eq!(entry, "tessera_stacked_lanes_64");
    assert_eq!(
        stacked_needs(&lanes).group_bytes,
        64 * 18 * 4 * 2 + 64 * 40 * 2
    );

    let wide = stacked_kernel(2048, 512, 192, true);
    assert!(wide.wide_64());
    assert!(wide.tiled_64());
    let Source::Fixed { entry, .. } = stacked_source_for(&wide) else {
        panic!("expected fixed source");
    };
    assert_eq!(entry, "tessera_stacked_wide_64");

    let decode_lanes = stacked_kernel(1, 256, 512, false);
    assert!(!decode_lanes.tiled_64());
    let Source::Fixed { entry, .. } = stacked_source_for(&decode_lanes) else {
        panic!("expected fixed source");
    };
    assert_eq!(entry, "tessera_stacked_lanes_tc");

    let decode_wide = stacked_kernel(1, 512, 192, true);
    assert!(!decode_wide.tiled_64());
    let Source::Fixed { entry, .. } = stacked_source_for(&decode_wide) else {
        panic!("expected fixed source");
    };
    assert_eq!(entry, "tessera_stacked_wide");
}

#[test]
fn stacked_source_has_64() {
    let src = stacked_source();
    assert!(src.contains("tessera_stacked_wide"));
    assert!(src.contains("tessera_stacked_lanes_64"));
    assert!(src.contains("tessera_stacked_wide_64"));
}
