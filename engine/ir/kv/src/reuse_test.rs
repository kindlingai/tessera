use crate::{Begin, Why, begin};

#[test]
fn skips_follow_rule() {
    let cases = [
        (1031, 1030, 191, 2191, 64, 128, Why::Hit),
        (1031, 1030, 63, 1999, 64, 0, Why::Grid),
        (100, 0, 100, 99, 0, 0, Why::NoLcp),
    ];
    for (h, valid, lcp, max, grid, skip, why) in cases {
        let got = begin(&vec![1; h], valid, &vec![1; lcp], max, grid);
        assert_eq!(got, Begin { skip, lcp, why });
    }
}
