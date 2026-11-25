//! The per-slot scan ragged kernels with kept state share.

/// A `#[kernel]` scanning slot `s`'s rows over its kept state.
macro_rules! slot_scan {
    (
        $(#[$doc:meta])*
        fn $name:ident($($params:tt)*),
        [$slots:ident, $positions:ident, $out:ident, $rows:ident, $peek:ident],
        [$s:ident, $r:ident, $pos:ident, $at:ident, $last:ident],
        setup: {$($setup:tt)*},
        open: {$($open:tt)*},
        zero: {$($zero:tt)*},
        load: {$($load:tt)*},
        store: {$($store:tt)*},
        row: {$($row:tt)*},
        step: {$($step:tt)*} $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        fn $name($($params)*) {
            $($setup)*
            let mut loaded = 0u32;
            let mut prev = 0xffff_ffffu32;
            let mut $last = 0.0f32;
            for $r in 0..$rows {
                if $slots[$r as u64] == $s {
                    let $pos = $positions[$r as u64];
                    if loaded == 0 {
                        loaded = 1;
                        $($open)*
                        if $pos == 0 {
                            $($zero)*
                        } else {
                            $($load)*
                        }
                    } else if $pos == 0 && prev != 0 {
                        $($zero)*
                    }
                    $($row)*
                    // A filler row repeats the last output.
                    if $pos == prev {
                        $out[$at] = $last;
                    } else {
                        $($step)*
                        prev = $pos;
                        $out[$at] = $last;
                    }
                }
            }
            if loaded != 0 && $peek == 0 {
                $($store)*
            }
        }
    };
}

pub(crate) use slot_scan;
