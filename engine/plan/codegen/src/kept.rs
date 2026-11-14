//! Kept-row layouts: where a slot's row at a position lives.

use crate::ir::{Body, Param, Ty};
use std::fmt;

/// Maps a slot and position to a kept row.
pub trait Layout: Sync {
    fn name(&self) -> &'static str;
    /// Rows the buffer holds for `slots` of `context` each.
    fn rows(&self, slots: usize, context: usize) -> usize;
    /// Unbroken rows of one slot, from an aligned start.
    fn run(&self, context: usize) -> usize;
    /// Names of the scalars in the kernel text.
    fn params(&self) -> &'static [&'static str];
    /// The scalar values, in the same order.
    fn values(&self, slots: usize, context: usize) -> Vec<u32>;
    /// Kernel text: the row of `position` in `slot`.
    fn row(&self, slot: &str, position: &str) -> String;
    /// The same on the host, given `values`.
    fn at(&self, slot: u32, position: u32, values: &[u32]) -> u32;
}

impl fmt::Debug for dyn Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl PartialEq for dyn Layout {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for dyn Layout {}

crate::form::gather! {
    #[gather]
    pub static LAYOUTS: scattered_collect::ScatteredSlice<&'static dyn Layout>;
}

/// Registers a layout. Recipes then name it.
#[macro_export]
macro_rules! layout {
    ($layout:expr) => {
        const _: () = {
            $crate::form::scatter! {
                #[scatter($crate::kept::LAYOUTS)]
                static LAYOUT: &'static dyn $crate::kept::Layout = &$layout;
            }
        };
    };
}

/// `body` reading kept rows through `layout`, its scalars last.
pub fn attach(mut body: Body, layout: &'static dyn Layout) -> Body {
    body.layout = Some(layout);
    let scalar = |name: &&str| Param::Scalar {
        name: (*name).into(),
        ty: Ty::U32,
    };
    body.params.extend(layout.params().iter().map(scalar));
    body
}

/// The registered layout called `name`.
pub fn named(name: &str) -> Option<&'static dyn Layout> {
    LAYOUTS.iter().copied().find(|l| l.name() == name)
}

/// A layout over one scalar, in kernel and host form.
macro_rules! plain {
    ($t:ident, $name:literal, $param:literal, $value:expr, run = $run:expr, row = $row:expr, at = $at:expr) => {
        impl Layout for $t {
            fn name(&self) -> &'static str {
                $name
            }
            fn rows(&self, slots: usize, context: usize) -> usize {
                slots * context
            }
            fn run(&self, context: usize) -> usize {
                let run: fn(usize) -> usize = $run;
                run(context)
            }
            fn params(&self) -> &'static [&'static str] {
                &[$param]
            }
            fn values(&self, slots: usize, context: usize) -> Vec<u32> {
                let value: fn(usize, usize) -> usize = $value;
                vec![value(slots, context) as u32]
            }
            fn row(&self, slot: &str, position: &str) -> String {
                let row: fn(&str, &str) -> String = $row;
                row(slot, position)
            }
            fn at(&self, slot: u32, position: u32, values: &[u32]) -> u32 {
                let at: fn(u32, u32, u32) -> u32 = $at;
                at(slot, position, values[0])
            }
        }
    };
}

/// Each slot's rows in order, slot after slot.
pub struct Dense;

plain!(
    Dense,
    "dense",
    "kept_stride",
    |_, context| context,
    run = |context| context,
    row = |s, p| format!("(({s}) * kept_stride + ({p}))"),
    at = |s, p, stride| s * stride + p
);

/// Each position's rows in order, slot by slot within it.
pub(crate) struct Interleaved;

plain!(
    Interleaved,
    "interleaved",
    "kept_slots",
    |slots, _| slots,
    run = |_| 1,
    row = |s, p| format!("(({p}) * kept_slots + ({s}))"),
    at = |s, p, slots| p * slots + s
);

crate::layout!(Dense);
crate::layout!(Interleaved);

pub const DENSE: &dyn Layout = &Dense;
pub const INTERLEAVED: &dyn Layout = &Interleaved;
