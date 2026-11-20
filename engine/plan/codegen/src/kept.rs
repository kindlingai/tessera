//! Kept-row layouts: where a slot's row at a position lives.

use crate::inline;
use crate::ir::{BinOp, Body, Expr, Intrinsic, Param, Ty};
use std::fmt;

/// Maps a slot and position to a kept row.
pub trait Layout: Sync {
    fn name(&self) -> &'static str;
    /// Rows the buffer holds for `slots` of `context` each.
    fn rows(&self, slots: usize, context: usize) -> usize;
    /// Unbroken rows of one slot, from an aligned start.
    fn run(&self, context: usize) -> usize;
    /// Rows per page, when a table names each slot's pages.
    fn page(&self) -> Option<usize> {
        None
    }
    /// Names of the scalars the row reads.
    fn params(&self) -> &'static [&'static str];
    /// The scalar values, in the same order.
    fn values(&self, slots: usize, context: usize) -> Vec<u32>;
    /// The row of `position` in `slot`.
    fn row(&self, slot: Expr, position: Expr) -> Expr;
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

/// Registers a layout for recipes to name.
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

/// Paged layouts read each slot's pages from this buffer.
pub const TABLE: &str = "kept_table";

/// Table entries for `slots` of `context` rows: none unless paged.
pub fn entries(layout: &dyn Layout, slots: usize, context: usize) -> usize {
    layout
        .page()
        .map_or(0, |page| slots * context.div_ceil(page))
}

/// `body` over `layout`: a table after its buffers, scalars last.
pub fn attach(mut body: Body, layout: &'static dyn Layout) -> Body {
    let row = |args: Vec<Expr>| {
        let [slot, position] = <[Expr; 2]>::try_from(args).expect("kept_row takes two");
        layout.row(slot, position)
    };
    body.stmts = inline::replace(&body.stmts, Intrinsic::KeptRow, &row);
    if layout.page().is_some() {
        let buffer = |p: &Param| matches!(p, Param::Buffer { .. });
        let last = body.params.iter().rposition(buffer);
        let table = Param::Buffer {
            name: TABLE.into(),
            form: crate::form::of(sensai_graph::DType::U32),
            writes: false,
        };
        body.params.insert(last.map_or(0, |at| at + 1), table);
    }
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

fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::bin(op, a, b)
}

/// A layout over one scalar.
macro_rules! plain {
    ($t:ident, $name:literal, $param:literal, $value:expr, run = $run:expr, row = $row:expr) => {
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
            fn row(&self, slot: Expr, position: Expr) -> Expr {
                let row: fn(Expr, Expr, Expr) -> Expr = $row;
                row(slot, position, Expr::var($param))
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
    row = |s, p, stride| bin(BinOp::Add, bin(BinOp::Mul, s, stride), p)
);

/// Each position's rows in order, slot by slot within it.
pub(crate) struct Interleaved;

plain!(
    Interleaved,
    "interleaved",
    "kept_slots",
    |slots, _| slots,
    run = |_| 1,
    row = |s, p, slots| bin(BinOp::Add, bin(BinOp::Mul, p, slots), s)
);

/// A table names each slot's pages.
pub struct Paged {
    name: &'static str,
    /// Rows per page, a power of two.
    page: usize,
}

impl Layout for Paged {
    fn name(&self) -> &'static str {
        self.name
    }
    fn rows(&self, slots: usize, context: usize) -> usize {
        slots * context.div_ceil(self.page) * self.page
    }
    fn run(&self, _: usize) -> usize {
        self.page
    }
    fn page(&self) -> Option<usize> {
        Some(self.page)
    }
    fn params(&self) -> &'static [&'static str] {
        &["kept_pages"]
    }
    fn values(&self, _: usize, context: usize) -> Vec<u32> {
        vec![context.div_ceil(self.page) as u32]
    }
    fn row(&self, slot: Expr, position: Expr) -> Expr {
        let shift = Expr::lit(self.page.trailing_zeros());
        let within = Expr::lit(self.page as i64 - 1);
        let first = bin(BinOp::Mul, slot, Expr::var("kept_pages"));
        let entry = bin(
            BinOp::Add,
            first,
            bin(BinOp::Shr, position.clone(), shift.clone()),
        );
        let page = Expr::index(Expr::var(TABLE), Expr::cast(Ty::U64, entry));
        let start = bin(BinOp::Shl, page, shift);
        bin(BinOp::BitOr, start, bin(BinOp::BitAnd, position, within))
    }
}

crate::layout!(Dense);
crate::layout!(Interleaved);
crate::layout!(PAGED);
crate::layout!(PAGED16);

pub const DENSE: &dyn Layout = &Dense;
pub const INTERLEAVED: &dyn Layout = &Interleaved;
/// Pages of 64 rows, the default page.
const PAGED: Paged = Paged {
    name: "paged",
    page: 64,
};
/// Pages of 16 rows, one flash key block each.
pub const PAGED16: Paged = Paged {
    name: "paged16",
    page: 16,
};
