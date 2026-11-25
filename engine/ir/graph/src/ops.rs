//! Ops: one table row each, with their facts.

use crate::dims::{DimExpr, Shape};
use crate::node::{Attend, Conv, Delta, Mask, Row, Select, View};
use crate::stages::{Pin, Stage};
use crate::types::DType;
use serde::{Deserialize, Serialize};

/// An op's arguments, state and pinned form site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpFacts {
    /// Arguments the op reads.
    pub args: usize,
    /// The op sums, so its node carries `Numerics`.
    pub reduces: bool,
    /// The result comes from outside the graph.
    pub source: bool,
    /// The op appends to kept rows.
    pub appends: bool,
    /// Kept state the op carries across launches.
    pub state: Option<u32>,
    pub pin: Pin,
}

impl OpFacts {
    /// Values for the columns a row leaves out.
    pub const PLAIN: OpFacts = OpFacts {
        args: 1,
        reduces: false,
        source: false,
        appends: false,
        state: None,
        pin: Pin::Plain,
    };
}

/// Defines `Op` and its facts from rows `Variant params => { columns }`.
macro_rules! ops {
    ($(
        $(#[$m:meta])*
        $name:ident
        $(($tb:ident: $tty:ty))?
        $({$($f:ident: $fty:ty),* $(,)?})?
        => {$($col:ident: $val:expr),* $(,)?}
    )*) => {
        #[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
        pub enum Op {$(
            $(#[$m])*
            $name $(($tty))? $({$($f: $fty),*})?,
        )*}

        impl Op {
            pub fn facts(&self) -> OpFacts {
                match self {$(
                    Op::$name $(($tb))? $({$($f),*})? => {
                        $(let _ = $tb;)?
                        $($(let _ = $f;)*)?
                        OpFacts {
                            $($col: $val,)*
                            ..OpFacts::PLAIN
                        }
                    }
                )*}
            }
        }
    };
}

ops! {
    /// An entry's input port.
    Input { entry: u32, port: u32 } => { args: 0, source: true }
    Weight(w: u32) => { args: 0, source: true }
    /// `x[.., K]` times `w[.., N, K]` transposed.
    Matmul => { args: 2, reduces: true, pin: Pin::Matmul }
    Attend(a: Attend) => {
        args: 3 + usize::from(a.picked) + 2 * usize::from(a.ragged),
        reduces: true,
        pin: Pin::Attention,
    }
    /// Queries, keys, values, log decays and write rates.
    Delta(d: Delta) => {
        args: 5 + 2 * usize::from(d.state.is_some()),
        reduces: true,
        state: d.state,
        pin: Pin::Attention,
    }
    /// Each channel sums its last `K` rows, weighted by taps.
    Conv(c: Conv) => { args: 2 + 2 * usize::from(c.state.is_some()), state: c.state }
    /// Scores `[.., Queries, Keys]`. Unseen keys become negative infinity.
    Mask(m: Mask) => {}
    /// Appends rows to kept rows. The result holds all.
    Append(kept: u32) => { args: 3, appends: true }
    Row(r: Row) => {
        args: 1 + r.0.iter().map(Stage::operands).sum::<usize>(),
        reduces: r.0.iter().any(Stage::reduces),
    }
    Select(s: Select) => { args: 2 - usize::from(matches!(s, Select::TopK { .. })) }
    View(v: View) => { args: 1 + usize::from(matches!(v, View::Concat)) }
    /// Truncates toward zero. The value must fit the target type.
    Cast(to: DType) => {}
    /// The mean over this axis. The axis goes away.
    Mean(axis: u32) => { reduces: true }
    /// The sum over this axis, slices weighted by argument two.
    Mix(axis: u32) => { args: 2, reduces: true, pin: Pin::Router }
}

impl Op {
    /// Whether this op sums, so it carries `Numerics`.
    pub fn reduces(&self) -> bool {
        self.facts().reduces
    }

    /// Whether the result reads state besides its arguments.
    pub fn stateful(&self) -> bool {
        self.facts().source || self.keeps()
    }

    /// Whether the op writes kept rows or state.
    pub fn keeps(&self) -> bool {
        self.facts().appends || self.recurrent()
    }

    /// Whether the op carries slot state across launches.
    pub fn recurrent(&self) -> bool {
        self.facts().state.is_some()
    }

    /// The op with `f` applied to each size it names.
    pub fn map_dims(&self, f: impl Fn(&DimExpr) -> DimExpr) -> Op {
        match self {
            Op::View(View::Reshape(s)) => {
                Op::View(View::Reshape(Shape(s.0.iter().map(f).collect())))
            }
            Op::View(View::Slice { axis, start, len }) => Op::View(View::Slice {
                axis: *axis,
                start: f(start),
                len: f(len),
            }),
            Op::View(View::Expand { axis, to }) => Op::View(View::Expand {
                axis: *axis,
                to: f(to),
            }),
            Op::Select(Select::TopK { k }) => Op::Select(Select::TopK { k: f(k) }),
            Op::View(View::Permute(_) | View::Concat)
            | Op::Select(Select::Gather | Select::Take)
            | Op::Input { .. }
            | Op::Weight(_)
            | Op::Matmul
            | Op::Attend(_)
            | Op::Delta(_)
            | Op::Conv(_)
            | Op::Mask(_)
            | Op::Append(_)
            | Op::Row(_)
            | Op::Cast(_)
            | Op::Mean(_)
            | Op::Mix(_) => self.clone(),
        }
    }
}
