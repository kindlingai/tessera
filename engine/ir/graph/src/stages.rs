//! Row stages: one table row each, with their facts.

use crate::node::{Act, Pairs};
use serde::{Deserialize, Serialize};

/// The argument a stage reads besides the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Takes {
    Nothing,
    /// Matches the row input's trailing axes and dtype.
    Broadcast,
    /// One replicated weight per row entry.
    RowWeight,
    /// Integer positions that lead the rows.
    Positions,
    /// Integer coordinates per position, with a trailing axis count.
    Axes,
    /// A `u32` mask word per 32 entries, then edits.
    Bits,
}

/// How a stage changes the row's placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    Keep,
    /// Joins the operand's placement as a sum.
    Sum,
    /// Joins the operand's placement as a product.
    Product,
    /// Reads replicated values only.
    Replicated,
    /// No partial sums. Routed only if zero maps to zero.
    Nonlinear(bool),
    /// Partial sums become replicated.
    Reduce,
    /// A split last axis becomes replicated.
    Gather,
    /// Partial sums become each rank's own rows.
    Scatter,
    /// Each rank's own rows become replicated.
    Join,
    /// Replicated rows become each rank's own rows.
    Own,
}

/// A check on a stage's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Check {
    Pass,
    Finite(f32, &'static str),
    /// Finite and above zero.
    Positive(f32, &'static str),
    PowerOfTwo(u32),
}

/// How lowering treats a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lower {
    /// Fuses into element-wise maps.
    Map,
    /// Interpreter only. Lowering needs the stage folded away.
    Fold,
    /// Moves values between ranks. One rank skips it.
    Crossing,
    /// The named kernel rule runs it.
    Kernel(&'static str),
}

/// The pinned form site for a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pin {
    Plain,
    Norm,
    /// Probabilities: softmax, or the router inside MoE.
    Prob,
    Residual,
    Matmul,
    Attention,
    Router,
}

/// A stage's operand, checks, placement rule and lowering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Facts {
    pub operand: Takes,
    /// Each rank must hold whole rows.
    pub whole: bool,
    /// The stage sums, so the row carries `Numerics`.
    pub reduces: bool,
    pub check: Check,
    pub place: Place,
    /// Rows must hold whole blocks of this many entries.
    pub blocks: u32,
    pub pin: Pin,
    pub lower: Lower,
}

impl Facts {
    /// Values for the columns a row leaves out.
    pub const PLAIN: Facts = Facts {
        operand: Takes::Nothing,
        whole: false,
        reduces: false,
        check: Check::Pass,
        place: Place::Keep,
        blocks: 1,
        pin: Pin::Plain,
        lower: Lower::Fold,
    };
}

/// Expands `$cb` over every row stage.
// Rows: `Variant params as builder => lowering { columns }`.
#[macro_export]
macro_rules! stages {
    ($cb:ident) => {
        $cb! {
            Add as add => Map { operand: Broadcast, place: Sum, pin: Residual }
            Mul as mul => Map { operand: Broadcast, place: Product }
            Scale(by: f32 = 0.5) as scale => Map { check: Finite(by, "scale") }
            /// Adds constant `by` to every element.
            Shift(by: f32 = 0.5) as shift => Map {
                check: Finite(by, "shift"),
                place: Nonlinear(by == 0.0),
            }
            /// Clamps each element to at most `hi`.
            Min(hi: f32 = 1.0) as min => Map {
                check: Finite(hi, "min bound"),
                place: Nonlinear(hi >= 0.0),
            }
            /// Clamps each element to at least `lo`.
            Max(lo: f32 = -1.0) as max => Map {
                check: Finite(lo, "max bound"),
                place: Nonlinear(lo <= 0.0),
            }
            RmsNorm { eps: f32 = 1e-6 } as rms_norm => "rms_norm" {
                operand: RowWeight,
                whole: true,
                reduces: true,
                check: Finite(eps, "norm epsilon"),
                pin: Norm,
            }
            /// Divides the row by its root mean square.
            Rms { eps: f32 = 1e-6 } as rms => "rms_norm" {
                whole: true,
                reduces: true,
                check: Finite(eps, "norm epsilon"),
                pin: Norm,
            }
            /// Shrinks a row whose length is past `threshold`.
            L2Clip { threshold: f32 = 1.0, eps: f32 = 1e-6 } as l2_clip => "l2_clip" {
                whole: true,
                reduces: true,
                check: Positive(threshold, "l2 clip threshold"),
                pin: Norm,
            }
            /// Scales rows by `1 + alpha * ln(max(1, (position + 1) / floor))`.
            LogScale { floor: f32 = 8192.0, alpha: f32 = 0.1 } as log_scale => "log_scale" {
                operand: Positions,
                check: Positive(floor, "log scale floor"),
            }
            /// The scaled Walsh-Hadamard transform of each `block` of the row.
            Hadamard { block: u32 = 4 } as hadamard => "hadamard" {
                check: PowerOfTwo(block),
                blocks: block,
            }
            /// Subtracts the row's mean.
            Center as center => "by_sum" { whole: true, reduces: true, pin: Norm }
            /// Divides the row by its sum.
            UnitSum as unit_sum => "by_sum" { whole: true, reduces: true, pin: Prob }
            /// Rotates pairs of elements. The argument holds positions.
            Rope {
                /// Pair `j` turns by `position * theta^(-2j/width)`.
                theta: f32 = 10000.0,
                pairs: Pairs = Pairs::Halves,
                /// Pairs from this one on stay still. `None` turns all.
                #[serde(default, skip_serializing_if = "Option::is_none")]
                turned: Option<u32> = None,
            } => "rope" {
                operand: Positions,
                check: Positive(theta, "rope base"),
                blocks: 2,
            }
            /// Multi-axis rotary positions. Operand holds coordinates per position.
            Mrope {
                theta: f32 = 10000.0,
                /// Pairs per axis, zero-padded to three axes.
                sections: [u32; 3] = [2, 2, 0],
                /// Interleaved uses global frequencies. Contiguous resets per axis.
                interleaved: bool = false,
            } => "rope" {
                operand: Axes,
                check: Positive(theta, "mrope base"),
                blocks: 2,
            }
            Act(a: Act = Act::Silu) as act => Map { place: Nonlinear(a.keeps_zero()) }
            /// Sine of each element. Fold it away before lowering.
            Sin as sin => Fold { place: Replicated }
            /// Cosine of each element. Fold it away before lowering.
            Cos as cos => Fold { place: Replicated }
            Softmax as softmax => "softmax" { whole: true, reduces: true, pin: Prob }
            AllReduce => Crossing { reduces: true, place: Reduce }
            /// Joins each rank's part of the last axis.
            AllGather => Crossing { reduces: true, place: Gather }
            /// Sums the ranks' parts. Each rank keeps its own rows.
            ReduceScatter => Crossing { reduces: true, place: Scatter }
            /// Joins each rank's own rows, in rank order.
            GatherRows => Crossing { reduces: true, place: Join }
            /// Keeps each rank's own rows without communication.
            OwnRows => Crossing { place: Own }
            /// Entries past the position in the argument become -inf.
            Causal as causal => "causal" { operand: Positions, whole: true }
            /// Masks entries to -inf, then applies packed edits.
            Allow as allow => "allow" { operand: Bits, whole: true }
        }
    };
}

macro_rules! lowering {
    ($rule:literal) => {
        Lower::Kernel($rule)
    };
    ($class:ident) => {
        Lower::$class
    };
}

macro_rules! define {
    ($(
        $(#[$m:meta])*
        $name:ident
        $(($tb:ident: $tty:ty = $ts:expr))?
        $({$($(#[$fm:meta])* $f:ident: $fty:ty = $fs:expr),* $(,)?})?
        $(as $method:ident)?
        => $lower:tt {$($col:ident: $val:expr),* $(,)?}
    )*) => {
        /// One step over last-axis rows. Some take an argument.
        #[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
        pub enum Stage {$(
            $(#[$m])*
            $name $(($tty))? $({$($(#[$fm])* $f: $fty),*})?,
        )*}

        /// Each stage once, with sample parameters.
        pub const STAGES: &[Stage] = &[$(Stage::$name $(($ts))? $({$($f: $fs),*})?),*];

        impl Stage {
            pub fn facts(&self) -> Facts {
                use {Check::*, Pin::*, Place::*, Takes::*};
                match *self {$(
                    Stage::$name $(($tb))? $({$($f),*})? => {
                        $(let _ = $tb;)?
                        $($(let _ = $f;)*)?
                        Facts {
                            $($col: $val,)*
                            lower: lowering!($lower),
                            ..Facts::PLAIN
                        }
                    }
                )*}
            }
        }
    };
}

stages!(define);

impl Stage {
    /// Arguments this stage reads besides the row itself.
    pub fn operands(&self) -> usize {
        usize::from(self.facts().operand != Takes::Nothing)
    }

    pub fn reduces(&self) -> bool {
        self.facts().reduces
    }
}
