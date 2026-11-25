//! Kernel operations on the host, for shared kernel bodies.

use tessera_graph::{E2M1, E4M3, F16};

/// Bit fields of a word, below 32 bits wide.
pub trait Bits {
    /// The `n` bits from bit `low`.
    fn bits(self, low: u32, n: u32) -> u32;
    /// The same, sign-extended, as a float.
    fn signed(self, low: u32, n: u32) -> f32;
    /// The 16 bits from `low` read as an f16.
    fn f16(self, low: u32) -> f32;
    /// The 8 bits from `low` read as an e4m3.
    fn e4m3(self, low: u32) -> f32;
    /// The 4 bits from `low` read as an e2m1.
    fn e2m1(self, low: u32) -> f32;
}

impl Bits for u32 {
    fn bits(self, low: u32, n: u32) -> u32 {
        (self >> low) & ((1 << n) - 1)
    }
    fn signed(self, low: u32, n: u32) -> f32 {
        (((self >> low) << (32 - n)) as i32 >> (32 - n)) as f32
    }
    fn f16(self, low: u32) -> f32 {
        F16((self >> low) as u16).into()
    }
    fn e4m3(self, low: u32) -> f32 {
        E4M3((self >> low) as u8).into()
    }
    fn e2m1(self, low: u32) -> f32 {
        E2M1(self.bits(low, 4) as u8).into()
    }
}

/// `x` to f16, nearest even.
pub(crate) fn round_f16(x: f32) -> f32 {
    F16::from(x).into()
}

/// `x` to e4m3, nearest even, 448 at most.
pub(crate) fn round_e4m3(x: f32) -> f32 {
    E4M3::from(x.clamp(-448.0, 448.0)).into()
}

/// The host type for a kernel parameter type.
macro_rules! host_ty {
    (Buf<U8>) => {
        &[u8]
    };
    ($t:ident) => {
        $t
    };
}

/// Each shared body as a plain host function.
macro_rules! host_fns {
    ($($(#[$doc:meta])* $($kernel:literal)?
        fn $name:ident $(<const $c:ident: $ct:ty>)?
            ($($(#[$build:meta])* $arg:ident: $ty:ident $(<$g:ident>)?),*) -> $ret:tt $body:block)*) => {$(
        $(#[$doc])*
        fn $name $(<const $c: $ct>)? ($($arg: $crate::ops::host_ty!($ty $(<$g>)?)),*) -> $ret $body
    )*};
}

pub(crate) use {host_fns, host_ty};
