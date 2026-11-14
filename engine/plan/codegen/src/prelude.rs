//! The `#[kernel]` prelude. Nothing here runs on the host.

use crate::form;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut};

pub use crate::LANES;
pub use kernel_macro::{helper, kernel};

/// The register type one element reads into.
pub trait Elems {
    type Elem: Copy;
}

/// A buffer whose form the kernel text settles.
pub trait Form: Elems {
    const FORM: &'static dyn form::Form;
}

pub struct F32;
pub struct Bf16;
pub struct U32;
/// Bytes a decode helper walks. Reads widen to u32.
pub struct U8;
/// A float buffer whose `#[form(..)]` picks the form.
pub struct Dyn;

impl Elems for F32 {
    type Elem = f32;
}

/// Reads widen to f32.
impl Elems for Bf16 {
    type Elem = f32;
}

impl Elems for U32 {
    type Elem = u32;
}

impl Elems for U8 {
    type Elem = u32;
}

impl Elems for Dyn {
    type Elem = f32;
}

impl Form for F32 {
    const FORM: &'static dyn form::Form = form::F32_FORM;
}

impl Form for Bf16 {
    const FORM: &'static dyn form::Form = form::BF16_FORM;
}

impl Form for U32 {
    const FORM: &'static dyn form::Form = form::U32_FORM;
}

impl Form for U8 {
    const FORM: &'static dyn form::Form = form::U8_FORM;
}

/// A float4, loaded or stored in one go.
#[derive(Clone, Copy, Default)]
pub struct F32x4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl F32x4 {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

/// Runs only on the device.
pub(crate) fn device() -> ! {
    unreachable!("kernel bodies run on the device")
}

/// A buffer the kernel reads.
pub struct Buf<F: Elems>(PhantomData<F>);

// A derive would ask `F: Copy`.
impl<F: Elems> Clone for Buf<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: Elems> Copy for Buf<F> {}

/// A buffer the kernel writes.
pub struct BufMut<F: Elems>(PhantomData<F>);

impl<F: Elems> Clone for BufMut<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: Elems> Copy for BufMut<F> {}

impl<F: Elems> Index<u64> for Buf<F> {
    type Output = F::Elem;

    fn index(&self, _: u64) -> &F::Elem {
        device()
    }
}

impl<F: Elems> Buf<F> {
    /// Elements `at` to `at + 3`, within one line.
    pub fn load4(&self, _: u64) -> F32x4 {
        device()
    }

    /// The u32 at byte `offset`, a multiple of 4.
    pub fn word(&self, _: u32) -> u32 {
        device()
    }

    /// The u16 at byte `offset`, a multiple of 2.
    pub fn half(&self, _: u32) -> u32 {
        device()
    }
}

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
    fn bits(self, _: u32, _: u32) -> u32 {
        device()
    }
    fn signed(self, _: u32, _: u32) -> f32 {
        device()
    }
    fn f16(self, _: u32) -> f32 {
        device()
    }
    fn e4m3(self, _: u32) -> f32 {
        device()
    }
    fn e2m1(self, _: u32) -> f32 {
        device()
    }
}

/// The buffer from element `n` on.
impl<F: Elems> Add<u64> for Buf<F> {
    type Output = Self;

    fn add(self, _: u64) -> Self {
        device()
    }
}

impl<F: Elems> Index<u64> for BufMut<F> {
    type Output = F::Elem;

    fn index(&self, _: u64) -> &F::Elem {
        device()
    }
}

impl<F: Elems> IndexMut<u64> for BufMut<F> {
    fn index_mut(&mut self, _: u64) -> &mut F::Elem {
        device()
    }
}

impl<F: Elems> BufMut<F> {
    pub fn load4(&self, _: u64) -> F32x4 {
        device()
    }

    pub fn store4(&mut self, _: u64, _: F32x4) {
        device()
    }
}

/// Group memory, for the whole group.
pub struct Shared<T>(PhantomData<T>);

/// `len` elements of group memory. `len` is fixed when built.
pub fn shared<T>(_: u32) -> Shared<T> {
    device()
}

impl<T> Index<u32> for Shared<T> {
    type Output = T;

    fn index(&self, _: u32) -> &T {
        device()
    }
}

impl<T> IndexMut<u32> for Shared<T> {
    fn index_mut(&mut self, _: u32) -> &mut T {
        device()
    }
}

/// A SIMD group's block of sums on the matrix unit.
pub struct Tiles;

/// A zeroed `rows` by `cols` block, both fixed when built.
pub fn tiles(_: u32, _: u32) -> Tiles {
    device()
}

impl Tiles {
    /// Adds `depth` staged columns of `a` times `b`.
    pub fn block(&mut self, _: &Shared<f32>, _: u32, _: &Shared<f32>, _: u32, _: u32, _: u32) {
        device()
    }

    /// Calls `f` per sum this lane holds: row, column, value.
    pub fn each(&self, _: impl FnMut(u32, u32, f32)) {
        device()
    }

    /// Every sum back to zero.
    pub fn zero(&mut self) {
        device()
    }

    /// Every sum times `f` of its row.
    pub fn scale(&mut self, _: impl Fn(u32) -> f32) {
        device()
    }
}

/// A free function's stub. Others are methods or associated.
macro_rules! stub {
    (fn [$($doc:tt)*] $name:ident ($($arg:ident),*) -> $ret:ident) => {
        $($doc)*
        pub fn $name($(_: $arg),*) -> $ret {
            device()
        }
    };
    ($($other:tt)*) => {};
}

/// The intrinsics called as free functions, from the table.
macro_rules! stubs {
    ($($(#[$doc:meta])* $var:ident = $kind:ident $($name:ident)::+ ($($arg:tt)*) $(-> $ret:ident)?
        { $($body:tt)* })*) => {
        $(stub!($kind [$(#[$doc])*] $($name)::+ ($($arg)*) $(-> $ret)?);)*
    };
}

crate::intrinsic::table!(stubs);

/// Waits for every thread in the group.
pub fn barrier() {
    device()
}

/// `for i in (a..b).by(s).unroll(n)`: step `s`, `n` trips unrolled.
pub trait Stride: Sized {
    fn by(self, _: u32) -> Self {
        self
    }

    fn unroll(self, _: u32) -> Self {
        self
    }
}

impl<I: Iterator> Stride for I {}
