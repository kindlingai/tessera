//! The `#[kernel]` prelude. Nothing here runs on the host.

use graph::DType;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub use crate::LANES;
pub use kernel_macro::kernel;

/// How a buffer stores elements, and what a read gives.
pub trait Form {
    const DTYPE: DType;
    type Elem: Copy;
}

pub struct F32;
pub struct Bf16;
pub struct U32;

impl Form for F32 {
    const DTYPE: DType = DType::F32;
    type Elem = f32;
}

/// Reads widen to f32.
impl Form for Bf16 {
    const DTYPE: DType = DType::Bf16;
    type Elem = f32;
}

impl Form for U32 {
    const DTYPE: DType = DType::U32;
    type Elem = u32;
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
}

/// Runs only on the device.
fn device() -> ! {
    unreachable!("kernel bodies run on the device")
}

/// A buffer the kernel reads.
pub struct Buf<F: Form>(PhantomData<F>);

/// A buffer the kernel writes.
pub struct BufMut<F: Form>(PhantomData<F>);

impl<F: Form> Index<u64> for Buf<F> {
    type Output = F::Elem;

    fn index(&self, _: u64) -> &F::Elem {
        device()
    }
}

impl<F: Form> Buf<F> {
    /// Elements `at` to `at + 3`, within one line.
    pub fn load4(&self, _: u64) -> F32x4 {
        device()
    }
}

impl<F: Form> Index<u64> for BufMut<F> {
    type Output = F::Elem;

    fn index(&self, _: u64) -> &F::Elem {
        device()
    }
}

impl<F: Form> IndexMut<u64> for BufMut<F> {
    fn index_mut(&mut self, _: u64) -> &mut F::Elem {
        device()
    }
}

impl<F: Form> BufMut<F> {
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

/// A warp's block of sums on the matrix unit.
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

/// The thread's index in the grid.
pub fn gid() -> u32 {
    device()
}

/// The thread's index in its group.
pub fn lid() -> u32 {
    device()
}

/// The thread's lane among the 32 that shuffle together.
pub fn lane() -> u32 {
    device()
}

/// The value the lane `mask` away holds.
pub fn shfl_xor(_: f32, _: u32) -> f32 {
    device()
}

/// The sum over the 32 lanes, in every lane.
pub fn sum32(_: f32) -> f32 {
    device()
}

pub fn rsqrt(_: f32) -> f32 {
    device()
}

pub fn exp(_: f32) -> f32 {
    device()
}

pub fn tanh(_: f32) -> f32 {
    device()
}

pub fn max(_: f32, _: f32) -> f32 {
    device()
}

pub fn clamp(_: f32, _: f32, _: f32) -> f32 {
    device()
}

pub fn pow(_: f32, _: f32) -> f32 {
    device()
}

pub fn cos(_: f32) -> f32 {
    device()
}

pub fn sin(_: f32) -> f32 {
    device()
}

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
