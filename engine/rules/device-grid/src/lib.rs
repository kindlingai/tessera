//! Persistent-grid collectives and the resident whole-step decoder.

pub mod arx;
pub mod p2p;
#[cfg(test)]
mod p2p_test;
pub mod sums;
pub mod whole;
#[cfg(test)]
mod whole_test;

use tessera_device::resident;

/// Bytes of `prim::Timing` (`enabled`, `waited`, `moved`, `t0`, `t1`).
pub(crate) const TIMING_BYTES: usize = 40;

pub(crate) fn write_timed<D: tessera_gpukit::device::Device>(ctx: &D, ptr: u64, on: bool) {
    let _ = ctx.upload(ptr, &u64::from(on).to_ne_bytes());
}

/// Reads `[waited, moved]` ticks at `ptr`, returning seconds since `last`.
pub(crate) fn read_ticks<D: tessera_gpukit::device::Device>(
    ctx: &D,
    ptr: u64,
    last: &mut [u64; 2],
) -> Result<[f64; 2], resident::Error> {
    ctx.finish()?;
    let mut raw = [0u8; 16];
    ctx.download(ptr, &mut raw)?;
    let words = raw.as_chunks::<8>().0;
    let now = [u64::from_ne_bytes(words[0]), u64::from_ne_bytes(words[1])];
    let ticks = ctx.ticks();
    let spent = [0, 1].map(|i| now[i].wrapping_sub(last[i]) as f64 / ticks);
    *last = now;
    Ok(spent)
}
