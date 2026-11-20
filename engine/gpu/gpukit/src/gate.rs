//! Holds the default stream, so stamps skip host launch gaps.

use crate::Error;
use crate::api::{Api, Call, Context, Kernel};
use crate::launch::KernelArgs;

const SOURCE: &str = include_str!("gate.cu");
/// Flag reads before the hold gives up: about a second.
const SPINS: u32 = 1 << 21;

/// A kernel waiting on a flag in pinned host memory.
pub(crate) struct Gate<A: Api> {
    api: &'static A,
    hold: Kernel<A>,
    host: u64,
    device: u64,
}

// Thread-safe for the same reason as the context.
unsafe impl<A: Api> Send for Gate<A> {}
unsafe impl<A: Api> Sync for Gate<A> {}

impl<A: Api> Gate<A> {
    /// Call with the device current.
    pub(crate) fn new(ctx: &Context<A>) -> Result<Self, Error> {
        let api = ctx.api();
        let hold = ctx.compile_kernel(SOURCE, "tessera_hold")?;
        let (host, device) = api.host_alloc(4)?;
        let gate = Self {
            api,
            hold,
            host,
            device,
        };
        gate.set(0);
        Ok(gate)
    }

    fn set(&self, value: u32) {
        unsafe { std::ptr::write_volatile(self.host as *mut u32, value) };
    }

    /// Queues the hold on the default stream.
    pub(crate) fn close(&self) -> Result<(), Error> {
        A::programmatic(false);
        let mut args = KernelArgs::new();
        args.device_ptr(self.device).u32(SPINS);
        let mut params = args.pointers();
        self.api.launch(&Call {
            kernel: self.hold.raw,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_bytes: 0,
            stream: std::ptr::null_mut(),
            params: params.as_mut_ptr(),
        })
    }

    /// Ends the hold. The kernel clears the flag.
    pub(crate) fn open(&self) {
        self.set(1);
    }
}

impl<A: Api> Drop for Gate<A> {
    fn drop(&mut self) {
        self.api.host_free(self.host);
    }
}
