//! KV pages whose bytes copy between tiers.

/// A device copy of `(vram, host)` page pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    /// Copies host pages to VRAM, then waits.
    Promote,
    /// Copies VRAM pages to host, then waits.
    Demote,
    /// Starts copies of VRAM pages to host.
    Depart,
    /// Waits for every started copy. Its pairs are empty.
    Settle,
}

/// A VRAM page whose bytes copy to a host page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Leaving {
    pub page: u32,
    pub vram: u32,
    pub host: u32,
    /// Whether the pool still holds the page.
    pub live: bool,
}

/// Two-phase demotes: copy pages out, then free VRAM.
#[derive(Default)]
pub(crate) struct Flights(Vec<Leaving>);

impl Flights {
    /// `page` starts its copy from `vram` to `host`.
    pub(crate) fn depart(&mut self, page: u32, vram: u32, host: u32) {
        self.0.push(Leaving {
            page,
            vram,
            host,
            live: true,
        });
    }

    /// The pool freed `page`. Its two pages stay held.
    pub(crate) fn free(&mut self, page: u32) {
        for leaving in &mut self.0 {
            if leaving.page == page {
                leaving.live = false;
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The copies that are done, once the device waited.
    pub(crate) fn land(&mut self) -> Vec<Leaving> {
        std::mem::take(&mut self.0)
    }
}
