//! Admission of connections to the server.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Chooses the connections to serve. Threads share one gate.
pub trait Gate: Send + Sync {
    /// Whether to serve `peer`. A yes counts it open.
    fn enter(&self, peer: Option<SocketAddr>) -> bool;
    /// A served connection from `peer` ended.
    fn leave(&self, peer: Option<SocketAddr>);
}

/// Serves at most `max` connections at once.
pub(crate) struct Capped {
    max: usize,
    open: AtomicUsize,
}

impl Capped {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            max,
            open: AtomicUsize::new(0),
        }
    }
}

impl Gate for Capped {
    fn enter(&self, _peer: Option<SocketAddr>) -> bool {
        let entered = self.open.fetch_add(1, Ordering::Relaxed) < self.max;
        if !entered {
            self.open.fetch_sub(1, Ordering::Relaxed);
        }
        entered
    }

    fn leave(&self, _peer: Option<SocketAddr>) {
        self.open.fetch_sub(1, Ordering::Relaxed);
    }
}
