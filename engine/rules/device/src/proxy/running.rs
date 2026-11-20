//! The proxy on a thread of its own.

use super::page::Page;
use super::wired::Wired;
use super::{Error, Fetched, Fetcher, Proxy};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tessera_engine_comm::Link as Comm;

/// Idle passes before the thread yields.
const SPINS: u32 = 4096;
/// How long a stop waits for puts in flight.
const DRAIN: Duration = Duration::from_secs(10);

/// A rank with no fetch tier.
struct NoFetch;

impl Fetcher for NoFetch {
    fn fetch(&mut self, pool: u32, _: u32, ordinal: u64, _: u32) -> Result<(), String> {
        Err(format!(
            "pool {pool} fetched ordinal {ordinal}, and this rank has no fetch tier"
        ))
    }

    fn fetched(&mut self) -> Vec<Fetched> {
        Vec::new()
    }

    fn reset(&mut self) {}
}

/// The link, unless the thread panicked, and how it ended.
pub type Stopped<L> = (Option<Wired<L>>, Result<(), Error>);
type Ended<L> = (Wired<L>, Result<(), Error>);

/// A proxy thread. `stop` gives the link back.
pub struct Running<L> {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Ended<L>>>,
}

fn serve<L: Comm>(
    mut proxy: Proxy,
    mut page: Page,
    wired: &mut Wired<L>,
    stop: &AtomicBool,
) -> Result<(), Error> {
    let mut idle = 0u32;
    let mut asked: Option<Instant> = None;
    loop {
        match proxy.step(&mut page, wired, &mut NoFetch)? {
            true => idle = 0,
            false => idle = idle.saturating_add(1),
        }
        if asked.is_none() && stop.load(Ordering::Acquire) {
            asked = Some(Instant::now());
        }
        if let Some(at) = asked {
            if !proxy.open() {
                return Ok(());
            }
            if at.elapsed() > DRAIN {
                return Err(Error::Link("puts still in flight at stop".into()));
            }
        }
        match idle > SPINS {
            true => std::thread::yield_now(),
            false => std::hint::spin_loop(),
        }
    }
}

impl<L: Comm + 'static> Running<L> {
    pub fn spawn(proxy: Proxy, page: Page, wired: Wired<L>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let asked = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("tessera-proxy".into())
            .spawn(move || {
                let mut wired = wired;
                let ended = serve(proxy, page, &mut wired, &asked);
                (wired, ended)
            })
            .expect("a proxy thread");
        Running {
            stop,
            thread: Some(thread),
        }
    }
}

impl<L> Running<L> {
    /// Whether the thread stopped, which only an error does.
    pub fn failed(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Waits for puts in flight, then ends the thread.
    pub fn stop(mut self) -> Stopped<L> {
        self.stop.store(true, Ordering::Release);
        let thread = self.thread.take().expect("a running thread");
        match thread.join() {
            Ok((wired, ended)) => (Some(wired), ended),
            Err(_) => (None, Err(Error::Link("the proxy thread panicked".into()))),
        }
    }
}

impl<L> Drop for Running<L> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
