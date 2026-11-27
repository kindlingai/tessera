//! One pool of bytes that a box's ranks map.

use crate::Error;
use crate::shm::{BLOCK, Mapped, ROUND};
use std::fs::File;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Head words: write batches done, then the token.
const DONE: usize = 0;
const TOKEN: usize = 8;

/// Shared memory of `len` bytes, and the path peers open.
#[cfg(target_os = "linux")]
fn memory(len: usize) -> Result<(File, String), Error> {
    use std::os::fd::FromRawFd;
    unsafe extern "C" {
        fn memfd_create(name: *const std::ffi::c_char, flags: std::ffi::c_uint) -> std::ffi::c_int;
    }
    const CLOSE_ON_EXEC: std::ffi::c_uint = 1;
    let fd = unsafe { memfd_create(c"tessera-kv".as_ptr(), CLOSE_ON_EXEC) };
    if fd < 0 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    // SAFETY: the call returned a descriptor nothing else owns.
    let file = unsafe { File::from_raw_fd(fd) };
    file.set_len(len as u64)?;
    Ok((file, format!("/proc/{}/fd/{fd}", std::process::id())))
}

#[cfg(not(target_os = "linux"))]
fn memory(_len: usize) -> Result<(File, String), Error> {
    Err(Error::Fit("this host cannot share memory by path".into()))
}

/// Who backs a rank's host KV pool.
#[derive(Clone, Copy, Debug)]
pub enum Backed<'a> {
    /// This rank pins its own pool.
    Own,
    /// This rank makes the pool its box's ranks map.
    Lead,
    /// This rank maps the pool rank 0 offers.
    Follow(&'a str),
}

/// The host KV pool a rank opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostPool {
    None,
    Own,
    /// Shared. Holds the offer the other ranks map.
    Shared(String),
}

impl HostPool {
    /// The pool a rank opened, shared or its own.
    pub fn of(shared: Option<&SharedPool>) -> Self {
        shared.map_or(Self::Own, |pool| Self::Shared(pool.offer.clone()))
    }
}

/// A pool one rank writes and its box's ranks read.
pub struct SharedPool {
    map: Mapped,
    writes: bool,
    /// Write batches this rank passed.
    seen: u64,
    wait: Duration,
    /// The pool's device address, once registered.
    dev: u64,
    offer: String,
}

impl SharedPool {
    /// The pool's offset in its mapping.
    pub const HEAD: usize = BLOCK;

    /// A pool over `file`. A token marks a reader.
    pub fn over(file: File, len: usize, token: Option<u64>, wait: Duration) -> Result<Self, Error> {
        let room = (Self::HEAD + len).next_multiple_of(BLOCK);
        let have = file.metadata()?.len() as usize;
        if have != room {
            return Err(Error::Fit(format!("{have} bytes for a {room} byte pool")));
        }
        let map = Mapped::over(file, room)?;
        let pool = SharedPool {
            map,
            writes: token.is_none(),
            seen: 0,
            wait,
            dev: 0,
            offer: String::new(),
        };
        // Another box may hold some other file at the path.
        if token.is_some_and(|t| pool.map.word(TOKEN).load(Ordering::Acquire) != t) {
            return Err(Error::Fit("the pool is not rank 0's".into()));
        }
        Ok(pool)
    }

    /// The writer's pool over new shared memory, and its offer.
    pub fn lead(len: usize, wait: Duration) -> Result<(Self, String), Error> {
        let room = (Self::HEAD + len).next_multiple_of(BLOCK);
        let (file, path) = memory(room)?;
        let pool = Self::over(file, len, None, wait)?;
        let clock = std::time::UNIX_EPOCH.elapsed().unwrap_or_default();
        let token = clock.as_nanos() as u64 ^ u64::from(std::process::id()).rotate_left(40);
        pool.map.word(TOKEN).store(token, Ordering::Release);
        Ok((pool, format!("{token:016x}{path}")))
    }

    /// A reader's pool: the one the writer's `offer` names.
    pub fn follow(offer: &str, len: usize, wait: Duration) -> Result<Self, Error> {
        let bad = || Error::Fit(format!("{offer:?} names no pool"));
        let (token, path) = offer.split_at_checked(16).ok_or_else(bad)?;
        let token = u64::from_str_radix(token, 16).map_err(|_| bad())?;
        let file = File::options().read(true).write(true).open(path)?;
        Self::over(file, len, Some(token), wait)
    }

    /// The shared pool `by` asks for, mapped by `register`.
    pub fn open(
        by: Backed,
        len: usize,
        register: impl FnOnce(*mut u8, usize) -> Option<u64>,
    ) -> Option<Self> {
        let (mut pool, offer) = match by {
            Backed::Own => return None,
            Backed::Lead => Self::lead(len, ROUND).ok()?,
            Backed::Follow(offer) => (Self::follow(offer, len, ROUND).ok()?, offer.into()),
        };
        // Rank 0 registers first, so its node holds the pages.
        pool.dev = register(pool.map.base(), pool.map.len())? + Self::HEAD as u64;
        pool.offer = offer;
        Some(pool)
    }

    /// The pool's host and device addresses.
    pub fn at(&self) -> (u64, u64) {
        (self.map.base() as u64 + Self::HEAD as u64, self.dev)
    }

    /// The whole mapping: its address and length.
    pub fn mapping(&self) -> (*mut u8, usize) {
        (self.map.base(), self.map.len())
    }

    /// Whether this rank writes the pool.
    pub fn writes(&self) -> bool {
        self.writes
    }

    /// Ends a write batch. A reader waits for the writer's.
    pub fn pass(&mut self) -> Result<(), Error> {
        let (want, done) = (self.seen + 1, self.map.word(DONE));
        if self.writes {
            done.store(want, Ordering::Release);
        }
        let start = Instant::now();
        while done.load(Ordering::Acquire) < want {
            if start.elapsed() > self.wait {
                let wrote = done.load(Ordering::Acquire);
                return Err(Error::Link(format!(
                    "rank 0 wrote {wrote} of {want} batches"
                )));
            }
            std::thread::yield_now();
        }
        self.seen = want;
        Ok(())
    }
}
