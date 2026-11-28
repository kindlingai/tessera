//! Thread-local command buffer and compute encoder state.

use crate::Error;

/// Command buffer teardown hook for safe drop cleanup.
pub(crate) trait PendingCmd {
    fn wait_done(&self);
}

/// Compute encoder teardown hook for safe drop cleanup.
pub(crate) trait PendingEnc {
    fn end_open(&self);
}

/// Backend hooks for a pending command buffer and encoder.
pub(crate) trait PendingOps {
    type Cmd: PendingCmd;
    type Enc: PendingEnc + Clone;

    fn new_cmd(&mut self, unretained: bool) -> Result<Self::Cmd, Error>;
    fn open_enc(&mut self, cmd: &Self::Cmd, concurrent: bool) -> Result<Self::Enc, Error>;
    fn end_enc(&mut self, enc: Self::Enc);
    fn commit(&mut self, cmd: &Self::Cmd);
    fn wait(&mut self, cmd: &Self::Cmd) -> (f64, f64);
}

/// Launches encoded on one queue since the last `finish`.
pub(crate) struct Pending<C: PendingCmd, E: PendingEnc, K = ()> {
    /// The context whose queue holds this batch.
    pub(crate) owner: usize,
    pub(crate) cmd: C,
    pub(crate) enc: Option<E>,
    pub(crate) concurrent: bool,
    /// True when `cmd` uses unretained references.
    pub(crate) unretained: bool,
    pub(crate) dirty: bool,
    /// Dispatches encoded into the open `cmd`.
    pub(crate) dispatches: usize,
    /// Earlier chunked command buffers committed on this queue.
    pub(crate) flushed: Vec<C>,
    /// Handles kept alive until every committed chunk finishes.
    pub(crate) keep: Vec<K>,
}

impl<C: PendingCmd, E: PendingEnc, K> Drop for Pending<C, E, K> {
    fn drop(&mut self) {
        if let Some(enc) = self.enc.take() {
            enc.end_open();
        }
        // Drain committed unretained chunks before `keep` drops.
        if let Some(last) = self.flushed.last() {
            last.wait_done();
        }
    }
}

impl<C: PendingCmd, E: PendingEnc + Clone, K> Pending<C, E, K> {
    /// Opens a fresh command buffer and encoder for `owner`.
    pub(crate) fn open<O: PendingOps<Cmd = C, Enc = E>>(
        owner: usize,
        want: bool,
        unretained: bool,
        ops: &mut O,
    ) -> Result<Self, Error> {
        let cmd = ops.new_cmd(unretained)?;
        let enc = ops.open_enc(&cmd, want)?;
        Ok(Self {
            owner,
            cmd,
            enc: Some(enc),
            concurrent: want,
            unretained,
            dirty: false,
            dispatches: 0,
            flushed: Vec::new(),
            keep: Vec::new(),
        })
    }

    /// Commits the open command buffer and opens a fresh one.
    pub(crate) fn rotate<O: PendingOps<Cmd = C, Enc = E>>(
        &mut self,
        want: bool,
        unretained: bool,
        ops: &mut O,
    ) -> Result<(), Error> {
        // Open replacement first so failure preserves `self`.
        let cmd = ops.new_cmd(unretained)?;
        let enc = ops.open_enc(&cmd, want)?;
        if let Some(old_enc) = self.enc.take() {
            ops.end_enc(old_enc);
        }
        ops.commit(&self.cmd);
        let prev = std::mem::replace(&mut self.cmd, cmd);
        self.flushed.push(prev);
        self.enc = Some(enc);
        self.concurrent = want;
        self.unretained = unretained;
        self.dirty = false;
        self.dispatches = 0;
        Ok(())
    }

    /// Matches `want` dispatch mode on the open buffer.
    pub(crate) fn ensure_mode<O: PendingOps<Cmd = C, Enc = E>>(
        &mut self,
        want: bool,
        ops: &mut O,
    ) -> Result<E, Error> {
        if self.concurrent == want
            && let Some(enc) = &self.enc
        {
            return Ok(enc.clone());
        }
        if let Some(old) = self.enc.take() {
            ops.end_enc(old);
        }
        self.dirty = false;
        let enc = ops.open_enc(&self.cmd, want)?;
        self.concurrent = want;
        Ok(self.enc.insert(enc).clone())
    }

    /// Records one encoded dispatch on the open command buffer.
    pub(crate) fn mark_dispatch(&mut self) {
        self.dispatches += 1;
        if self.concurrent {
            self.dirty = true;
        }
    }

    /// Ends open encoder, commits work, and waits.
    pub(crate) fn finish<O: PendingOps<Cmd = C, Enc = E>>(mut self, ops: &mut O) -> f64 {
        if let Some(enc) = self.enc.take() {
            ops.end_enc(enc);
        }
        let mut first_start = f64::INFINITY;
        let mut last_end = 0.0f64;
        let mut fold = |(start, end): (f64, f64)| {
            if start > 0.0 && end >= start {
                first_start = first_start.min(start);
                last_end = last_end.max(end);
            }
        };
        if self.dispatches > 0 || self.flushed.is_empty() {
            ops.commit(&self.cmd);
            fold(ops.wait(&self.cmd));
        }
        for prev in &self.flushed {
            fold(ops.wait(prev));
        }
        self.flushed.clear();
        if first_start.is_finite() {
            (last_end - first_start).max(0.0)
        } else {
            0.0
        }
    }

    /// Fences outstanding concurrent dispatches on the open encoder.
    pub(crate) fn fence(&mut self, barrier: impl FnOnce(&E)) {
        if self.concurrent
            && self.dirty
            && let Some(enc) = &self.enc
        {
            barrier(enc);
            self.dirty = false;
        }
    }

    /// Returns an open retained encoder for `owner` in `want` mode.
    pub(crate) fn acquire<O: PendingOps<Cmd = C, Enc = E>>(
        slot: &mut Option<Self>,
        owner: usize,
        want: bool,
        ops: &mut O,
    ) -> Result<E, Error> {
        match slot {
            Some(p) if p.owner == owner && p.unretained => {
                p.rotate(want, false, ops)?;
                return Ok(p.enc.as_ref().expect("opened encoder").clone());
            }
            Some(p) if p.owner == owner => {
                return p.ensure_mode(want, ops);
            }
            // Another context's work ends before this one's starts.
            Some(_) => {
                if let Some(other) = slot.take() {
                    let _ = other.finish(ops);
                }
            }
            None => {}
        }
        let made = slot.insert(Self::open(owner, want, false, ops)?);
        Ok(made.enc.as_ref().expect("opened encoder").clone())
    }
}
