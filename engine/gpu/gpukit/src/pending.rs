//! Thread-local command buffer and compute encoder state.

use crate::Error;

/// Command buffer lifecycle hooks for drop teardown.
pub(crate) trait PendingCmd {
    fn commit_open(&self);
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
    pub(crate) owner: u64,
    pub(crate) cmd: Option<C>,
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
        // Submit any encoded tail so teardown finishes the batch.
        if let Some(cmd) = self.cmd.take()
            && self.dispatches > 0
        {
            cmd.commit_open();
            self.flushed.push(cmd);
        }
        // Drain every committed chunk before `keep` drops.
        for cmd in &self.flushed {
            cmd.wait_done();
        }
    }
}

impl<C: PendingCmd, E: PendingEnc + Clone, K> Pending<C, E, K> {
    /// Opens a fresh command buffer and encoder for `owner`.
    pub(crate) fn open<O: PendingOps<Cmd = C, Enc = E>>(
        owner: u64,
        want: bool,
        unretained: bool,
        ops: &mut O,
    ) -> Result<Self, Error> {
        let cmd = ops.new_cmd(unretained)?;
        let enc = ops.open_enc(&cmd, want)?;
        Ok(Self {
            owner,
            cmd: Some(cmd),
            enc: Some(enc),
            concurrent: want,
            unretained,
            dirty: false,
            dispatches: 0,
            flushed: Vec::new(),
            keep: Vec::new(),
        })
    }

    /// Commits the open buffer before acquiring its replacement.
    pub(crate) fn rotate<O: PendingOps<Cmd = C, Enc = E>>(
        &mut self,
        want: bool,
        unretained: bool,
        ops: &mut O,
    ) -> Result<(), Error> {
        // Release the current queue slot before requesting another buffer.
        if let Some(old_enc) = self.enc.take() {
            ops.end_enc(old_enc);
        }
        if let Some(prev) = self.cmd.take()
            && self.dispatches > 0
        {
            ops.commit(&prev);
            self.flushed.push(prev);
        }
        self.dirty = false;
        self.dispatches = 0;
        self.concurrent = want;
        self.unretained = unretained;
        let cmd = ops.new_cmd(unretained)?;
        self.cmd = Some(cmd);
        let cmd = self.cmd.as_ref().expect("open command buffer");
        let enc = ops.open_enc(cmd, want)?;
        self.enc = Some(enc);
        Ok(())
    }

    /// Matches `want` dispatch mode on the open buffer.
    pub(crate) fn ensure_mode<O: PendingOps<Cmd = C, Enc = E>>(
        &mut self,
        want: bool,
        unretained: bool,
        ops: &mut O,
    ) -> Result<E, Error> {
        if self.cmd.is_none() {
            let cmd = ops.new_cmd(unretained)?;
            self.unretained = unretained;
            self.cmd = Some(cmd);
        }
        if self.concurrent == want
            && let Some(enc) = &self.enc
        {
            return Ok(enc.clone());
        }
        if let Some(old) = self.enc.take() {
            ops.end_enc(old);
        }
        self.dirty = false;
        let cmd = self.cmd.as_ref().expect("open command buffer");
        let enc = ops.open_enc(cmd, want)?;
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
        if let Some(cmd) = self.cmd.take()
            && (self.dispatches > 0 || self.flushed.is_empty())
        {
            ops.commit(&cmd);
            fold(ops.wait(&cmd));
        }
        self.dispatches = 0;
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
        owner: u64,
        want: bool,
        ops: &mut O,
    ) -> Result<E, Error> {
        match slot {
            Some(p) if p.owner == owner && p.unretained => {
                p.rotate(want, false, ops)?;
                return Ok(p.enc.as_ref().expect("opened encoder").clone());
            }
            Some(p) if p.owner == owner => {
                return p.ensure_mode(want, false, ops);
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
