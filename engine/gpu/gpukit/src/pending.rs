//! Thread-local command buffer and compute encoder state.

use crate::Error;

/// Backend hooks for a pending command buffer and encoder.
pub(crate) trait PendingOps {
    type Cmd;
    type Enc: Clone;

    fn new_cmd(&mut self) -> Result<Self::Cmd, Error>;
    fn open_enc(&mut self, cmd: &Self::Cmd, concurrent: bool) -> Result<Self::Enc, Error>;
    fn end_enc(&mut self, enc: Self::Enc);
    fn commit(&mut self, cmd: Self::Cmd) -> f64;
}

/// Launches encoded on one command buffer since the last `finish`.
pub(crate) struct Pending<C, E> {
    /// The context whose queue holds it.
    pub(crate) owner: usize,
    pub(crate) cmd: C,
    pub(crate) enc: Option<E>,
    pub(crate) concurrent: bool,
    pub(crate) dirty: bool,
}

impl<C, E: Clone> Pending<C, E> {
    /// Ends any open encoder, commits, and waits.
    pub(crate) fn finish<O: PendingOps<Cmd = C, Enc = E>>(self, ops: &mut O) -> f64 {
        if let Some(enc) = self.enc {
            ops.end_enc(enc);
        }
        ops.commit(self.cmd)
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

    /// Returns an open encoder for `owner` in `want` mode.
    pub(crate) fn acquire<O: PendingOps<Cmd = C, Enc = E>>(
        slot: &mut Option<Self>,
        owner: usize,
        want: bool,
        ops: &mut O,
    ) -> Result<E, Error> {
        match slot {
            Some(p) if p.owner == owner && p.concurrent == want => {
                if let Some(enc) = &p.enc {
                    return Ok(enc.clone());
                }
                let enc = ops.open_enc(&p.cmd, want)?;
                p.dirty = false;
                return Ok(p.enc.insert(enc).clone());
            }
            Some(p) if p.owner == owner => {
                if let Some(old) = p.enc.take() {
                    ops.end_enc(old);
                }
                p.dirty = false;
                let enc = ops.open_enc(&p.cmd, want)?;
                p.concurrent = want;
                return Ok(p.enc.insert(enc).clone());
            }
            // Another context's work ends before this one's starts.
            Some(_) => {
                if let Some(other) = slot.take() {
                    let _ = other.finish(ops);
                }
            }
            None => {}
        }
        let cmd = ops.new_cmd()?;
        let enc = ops.open_enc(&cmd, want)?;
        let made = slot.insert(Self {
            owner,
            cmd,
            enc: Some(enc),
            concurrent: want,
            dirty: false,
        });
        Ok(made.enc.as_ref().expect("opened encoder").clone())
    }
}
