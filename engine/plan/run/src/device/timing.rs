//! Step timing by device stamps or by waiting.

use super::Runner;
use crate::Error;
use crate::marks;
use crate::profile::Marks;
use crate::rank::Sum;
use crate::run::Run;
use std::fmt;
use std::ops::Range;
use std::time::Instant;
use tessera_gpukit::device::Device;
use tessera_gpukit::launch::Launch;

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Runner<D> {
    /// Each step's seconds by waiting, or `None` when stamped.
    pub(super) fn walk(&self, steps: Range<usize>) -> Result<Option<Vec<f64>>, Error> {
        let stamped = self.ctx.stamp()?;
        let mut waited = Vec::new();
        for i in steps {
            let start = Instant::now();
            for launch in self.base.launches(i)? {
                self.launch(i, &launch)?;
            }
            match stamped {
                true => _ = self.ctx.stamp()?,
                false => {
                    let clocked = self.ctx.finish()?;
                    waited.push(clocked.unwrap_or_else(|| start.elapsed().as_secs_f64()));
                }
            }
        }
        Ok((!stamped).then_some(waited))
    }

    /// Opens the gate, then reads stamps. Errors discard stamps.
    pub(super) fn opened<T>(&self, done: Result<T, Error>) -> Result<(T, Vec<f64>), Error> {
        self.open();
        match done {
            Ok(done) => Ok((done, self.ctx.stamps()?)),
            Err(e) => {
                _ = self.ctx.stamps();
                Err(e)
            }
        }
    }

    pub(super) fn grid_marks(&mut self) -> Vec<(String, Marks)> {
        self.grid
            .get_mut()
            .as_mut()
            .map_or_else(Vec::new, |g| g.take_marks())
    }

    /// Times each step by device stamps, else by waiting.
    pub(super) fn profiled(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.close()?;
        let walked = self.walk(steps.clone());
        let (waited, stamps) = self.opened(walked)?;
        for (i, s) in steps.zip(waited.unwrap_or(stamps)) {
            if let Some(p) = self.base.profile.as_mut() {
                let attr = self.attrs.get(i).copied().flatten();
                p.step_with(&self.base.plan.steps[i], s, 1, attr);
            }
        }
        for (entry, found) in self.grid_marks() {
            for region in &found.regions {
                let (middle, _) = region.spread();
                let label = format!("mark {entry}: L{}-L{}", region.from, region.to);
                self.note(&label, middle);
            }
        }
        Ok(())
    }

    /// Runs `work`, optionally waiting. Returns the device's seconds.
    pub(super) fn waited(
        &self,
        wait: bool,
        timed: bool,
        work: impl FnOnce(&Self) -> Result<(), Error>,
    ) -> Result<Option<f64>, Error> {
        if !wait || !timed {
            work(self)?;
            return Ok(if wait { self.ctx.finish()? } else { None });
        }
        self.close()?;
        let stamped = self.ctx.stamp().map_err(Error::from).and_then(|stamped| {
            work(self)?;
            if stamped {
                self.ctx.stamp()?;
            }
            Ok(stamped)
        });
        match self.opened(stamped)? {
            (true, stamps) => Ok(stamps.first().copied()),
            (false, _) => Ok(self.ctx.finish()?),
        }
    }

    pub(super) fn flush_stamps(&mut self) -> Result<(), Error> {
        let Some(stamps) = &mut self.stamps else {
            return Ok(());
        };
        stamps.flush(
            self.ctx,
            &self.base.plan,
            &self.attrs,
            self.base.profile.as_mut(),
        )
    }

    pub(super) fn note_cold(&mut self, phase: &str, since: Instant, detail: impl fmt::Display) {
        let secs = since.elapsed().as_secs_f64();
        self.note(phase, secs);
        eprintln!("tessera: cold {phase} ({detail}) {:.2}ms", secs * 1e3);
    }

    pub(super) fn ensure_stamps(
        &mut self,
        len: usize,
        sums: &mut Option<&mut (dyn Sum + 'static)>,
    ) -> Result<(), Error> {
        let need = self.stamps.as_ref().is_none_or(|s| s.needs_grow(len));
        if !need {
            return Ok(());
        }
        self.flush_stamps()?;
        if let Some(grid) = sums.as_deref_mut() {
            grid.pause()?;
        }
        match &mut self.stamps {
            Some(stamps) => {
                stamps.grow(self.ctx, len)?;
                self.clear_graphs();
            }
            None => self.stamps = Some(marks::Stamps::open(self.ctx, len)?),
        }
        Ok(())
    }
}
