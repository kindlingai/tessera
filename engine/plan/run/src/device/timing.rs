//! Step timing, deferred graph/buffer retirement, and staged reloads.

use super::{Runner, device_buf};
use crate::Error;
use crate::cache::Lru;
use crate::kernels;
use crate::marks;
use crate::overrides::{self, Pending, Ready};
use crate::profile::Marks;
use crate::rank::Sum;
use crate::run::{Reload, Run};
use std::collections::HashSet;
use std::fmt;
use std::ops::Range;
use std::time::Instant;
use tessera_gpukit::device::Device;
use tessera_gpukit::launch::Launch;
use tessera_lower::Step;

impl<D: Device> Runner<D> {
    pub(super) fn override_launch(&self, step: usize) -> Option<&overrides::LaunchOverride> {
        let files = self.overrides.as_ref()?;
        let idx = *self.uses.get(step)?;
        files.launch(&self.entries[idx].0)
    }

    /// Passes profiling state and resident overrides to the grid.
    pub(super) fn sync_grid(&mut self) {
        let Some(grid) = self.grid.get_mut().as_mut() else {
            return;
        };
        grid.profiling(self.base.profile.is_some());
        let Some(files) = &self.overrides else {
            return;
        };
        for (entry, step) in &self.entries {
            if self.resident[*step] {
                grid.replace(entry, files.text(entry).map(str::to_string));
            }
        }
    }

    /// Retains owned handles when teardown cannot quiesce.
    pub(crate) fn retain(&mut self) {
        std::mem::forget(std::mem::replace(&mut self.graphs, Lru::new(0)));
        std::mem::forget(std::mem::take(&mut self.evicted));
        std::mem::forget(std::mem::take(&mut self.retired));
        std::mem::forget(std::mem::take(&mut self.kernels));
        std::mem::forget(std::mem::take(&mut self.planned));
        std::mem::forget(self.overrides.take());
        std::mem::forget(std::mem::take(&mut self.marked));
        std::mem::forget(std::mem::take(&mut self.compiled));
        std::mem::forget(std::mem::replace(&mut self.plans, Lru::new(0)));
        std::mem::forget(std::mem::take(&mut self.base.held));
        std::mem::forget(self.ledger.take());
        std::mem::forget(self.stamps.take());
        std::mem::forget(self.pager.take());
        std::mem::forget(std::mem::take(&mut self.kv));
        std::mem::forget(self.grid.get_mut().take());
    }
}

impl<D: Device> Drop for Runner<D> {
    fn drop(&mut self) {
        let live = !self.graphs.is_empty()
            || !self.evicted.is_empty()
            || !self.retired.is_empty()
            || !self.kernels.is_empty()
            || !self.planned.is_empty()
            || self.overrides.is_some()
            || !self.marked.is_empty()
            || !self.compiled.is_empty()
            || !self.plans.is_empty()
            || self.base.held.total() > 0
            || self.ledger.is_some()
            || self.stamps.is_some()
            || self.pager.is_some()
            || self.kv.stage.is_some()
            || self.kv.host.is_some()
            || self.grid.get_mut().is_some();
        if !live {
            return;
        }
        if self.ctx.finish().is_err() {
            self.retain();
            return;
        }
        // Drop captured graphs and grid before buffers and kernels.
        self.graphs.clear();
        self.evicted.clear();
        self.retired.clear();
        let _ = self.grid.get_mut().take();
    }
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Runner<D> {
    /// Each step's seconds by waiting, or `None` when stamped.
    pub(super) fn walk(&self, steps: Range<usize>) -> Result<Option<Vec<f64>>, Error> {
        let stamped = self.ctx.stamp()?;
        let mut waited = Vec::new();
        for (i, step) in self.base.plan.sequence(steps) {
            let start = Instant::now();
            let Some(i) = i else {
                self.stage(step, None)?;
                self.clock(stamped, start, &mut waited)?;
                continue;
            };
            for launch in self.base.launches(i)? {
                self.launch(i, &launch)?;
            }
            self.clock(stamped, start, &mut waited)?;
        }
        Ok((!stamped).then_some(waited))
    }

    /// Ends one step's time: a stamp, else a wait.
    fn clock(&self, stamped: bool, start: Instant, waited: &mut Vec<f64>) -> Result<(), Error> {
        if stamped {
            self.ctx.stamp()?;
            return Ok(());
        }
        let clocked = self.ctx.finish()?;
        waited.push(clocked.unwrap_or_else(|| start.elapsed().as_secs_f64()));
        Ok(())
    }

    /// Queues a copy step, stamped at `slot` when given.
    pub(super) fn stage(
        &self,
        step: &Step,
        stamp: Option<(&marks::Stamps<D>, usize)>,
    ) -> Result<(), Error> {
        if let Some((stamps, slot)) = stamp {
            stamps.emit(self.ctx, slot, marks::LANE_BEGIN)?;
        }
        if let (Some(batch), Some(pager)) = (step.kernel.staged(), &self.pager) {
            pager.stage(self.ctx, batch as usize, self.base.plan.id)?;
        }
        if let Some((stamps, slot)) = stamp {
            stamps.emit(self.ctx, slot, marks::LANE_END)?;
        }
        Ok(())
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
        let base = &mut self.base;
        if let Some(p) = base.profile.as_mut() {
            let order = base.plan.sequence(steps);
            for ((i, step), s) in order.into_iter().zip(waited.unwrap_or(stamps)) {
                let attr = i.and_then(|i| self.attrs.get(i).copied().flatten());
                p.step_with(step, s, 1, attr);
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
        steps: &Range<usize>,
        sums: &mut Option<&mut (dyn Sum + 'static)>,
    ) -> Result<(), Error> {
        let len = marks::slots(&self.base.plan, steps.clone()).len();
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
                self.clear_graphs()?;
            }
            None => self.stamps = Some(marks::Stamps::open(self.ctx, len)?),
        }
        Ok(())
    }

    /// Rejects launches while prior binding or retirement is incomplete.
    pub(super) fn check_ready(&self) -> Result<(), Error> {
        if self.dirty || !self.evicted.is_empty() || !self.retired.is_empty() {
            return Err(Error::Inconsistent);
        }
        Ok(())
    }

    /// Drains the stream before dropping evicted graphs and retired buffers.
    pub(super) fn finish_evicted(&mut self) -> Result<(), Error> {
        if self.evicted.is_empty() && self.retired.is_empty() {
            return Ok(());
        }
        self.ctx.finish()?;
        // Drop graphs before freeing buffers their captures referenced.
        self.evicted.clear();
        self.retired.clear();
        Ok(())
    }

    /// Pauses `sum`, evicts stale graphs, syncs, then drops retired state.
    pub(super) fn retire(
        &mut self,
        sum: Option<&mut (dyn Sum + 'static)>,
        clear_all: bool,
    ) -> Result<(), Error> {
        if self.retired.is_empty()
            && self.evicted.is_empty()
            && !(clear_all && !self.graphs.is_empty())
        {
            return Ok(());
        }
        if let Some(g) = sum {
            g.pause()?;
        }
        if clear_all {
            for set in self.graphs.drain_if(|_| true) {
                self.evicted.extend(set.ranges.into_values());
            }
        } else if !self.retired.is_empty() {
            let dead: HashSet<u64> = self
                .retired
                .iter()
                .map(|b| device_buf::<D>(b).device_ptr())
                .collect();
            // Drop only graphs referencing retired device addresses.
            for set in self
                .graphs
                .drain_if(|set| set.ptrs.iter().any(|p| dead.contains(p)))
            {
                self.evicted.extend(set.ranges.into_values());
            }
        }
        self.finish_evicted()
    }

    /// Clears all graphs after syncing. Keeps ownership on sync error.
    pub(super) fn clear_graphs(&mut self) -> Result<(), Error> {
        if self.graphs.is_empty() && self.evicted.is_empty() {
            return Ok(());
        }
        for set in self.graphs.drain_if(|_| true) {
            self.evicted.extend(set.ranges.into_values());
        }
        if let Err(e) = self.finish_evicted() {
            self.dirty = true;
            return Err(e);
        }
        Ok(())
    }

    /// Clears graphs from non-fallible setters without unsafe drops.
    pub(super) fn drop_graphs(&mut self) {
        if self.clear_graphs().is_err() {
            self.dirty = true;
        }
    }

    /// Recompiles changed override files after pausing any running grid.
    pub(super) fn reload_with(
        &mut self,
        sum: Option<&mut (dyn Sum + 'static)>,
    ) -> Result<Vec<Reload>, Error> {
        let Some(files) = self.overrides.as_ref() else {
            return Ok(Vec::new());
        };
        let plan = &self.base.plan;
        let mut pending = Vec::new();
        for (entry, step) in &self.entries {
            let planned = || overrides::text(&plan.steps[*step], D::BACKEND);
            if let Some(change) = files.scan(entry, self.resident[*step], planned) {
                pending.push(change);
            }
        }
        let needs_pause = pending
            .iter()
            .any(|p| matches!(p, Pending::Revert { .. } | Pending::Compile { .. }));
        if needs_pause && let Some(grid) = sum {
            grid.pause()?;
        }
        let ctx = self.ctx;
        let mut ready = Vec::with_capacity(pending.len());
        let mut changed = false;
        for item in pending {
            match item {
                Pending::Unchanged { entry, seen } => ready.push(Ready::unchanged(entry, seen)),
                Pending::Revert { entry, path, seen } => {
                    changed = true;
                    ready.push(Ready::reverted(entry, path, seen));
                }
                Pending::Failed {
                    entry,
                    path,
                    seen,
                    why,
                } => ready.push(Ready::failed(entry, path, seen, why)),
                Pending::Compile {
                    entry,
                    path,
                    seen,
                    text,
                    launch,
                } => {
                    let flags = launch
                        .as_ref()
                        .map_or_else(Vec::new, overrides::LaunchOverride::flags);
                    match ctx.compile_with(&format!("{}{text}", marks::OFF), &entry, &flags) {
                        Ok(kernel) => {
                            changed = true;
                            ready.push(Ready::loaded(entry, path, seen, kernel, text, launch));
                        }
                        Err(e) => ready.push(Ready::failed(entry, path, seen, e.to_string())),
                    }
                }
            }
        }
        if changed {
            // Recorded launches and eager kernels must finish before unload.
            if self.graphs.is_empty() && self.evicted.is_empty() {
                if let Err(e) = self.ctx.finish() {
                    self.dirty = true;
                    return Err(e.into());
                }
            } else {
                self.clear_graphs()?;
            }
        }
        let files = self.overrides.as_mut().expect("checked above");
        let mut reloads = Vec::new();
        for item in ready {
            if let Some(reload) = files.commit(item) {
                reloads.push(reload);
            }
        }
        if changed {
            self.apply();
            self.attrs = kernels::step_attrs(
                ctx,
                &self.base.plan,
                &self.kernels,
                &self.resident,
                self.overrides.as_ref(),
            );
            self.sync_grid();
        }
        Ok(reloads)
    }
}
