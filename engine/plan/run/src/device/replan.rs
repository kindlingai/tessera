//! Plan switching: cached tables, binding, and kernel compilation.

use super::Runner;
use crate::Error;
use crate::cache::Prepared;
use crate::kernels;
use crate::marks;
use crate::paged;
use crate::rank::Sum;
use std::time::Instant;
use tessera_gpukit::device::{Device, cached_prefixed, cached_source};
use tessera_gpukit::launch::Launch;
use tessera_lower::Plan;
use tessera_mem::Buffer;

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Runner<D> {
    /// Moves the active plan's tables into the plan cache.
    fn stash_plan(&mut self) {
        let id = self.base.plan.id;
        if id == 0 {
            return;
        }
        let mut plan = Plan::default();
        let mut launches = Vec::new();
        self.base.swap(&mut plan, &mut launches);
        let saved = Prepared {
            plan,
            launches,
            kernels: std::mem::take(&mut self.kernels),
            planned: std::mem::take(&mut self.planned),
            entries: std::mem::take(&mut self.entries),
            uses: std::mem::take(&mut self.uses),
            waits: std::mem::take(&mut self.waits),
            resident: std::mem::take(&mut self.resident),
            attrs: std::mem::take(&mut self.attrs),
            range_dims: std::mem::take(&mut self.range_dims),
        };
        self.plans.insert(id, saved);
    }

    /// Installs a cached plan's tables as active.
    fn restore_plan(&mut self, mut cached: Prepared<<D as Device>::Kernel>) {
        self.base.swap(&mut cached.plan, &mut cached.launches);
        self.kernels = cached.kernels;
        self.planned = cached.planned;
        self.entries = cached.entries;
        self.uses = cached.uses;
        self.waits = cached.waits;
        self.resident = cached.resident;
        self.attrs = cached.attrs;
        self.range_dims = cached.range_dims;
    }

    /// Reads each step's compiled resource counts.
    fn read_attrs(&mut self, plan: &Plan) {
        self.attrs = kernels::step_attrs(
            self.ctx,
            plan,
            &self.kernels,
            &self.resident,
            self.overrides.as_ref(),
        );
    }

    /// Finishes switching to a plan whose kernels are already built.
    fn rebind(&mut self, plan: &Plan) -> Result<(), Error> {
        self.apply();
        if self.overrides.is_some() {
            self.read_attrs(plan);
        }
        self.sync_grid();
        self.ready(plan)?;
        self.dirty = false;
        Ok(())
    }

    /// Replans, pausing `sum` before any allocation or compilation.
    pub(crate) fn replan_with(
        &mut self,
        plan: &Plan,
        mut sum: Option<&mut (dyn Sum + 'static)>,
    ) -> Result<(), Error> {
        let pending = self.dirty || !self.retired.is_empty() || !self.evicted.is_empty();
        if !pending && plan.id != 0 && plan.id == self.base.plan.id {
            return Ok(());
        }
        if !self.dirty {
            self.flush_stamps()?;
        }
        if self.pager.is_none()
            && self
                .base
                .budget
                .is_some_and(|l| plan.buffers.iter().map(|&b| b as u64).sum::<u64>() > l)
            && let Some(g) = sum.as_deref_mut()
        {
            g.pause()?;
        }
        self.base.shared_bus |= sum.as_deref().is_some_and(|s| s.shares_bus());
        let (plan, extra) = paged::prepare(&mut self.pager, self.ctx, plan, &self.base)?;
        self.base.beside = extra;
        let pager = self.pager.as_ref();
        let sized = |w, p| pager.and_then(|g| g.slot_bytes(w, p));
        let plan = self.base.held.reuse_weights(plan.as_ref(), sized);
        let plan = plan.as_ref();
        if pending {
            self.base.validate(plan, extra)?;
        } else if !self.base.fresh_with(plan, extra)? {
            return Ok(());
        }
        let (ctx, unified) = (self.ctx, self.ctx.unified());
        let alloc_start = self.base.profile.is_some().then(Instant::now);
        let mut allocated = false;
        let pager = self.pager.as_ref();
        let sized = |w, p| pager.and_then(|g| g.slot_bytes(w, p));
        self.dirty = true;
        let before = self.retired.len();
        let bound =
            self.base
                .held
                .bind_sized(plan, sized, &mut self.retired, |zone, bytes, holds| {
                    allocated = true;
                    if let Some(g) = sum.as_deref_mut() {
                        g.pause()?;
                    }
                    let buffer = crate::held::alloc_buf(ctx, unified, bytes, holds)?;
                    Ok(Buffer::new(zone, buffer))
                });
        if let Err(e) = bound {
            // Keep retired buffers alive until graphs evict and stream syncs.
            let _ = self.retire(sum.as_deref_mut(), true);
            return Err(e);
        }
        let retired_count = self.retired.len() - before;
        if !self.retired.is_empty() || !self.evicted.is_empty() {
            self.retire(sum.as_deref_mut(), false)?;
        }
        if (allocated || retired_count > 0)
            && let Some(t0) = alloc_start
        {
            self.note_cold(
                "plan: alloc",
                t0,
                format_args!("plan={} retired={retired_count}", plan.id),
            );
        }
        self.ptrs = self.bound_ptrs();
        self.kv.bind(plan);
        if plan.id != 0 && plan.id == self.base.plan.id && self.kernels.len() == plan.steps.len() {
            return self.rebind(plan);
        }
        if plan.id != 0
            && let Some(cached) = self.plans.remove(&plan.id)
        {
            self.stash_plan();
            self.restore_plan(cached);
            return self.rebind(plan);
        }
        let compile_start = self.base.profile.is_some().then(Instant::now);
        if let Some(g) = sum {
            g.pause()?;
        }
        let resident: Vec<bool> = plan
            .steps
            .iter()
            .map(|s| s.kernel.resident().is_some())
            .collect();
        let mut kernels = Vec::with_capacity(plan.steps.len());
        let mut waits = Vec::with_capacity(plan.steps.len());
        for step in &plan.steps {
            let fixed = |text: &'static str, entry: &'static str| -> Result<_, Error> {
                match text.contains("TS_MARK") {
                    true => Ok(cached_prefixed::<D>(marks::OFF, text, entry)?),
                    false => Ok(cached_source::<D>(text, entry)?),
                }
            };
            let made = |s: &str, e: &str| -> Result<_, Error> { Ok(ctx.compile(s, e)?) };
            // Generated kernels wait on `griddepcontrol`. Fixed ones do not.
            let (kernel, generated) = self.compiled.get(step, D::BACKEND, fixed, made)?;
            kernels.push(kernel);
            waits.push(generated);
        }
        if self.base.plan.id != plan.id {
            self.stash_plan();
        }
        self.range_dims.clear();
        self.planned = kernels.clone();
        self.kernels = kernels;
        self.waits = waits;
        self.resident = resident;
        self.index(plan);
        self.apply();
        self.read_attrs(plan);
        if self.log_attrs || self.base.profile.is_some() {
            kernels::log_spills(plan, &self.attrs, &mut self.spills);
        }
        self.sync_grid();
        self.ready(plan)?;
        self.base.adopt(plan);
        self.dirty = false;
        if let Some(t0) = compile_start {
            self.note_cold(
                "plan: compile",
                t0,
                format_args!("plan={} steps={}", plan.id, plan.steps.len()),
            );
        }
        Ok(())
    }
}
