//! The engine as a speculative decoder's target.

use crate::kept::Kept;
use crate::lanes::Lanes;
use crate::sample::{Sampler, Sampling};
use crate::{Allowed, Edit, Engine, Error, Head, Hooks, Row};
use tessera_engine_comm::Transport;
use tessera_engine_hooks::{Rules, TapsBuf, VerifyOut};
use tessera_graph::Role;
use tessera_lower::{Slot, Wants};
use tessera_run::{Error as RunError, Profile, Run};

#[cfg(test)]
mod vote_test;

/// Plan availability and preference bits exchanged across ranks.
const CAN_OFF: u8 = 1;
const CAN_ON: u8 = 2;
const PREF_ON: u8 = 4;

/// Coordinates `(headless, with_head)` plan selection across ranks before launch.
fn decide_plan(
    control: Option<&mut (dyn Transport + '_)>,
    off: Result<usize, Error>,
    on: Result<usize, Error>,
) -> Result<bool, Error> {
    let (vote, err) = match (off, on) {
        (Ok(h), Ok(w)) => (CAN_OFF | CAN_ON | if h > w { PREF_ON } else { 0 }, None),
        (Ok(_), Err(e @ Error::Lower(_))) => (CAN_OFF, Some(e)),
        (Err(e @ Error::Lower(_)), Ok(_)) => (CAN_ON, Some(e)),
        (Err(Error::Lower(_)), Err(e)) | (Err(e), _) | (_, Err(e)) => (0, Some(e)),
    };
    let votes = match control {
        Some(c) => c
            .exchange(&[vote])
            .map_err(RunError::from)?
            .iter()
            .map(|p| (p.len() == 1).then(|| p[0]))
            .collect::<Option<Vec<u8>>>()
            .unwrap_or_default(),
        None => vec![vote],
    };
    let all = |bit| !votes.is_empty() && votes.iter().all(|v| v & bit != 0);
    match (all(CAN_OFF), all(CAN_ON)) {
        (true, true) => Ok(votes.iter().any(|v| v & PREF_ON != 0)),
        (false, true) => Ok(true),
        (true, false) => Ok(false),
        (false, false) => Err(err.unwrap_or_else(|| RunError::OutputsDisagree.into())),
    }
}

impl Engine {
    /// A local launch picking after every row, with taps.
    fn step_taps_local(&mut self, rows: &[Row]) -> Result<(Vec<u32>, TapsBuf), Error> {
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        let (ids, taps) = self.launch_local(rows, &all, Head::Full(None), true, false)?;
        Ok((ids, taps.expect("taps were asked for")))
    }

    /// Whether all ranks should run `with_head` instead of `headless`.
    fn prefer_head(
        &mut self,
        rows: &[Row],
        last: &[u32],
        headless: Wants,
        with_head: Wants,
    ) -> Result<bool, Error> {
        // Probe both plans before voting across ranks.
        let off = self.steps(rows, &[], headless);
        let on = self.steps(rows, last, with_head);
        let ctl = self.control.as_mut().map(|c| c.as_mut());
        decide_plan(ctl, off, on)
    }

    /// Local launch picking the last row with taps.
    fn step_taps_last(&mut self, rows: &[Row]) -> Result<(u32, TapsBuf), Error> {
        self.check_rows(rows)?;
        let last = [rows.len() as u32 - 1];
        let (ids, taps) = self.launch_local(rows, &last, Head::Full(None), true, false)?;
        Ok((ids[0], taps.expect("taps were asked for")))
    }

    /// Runs `rows` locally: the `first` output, and taps if asked.
    fn tapped(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        (head, peek): (Head, bool),
        first: Role,
        taps: bool,
    ) -> Result<(Slot, Option<TapsBuf>), Error> {
        let width = match taps {
            true => self.taps.ok_or(Error::Taps)?,
            false => 0,
        };
        let roles = &[first, Role::Taps][..1 + taps as usize];
        let wants = Wants::roles(&self.graph, self.full, roles);
        // Outputs come in the entry's order.
        let order = &self.graph.entries[self.full].outputs;
        let wanted = order.iter().filter(|o| roles.contains(&o.role));
        let at = |role| wanted.clone().position(|o| o.role == role).unwrap_or(0);
        let (out_at, taps_at) = (at(first), at(Role::Taps));
        let outputs = self.run_local(rows, picks, head, wants, peek)?;
        let out = outputs[out_at];
        if !taps {
            return Ok((out, None));
        }
        let buf = self.taps_buf(outputs[taps_at], rows.len(), width)?;
        Ok((out, Some(buf)))
    }

    /// Wraps a tap output slot on device or host.
    fn taps_buf(&mut self, slot: Slot, rows: usize, width: usize) -> Result<TapsBuf, Error> {
        if slot.form.name() == "f32"
            && let Some(addr) = self.runner.address(slot)
        {
            return Ok(TapsBuf::Device { slot, addr, rows });
        }
        Ok(TapsBuf::Host(self.read_floats(slot, rows * width)?))
    }
}

impl Hooks for Engine {
    fn context(&self) -> usize {
        self.context
    }

    fn slots(&self) -> usize {
        self.slots
    }

    fn words(&self) -> Option<usize> {
        self.words
    }

    fn edits_fit(&self, n: usize) -> bool {
        Engine::edits_fit(self, n)
    }

    fn stateful(&self) -> bool {
        self.stateful
    }

    fn tap_width(&self) -> Option<usize> {
        self.taps
    }

    fn draft_taps(&self) -> Result<usize, Error> {
        if self.stateful {
            return Err(Error::Stateful);
        }
        self.taps.ok_or(Error::Taps)
    }

    fn rank(&self) -> Option<u32> {
        self.control.as_ref().map(|c| c.rank())
    }

    fn lanes(&self) -> Option<&Lanes> {
        self.lanes.as_ref()
    }

    fn lanes_mut(&mut self) -> Option<&mut Lanes> {
        self.lanes.as_mut()
    }

    fn runner(&self) -> &dyn Run {
        self.runner.as_ref()
    }

    fn copy_from(&mut self, dst: Slot, offset: usize, src: u64, bytes: usize) -> Result<(), Error> {
        Ok(self.runner.copy_from(dst, offset, src, bytes)?)
    }

    fn edited(&self, allowed: &[&Allowed], edits: &[&[Edit]]) -> Result<Head, Error> {
        Engine::edited(self, allowed, edits)
    }

    fn settle(&mut self) -> Result<(), Error> {
        Engine::settle(self)
    }

    fn told(&mut self, mine: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        Engine::told(self, mine)
    }

    fn take_held(&mut self) -> Option<Profile> {
        self.held.take().map(|(_, p)| p)
    }

    fn sync(&mut self, stopped: bool, hash: u32) -> Result<bool, Error> {
        let Some(control) = &mut self.control else {
            return Ok(stopped);
        };
        let lead = control.rank() == 0;
        let mut mine = Vec::with_capacity(5);
        mine.push(u8::from(lead && stopped));
        mine.extend_from_slice(&hash.to_le_bytes());
        if !lead && let Some((_, p)) = self.held.take() {
            mine.extend_from_slice(&p.to_bytes());
        }
        let parts = control.exchange(&mine).map_err(RunError::from)?;
        let first = parts.first().ok_or(RunError::OutputsDisagree)?;
        if first.len() < 5 {
            return Err(RunError::OutputsDisagree.into());
        }
        let stop = first[0] != 0;
        let want = u32::from_le_bytes(first[1..5].try_into().unwrap());
        for p in &parts[1..] {
            if p.len() < 5 || u32::from_le_bytes(p[1..5].try_into().unwrap()) != want {
                return Err(RunError::OutputsDisagree.into());
            }
        }
        if lead {
            self.gather(parts[1..].iter().map(|p| Profile::from_bytes(&p[5..])));
        }
        Ok(stop)
    }

    fn commit(&mut self, rows: usize) -> Result<(), Error> {
        Engine::commit(self, rows)
    }

    fn grow_kept(&mut self, kept: &mut Kept, rows: &[Row]) -> Result<(), Error> {
        let Some(lanes) = &mut self.lanes else {
            return Ok(());
        };
        let runner = &mut self.runner;
        kept.grow_rows_tiered(rows, lanes, &mut |promote, pairs| {
            Ok(runner.move_kv_pages(promote, pairs)?)
        })
    }

    fn restore_kv(
        &mut self,
        kept: Option<&mut Kept>,
        slot: u32,
        tokens: &[u32],
        skip: usize,
        lead: bool,
    ) -> Result<usize, Error> {
        Engine::restore_kv(self, kept, slot, tokens, skip, lead)
    }

    fn spill_kv(&mut self, kept: Option<&Kept>, slot: u32, history: &[u32], lead: bool) {
        Engine::spill_kv(self, kept, slot, history, lead)
    }

    fn fill_local(&mut self, rows: &[Row]) -> Result<(), Error> {
        self.check_rows(rows)?;
        let tokens = Wants::role(&self.graph, self.full, Role::Tokens);
        let last = [rows.len() as u32 - 1];
        if self.prefer_head(rows, &last, Wants::NONE, tokens)? {
            self.launch_local(rows, &last, Head::Full(None), false, false)?;
            return Ok(());
        }
        self.run_local(rows, &[], Head::Full(None), Wants::NONE, false)?;
        self.report();
        Ok(())
    }

    fn step_taps_only(&mut self, rows: &[Row]) -> Result<TapsBuf, Error> {
        self.check_rows(rows)?;
        let width = self.taps.ok_or(Error::Taps)?;
        let wants = Wants::role(&self.graph, self.full, Role::Taps);
        let both = Wants::roles(&self.graph, self.full, &[Role::Tokens, Role::Taps]);
        let last = [rows.len() as u32 - 1];
        if self.prefer_head(rows, &last, wants, both)? {
            let (_, buf) = self.launch_local(rows, &last, Head::Full(None), true, false)?;
            return Ok(buf.expect("taps were asked for"));
        }
        let outputs = self.run_local(rows, &[], Head::Full(None), wants, false)?;
        let buf = self.taps_buf(outputs[0], rows.len(), width)?;
        self.report();
        Ok(buf)
    }

    fn step_verify_local(
        &mut self,
        rows: &[Row],
        sampling: bool,
    ) -> Result<(VerifyOut, TapsBuf), Error> {
        if !sampling {
            let (ids, taps) = self.step_taps_local(rows)?;
            return Ok((VerifyOut::Ids(ids), taps));
        }
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        let (logits, vocab, taps) = self.logits_local(rows, &all, true, false)?;
        Ok((
            VerifyOut::Logits(logits, vocab),
            taps.expect("taps were asked for"),
        ))
    }

    fn prefill_pick(
        &mut self,
        rows: &[Row],
        sampler: &mut Sampler,
    ) -> Result<(u32, TapsBuf), Error> {
        if sampler.greedy() {
            return self.step_taps_last(rows);
        }
        let (mut logits, _, taps) = self.logits_local(rows, &[0], true, false)?;
        let anchor = sampler.draw_row(&mut logits, None, &[]);
        Ok((anchor, taps.expect("taps were asked for")))
    }

    fn logits_local(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        taps: bool,
        peek: bool,
    ) -> Result<(Vec<f32>, usize, Option<TapsBuf>), Error> {
        let vocab = self.vocab.ok_or(Error::Scores)?;
        let (out, taps) = self.tapped(rows, picks, (Head::Full(None), peek), Role::Logits, taps)?;
        let logits = self.read_floats(out, picks.len() * vocab)?;
        self.report();
        Ok((logits, vocab, taps))
    }

    fn launch_local(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        head: Head,
        taps: bool,
        peek: bool,
    ) -> Result<(Vec<u32>, Option<TapsBuf>), Error> {
        let (out, taps) = self.tapped(rows, picks, (head, peek), Role::Tokens, taps)?;
        let mut ids = self.runner.ids(out)?;
        ids.truncate(picks.len());
        self.report();
        Ok((ids, taps))
    }

    fn pick_last(
        &mut self,
        rows: &[Row],
        rules: &Rules,
        taps: bool,
    ) -> Result<(u32, Option<TapsBuf>), Error> {
        let last = [rows.len() as u32 - 1];
        if self.on_device(rules) {
            let head = self.head_of(std::slice::from_ref(rules))?;
            let (ids, taps) = self.launch_local(rows, &last, head, taps, false)?;
            return Ok((ids[0], taps));
        }
        // Rules the device head cannot apply run on the host.
        let (mut row, _, taps) = self.logits_local(rows, &last, taps, false)?;
        let mut sampler = Sampler::new(Sampling::default());
        Ok((sampler.draw_row(&mut row, Some(&rules.0), &rules.1), taps))
    }
}
