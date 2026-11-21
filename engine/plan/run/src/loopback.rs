//! Every rank in one process, crossing through host memory.

use crate::Error;
use crate::Run;
use crate::open;
use tessera_engine_comm::{gathered, summed};
use tessera_engine_settings::Settings;
use tessera_lower::kernels::{AllGather, crosses};
use tessera_lower::{Plan, Weight};
use tessera_mem::{Tensor, View};

/// One runner per rank. Sums run in rank order.
pub struct Loopback {
    ranks: Vec<Box<dyn Run>>,
}

impl Loopback {
    /// A runner per plan, one per rank in order.
    pub fn open(plans: &[Plan], settings: &Settings) -> Result<Self, Error> {
        let mut ranks = Vec::with_capacity(plans.len());
        for (rank, plan) in plans.iter().enumerate() {
            if plan.rank as usize != rank || plan.ranks.max(1) as usize != plans.len() {
                return Err(Error::RankOrder);
            }
            ranks.push(open(plan.clone(), settings)?);
        }
        Ok(Self { ranks })
    }

    pub fn rank(&self, r: usize) -> &dyn Run {
        &*self.ranks[r]
    }

    /// Rank `r` alone, to take bytes cut for it.
    pub fn rank_mut(&mut self, r: usize) -> &mut dyn Run {
        &mut *self.ranks[r]
    }

    /// Every rank takes its slice of the whole weight.
    pub fn load(&mut self, weight: u32, data: View) -> Result<(), Error> {
        for r in &mut self.ranks {
            r.load(weight, data.clone())?;
        }
        Ok(())
    }

    /// Every rank takes its part of the weight as stored.
    pub fn load_weight(&mut self, weight: u32, w: &Weight) -> Result<(), Error> {
        for r in &mut self.ranks {
            r.load_weight(weight, w)?;
        }
        Ok(())
    }

    /// Runs each rank's own steps. Sums meet between steps.
    pub fn run(&mut self, inputs: &[View]) -> Result<Vec<Tensor>, Error> {
        for r in &mut self.ranks {
            r.feed(inputs)?;
        }
        // Uneven ranks lower to different steps.
        let sums: Vec<Vec<usize>> = self.ranks.iter().map(|r| sums(r.plan())).collect();
        if sums.iter().any(|s| s.len() != sums[0].len()) {
            return Err(Error::RankOrder);
        }
        let mut from = vec![0; self.ranks.len()];
        for at in 0..=sums[0].len() {
            for (r, rank) in self.ranks.iter_mut().enumerate() {
                let to = sums[r].get(at).copied();
                let to = to.unwrap_or(rank.plan().steps.len());
                rank.submit(from[r]..to)?;
                from[r] = to + 1;
            }
            if at == sums[0].len() {
                break;
            }
            let step = |r: usize| &self.ranks[r].plan().steps[sums[r][at]];
            let mut parts = Vec::with_capacity(self.ranks.len());
            for (r, rank) in self.ranks.iter().enumerate() {
                parts.push(rank.read(step(r).reads[0])?);
            }
            let first = step(0);
            let whole = match first.kernel.name() == AllGather::NAME {
                true => {
                    let joined = gathered(
                        &parts.iter().map(Tensor::bytes).collect::<Vec<_>>(),
                        first.kernel.sizes()[0],
                    );
                    Tensor::from_bytes(vec![first.write.len], parts[0].dtype(), &joined)
                }
                false => {
                    let sum = summed(parts.iter().map(|p| p.floats().expect("floats")));
                    Tensor::f32(vec![first.write.len], sum)
                }
            };
            for (r, rank) in self.ranks.iter_mut().enumerate() {
                let write = rank.plan().steps[sums[r][at]].write;
                rank.write(write, whole.view())?;
            }
        }
        let outputs = self.ranks[0].plan().outputs.clone();
        let mut out = Vec::with_capacity(outputs.len());
        for (at, o) in outputs.iter().enumerate() {
            let first = self.ranks[0].read(o.slot)?;
            for r in &self.ranks[1..] {
                if r.read(r.plan().outputs[at].slot)? != first {
                    return Err(Error::OutputsDisagree);
                }
            }
            out.push(first.reshaped(o.shape.clone()));
        }
        Ok(out)
    }
}

/// The steps of `plan` that cross ranks, in order.
fn sums(plan: &Plan) -> Vec<usize> {
    let steps = plan.steps.iter().enumerate();
    let sums = steps.filter(|(_, s)| crosses(s.kernel.name()));
    sums.map(|(i, _)| i).collect()
}
