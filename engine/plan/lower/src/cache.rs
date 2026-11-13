//! Plans kept by their sizes, for launches that repeat.

use crate::cost::Device;
use crate::plan::Plan;
use crate::{Error, lower_upto};
use graph::{DimName, Graph};
use std::collections::HashMap;

/// A plan holds while its graph and device stay unchanged.
type Key = (u64, usize, Vec<(DimName, u64)>, Vec<DimName>, u64);

/// Runtime dimensions are lowered at the next power of two.
const LEAST: u64 = 16;

/// Each runtime dimension's bucket, bound at launch.
pub type Bound = Vec<(DimName, u32)>;

/// Lowered plans by graph, entry and sizes.
#[derive(Default)]
pub struct Cache {
    plans: HashMap<Key, Plan>,
    /// Each graph seen, by address, with its hash.
    graphs: Vec<(*const Graph, u64)>,
    pub hits: usize,
}

impl Cache {
    /// The graph's hash, computed once per graph it is given.
    fn graph(&mut self, g: &Graph) -> u64 {
        let at = std::ptr::from_ref(g);
        match self.graphs.iter().find(|(seen, _)| *seen == at) {
            Some((_, hash)) => *hash,
            None => {
                let hash = g.hash();
                self.graphs.push((at, hash));
                hash
            }
        }
    }

    /// The plan for these sizes, lowered once per device state.
    pub fn lower(
        &mut self,
        g: &Graph,
        entry: usize,
        sizes: &[(DimName, u64)],
        device: &Device,
    ) -> Result<Plan, Error> {
        Ok(self.upto(g, entry, sizes, &[], device)?.0.clone())
    }

    /// A plan reading `upto` at launch, bound at their buckets.
    pub fn upto(
        &mut self,
        g: &Graph,
        entry: usize,
        sizes: &[(DimName, u64)],
        upto: &[DimName],
        device: &Device,
    ) -> Result<(&Plan, Bound), Error> {
        let mut bound = Vec::with_capacity(upto.len());
        let bucket = |(name, n): &(DimName, u64)| match upto.contains(name) {
            // Segment counts start at one, contexts at the least plan.
            true if *name == DimName::Segments => (*name, n.next_power_of_two()),
            true => (*name, n.next_power_of_two().max(LEAST)),
            false => (*name, *n),
        };
        let rounded: Vec<(DimName, u64)> = sizes.iter().map(bucket).collect();
        for (name, n) in rounded.iter().filter(|(name, _)| upto.contains(name)) {
            bound.push((*name, *n as u32));
        }
        let key = (
            self.graph(g),
            entry,
            rounded,
            upto.to_vec(),
            device.generation(),
        );
        if !self.plans.contains_key(&key) {
            let plan = lower_upto(g, entry, &key.2, upto, device)?;
            self.plans.insert(key.clone(), plan);
        } else {
            self.hits += 1;
        }
        Ok((&self.plans[&key], bound))
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }
}
