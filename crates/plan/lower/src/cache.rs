//! Plans kept by their sizes, for launches that repeat.

use crate::cost::Device;
use crate::plan::Plan;
use crate::{Error, lower};
use graph::{DimName, Graph};
use std::collections::HashMap;

/// A plan holds while its graph and device stay unchanged.
type Key = (u64, usize, Vec<(DimName, u64)>, u64);

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
        let key = (self.graph(g), entry, sizes.to_vec(), device.generation());
        if let Some(plan) = self.plans.get(&key) {
            self.hits += 1;
            return Ok(plan.clone());
        }
        let plan = lower(g, entry, sizes, device)?;
        self.plans.insert(key, plan.clone());
        Ok(plan)
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }
}
