//! Presence, frequency and repetition penalties as a guide.

use crate::guide::{Allowed, Edit, Fault, Guide};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Penalties for one answer. The default changes nothing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Penalties {
    /// Subtracted once from each generated id.
    pub presence: f32,
    /// Subtracted from each generated id per use.
    pub frequency: f32,
    /// Divides positive, multiplies negative logits of prompt and answer ids.
    pub repetition: f32,
}

impl Default for Penalties {
    fn default() -> Self {
        Self {
            presence: 0.0,
            frequency: 0.0,
            repetition: 1.0,
        }
    }
}

impl Penalties {
    /// Whether these penalties change nothing.
    pub fn neutral(&self) -> bool {
        *self == Self::default()
    }

    /// A guide for an answer to `prompt`. None when neutral.
    pub fn guide(self, prompt: &[u32]) -> Option<Box<dyn Guide>> {
        if self.neutral() {
            return None;
        }
        let mut guide = Penalize {
            penalties: self,
            edits: Vec::new(),
            at: HashMap::new(),
        };
        if self.repetition != 1.0 {
            for &id in prompt {
                guide.entry(id);
            }
        }
        Some(Box::new(guide))
    }
}

/// One edit per seen id, updated as tokens arrive.
struct Penalize {
    penalties: Penalties,
    edits: Vec<Edit>,
    /// Each id's index in `edits` and its answer uses.
    at: HashMap<u32, (usize, u32)>,
}

impl Penalize {
    /// The seen id's index and answer uses, added if new.
    fn entry(&mut self, id: u32) -> &mut (usize, u32) {
        let next = self.edits.len();
        let entry = self.at.entry(id).or_insert((next, 0));
        if entry.0 == next {
            self.edits.push(Edit {
                id,
                add: 0.0,
                rep: self.penalties.repetition,
            });
        }
        entry
    }
}

impl Guide for Penalize {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        Ok(Allowed::Any)
    }

    fn accept(&mut self, id: u32) -> Result<(), Fault> {
        let Penalties {
            presence,
            frequency,
            ..
        } = self.penalties;
        let (at, uses) = self.entry(id);
        *uses += 1;
        let (at, uses) = (*at, *uses);
        self.edits[at].add = -(presence + frequency * uses as f32);
        Ok(())
    }

    fn edits(&mut self) -> Vec<Edit> {
        self.edits.clone()
    }
}
