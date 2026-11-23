//! Bounded least-recently-used map for cached plans and graphs.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::OnceLock;
use tessera_graph::DimName;
use tessera_lower::{Launch as Call, Plan};

/// Most inactive plans kept ready for instant switching.
pub(crate) const MAX_PLANS: usize = 32;

/// Most `(plan, bound)` graph sets kept alive at once.
pub(crate) const MAX_GRAPHS: usize = 32;

/// Plan id and bound dims sharing one graph set.
pub(crate) type GraphKey = (u64, Vec<(DimName, u32)>);

/// Recorded step ranges for one binding and its addresses.
pub(crate) struct GraphSet<G> {
    pub(crate) ptrs: Vec<u64>,
    pub(crate) ranges: HashMap<(usize, usize), G>,
}

/// Compiled kernel and dispatch tables for one inactive plan.
pub(crate) struct Prepared<K> {
    pub(crate) plan: Plan,
    pub(crate) launches: Vec<OnceLock<Vec<Call>>>,
    pub(crate) kernels: Vec<K>,
    pub(crate) planned: Vec<K>,
    pub(crate) entries: Vec<(String, usize)>,
    pub(crate) uses: Vec<usize>,
    pub(crate) waits: Vec<bool>,
    pub(crate) resident: Vec<bool>,
}

/// Bounded map evicting the least recently used entry.
pub(crate) struct Lru<K, V> {
    cap: usize,
    entries: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: Eq + Hash + Clone, V> Lru<K, V> {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            cap,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.entries.get(key)
    }

    pub(crate) fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if self.entries.contains_key(key) {
            self.touch(key);
        }
        self.entries.get_mut(key)
    }

    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        let value = self.entries.remove(key)?;
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        Some(value)
    }

    /// Inserts `value`, returning any evicted entry.
    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        let evicted = if !self.entries.contains_key(&key) && self.entries.len() >= self.cap {
            self.order
                .pop_front()
                .and_then(|old| self.entries.remove(&old))
        } else {
            None
        };
        self.touch(&key);
        self.entries.insert(key, value);
        evicted
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    fn touch(&mut self, key: &K) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
    }
}
