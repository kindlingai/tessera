//! Bounded least-recently-used map for cached plans and graphs.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

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
