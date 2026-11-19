//! A host copy of the page table, by slot.

/// Each slot's lane names its pages, in order.
pub(crate) struct Lanes {
    /// Rows per page.
    pub page: usize,
    /// Entries per slot.
    width: usize,
    table: Vec<u32>,
    /// Whether the device's copy is old.
    stale: bool,
}

impl Lanes {
    /// Each slot owns its own pages, in order.
    pub fn new(page: usize, entries: usize, slots: usize) -> Self {
        Self {
            page,
            width: entries / slots,
            table: (0..entries as u32).collect(),
            stale: true,
        }
    }

    /// Pages in the table.
    pub fn pages(&self) -> usize {
        self.table.len()
    }

    /// Gives each slot its own pages again.
    pub fn reset(&mut self) {
        let own = 0..self.table.len() as u32;
        if !own.clone().eq(self.table.iter().copied()) {
            self.table = own.collect();
            self.stale = true;
        }
    }

    /// Writes a slot's first pages.
    pub fn set(&mut self, slot: u32, pages: &[u32]) {
        let first = slot as usize * self.width;
        self.table[first..][..pages.len()].copy_from_slice(pages);
        self.stale = true;
    }

    /// The table's bytes, if the device's copy is old.
    pub fn changed(&mut self) -> Option<Vec<u8>> {
        let stale = std::mem::take(&mut self.stale);
        let bytes = self.table.iter().flat_map(|p| p.to_le_bytes());
        stale.then(|| bytes.collect())
    }
}
