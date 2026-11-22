//! Steps one persistent grid runs, with its sums inside.

use crate::Error;
use crate::profile::Marks;
use tessera_lower::KernelRef;

/// Role of one table item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemKind {
    Input,
    Output,
    Scratch,
}

/// Companion metadata for a packed storage form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormMeta {
    pub name: &'static str,
    pub rows: u32,
    pub cols: u32,
    pub suh: Option<u64>,
    pub svh: Option<u64>,
}

/// One table item: address, role, shape, share and form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    pub at: u64,
    pub bytes: usize,
    pub kind: ItemKind,
    pub elem: &'static str,
    pub dims: u32,
    pub shape: [u32; 4],
    pub share: [u32; 4],
    pub form: Option<FormMeta>,
}

/// One kept-row descriptor for the KV convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeptDesc {
    pub base: u64,
    pub pages: Option<u64>,
    pub len: Option<u64>,
    pub slots: u32,
    pub context: u32,
    pub pages_per_slot: u32,
    pub page_rows: u32,
    pub row_bytes: u32,
    pub elem: &'static str,
    pub interleaved: bool,
}

/// Table counts a grid opens for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub items: usize,
    pub forms: usize,
    pub kept: usize,
    pub pages: usize,
}

/// Runs resident steps. One launch per step, ended each time.
pub trait Grid {
    /// Readies `kernel` for launches matching `layout`.
    fn open(&mut self, kernel: &KernelRef, layout: Layout) -> Result<(), Error>;
    /// One launch of `kernel`. It returns once the grid ends.
    fn launch(
        &mut self,
        kernel: &KernelRef,
        items: &[Item],
        kept: &[KeptDesc],
        values: &[u32],
    ) -> Result<(), Error>;
    /// Enables or disables resident profiling telemetry and marks.
    fn profiling(&mut self, _on: bool) {}
    /// Replaces `entry`'s source with `source`, or restores planned text.
    fn replace(&mut self, _entry: &str, _source: Option<String>) {}
    /// Drains mark samples gathered by recent launches.
    fn take_marks(&mut self) -> Vec<(String, Marks)> {
        Vec::new()
    }
}
