//! Times between `TS_MARK()` lines inside one kernel.

use crate::profile::{Marks, Region};

/// Put before kernel text to record its marks.
pub(crate) const ON: &str = include_str!("marks.cuh");
/// Put before kernel text that runs unmarked.
pub(crate) const OFF: &str = "#ifndef TS_MARK\n#define TS_MARK()\n#endif\n#line 1\n";

/// Blocks that record, and records each one keeps.
pub(crate) const BLOCKS: usize = 4096;
pub(crate) const ROOM: usize = 256;
/// Bytes of counts before the records.
pub(crate) const COUNTS: usize = BLOCKS * 4;

/// Bytes of records for the first `blocks` blocks.
pub(crate) fn records(blocks: usize) -> usize {
    blocks * ROOM * 16
}

/// Adds the gaps between each block's marks to `marks`.
pub(crate) fn read(counts: &[u32], records: &[u64], ticks: f64, marks: &mut Marks) {
    for (block, &count) in counts.iter().enumerate() {
        let kept = (count as usize).min(ROOM);
        marks.lost += count as u64 - kept as u64;
        let at = |i: usize| {
            (
                records[2 * (block * ROOM + i)],
                records[2 * (block * ROOM + i) + 1],
            )
        };
        for i in 1..kept {
            let ((from, start), (to, end)) = (at(i - 1), at(i));
            let seconds = end.wrapping_sub(start) as f64 / ticks;
            let (from, to) = (from as u32, to as u32);
            match marks
                .regions
                .iter_mut()
                .find(|r| (r.from, r.to) == (from, to))
            {
                Some(region) => region.seconds.push(seconds),
                None => marks.regions.push(Region {
                    from,
                    to,
                    seconds: vec![seconds],
                }),
            }
        }
    }
}
