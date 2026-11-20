//! A prompt's rows already in kept rows.

/// Why a prompt skips the rows it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// There is no history.
    First,
    /// No common row may be skipped.
    NoLcp,
    /// The common rows end below one grid step.
    Grid,
    Hit,
}

/// The rows to skip, and the common prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Begin {
    pub skip: usize,
    pub lcp: usize,
    pub why: Why,
}

/// The rows of `prompt` that `valid` kept rows spare.
pub fn begin(history: &[u32], valid: usize, prompt: &[u32], limit: usize, grid: usize) -> Begin {
    let common = history.iter().zip(prompt).take_while(|(a, b)| a == b);
    let lcp = common.count();
    let rows = lcp.min(valid).min(limit);
    let step = grid.max(1);
    let skip = rows / step * step;
    let why = match (history.is_empty(), rows, skip) {
        (true, ..) => Why::First,
        (_, 0, _) => Why::NoLcp,
        (_, _, 0) => Why::Grid,
        _ => Why::Hit,
    };
    Begin { skip, lcp, why }
}
