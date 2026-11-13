//! Token ids to text, a piece at a time.

/// Decoded text, and how much of it was handed out.
#[derive(Default)]
pub struct Spool {
    ids: Vec<u32>,
    shown: usize,
}

impl Spool {
    /// Adds `id`. Returns the text it completes, if any.
    pub fn push(&mut self, id: u32, decode: impl Fn(&[u32]) -> String) -> String {
        self.ids.push(id);
        let text = decode(&self.ids);
        // A token can end mid-character: the replacement character.
        if text.ends_with('\u{fffd}') || text.len() < self.shown {
            return String::new();
        }
        let fresh = text[self.shown..].to_string();
        self.shown = text.len();
        fresh
    }
}
