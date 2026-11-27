//! Token ids to text, a piece at a time.

/// Sliding decoded text window for O(1) incremental token pushes.
#[derive(Default)]
pub(crate) struct Spool {
    buf: Vec<u32>,
    /// Tokens in `buf` before the unresolved tail.
    anchor: usize,
    /// Prefix of `decode(&self.buf)` already returned.
    sent: String,
}

fn fresh_tail(safe: &str, sent: &str) -> String {
    match safe.strip_prefix(sent) {
        Some(rest) => rest.to_string(),
        None => {
            let same = safe
                .chars()
                .zip(sent.chars())
                .take_while(|(a, b)| a == b)
                .map(|(c, _)| c.len_utf8())
                .sum();
            safe[same..].to_string()
        }
    }
}

impl Spool {
    /// Adds `id`. Returns the text it completes, if any.
    pub fn push(&mut self, id: u32, decode: impl Fn(&[u32]) -> String) -> String {
        self.buf.push(id);
        let full = decode(&self.buf);
        let trailing = full.ends_with('\u{fffd}');
        let safe = match trailing {
            true => &full[..full.len() - '\u{fffd}'.len_utf8()],
            false => full.as_str(),
        };
        if safe.len() < self.sent.len() {
            return String::new();
        }
        let fresh = fresh_tail(safe, &self.sent);
        if fresh.is_empty() {
            self.compact_idle(id, &full, trailing, &decode);
            return fresh;
        }
        match trailing {
            false => self.compact_clean(full, &decode),
            true => self.compact_trailing(safe, &decode),
        }
        fresh
    }

    fn compact_idle(
        &mut self,
        id: u32,
        full: &str,
        trailing: bool,
        decode: &impl Fn(&[u32]) -> String,
    ) {
        if self.buf.len().saturating_sub(self.anchor) < 8 {
            return;
        }
        let keep = self.anchor.max(1);
        if !trailing && full == self.sent {
            if keep < self.buf.len() && decode(&self.buf[..keep]) == self.sent {
                self.buf.truncate(keep);
            }
            return;
        }
        let prev = self.buf.len() - 1;
        if trailing && prev > keep && decode(&[id]).is_empty() && decode(&self.buf[..prev]) == full
        {
            self.buf.pop();
        }
    }

    fn compact_clean(&mut self, full: String, decode: &impl Fn(&[u32]) -> String) {
        if self.anchor == 0 {
            self.anchor = self.buf.len();
            self.sent = full;
            return;
        }
        let tail = decode(&self.buf[self.anchor..]);
        if !tail.is_empty() && !tail.starts_with('\u{fffd}') && !tail.ends_with('\u{fffd}') {
            self.buf.drain(..self.anchor);
            self.anchor = self.buf.len();
            self.sent = tail;
            return;
        }
        if self.buf.len() >= 16 {
            let cut = self.buf.len() - 2;
            let cand = decode(&self.buf[cut..]);
            if !cand.is_empty() && !cand.starts_with('\u{fffd}') && !cand.ends_with('\u{fffd}') {
                self.buf.drain(..cut);
                self.anchor = self.buf.len();
                self.sent = cand;
                return;
            }
        }
        self.anchor = self.buf.len();
        self.sent = full;
    }

    fn compact_trailing(&mut self, safe: &str, decode: &impl Fn(&[u32]) -> String) {
        self.sent = safe.to_string();
        if self.anchor == 0 {
            let lo = self.buf.len().saturating_sub(4).max(1);
            for cut in (lo..self.buf.len()).rev() {
                if decode(&self.buf[..cut]) == safe {
                    self.anchor = cut;
                    break;
                }
            }
        }
        if self.buf.len() < 4 {
            return;
        }
        let keep = self.anchor.clamp(1, 4);
        let last = self.buf.len() - 1;
        if keep >= last {
            return;
        }
        let head = decode(&self.buf[..keep]);
        let tail = decode(&self.buf[last..]);
        let mut cand = Vec::with_capacity(keep + 1);
        cand.extend_from_slice(&self.buf[..keep]);
        cand.push(self.buf[last]);
        let combined = decode(&cand);
        if combined.ends_with('\u{fffd}') && combined.strip_prefix(&head) == Some(tail.as_str()) {
            self.buf = cand;
            self.anchor = keep;
            self.sent = combined[..combined.len() - '\u{fffd}'.len_utf8()].to_string();
        }
    }
}

/// Text that stop strings can cut, held until they cannot.
pub(crate) struct Stops {
    stops: Vec<String>,
    text: String,
    /// Bytes of `text` handed out.
    sent: usize,
    /// Characters held back: the longest stop, less one.
    hold: usize,
}

/// The byte index `n` characters before byte `end`.
fn back(text: &str, end: usize, n: usize) -> usize {
    match n {
        0 => end,
        _ => text[..end]
            .char_indices()
            .rev()
            .nth(n - 1)
            .map_or(0, |(i, _)| i),
    }
}

impl Stops {
    pub fn new(stops: Vec<String>) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let longest = stops.iter().map(|s| s.chars().count()).max();
        Self {
            stops,
            text: String::new(),
            sent: 0,
            hold: longest.unwrap_or(1) - 1,
        }
    }

    /// Adds `piece`. Returns text safe to show, and if cut.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        let before = self.text.len();
        self.text.push_str(piece);
        // Only a match that ends in `piece` is new.
        let from = back(&self.text, before, self.hold);
        let mut cut: Option<usize> = None;
        for stop in &self.stops {
            if let Some(at) = self.text[from..].find(stop.as_str()) {
                cut = Some(cut.map_or(from + at, |c| c.min(from + at)));
            }
        }
        let end = match cut {
            Some(at) => {
                self.text.truncate(at);
                at
            }
            None => back(&self.text, self.text.len(), self.hold).max(self.sent),
        };
        let out = self.text[self.sent..end].to_string();
        self.sent = end;
        (out, cut.is_some())
    }

    /// The text still held, once the answer ends.
    pub fn rest(&mut self) -> String {
        let out = self.text[self.sent..].to_string();
        self.sent = self.text.len();
        out
    }
}
