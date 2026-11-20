//! Guides: per-answer rules on which tokens come next.

pub type Fault = Box<dyn std::error::Error + Send + Sync>;

/// The most ids a sparse pick reads.
pub const FEW: usize = sensai_models::FEW as usize;

/// The tokens one pick may choose from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Allowed {
    Any,
    /// The host makes this pick.
    One(u32),
    /// Up to `FEW` ids, each read from its head row.
    Few(Vec<u32>),
    /// Every id but these.
    Except(Vec<u32>),
    /// One bit per vocabulary row, set where allowed.
    Bits(Vec<u32>),
}

impl Allowed {
    /// A mask, stored in the smallest tier that fits.
    pub fn from_bits(bits: Vec<u32>) -> Self {
        let mut ids = Vec::new();
        for (w, &word) in bits.iter().enumerate() {
            let mut rest = word;
            while rest != 0 && ids.len() <= FEW {
                ids.push(w as u32 * 32 + rest.trailing_zeros());
                rest &= rest - 1;
            }
        }
        match ids.len() {
            1 => Allowed::One(ids[0]),
            n if n <= FEW => Allowed::Few(ids),
            _ => Allowed::Bits(bits),
        }
    }

    /// One bit per row, over `words` words.
    pub(crate) fn bits(&self, words: usize) -> Vec<u32> {
        let set = |out: &mut [u32], id: u32, on: bool| {
            let (w, b) = (id as usize / 32, id % 32);
            if let Some(word) = out.get_mut(w) {
                *word = if on {
                    *word | 1 << b
                } else {
                    *word & !(1 << b)
                };
            }
        };
        let mut out = vec![u32::MAX; words];
        match self {
            Allowed::Any => {}
            Allowed::One(id) => {
                out.fill(0);
                set(&mut out, *id, true);
            }
            Allowed::Few(ids) => {
                out.fill(0);
                ids.iter().for_each(|&id| set(&mut out, id, true));
            }
            Allowed::Except(ids) => ids.iter().for_each(|&id| set(&mut out, id, false)),
            Allowed::Bits(bits) => {
                let n = bits.len().min(words);
                out[..n].copy_from_slice(&bits[..n]);
            }
        }
        out
    }
}

/// One answer's rules on its next tokens.
pub trait Guide: Send {
    /// The tokens the next pick may choose from.
    fn allowed(&mut self) -> Result<Allowed, Fault>;

    /// Answer tokens to add with no pick, already accepted.
    fn forced(&mut self) -> Vec<u32> {
        Vec::new()
    }

    /// Moves past a picked token.
    fn accept(&mut self, id: u32) -> Result<(), Fault>;

    /// Whether the answer ends here.
    fn done(&self) -> bool {
        false
    }
}

/// Fixed tokens first, then any.
pub struct Prefix(pub Vec<u32>);

impl Guide for Prefix {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        Ok(Allowed::Any)
    }

    fn forced(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.0)
    }

    fn accept(&mut self, _: u32) -> Result<(), Fault> {
        Ok(())
    }
}
