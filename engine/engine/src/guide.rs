//! Guides: per-answer rules on which tokens come next.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Fault = Box<dyn std::error::Error + Send + Sync>;

/// The most ids a sparse pick reads.
pub const FEW: usize = tessera_models::FEW as usize;

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

    /// Whether the set holds `id`.
    pub fn allows(&self, id: u32) -> bool {
        let bit = |bits: &[u32]| {
            bits.get(id as usize / 32)
                .is_none_or(|w| w >> (id % 32) & 1 == 1)
        };
        match self {
            Allowed::Any => true,
            Allowed::One(one) => *one == id,
            Allowed::Few(ids) => ids.contains(&id),
            Allowed::Except(ids) => !ids.contains(&id),
            Allowed::Bits(bits) => bit(bits),
        }
    }

    /// The ids both sets hold. Empty is an empty `Few`.
    pub fn meet(self, other: Allowed) -> Allowed {
        match (self, other) {
            (Allowed::Any, a) | (a, Allowed::Any) => a,
            (Allowed::One(id), a) | (a, Allowed::One(id)) => match a.allows(id) {
                true => Allowed::One(id),
                false => Allowed::Few(Vec::new()),
            },
            (Allowed::Few(mut ids), a) | (a, Allowed::Few(mut ids)) => {
                ids.retain(|&id| a.allows(id));
                match ids[..] {
                    [id] => Allowed::One(id),
                    _ => Allowed::Few(ids),
                }
            }
            (Allowed::Except(mut a), Allowed::Except(b)) => {
                a.extend(b);
                a.sort_unstable();
                a.dedup();
                Allowed::Except(a)
            }
            (Allowed::Bits(mut bits), Allowed::Except(ids))
            | (Allowed::Except(ids), Allowed::Bits(mut bits)) => {
                for id in ids {
                    if let Some(word) = bits.get_mut(id as usize / 32) {
                        *word &= !(1 << (id % 32));
                    }
                }
                Allowed::from_bits(bits)
            }
            (Allowed::Bits(a), Allowed::Bits(b)) => {
                let (mut long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
                for (word, other) in long.iter_mut().zip(short) {
                    *word &= other;
                }
                Allowed::from_bits(long)
            }
        }
    }

    /// Whether no id is allowed.
    pub fn empty(&self) -> bool {
        matches!(self, Allowed::Few(ids) if ids.is_empty())
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

/// Divides positive logits by `rep`, multiplies others, adds `add`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edit {
    pub id: u32,
    pub add: f32,
    pub rep: f32,
}

/// Applies `edits` to a row, skipping ids past its end.
pub(crate) fn edit(row: &mut [f32], edits: &[Edit]) {
    for e in edits {
        if let Some(x) = row.get_mut(e.id as usize) {
            let scaled = if *x > 0.0 { *x / e.rep } else { *x * e.rep };
            *x = scaled + e.add;
        }
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

    /// Changes to the next pick's logits.
    fn edits(&mut self) -> Vec<Edit> {
        Vec::new()
    }
}

/// Several guides as one. Picks must suit every member.
pub struct Stacked {
    members: Vec<Box<dyn Guide>>,
    /// A conflict over forced tokens, reported by the next `allowed`.
    fault: Option<Fault>,
}

impl Stacked {
    /// None, the only member, or a stack of `members`.
    pub fn of(mut members: Vec<Box<dyn Guide>>) -> Option<Box<dyn Guide>> {
        match members.len() {
            0 => None,
            1 => members.pop(),
            _ => Some(Box::new(Stacked {
                members,
                fault: None,
            })),
        }
    }
}

impl Guide for Stacked {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        if let Some(fault) = self.fault.take() {
            return Err(fault);
        }
        let mut out = Allowed::Any;
        for m in &mut self.members {
            out = out.meet(m.allowed()?);
        }
        match out.empty() {
            true => Err("the guides allow disjoint token sets".into()),
            false => Ok(out),
        }
    }

    /// Two members that force different tokens are a fault.
    fn forced(&mut self) -> Vec<u32> {
        let lists: Vec<Vec<u32>> = self.members.iter_mut().map(|m| m.forced()).collect();
        let Some(first) = lists.iter().find(|l| !l.is_empty()).cloned() else {
            return Vec::new();
        };
        for (m, list) in self.members.iter_mut().zip(&lists) {
            if list.is_empty() {
                for &id in &first {
                    if let Err(e) = m.accept(id) {
                        self.fault.get_or_insert(e);
                    }
                }
            } else if *list != first {
                self.fault
                    .get_or_insert("the guides force different tokens".into());
            }
        }
        first
    }

    fn accept(&mut self, id: u32) -> Result<(), Fault> {
        let mut out = Ok(());
        for m in &mut self.members {
            let each = m.accept(id);
            if out.is_ok() {
                out = each;
            }
        }
        out
    }

    /// A pending fault waits for `allowed` to report it.
    fn done(&self) -> bool {
        self.fault.is_none() && self.members.iter().any(|m| m.done())
    }

    fn edits(&mut self) -> Vec<Edit> {
        merged(self.members.iter_mut().flat_map(|m| m.edits()).collect())
    }
}

/// One edit per id, sorted: adds sum, reps multiply.
pub(crate) fn merged(mut all: Vec<Edit>) -> Vec<Edit> {
    all.sort_by_key(|e| e.id);
    let mut out: Vec<Edit> = Vec::with_capacity(all.len());
    for e in all {
        match out.last_mut() {
            Some(last) if last.id == e.id => {
                last.add += e.add;
                last.rep *= e.rep;
            }
            _ => out.push(e),
        }
    }
    out
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

    /// A program with these values in every scope.
    pub fn program(self) -> Program {
        Program {
            moves: Vec::new(),
            terms: vec![
                Term::Count {
                    presence: vec![self.presence],
                    frequency: vec![self.frequency],
                },
                Term::Repeat {
                    repetition: vec![self.repetition],
                },
            ],
        }
    }

    /// A guide for an answer to `prompt`. None when neutral.
    pub fn guide(self, prompt: &[u32]) -> Option<Box<dyn Guide>> {
        match self.neutral() {
            true => None,
            false => Some(Box::new(Scoped::new(self.program(), prompt, 0))),
        }
    }
}

/// Token `id` moves an answer from scope `from` to `to`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Move {
    pub from: u8,
    pub id: u32,
    pub to: u8,
}

/// A program part. Rows hold one value per scope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Term {
    /// Each generated id: `add = -(presence + frequency * count)`.
    Count {
        presence: Vec<f32>,
        frequency: Vec<f32>,
    },
    /// In `scope`, raises `id` by `slope` per token past `start`.
    Ramp {
        id: u32,
        scope: u8,
        start: u32,
        slope: f32,
        /// The highest raise.
        max: f32,
    },
    /// Each id in the prompt or answer: `rep = repetition`.
    Repeat { repetition: Vec<f32> },
}

/// Scope moves and terms, as plain data scripts can build.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub moves: Vec<Move>,
    pub terms: Vec<Term>,
}

/// Row value for `scope`. Missing values are `none`.
fn at(row: &[f32], scope: u8, none: f32) -> f32 {
    row.get(scope as usize).copied().unwrap_or(none)
}

impl Program {
    /// Whether any term reads the prompt ids.
    fn repeats(&self) -> bool {
        let reads = |t: &Term| matches!(t, Term::Repeat { repetition } if repetition.iter().any(|&r| r != 1.0));
        self.terms.iter().any(reads)
    }

    /// Non-neutral edits for the next pick.
    pub fn edits(&self, tally: &Tally) -> Vec<Edit> {
        let scope = tally.scope;
        let mut out = Vec::with_capacity(tally.seen.len());
        for &id in &tally.seen {
            let uses = tally.counts[&id];
            let mut e = Edit {
                id,
                add: 0.0,
                rep: 1.0,
            };
            for term in &self.terms {
                match term {
                    Term::Count {
                        presence,
                        frequency,
                    } if uses > 0 => {
                        let once = at(presence, scope, 0.0);
                        e.add -= once + at(frequency, scope, 0.0) * uses as f32;
                    }
                    Term::Repeat { repetition } => e.rep *= at(repetition, scope, 1.0),
                    _ => {}
                }
            }
            if e.add != 0.0 || e.rep != 1.0 {
                out.push(e);
            }
        }
        for term in &self.terms {
            let Term::Ramp {
                id,
                scope: on,
                start,
                slope,
                max,
            } = *term
            else {
                continue;
            };
            if on != scope || tally.since <= start {
                continue;
            }
            let add = max.min(slope * (tally.since - start) as f32);
            match out.iter_mut().find(|e| e.id == id) {
                Some(e) => e.add += add,
                None => out.push(Edit { id, add, rep: 1.0 }),
            }
        }
        out
    }
}

/// One answer's scope and token counts, read by a program.
#[derive(Clone, Debug, Default)]
pub struct Tally {
    pub scope: u8,
    /// Tokens since the answer entered `scope`.
    pub since: u32,
    /// Answer uses of each seen id. Prompt ids hold zero.
    counts: HashMap<u32, u32>,
    /// Seen ids, in first-seen order.
    seen: Vec<u32>,
}

impl Tally {
    /// Starts in `scope`. Prompt ids count only for `Repeat`.
    pub fn new(program: &Program, prompt: &[u32], scope: u8) -> Self {
        let mut tally = Tally {
            scope,
            ..Tally::default()
        };
        if program.repeats() {
            for &id in prompt {
                tally.see(id);
            }
        }
        tally
    }

    /// The answer uses of `id`, added as seen if new.
    fn see(&mut self, id: u32) -> &mut u32 {
        let seen = &mut self.seen;
        self.counts.entry(id).or_insert_with(|| {
            seen.push(id);
            0
        })
    }

    /// Counts an answer token, then follows any scope move.
    pub fn step(&mut self, program: &Program, id: u32) {
        *self.see(id) += 1;
        self.since += 1;
        let scope = self.scope;
        if let Some(m) = program.moves.iter().find(|m| m.from == scope && m.id == id) {
            self.scope = m.to;
            self.since = 0;
        }
    }
}

/// A program and its tally, as a guide.
pub struct Scoped {
    program: Program,
    tally: Tally,
}

impl Scoped {
    /// Starts in `scope`, as the prompt may leave one open.
    pub fn new(program: Program, prompt: &[u32], scope: u8) -> Self {
        let tally = Tally::new(&program, prompt, scope);
        Self { program, tally }
    }
}

impl Guide for Scoped {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        Ok(Allowed::Any)
    }

    fn accept(&mut self, id: u32) -> Result<(), Fault> {
        self.tally.step(&self.program, id);
        Ok(())
    }

    fn edits(&mut self) -> Vec<Edit> {
        self.program.edits(&self.tally)
    }
}
