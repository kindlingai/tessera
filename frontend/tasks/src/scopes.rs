//! Logit programs over answer, thinking and tool-call scopes.

use crate::api::{Params, Refused};
use tessera_bundle::tokenize::hf::Hf;
use tessera_bundle::{Markers, ResponseParser};
use tessera_engine::{Guide, Move, Penalties, Program, Scoped, Term};

const ANSWER: u8 = 0;
const THINKING: u8 = 1;
/// Neutral: tool calls repeat names, quotes and braces by design.
const TOOL: u8 = 2;

/// Tokens, a share of `max_tokens`, or the smaller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Limit {
    Tokens(u32),
    Share(f32),
    Least(u32, f32),
}

impl Limit {
    /// `16384`, `0.5x` or `16384|0.5x`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let want =
            || format!("{s:?}: want a token count, a share like 0.5x, or both like 16384|0.5x");
        let tokens = |t: &str| t.trim().parse::<u32>().map_err(|_| want());
        let share = |t: &str| match t.trim().strip_suffix('x').map(str::parse::<f32>) {
            Some(Ok(f)) if (0.0..=1.0).contains(&f) => Ok(f),
            Some(Ok(_)) => Err(format!("{s:?}: want a share from 0x to 1x")),
            _ => Err(want()),
        };
        match s.split_once('|') {
            // A count of 0 would always win over the share.
            Some((n, _)) if tokens(n)? == 0 => Err(format!("{s:?}: a pair needs a count above 0")),
            Some((n, f)) => Ok(Limit::Least(tokens(n)?, share(f)?)),
            None if s.trim().ends_with('x') => Ok(Limit::Share(share(s)?)),
            None => Ok(Limit::Tokens(tokens(s)?)),
        }
    }

    /// The limit in tokens, given `max_tokens`.
    pub fn tokens(self, max_tokens: usize) -> u32 {
        let part = |f: f32| (f * max_tokens as f32).round() as u32;
        match self {
            Limit::Tokens(n) => n,
            Limit::Share(f) => part(f),
            Limit::Least(n, f) => n.min(part(f)),
        }
    }
}

/// Raises the think-end logit as thinking runs past `start`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boost {
    pub start: Limit,
    /// Logits per thousand tokens past `start`.
    pub per1k: f32,
    /// The highest raise.
    pub max: f32,
}

impl Boost {
    /// `start=S,per1k=P,max=M`. None when `per1k` or `max` is 0.
    pub fn parse(s: &str) -> Result<Option<Self>, String> {
        let (mut start, mut per1k, mut max) = (None, None, None);
        for part in s.split(',') {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("{part:?}: want key=value"))?;
            let (key, value) = (key.trim(), value.trim());
            let num = |hi: f32| match value.parse::<f32>() {
                Ok(v) if v.is_finite() && (0.0..=hi).contains(&v) => Ok(v),
                _ => Err(format!("{key}={value:?}: want a number from 0 to {hi}")),
            };
            let slot = match key {
                "start" => {
                    let limit = Limit::parse(value).map_err(|e| format!("start={e}"))?;
                    start.replace(limit).is_some()
                }
                "per1k" => per1k.replace(num(f32::MAX)?).is_some(),
                "max" => max.replace(num(20.0)?).is_some(),
                _ => return Err(format!("{key:?}: want start, per1k or max")),
            };
            if slot {
                return Err(format!("{key} is given twice"));
            }
        }
        let (per1k, max) = (per1k.unwrap_or(0.0), max.unwrap_or(0.0));
        Ok((per1k > 0.0 && max > 0.0).then_some(Boost {
            start: start.unwrap_or(Limit::Tokens(0)),
            per1k,
            max,
        }))
    }
}

/// Changes inside a think block, beyond the answer's penalties.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Thinking {
    /// Added to the presence penalty.
    pub presence: f32,
    /// Added to the frequency penalty.
    pub frequency: f32,
    pub boost: Option<Boost>,
}

impl Thinking {
    /// The request key of the first value that changes anything.
    pub fn key(&self) -> Option<&'static str> {
        let keys = [
            (self.presence != 0.0, "thinking_presence_penalty"),
            (self.frequency != 0.0, "thinking_frequency_penalty"),
            (self.boost.is_some(), "think_end_boost"),
        ];
        keys.into_iter().find(|k| k.0).map(|k| k.1)
    }
}

/// The ids of markers that are one token.
type Marks = Markers<Option<u32>>;

/// Each marker's first candidate that is one token.
fn marks(parser: &ResponseParser, tokenizer: &Hf) -> Marks {
    let one = |candidates: Vec<&str>| {
        for text in candidates {
            if let Ok(ids) = tokenizer.encode(text.trim())
                && let [id] = ids[..]
            {
                return Some(id);
            }
        }
        None
    };
    let m = parser.markers();
    Markers {
        think_open: one(m.think_open),
        think_close: one(m.think_close),
        call_open: one(m.call_open),
        call_close: one(m.call_close),
    }
}

/// The scoped guide for an answer to `prompt`.
pub fn guide(
    parser: Option<&ResponseParser>,
    tokenizer: &Hf,
    params: &Params,
    prompt: &[u32],
) -> Result<Box<dyn Guide>, Refused> {
    let marks = parser.map_or_else(Marks::default, |p| marks(p, tokenizer));
    if let Some(key) = params.thinking.key()
        && marks.think_close.is_none()
    {
        let why = format!("{key} needs a model whose think-end marker is one token");
        return Err(Refused(why, key));
    }
    let start = match params.starts_thinking && marks.think_close.is_some() {
        true => THINKING,
        false => ANSWER,
    };
    let program = program(params.penalties, params.thinking, marks, params.max_tokens);
    Ok(Box::new(Scoped::new(program, prompt, start)))
}

/// The program for one answer of up to `max_tokens`.
fn program(penalties: Penalties, thinking: Thinking, marks: Marks, max_tokens: usize) -> Program {
    let mut moves = Vec::new();
    let mut step = |from, id: Option<u32>, to| moves.extend(id.map(|id| Move { from, id, to }));
    step(THINKING, marks.think_close, ANSWER);
    step(ANSWER, marks.think_open, THINKING);
    step(ANSWER, marks.call_open, TOOL);
    step(TOOL, marks.call_close, ANSWER);
    let p = penalties;
    let mut terms = vec![
        Term::Count {
            presence: vec![p.presence, p.presence + thinking.presence, 0.0],
            frequency: vec![p.frequency, p.frequency + thinking.frequency, 0.0],
        },
        Term::Repeat {
            repetition: vec![p.repetition, p.repetition, 1.0],
        },
    ];
    if let (Some(b), Some(id)) = (thinking.boost, marks.think_close) {
        terms.push(Term::Ramp {
            id,
            scope: THINKING,
            start: b.start.tokens(max_tokens),
            slope: b.per1k / 1000.0,
            max: b.max,
        });
    }
    Program { moves, terms }
}
