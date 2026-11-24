//! `/v1/systemone`: Jev decisions from label-token logits.

use crate::http::Response;
use crate::wire::{self, Bag, Bagged, Refusal, refuse};
use crate::{Model, Slot, reply};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tessera_bundle::Message;
use tessera_tasks::api::{Read, Tokens};

/// The alternative count limit per question.
const MOST: usize = 26;
/// Past this many questions, each id runs into its label.
const LINES: usize = 10;
/// The fields of a question this route reads.
const ASKED: [&str; 3] = ["type", "instructions", "criteria"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind {
    Noul,
    Choice,
    Score,
}

/// One question, ready to lay out.
#[derive(Debug, PartialEq)]
pub(crate) struct Question {
    pub(crate) id: String,
    pub(crate) kind: Kind,
    pub(crate) instructions: String,
    /// Each alternative's name and description.
    pub(crate) choices: Vec<(String, Option<String>)>,
    /// The text each alternative answers with.
    pub(crate) labels: Vec<String>,
}

/// A question's label row in the answer, and label ids.
#[derive(Debug, PartialEq)]
pub(crate) struct Place {
    pub(crate) at: usize,
    pub(crate) ids: Vec<u32>,
}

/// The request body. `model` names any model.
#[derive(Deserialize)]
pub(crate) struct Decide {
    state: Option<Value>,
    questions: Option<Value>,
    instructions: Option<Value>,
    #[serde(flatten)]
    extra: Bag,
}

impl Bagged for Decide {
    fn bags(&self) -> (&Bag, Vec<String>) {
        let mut nested = Vec::new();
        if let Some(Value::Object(questions)) = &self.questions {
            for (id, q) in questions {
                let Value::Object(q) = q else { continue };
                let unread = q.keys().filter(|k| !ASKED.contains(&k.as_str()));
                nested.extend(unread.map(|k| format!("questions.{id}.{k}")));
            }
        }
        (&self.extra, nested)
    }
}

/// A request that breaks the question schema.
pub(crate) fn invalid(message: impl Into<String>) -> Refusal {
    refuse(422, message, None)
}

/// A value as prompt text. Null and blank give `None`.
fn text(value: Option<&Value>) -> Option<String> {
    let text = match value? {
        Value::Null => return None,
        Value::String(s) => s.trim().to_string(),
        other => other.to_string(),
    };
    (!text.is_empty()).then_some(text)
}

/// `n` labels: `A`, `B`, ..., or `1`, `2`, ... when `digits`.
fn labels(n: usize, digits: bool) -> Vec<String> {
    let first = if digits { b'1' } else { b'A' };
    (0..n as u8)
        .map(|i| char::from(first + i).to_string())
        .collect()
}

/// The questions in request order, checked.
pub(crate) fn questions(value: Option<&Value>) -> Result<Vec<Question>, Refusal> {
    let map = match value {
        Some(Value::Object(map)) if !map.is_empty() => map,
        _ => {
            return Err(invalid(
                "questions: needs a non-empty map of id -> question",
            ));
        }
    };
    map.iter().map(|(id, q)| question(id, q)).collect()
}

fn question(id: &str, q: &Value) -> Result<Question, Refusal> {
    let id = id.trim();
    if id.is_empty() || id.contains([':', '\n']) {
        let message = format!("question id {id:?} must be non-empty, no ':' or newline");
        return Err(invalid(message));
    }
    let why = |what: &str| invalid(format!("question {id:?}: {what}"));
    let Value::Object(q) = q else {
        return Err(why("must be an object"));
    };
    let criteria = q.get("criteria").filter(|c| !c.is_null());
    let (kind, choices) = match (q.get("type").and_then(Value::as_str), criteria) {
        (Some("noul"), None) => (Kind::Noul, vec![("yes".into(), None), ("no".into(), None)]),
        (Some("noul"), Some(Value::Object(c))) => {
            let both = [("yes", "true"), ("no", "false")];
            let choices = both.map(|(name, key)| (name.to_string(), text(c.get(key))));
            (Kind::Noul, choices.into())
        }
        (Some("noul"), _) => {
            return Err(why("noul criteria must be an object with true and false"));
        }
        (Some("choice"), Some(Value::Object(c))) if !c.is_empty() => {
            let options = c.iter().map(|(name, d)| (name.clone(), text(Some(d))));
            (Kind::Choice, options.collect())
        }
        (Some("choice"), _) => {
            return Err(why("choice criteria must map option names to descriptions"));
        }
        (Some("score"), Some(Value::Array(levels))) => {
            let name = |l: &Value| text(Some(l)).unwrap_or_default();
            (
                Kind::Score,
                levels.iter().map(|l| (name(l), None)).collect(),
            )
        }
        (Some("score"), _) => {
            return Err(why("score criteria must be an ordered list of levels"));
        }
        _ => {
            let kind = q.get("type").map_or("null".into(), Value::to_string);
            return Err(why(&format!("unknown type {kind}")));
        }
    };
    let n = choices.len();
    if n < 2 {
        return Err(why("needs at least two alternatives"));
    }
    if n > MOST {
        return Err(why(&format!("at most {MOST} alternatives")));
    }
    let labels = match kind {
        Kind::Noul => vec!["yes".into(), "no".into()],
        Kind::Choice => labels(n, false),
        Kind::Score => labels(n, n <= 9),
    };
    Ok(Question {
        id: id.to_string(),
        kind,
        instructions: text(q.get("instructions")).unwrap_or_default(),
        choices,
        labels,
    })
}

/// The system message: the questions, then how to reply.
pub(crate) fn system(questions: &[Question], instructions: Option<&str>) -> String {
    let mut s = String::from(
        "Answer a fixed set of questions about the state the user provides. \
         Each question lists its allowed answers; reply with exactly one label per question.\n",
    );
    for q in questions {
        s += &format!("\nQuestion {}: {}\n", q.id, q.instructions);
        for ((name, description), label) in q.choices.iter().zip(&q.labels) {
            s += &match (q.kind, description) {
                (Kind::Noul, Some(d)) => format!("  {label}: {d}\n"),
                (Kind::Noul, None) => format!("  {label}\n"),
                (_, Some(d)) => format!("  {label}: {name} ({d})\n"),
                (_, None) => format!("  {label}: {name}\n"),
            };
        }
    }
    s += match questions.len() <= LINES {
        true => "\nReply with one line per question, in this order, formatted as \"id: label\".",
        false => {
            "\nReply on one line with each question's id immediately followed by its label, \
             separated by single spaces."
        }
    };
    // Last, so requests that differ only here share a prefix.
    if let Some(i) = instructions {
        s += "\n\n";
        s += i;
    }
    s
}

/// The answer text with label `picks[i]` for question `i`.
pub(crate) fn answers(questions: &[Question], picks: &[usize]) -> String {
    let lines = questions.len() <= LINES;
    let mut out = String::new();
    for (i, (q, &pick)) in questions.iter().zip(picks).enumerate() {
        let gap = match (i, lines) {
            (0, _) => "",
            (_, true) => "\n",
            (_, false) => " ",
        };
        let colon = if lines { ": " } else { "" };
        out += &format!("{gap}{}{colon}{}", q.id, q.labels[pick]);
    }
    out
}

/// The answer ids with first labels, and each question's place.
pub(crate) fn template(
    tokens: &dyn Tokens,
    questions: &[Question],
) -> Result<(Vec<u32>, Vec<Place>), Refusal> {
    let encode = |picks: &[usize]| {
        let text = answers(questions, picks);
        tokens
            .encode(&text)
            .map_err(|e| refuse(400, e.to_string(), None))
    };
    let mut picks = vec![0; questions.len()];
    let base = encode(&picks)?;
    let mut places = Vec::with_capacity(questions.len());
    for (qi, q) in questions.iter().enumerate() {
        let why = |what: String| invalid(format!("question {:?}: {what}", q.id));
        let mut others = Vec::with_capacity(q.labels.len());
        let mut at = None;
        for (li, label) in q.labels.iter().enumerate().skip(1) {
            picks[qi] = li;
            let other = encode(&picks)?;
            picks[qi] = 0;
            if other.len() != base.len() {
                return Err(why(format!("label {label:?} is not a single token")));
            }
            let mut diffs = (0..base.len()).filter(|&i| other[i] != base[i]);
            match (diffs.next(), diffs.next(), at) {
                (Some(d), None, None) => at = Some(d),
                (Some(d), None, Some(a)) if d == a => {}
                _ => return Err(why("labels do not share one template slot".into())),
            }
            others.push(other[at.unwrap_or_default()]);
        }
        let at = at.unwrap_or_default();
        let ids: Vec<u32> = std::iter::once(base[at]).chain(others).collect();
        let mut seen = ids.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != ids.len() {
            return Err(why("two labels tokenize to the same id".into()));
        }
        places.push(Place { at, ids });
    }
    Ok((base, places))
}

/// A question's answer in Jev's shape, from its label logits.
pub(crate) fn answer(q: &Question, logits: &[f32]) -> Value {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f64> = logits.iter().map(|&l| f64::from(l - top).exp()).collect();
    let sum: f64 = exp.iter().sum();
    let p: Vec<f64> = exp.iter().map(|e| e / sum).collect();
    let best = (0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b });
    let names = q.choices.iter().map(|(name, _)| name);
    match q.kind {
        Kind::Noul => json!({"type": "noul", "noul": p[0]}),
        Kind::Choice => {
            let probabilities: Map<String, Value> =
                names.zip(&p).map(|(n, p)| (n.clone(), json!(p))).collect();
            json!({
                "type": "choice",
                "choice": q.choices[best].0,
                "probabilities": probabilities,
                "confidence": p[best],
            })
        }
        Kind::Score => {
            let (mut legend, mut probabilities) = (Map::new(), Map::new());
            for (i, (name, p)) in names.zip(&p).enumerate() {
                legend.insert(i.to_string(), json!(name));
                probabilities.insert(i.to_string(), json!(p));
            }
            let score: f64 = p.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
            json!({
                "type": "score",
                "score": score,
                "legend": legend,
                "probabilities": probabilities,
                "confidence": p[best],
            })
        }
    }
}

/// The prompt and answer ids, and reads in row order.
fn layout(model: &Model, d: Decide) -> Result<(Vec<Question>, Laid), Refusal> {
    let state = match d.state {
        None | Some(Value::Null) => return Err(invalid("state: required")),
        Some(Value::String(s)) => s,
        Some(other) => other.to_string(),
    };
    let questions = questions(d.questions.as_ref())?;
    let chat = model.chat.as_ref();
    let chat = chat.ok_or_else(|| refuse(404, "the model has no chat template", None))?;
    let system = system(&questions, text(d.instructions.as_ref()).as_deref());
    let said = |role: &str, content: String| Message::text(role, content);
    let failed = |e: tessera_tasks::Error| refuse(400, e.to_string(), None);
    let prompt = chat.render(&[said("system", system), said("user", state)], false);
    let mut tokens = model
        .tokens
        .encode(&prompt.map_err(failed)?)
        .map_err(failed)?;
    let input = tokens.len();
    let (answer, places) = template(&*model.tokens, &questions)?;
    let mut order: Vec<usize> = (0..places.len()).collect();
    order.sort_by_key(|&i| places[i].at);
    let mut reads = Vec::with_capacity(order.len());
    for &i in &order {
        // A label's logits come from the row before it.
        let row = (input + places[i].at).saturating_sub(1) as u32;
        reads.push(Read {
            row,
            ids: places[i].ids.clone(),
        });
    }
    let last = places.iter().map(|p| p.at).max().unwrap_or_default();
    tokens.extend_from_slice(&answer[..last]);
    let context = model.info.context;
    if tokens.len() > context {
        let n = tokens.len();
        let message = format!("the prompt is {n} tokens, past the {context}-token context");
        return Err(refuse(400, message, Some("state")));
    }
    let usage = json!({"input_tokens": input, "output_tokens": answer.len()});
    let laid = Laid {
        tokens,
        reads,
        order,
        usage,
    };
    Ok((questions, laid))
}

/// The tokens and reads of one decision.
struct Laid {
    tokens: Vec<u32>,
    reads: Vec<Read>,
    /// The question each read answers.
    order: Vec<usize>,
    usage: Value,
}

fn decide(model: &Model, body: &[u8]) -> Result<Value, Refusal> {
    let mut body = wire::object(body)?;
    body.remove("model");
    let score = model.score.clone();
    let score = score.ok_or_else(|| refuse(404, "the model does not score labels", None))?;
    let (questions, laid) = layout(model, wire::fields(model, body)?)?;
    let scores = score.score(laid.tokens, laid.reads);
    let scores = scores.map_err(|e| refuse(503, e.to_string(), None))?;
    let mut got = vec![Value::Null; questions.len()];
    for (&i, logits) in laid.order.iter().zip(&scores) {
        got[i] = answer(&questions[i], logits);
    }
    let ids = questions.iter().map(|q| q.id.clone());
    let answers: Map<String, Value> = ids.zip(got).collect();
    Ok(json!({"model": model.info.name, "answers": answers, "usage": laid.usage}))
}

pub(crate) fn systemone(slot: &Slot, body: &[u8]) -> Response {
    reply(slot, |m| decide(m, body))
}
