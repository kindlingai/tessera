//! OpenAI-shaped HTTP routes over the `tasks::api` traits.

mod answer;
#[cfg(test)]
mod answer_test;
mod decide;
#[cfg(test)]
mod decide_test;
pub mod http;
#[cfg(test)]
mod http_test;
mod images;
#[cfg(test)]
mod images_test;
#[cfg(test)]
mod lib_test;
mod messages;
mod metrics;
#[cfg(test)]
mod mocks;
pub mod ranks;
#[cfg(test)]
mod ranks_test;
pub mod start;
pub mod task;

mod wire;
#[cfg(test)]
mod wire_test;

use http::{Request, Response};
pub use images::base64;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};
use tessera_bundle::RenderOptions;
use tessera_engine::progress::{Event, Open, Unit};
use tessera_engine::turns::Turns;
pub use tessera_tasks::api::Model;
use wire::{Detokenize, Embeddings, Messages, Refusal, Tokenize, bad, refuse};

/// The model once it loads. Until then routes answer 503.
pub type Slot = Arc<OnceLock<Model>>;

/// Each served model by name. The first is the default.
pub(crate) struct Models {
    pub(crate) named: Vec<(String, Slot)>,
    pub(crate) turns: Arc<Turns>,
}

/// The one field of a request that picks its model.
#[derive(Deserialize)]
struct Named {
    model: Option<String>,
}

impl Models {
    /// The slot the body's `model` names, else the first.
    fn pick(&self, body: &[u8]) -> &Slot {
        let first = &self.named[0].1;
        if self.named.len() == 1 {
            return first;
        }
        let asked = serde_json::from_slice::<Named>(body).ok();
        let asked = asked.and_then(|n| n.model);
        let found = asked.and_then(|m| self.named.iter().find(|(name, _)| *name == m));
        found.map_or(first, |(_, slot)| slot)
    }

    /// Answers one request. Health lists `open`'s load steps.
    pub(crate) fn route(&self, open: Option<&Open>, request: &Request) -> Response {
        let mut slots = self.named.iter().map(|(_, slot)| slot);
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", path) if HEALTH.contains(&path) => {
                health(slots.all(|s| s.get().is_some()), open)
            }
            ("GET", "/v1/models") => cards(slots, Some(&self.turns)),
            ("GET", path) if path.starts_with("/v1/models/") => {
                model_card(slots, &path["/v1/models/".len()..], Some(&self.turns))
            }
            _ => route(self.pick(&request.body), request),
        }
    }
}

/// Runs `f` on the model, or refuses while it loads.
fn with_model<T>(slot: &Slot, f: impl FnOnce(&Model) -> Result<T, Refusal>) -> Result<T, Refusal> {
    let loading = || refuse(503, "the model is loading", None);
    f(slot.get().ok_or_else(loading)?)
}

/// The messages as one prompt, by the model's template.
fn render(model: &Model, m: Messages, allow_tools: bool) -> Result<(String, bool), Refusal> {
    let chat = model
        .chat
        .as_ref()
        .ok_or_else(|| bad("the model has no chat template", "messages"))?;
    let mut messages = Vec::with_capacity(m.messages.len());
    for said in m.messages {
        messages.push(said.into_message()?);
    }
    let mut tools = Vec::new();
    if allow_tools && let Some(given) = m.tools {
        for tool in given {
            tools.push(tool.to_value()?);
        }
    }
    let effort = m
        .reasoning_effort
        .as_deref()
        .or(m.chat_template_kwargs.reasoning_effort.as_deref());
    let think = m
        .chat_template_kwargs
        .enable_thinking
        .or_else(|| (effort == Some("none")).then_some(false));
    let options = RenderOptions {
        think,
        clear_thinking: m.chat_template_kwargs.clear_thinking,
        reasoning_effort: effort,
        tools: &tools,
    };
    let prompt = chat
        .render_options(&messages, &options)
        .map_err(|e| bad(e.to_string(), "messages"))?;
    Ok((prompt, !tools.is_empty()))
}

/// A JSON answer, or the refusal.
fn reply(slot: &Slot, f: impl FnOnce(&Model) -> Result<Value, Refusal>) -> Response {
    match with_model(slot, f) {
        Ok(v) => Response::json(200, &v),
        Err(refused) => refused.into(),
    }
}

/// Paths that answer `health`.
const HEALTH: [&str; 4] = ["/health", "/v1/health", "/v1/health/ready", "/ping"];

/// Ready or loading, matching llama.cpp's status, with load steps.
fn health(ready: bool, open: Option<&Open>) -> Response {
    if ready {
        return Response::json(200, &json!({"status": "ok"}));
    }
    let error =
        json!({"code": 503, "message": "the model is loading", "type": "unavailable_error"});
    let mut body = json!({"error": error});
    if let Some(open) = open {
        body["progress"] = open.steps().iter().map(step).collect();
    }
    Response::json(503, &body)
}

/// One open load step as JSON.
pub fn step(event: &Event) -> Value {
    let unit = match event.unit {
        Unit::Bytes => "bytes",
        Unit::Items => "items",
    };
    json!({
        "id": event.id,
        "parent": event.parent,
        "name": &*event.name,
        "unit": unit,
        "done": event.done,
        "total": event.total,
    })
}

/// One loaded model's OpenAI card, with its seat's numbers.
fn card(m: &Model, turns: Option<&Turns>) -> Value {
    let mut out = json!({
        "id": m.info.name,
        "object": "model",
        "created": wire::now(),
        "owned_by": "tessera",
        "root": m.info.name,
        "max_model_len": m.info.context,
    });
    if let Some(seat) = turns.and_then(|t| t.status(&m.info.name)) {
        out["resident"] = seat.resident.into();
        out["bytes"] = seat.bytes.into();
        out["load_seconds"] = seat.seconds.into();
        out["loads"] = seat.loads.into();
    }
    out
}

/// Decodes `%XX` escapes in a URL path segment.
fn decode_id(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3])
            && let Ok(b) = u8::from_str_radix(hex, 16)
        {
            out.push(b);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| raw.to_string())
}

/// A card for each loaded model, with its seat's numbers.
fn cards<'a>(slots: impl Iterator<Item = &'a Slot>, turns: Option<&Turns>) -> Response {
    let data: Vec<Value> = slots
        .filter_map(|slot| slot.get())
        .map(|m| card(m, turns))
        .collect();
    match data.is_empty() {
        true => refuse(503, "the model is loading", None).into(),
        false => Response::json(200, &json!({"object": "list", "data": data})),
    }
}

/// One model card by id, or 404 when absent.
fn model_card<'a>(
    slots: impl Iterator<Item = &'a Slot>,
    raw_id: &str,
    turns: Option<&Turns>,
) -> Response {
    let loaded: Vec<&Model> = slots.filter_map(|slot| slot.get()).collect();
    if loaded.is_empty() {
        return refuse(503, "the model is loading", None).into();
    }
    let id = decode_id(raw_id);
    if id.is_empty() {
        return refuse(404, "no such route", None).into();
    }
    match loaded.into_iter().find(|m| m.info.name == id) {
        Some(m) => Response::json(200, &card(m, turns)),
        None => refuse(
            404,
            format!("The model `{id}` does not exist."),
            Some("model"),
        )
        .into(),
    }
}

fn tokenize(slot: &Slot, body: &[u8]) -> Response {
    reply(slot, |m| {
        let text = match wire::parse(m, body)? {
            Tokenize::Prompt { prompt, .. } => prompt,
            Tokenize::Chat { messages, .. } => render(m, messages, true)?.0,
        };
        let tokens = m
            .tokens
            .encode(&text)
            .map_err(|e| bad(e.to_string(), "prompt"))?;
        Ok(json!({"count": tokens.len(), "max_model_len": m.info.context, "tokens": tokens}))
    })
}

fn detokenize(slot: &Slot, body: &[u8]) -> Response {
    reply(slot, |m| {
        let d: Detokenize = wire::parse(m, body)?;
        Ok(json!({"prompt": m.tokens.decode(&d.tokens)}))
    })
}

fn embeddings(slot: &Slot, body: &[u8]) -> Response {
    reply(slot, |m| {
        let embed = m
            .embed
            .as_ref()
            .ok_or_else(|| refuse(404, "the model does not embed text", None))?;
        let e: Embeddings = wire::parse(m, body)?;
        let texts = e.input.list();
        let got = embed
            .embed(&texts)
            .map_err(|e| bad(e.to_string(), "input"))?;
        let mut data = Vec::with_capacity(got.vectors.len());
        for (index, v) in got.vectors.iter().enumerate() {
            data.push(json!({"object": "embedding", "index": index, "embedding": v}));
        }
        let usage = json!({"prompt_tokens": got.tokens, "total_tokens": got.tokens});
        Ok(json!({"object": "list", "data": data, "model": m.info.name, "usage": usage}))
    })
}

tessera_engine::metric_registry!(API = "api");
tessera_engine::metric!(COMPLETIONS in API = Counter "tessera:completion_requests_total", "Requests to /v1/completions.");
tessera_engine::metric!(CHATS in API = Counter "tessera:chat_requests_total", "Requests to /v1/chat/completions.");
tessera_engine::metric!(MESSAGES in API = Counter "tessera:messages_requests_total", "Requests to /v1/messages.");
tessera_engine::metric!(EMBEDS in API = Counter "tessera:embedding_requests_total", "Requests to /v1/embeddings.");
tessera_engine::metric!(DECISIONS in API = Counter "tessera:systemone_requests_total", "Requests to /v1/systemone.");
tessera_engine::metric!(IMAGES in API = Counter "tessera:image_requests_total", "Requests to /v1/images/generations.");

/// Answers one request.
pub fn route(slot: &Slot, request: &Request) -> Response {
    let body = request.body.as_slice();
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", path) if HEALTH.contains(&path) => health(slot.get().is_some(), None),
        ("GET", "/version") => Response::json(200, &json!({"version": env!("CARGO_PKG_VERSION")})),
        ("GET", "/v1/models") => cards(std::iter::once(slot), None),
        ("GET", path) if path.starts_with("/v1/models/") => {
            model_card(std::iter::once(slot), &path["/v1/models/".len()..], None)
        }
        ("GET", "/metrics") => metrics::metrics(),
        ("POST", "/tokenize") => tokenize(slot, body),
        ("POST", "/detokenize") => detokenize(slot, body),
        ("POST", "/v1/embeddings") => {
            EMBEDS.add(1);
            embeddings(slot, body)
        }
        ("POST", "/v1/images/generations") => {
            IMAGES.add(1);
            images::generations(slot, body)
        }
        ("POST", "/v1/completions") => {
            COMPLETIONS.add(1);
            answer::completions(slot, request)
        }
        ("POST", "/v1/chat/completions") => {
            CHATS.add(1);
            answer::chat_completions(slot, request)
        }
        ("POST", "/v1/messages") => {
            MESSAGES.add(1);
            messages::messages(slot, request)
        }
        ("POST", "/v1/systemone") => {
            DECISIONS.add(1);
            decide::systemone(slot, body)
        }
        _ => refuse(404, "no such route", None).into(),
    }
}
