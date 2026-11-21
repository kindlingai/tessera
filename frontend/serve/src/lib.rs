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
mod metrics;
#[cfg(test)]
mod mocks;
pub mod ranks;
#[cfg(test)]
mod ranks_test;
pub mod start;

mod wire;
#[cfg(test)]
mod wire_test;

use http::{Request, Response};
pub use images::{fresh, square};
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};
use tessera_bundle::Message;
use tessera_tasks::api::{Chat, Embed, GenerateImage, Generator, Info, Score, Tokens};
use wire::{Detokenize, Embeddings, Messages, Refusal, Tokenize, bad, refuse};

/// The loaded model, from the routes' side.
pub struct Model {
    pub info: Info,
    pub tokens: Arc<dyn Tokens>,
    pub chat: Option<Arc<dyn Chat>>,
    pub generator: Option<Arc<dyn Generator>>,
    pub embed: Option<Arc<dyn Embed>>,
    pub image: Option<Arc<dyn GenerateImage>>,
    pub score: Option<Arc<dyn Score>>,
    /// Unknown request fields refuse the request. Else they warn.
    pub strict: bool,
}

/// The model once it loads. Until then routes answer 503.
pub type Slot = Arc<OnceLock<Model>>;

/// Runs `f` on the model, or refuses while it loads.
fn with_model<T>(slot: &Slot, f: impl FnOnce(&Model) -> Result<T, Refusal>) -> Result<T, Refusal> {
    let loading = || refuse(503, "the model is loading", None);
    f(slot.get().ok_or_else(loading)?)
}

/// The messages as one prompt, by the model's template.
fn render(model: &Model, m: Messages) -> Result<String, Refusal> {
    let chat = model
        .chat
        .as_ref()
        .ok_or_else(|| bad("the model has no chat template", "messages"))?;
    let messages: Vec<Message> = m.messages.into_iter().map(Message::from).collect();
    let think = m.chat_template_kwargs.enable_thinking.unwrap_or(true);
    chat.render(&messages, think)
        .map_err(|e| bad(e.to_string(), "messages"))
}

/// A JSON answer, or the refusal.
fn reply(slot: &Slot, f: impl FnOnce(&Model) -> Result<Value, Refusal>) -> Response {
    match with_model(slot, f) {
        Ok(v) => Response::json(200, &v),
        Err(refused) => refused.into(),
    }
}

/// Ready or loading, matching llama.cpp's status.
fn health(slot: &Slot) -> Response {
    match slot.get() {
        Some(_) => Response::json(200, &json!({"status": "ok"})),
        None => {
            let error = json!({"code": 503, "message": "the model is loading", "type": "unavailable_error"});
            Response::json(503, &json!({"error": error}))
        }
    }
}

fn models(slot: &Slot) -> Response {
    reply(slot, |m| {
        let card = json!({
            "id": m.info.name,
            "object": "model",
            "created": wire::now(),
            "owned_by": "tessera",
            "root": m.info.name,
            "max_model_len": m.info.context,
        });
        Ok(json!({"object": "list", "data": [card]}))
    })
}

fn tokenize(slot: &Slot, body: &[u8]) -> Response {
    reply(slot, |m| {
        let text = match wire::parse(m, body)? {
            Tokenize::Prompt { prompt, .. } => prompt,
            Tokenize::Chat { messages, .. } => render(m, messages)?,
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
tessera_engine::metric!(EMBEDS in API = Counter "tessera:embedding_requests_total", "Requests to /v1/embeddings.");
tessera_engine::metric!(DECISIONS in API = Counter "tessera:systemone_requests_total", "Requests to /v1/systemone.");
tessera_engine::metric!(IMAGES in API = Counter "tessera:image_requests_total", "Requests to /v1/images/generations.");

/// Answers one request.
pub fn route(slot: &Slot, request: &Request) -> Response {
    let body = request.body.as_slice();
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health" | "/v1/health" | "/v1/health/ready" | "/ping") => health(slot),
        ("GET", "/version") => Response::json(200, &json!({"version": env!("CARGO_PKG_VERSION")})),
        ("GET", "/v1/models") => models(slot),
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
            answer::completions(slot, body)
        }
        ("POST", "/v1/chat/completions") => {
            CHATS.add(1);
            answer::chat_completions(slot, body)
        }
        ("POST", "/v1/systemone") => {
            DECISIONS.add(1);
            decide::systemone(slot, body)
        }
        _ => refuse(404, "no such route", None).into(),
    }
}
