//! Mocks of each trait, and a client for the routes.

mod chat;
mod embed;
mod generator;
mod image;
mod tokens;

use crate::http::{Body, Request};
use crate::{Model, Slot, route};
use chat::MockChat;
use embed::MockEmbed;
pub use generator::MockGenerator;
use image::MockImage;
use serde_json::Value;
use std::sync::{Arc, OnceLock};
use tessera_bundle::ResponseParser;
use tessera_tasks::api::Info;
pub(crate) use tokens::MockTokens;

/// The mocks as model `m`, 16 tokens of context.
pub fn loaded(generator: &Arc<MockGenerator>, strict: bool) -> Slot {
    loaded_with(generator, strict, None)
}

/// Like `loaded`, with an optional response parser.
pub(crate) fn loaded_with(
    generator: &Arc<MockGenerator>,
    strict: bool,
    response: Option<ResponseParser>,
) -> Slot {
    let slot: Slot = Arc::new(OnceLock::new());
    let _ = slot.set(Model {
        info: Info {
            name: "m".into(),
            context: 16,
        },
        tokens: Arc::new(MockTokens),
        chat: Some(Arc::new(MockChat { response })),
        generator: Some(generator.clone()),
        embed: Some(Arc::new(MockEmbed)),
        image: Some(Arc::new(MockImage)),
        score: Some(generator.clone()),
        strict,
        asks: None,
    });
    slot
}

/// The status and body of a call. No body: GET.
pub fn call(slot: &Slot, path: &str, body: Option<Value>) -> (u16, String) {
    let request = Request {
        method: if body.is_some() { "POST" } else { "GET" }.into(),
        path: path.into(),
        headers: Vec::new(),
        body: body.map_or_else(Vec::new, |b| b.to_string().into_bytes()),
    };
    let response = route(slot, &request);
    let text = match response.body {
        Body::Full { bytes, .. } => String::from_utf8(bytes).unwrap(),
        Body::Events(events) => events
            .iter()
            .map(|e| match e.starts_with("event: ") {
                true => format!("{e}\n\n"),
                false => format!("data: {e}\n\n"),
            })
            .collect(),
    };
    (response.status, text)
}

/// The same, with the body as JSON.
pub fn json(slot: &Slot, path: &str, body: Option<Value>) -> (u16, Value) {
    let (status, text) = call(slot, path, body);
    (status, serde_json::from_str(&text).unwrap())
}
