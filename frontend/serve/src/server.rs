//! The HTTP routes: health, completions and chat, plain or streamed.

use crate::text::Spool;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as Sse, KeepAlive, Sse as Stream};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bundle::{Message, Template};
use engine::{Event, Reason, Request};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::{Arc, Mutex, mpsc};
use tokenizers::Tokenizer;
use tokio::sync::mpsc as async_mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

/// A request for the engine thread, and its event channel.
pub type Job = (Request, async_mpsc::Sender<Event>);

/// What every handler shares.
pub struct Shared {
    pub tokenizer: Tokenizer,
    pub template: Option<Template>,
    pub stop: Vec<u32>,
    /// The engine thread's queue. Answers share each step.
    pub jobs: Mutex<mpsc::Sender<Job>>,
}

#[derive(Deserialize)]
pub struct Completion {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
}

#[derive(Deserialize)]
pub struct Chat {
    messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    /// Whether the model may think first. Left out: its default.
    #[serde(default = "yes")]
    think: bool,
}

fn yes() -> bool {
    true
}

fn default_max_tokens() -> usize {
    256
}

#[derive(Serialize)]
struct Answer {
    text: String,
    tokens: usize,
    finish_reason: &'static str,
}

/// One streamed piece: text, then the reason on the last.
#[derive(Serialize)]
struct Piece {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

fn reason(r: Reason) -> &'static str {
    match r {
        Reason::Stop => "stop",
        Reason::Length | Reason::Context => "length",
    }
}

pub fn router(shared: Arc<Shared>) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/completions", post(complete))
        .route("/v1/chat/completions", post(chat))
        .with_state(shared)
}

/// Why a request was refused, with its status.
type Refused = (StatusCode, String);

/// Queues the prompt. Returns the event stream, or why not.
fn submit(
    shared: &Shared,
    prompt: &str,
    max_tokens: usize,
) -> Result<async_mpsc::Receiver<Event>, Refused> {
    let bad = |what: String| (StatusCode::BAD_REQUEST, what);
    let encoded = shared.tokenizer.encode(prompt, false);
    let tokens = encoded.map_err(|e| bad(e.to_string()))?.get_ids().to_vec();
    if tokens.is_empty() {
        return Err(bad("an empty prompt".into()));
    }
    let request = Request {
        tokens,
        max_tokens,
        stop: shared.stop.clone(),
    };
    let (events, receiver) = async_mpsc::channel(64);
    let gone = || {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "the engine stopped".to_string(),
        )
    };
    let jobs = shared.jobs.lock().map_err(|_| gone())?;
    jobs.send((request, events)).map_err(|_| gone())?;
    Ok(receiver)
}

async fn complete(State(shared): State<Arc<Shared>>, Json(c): Json<Completion>) -> Response {
    match submit(&shared, &c.prompt, c.max_tokens) {
        Ok(events) => answer(shared, events, c.stream).await,
        Err(refused) => refused.into_response(),
    }
}

async fn chat(State(shared): State<Arc<Shared>>, Json(c): Json<Chat>) -> Response {
    let Some(template) = &shared.template else {
        return (StatusCode::BAD_REQUEST, "the model has no chat template").into_response();
    };
    if c.messages.is_empty() {
        return (StatusCode::BAD_REQUEST, "no messages").into_response();
    }
    let prompt = match template.render(&c.messages, c.think) {
        Ok(prompt) => prompt,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    match submit(&shared, &prompt, c.max_tokens) {
        Ok(events) => answer(shared, events, c.stream).await,
        Err(refused) => refused.into_response(),
    }
}

/// The events as text: one response, or streamed pieces.
async fn answer(
    shared: Arc<Shared>,
    mut events: async_mpsc::Receiver<Event>,
    stream: bool,
) -> Response {
    let decode = move |ids: &[u32]| shared.tokenizer.decode(ids, true).unwrap_or_default();
    let mut spool = Spool::default();
    if stream {
        let pieces = ReceiverStream::new(events).map(move |e| {
            let piece = match e {
                Event::Token(id) => Piece {
                    text: spool.push(id, &decode),
                    finish_reason: None,
                },
                Event::Done(r) => Piece {
                    text: String::new(),
                    finish_reason: Some(reason(r)),
                },
            };
            Ok::<Sse, Infallible>(Sse::default().json_data(piece).expect("a plain struct"))
        });
        return Stream::new(pieces)
            .keep_alive(KeepAlive::default())
            .into_response();
    }
    let (mut text, mut tokens, mut finish) = (String::new(), 0, "length");
    while let Some(e) = events.recv().await {
        match e {
            Event::Token(id) => {
                text += &spool.push(id, &decode);
                tokens += 1;
            }
            Event::Done(r) => finish = reason(r),
        }
    }
    Json(Answer {
        text,
        tokens,
        finish_reason: finish,
    })
    .into_response()
}
