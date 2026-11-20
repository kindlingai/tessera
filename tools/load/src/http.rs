//! Health checks and streamed completions over HTTP.

use crate::Failure;
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::time::Instant;

/// One completion, from the client's side.
pub(crate) struct Sample {
    pub tokens: usize,
    pub ttft_ms: f64,
    pub latency_ms: f64,
}

/// One `data:` chunk. The last choice carries a reason.
#[derive(Deserialize)]
struct Piece {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    finish_reason: Option<String>,
}

/// The response, or its status and body as the error.
fn send(request: ureq::Request, body: Option<&str>) -> Result<ureq::Response, Failure> {
    let sent = match body {
        Some(body) => request
            .set("Content-Type", "application/json")
            .send_string(body),
        None => request.call(),
    };
    match sent {
        Ok(response) => Ok(response),
        Err(ureq::Error::Status(code, response)) => {
            let text = response.into_string().unwrap_or_default();
            Err(format!("HTTP {code}: {text}").into())
        }
        Err(e) => Err(e.into()),
    }
}

pub(crate) fn health(agent: &ureq::Agent, url: &str) -> Result<(), Failure> {
    // A status other than 200 is an error here.
    send(agent.get(&format!("{url}/health")), None)?;
    Ok(())
}

/// Streams one completion. Times the first token and the end.
pub(crate) fn complete(agent: &ureq::Agent, url: &str, body: &str) -> Result<Sample, Failure> {
    let start = Instant::now();
    let response = send(agent.post(&format!("{url}/v1/completions")), Some(body))?;
    if response.content_type() != "text/event-stream" {
        return Err("expected text/event-stream".into());
    }
    let (mut tokens, mut first, mut done) = (0, None, None);
    for line in BufReader::new(response.into_reader()).lines() {
        let line = line?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim_start();
        if data == "[DONE]" {
            break;
        }
        if done.is_some() {
            return Err("data after stream completion".into());
        }
        let piece: Piece = serde_json::from_str(data)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let reason = piece
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.finish_reason);
        match reason {
            Some(_) => done = Some(ms),
            None => {
                tokens += 1;
                first.get_or_insert(ms);
            }
        }
    }
    let latency_ms = done.ok_or("stream ended without finish_reason")?;
    Ok(Sample {
        tokens,
        ttft_ms: first.unwrap_or(latency_ms),
        latency_ms,
    })
}
