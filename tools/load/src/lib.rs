//! Concurrent streamed completions against tessera-serve.

mod http;

use http::Sample;
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub type Failure = Box<dyn std::error::Error + Send + Sync>;

pub const USAGE: &str = "usage: tessera-load [--url <http://host:port>] [--clients <list>] \
[--requests <n>] [--rounds <n>] [--prompt-words <n>] [--prompt <text>] \
[--max-tokens <n>] [--warmup <n>]";

#[derive(Debug, Clone)]
pub struct Config {
    pub url: String,
    pub clients: Vec<usize>,
    pub requests: Option<usize>,
    pub rounds: usize,
    pub prompt_words: usize,
    pub prompt: Option<String>,
    pub max_tokens: usize,
    pub warmup: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:8080".into(),
            clients: vec![1],
            requests: None,
            rounds: 4,
            prompt_words: 32,
            prompt: None,
            max_tokens: 40,
            warmup: 1,
        }
    }
}

impl Config {
    pub fn parse(args: impl IntoIterator<Item = String>) -> Option<Self> {
        let mut config = Self::default();
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let value = args.next()?;
            match flag.as_str() {
                "--url" => config.url = value,
                "--clients" => {
                    config.clients = value
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .ok()?;
                }
                "--requests" => config.requests = Some(value.parse().ok()?),
                "--rounds" => config.rounds = value.parse().ok()?,
                "--prompt-words" => config.prompt_words = value.parse().ok()?,
                "--prompt" => config.prompt = Some(value),
                "--max-tokens" => config.max_tokens = value.parse().ok()?,
                "--warmup" => config.warmup = value.parse().ok()?,
                _ => return None,
            }
        }
        config.validate().ok()?;
        Some(config)
    }

    fn validate(&self) -> Result<(), Failure> {
        if self.clients.is_empty() || self.clients.contains(&0) {
            return Err("clients must be positive".into());
        }
        if self.requests == Some(0) || (self.requests.is_none() && self.rounds == 0) {
            return Err("request count must be positive".into());
        }
        if self.requests.is_none()
            && self
                .clients
                .iter()
                .any(|c| c.checked_mul(self.rounds).is_none())
        {
            return Err("request count is too large".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Span {
    pub p50: f64,
    pub p95: f64,
}

impl Span {
    pub fn samples(mut values: Vec<f64>) -> Self {
        values.sort_by(f64::total_cmp);
        let rank = |p: f64| {
            let n = values.len();
            if n == 0 {
                return 0.0;
            }
            values[((p * n as f64).ceil() as usize)
                .saturating_sub(1)
                .min(n - 1)]
        };
        Self {
            p50: rank(0.50),
            p95: rank(0.95),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Run {
    pub clients: usize,
    pub requests: usize,
    pub tokens: usize,
    pub elapsed_s: f64,
    pub tokens_per_s: f64,
    pub ttft_ms: Span,
    pub latency_ms: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub url: String,
    pub prompt_words: usize,
    pub prompt_chars: usize,
    pub max_tokens: usize,
    pub runs: Vec<Run>,
}

pub fn prompt(words: usize) -> String {
    "The quiet river flows through green fields while birds sing in the morning light."
        .split_whitespace()
        .cycle()
        .take(words)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Runs `clients` workers: a shared queue, or `rounds` each.
fn sweep(
    agent: &ureq::Agent,
    url: &str,
    body: &str,
    config: &Config,
    clients: usize,
) -> Result<Run, Failure> {
    let left = AtomicUsize::new(config.requests.unwrap_or(0));
    let more = |done: usize| match config.requests {
        Some(_) => {
            (left.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))).is_ok()
        }
        None => done < config.rounds,
    };
    let worker = || -> Result<Vec<Sample>, Failure> {
        let mut samples = Vec::new();
        while more(samples.len()) {
            samples.push(http::complete(agent, url, body)?);
        }
        Ok(samples)
    };
    let start = Instant::now();
    let results = thread::scope(|s| {
        let mut workers = Vec::new();
        for _ in 0..clients {
            workers.push(s.spawn(worker));
        }
        let mut results = Vec::new();
        for w in workers {
            results.push(w.join().expect("a worker panicked"));
        }
        results
    });
    let elapsed_s = start.elapsed().as_secs_f64();
    let mut samples = Vec::new();
    for result in results {
        samples.extend(result?);
    }
    let tokens = samples.iter().map(|s| s.tokens).sum();
    Ok(Run {
        clients,
        requests: samples.len(),
        tokens,
        elapsed_s,
        tokens_per_s: tokens as f64 / elapsed_s,
        ttft_ms: Span::samples(samples.iter().map(|s| s.ttft_ms).collect()),
        latency_ms: Span::samples(samples.iter().map(|s| s.latency_ms).collect()),
    })
}

/// Checks health, warms up, then sweeps each client count.
pub fn run(config: &Config) -> Result<Report, Failure> {
    config.validate()?;
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout_write(Duration::from_secs(10))
        .build();
    let url = config.url.trim_end_matches('/');
    let prompt = config
        .prompt
        .clone()
        .unwrap_or_else(|| prompt(config.prompt_words));
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": config.max_tokens,
        "stream": true,
    })
    .to_string();
    http::health(&agent, url)?;
    for _ in 0..config.warmup {
        http::complete(&agent, url, &body)?;
    }
    let mut runs = Vec::new();
    for clients in &config.clients {
        runs.push(sweep(&agent, url, &body, config, *clients)?);
    }
    Ok(Report {
        url: config.url.clone(),
        prompt_words: prompt.split_whitespace().count(),
        prompt_chars: prompt.len(),
        max_tokens: config.max_tokens,
        runs,
    })
}
