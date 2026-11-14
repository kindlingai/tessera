use load::{Config, Span};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::task::{JoinHandle, JoinSet};

#[derive(Clone, Copy)]
enum Mode {
    Chunked,
    Length,
    Empty,
    Refused,
    Truncated,
}

#[derive(Default)]
struct State {
    health: AtomicUsize,
    active: AtomicUsize,
    peak: AtomicUsize,
    bodies: Mutex<Vec<serde_json::Value>>,
}

struct Mock {
    url: String,
    state: Arc<State>,
    task: JoinHandle<()>,
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Mock {
    async fn start(mode: Mode) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(State::default());
        let shared = state.clone();
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                tokio::select! {
                    socket = listener.accept() => {
                        let (socket, _) = socket.unwrap();
                        connections.spawn(respond(socket, mode, shared.clone()));
                    }
                    result = connections.join_next(), if !connections.is_empty() => {
                        result.unwrap().unwrap();
                    }
                }
            }
        });
        Self { url, state, task }
    }

    fn config(&self) -> Config {
        Config {
            url: self.url.clone(),
            clients: vec![1, 4],
            rounds: 2,
            max_tokens: 5,
            ..Config::default()
        }
    }
}

async fn respond(socket: TcpStream, mode: Mode, state: Arc<State>) {
    let mut reader = BufReader::new(socket);
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();
    let health = line.starts_with("GET /health ");
    assert!(health || line.starts_with("POST /v1/completions "));
    let mut length = 0;
    loop {
        line.clear();
        reader.read_line(&mut line).await.unwrap();
        if line == "\r\n" {
            break;
        }
        if let Some(value) = line.strip_prefix("Content-Length:") {
            length = value.trim().parse().unwrap();
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).await.unwrap();
    let mut socket = reader.into_inner();
    if health {
        state.health.fetch_add(1, Ordering::SeqCst);
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            .await
            .unwrap();
        return;
    }
    let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(request["stream"], true);
    let tokens = request["max_tokens"].as_u64().unwrap() as usize;
    state.bodies.lock().await.push(request);
    let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
    state.peak.fetch_max(active, Ordering::SeqCst);
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let stream = events(tokens, mode);
    if matches!(mode, Mode::Refused) {
        socket
            .write_all(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 11\r\n\r\nengine down")
            .await
            .unwrap();
    } else if matches!(mode, Mode::Length) {
        socket
            .write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: text/event-stream\r\n\r\n{stream}", stream.len()).as_bytes())
            .await
            .unwrap();
    } else {
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        for part in stream.as_bytes().chunks(7) {
            socket
                .write_all(format!("{:x};test=yes\r\n", part.len()).as_bytes())
                .await
                .unwrap();
            socket.write_all(part).await.unwrap();
            socket.write_all(b"\r\n").await.unwrap();
        }
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    }
    state.active.fetch_sub(1, Ordering::SeqCst);
}

fn events(tokens: usize, mode: Mode) -> String {
    let mut stream = String::from(": keep-alive\n\n\r\n");
    if !matches!(mode, Mode::Empty) {
        for i in 0..tokens {
            let text = if i == 1 { "" } else { "café" };
            stream.push_str(&format!("data: {{\"text\":\"{text}\"}}\r\n\r\n"));
        }
    }
    if !matches!(mode, Mode::Truncated) {
        stream.push_str("data: {\"text\":\"\",\"finish_reason\":\"length\"}\n\n");
    }
    stream
}

#[tokio::test]
async fn concurrent_sweeps() {
    let server = Mock::start(Mode::Chunked).await;
    let report = load::run(&server.config()).await.unwrap();
    assert_eq!(report.runs.len(), 2);
    for (run, requests) in report.runs.iter().zip([2, 8]) {
        assert_eq!(run.requests, requests);
        assert_eq!(run.tokens, requests * 5);
        assert!(run.tokens_per_s > 0.0);
        assert!(run.ttft_ms.p50 <= run.ttft_ms.p95);
        assert!(run.latency_ms.p50 <= run.latency_ms.p95);
        assert!(run.ttft_ms.p50 <= run.latency_ms.p50);
    }
    assert_eq!(server.state.health.load(Ordering::SeqCst), 1);
    assert_eq!(server.state.bodies.lock().await.len(), 11);
    assert_eq!(server.state.peak.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn queued_requests() {
    let server = Mock::start(Mode::Length).await;
    let mut config = server.config();
    config.requests = Some(3);
    config.warmup = 0;
    config.prompt = Some("Hello café\nworld".into());
    let report = load::run(&config).await.unwrap();
    assert_eq!(report.prompt_words, 3);
    assert_eq!(report.prompt_chars, config.prompt.as_ref().unwrap().len());
    assert!(
        report
            .runs
            .iter()
            .all(|r| r.requests == 3 && r.tokens == 15)
    );
    let bodies = server.state.bodies.lock().await;
    assert_eq!(bodies.len(), 6);
    assert!(bodies.iter().all(|b| b["prompt"] == "Hello café\nworld"));
}

#[tokio::test]
async fn empty_stream() {
    let server = Mock::start(Mode::Empty).await;
    let report = load::run(&server.config()).await.unwrap();
    for run in report.runs {
        assert_eq!(run.tokens, 0);
        assert_eq!(run.ttft_ms.p50, run.latency_ms.p50);
        assert_eq!(run.ttft_ms.p95, run.latency_ms.p95);
    }
}

#[tokio::test]
async fn failed_streams() {
    for (mode, error) in [
        (Mode::Refused, "HTTP 503: engine down"),
        (Mode::Truncated, "stream ended without finish_reason"),
    ] {
        let server = Mock::start(mode).await;
        let result = load::run(&server.config()).await.unwrap_err();
        assert_eq!(result.to_string(), error);
    }
}

#[test]
fn config_inputs() {
    for flags in [
        "--clients 0",
        "--clients 1,0",
        "--requests 0",
        "--rounds 0",
        "--url https://localhost",
        "--url http://[::1",
        "--unknown x",
        "--warmup",
        "--clients 1,",
    ] {
        assert!(Config::parse(flags.split_whitespace().map(str::to_owned)).is_none());
    }
    for url in [
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
        "http://[::1]",
        "http://localhost",
    ] {
        assert!(Config::parse(["--url".into(), url.into()]).is_some());
    }
    let config = Config::parse(Vec::new()).unwrap();
    assert_eq!(config.clients, vec![1]);
    assert_eq!(config.rounds, 4);
    assert_eq!(load::prompt(32).split_whitespace().count(), 32);
    assert_eq!(load::prompt(0), "");
}

#[test]
fn nearest_rank() {
    let span = Span::samples((1..=20).rev().map(f64::from).collect());
    assert_eq!(span.p50, 10.0);
    assert_eq!(span.p95, 19.0);
    let single = Span::samples(vec![7.0]);
    assert_eq!(single.p50, 7.0);
    assert_eq!(single.p95, 7.0);
}

#[test]
fn cli_errors() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_tessera-load"))
        .args(["--clients", "0"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(
        String::from_utf8(output.stderr).unwrap().trim(),
        load::USAGE
    );
}
