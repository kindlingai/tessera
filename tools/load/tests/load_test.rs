use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tessera_load::{Config, Span};

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

/// A fake server on a free port.
struct Mock {
    url: String,
    state: Arc<State>,
}

impl Mock {
    fn start(mode: Mode) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(State::default());
        let shared = state.clone();
        thread::spawn(move || {
            for socket in listener.incoming() {
                let state = shared.clone();
                thread::spawn(move || respond(socket.unwrap(), mode, &state));
            }
        });
        Self { url, state }
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

const OK: &str = "HTTP/1.1 200 OK\r\nConnection: close\r\n";
const TIMEOUT: Duration = Duration::from_secs(5);

fn respond(socket: TcpStream, mode: Mode, state: &State) {
    socket.set_read_timeout(Some(TIMEOUT)).unwrap();
    socket.set_write_timeout(Some(TIMEOUT)).unwrap();
    let mut reader = BufReader::new(socket);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let health = line.starts_with("GET /health ");
    assert!(health || line.starts_with("POST /v1/completions "));
    let mut length = 0;
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap();
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    let mut socket = reader.into_inner();
    if health {
        state.health.fetch_add(1, Ordering::SeqCst);
        write!(socket, "{OK}Content-Length: 2\r\n\r\nok").unwrap();
        return;
    }
    let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(request["stream"], true);
    let tokens = request["max_tokens"].as_u64().unwrap() as usize;
    state.bodies.lock().unwrap().push(request);
    let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
    state.peak.fetch_max(active, Ordering::SeqCst);
    thread::sleep(Duration::from_millis(20));
    // Before the reply, so a client's next request never overlaps.
    state.active.fetch_sub(1, Ordering::SeqCst);
    let stream = events(tokens, mode);
    let sse = "Content-Type: text/event-stream\r\n";
    match mode {
        Mode::Refused => write!(
            socket,
            "HTTP/1.1 503 Unavailable\r\nConnection: close\r\nContent-Length: 11\r\n\r\nengine down"
        )
        .unwrap(),
        Mode::Length => write!(
            socket,
            "{OK}{sse}Content-Length: {}\r\n\r\n{stream}",
            stream.len()
        )
        .unwrap(),
        _ => {
            write!(socket, "{OK}{sse}Transfer-Encoding: chunked\r\n\r\n").unwrap();
            // Seven bytes at a time splits `é` across chunks.
            for part in stream.as_bytes().chunks(7) {
                write!(socket, "{:x}\r\n", part.len()).unwrap();
                socket.write_all(part).unwrap();
                socket.write_all(b"\r\n").unwrap();
            }
            socket.write_all(b"0\r\n\r\n").unwrap();
        }
    }
}

fn events(tokens: usize, mode: Mode) -> String {
    let mut stream = String::from(": keep-alive\n\n\r\n");
    if !matches!(mode, Mode::Empty) {
        for i in 0..tokens {
            let text = if i == 1 { "" } else { "café" };
            let chunk = format!(r#"{{"choices":[{{"text":"{text}","finish_reason":null}}]}}"#);
            stream.push_str(&format!("data: {chunk}\r\n\r\n"));
        }
    }
    if !matches!(mode, Mode::Truncated) {
        let last = r#"{"choices":[{"text":"","finish_reason":"length"}]}"#;
        stream.push_str(&format!("data: {last}\n\ndata: [DONE]\n\n"));
    }
    stream
}

#[test]
fn concurrent_sweeps() {
    let server = Mock::start(Mode::Chunked);
    let report = tessera_load::run(&server.config()).unwrap();
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
    assert_eq!(server.state.bodies.lock().unwrap().len(), 11);
    assert_eq!(server.state.peak.load(Ordering::SeqCst), 4);
}

#[test]
fn queued_requests() {
    let server = Mock::start(Mode::Length);
    let mut config = server.config();
    config.requests = Some(3);
    config.warmup = 0;
    config.prompt = Some("Hello café\nworld".into());
    let report = tessera_load::run(&config).unwrap();
    assert_eq!(report.prompt_words, 3);
    assert_eq!(report.prompt_chars, config.prompt.as_ref().unwrap().len());
    assert!(
        report
            .runs
            .iter()
            .all(|r| r.requests == 3 && r.tokens == 15)
    );
    let bodies = server.state.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 6);
    assert!(bodies.iter().all(|b| b["prompt"] == "Hello café\nworld"));
}

#[test]
fn empty_stream() {
    let server = Mock::start(Mode::Empty);
    let report = tessera_load::run(&server.config()).unwrap();
    for run in report.runs {
        assert_eq!(run.tokens, 0);
        assert_eq!(run.ttft_ms.p50, run.latency_ms.p50);
        assert_eq!(run.ttft_ms.p95, run.latency_ms.p95);
    }
}

#[test]
fn failed_streams() {
    for (mode, error) in [
        (Mode::Refused, "HTTP 503: engine down"),
        (Mode::Truncated, "stream ended without finish_reason"),
    ] {
        let server = Mock::start(mode);
        let result = tessera_load::run(&server.config()).unwrap_err();
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
        "--unknown x",
        "--warmup",
        "--clients 1,",
    ] {
        assert!(Config::parse(flags.split_whitespace().map(str::to_owned)).is_none());
    }
    let config = Config::parse(Vec::new()).unwrap();
    assert_eq!(config.clients, vec![1]);
    assert_eq!(config.rounds, 4);
    assert_eq!(tessera_load::prompt(32).split_whitespace().count(), 32);
    assert_eq!(tessera_load::prompt(0), "");
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
        tessera_load::USAGE
    );
}
