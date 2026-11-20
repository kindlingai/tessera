//! HTTP/1.1 on std threads: one thread per connection.

use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

sensai_engine::metric_registry!(HTTP = "http");
sensai_engine::metric!(OPEN in HTTP = Gauge "tessera:http_connections_open", "Connections open now.");
sensai_engine::metric!(REFUSED in HTTP = Counter "tessera:http_refused_total", "Connections refused past the limit.");
sensai_engine::metric!(REQUESTS in HTTP = Counter "tessera:http_requests_total", "Requests read and routed.");
sensai_engine::metric!(MALFORMED in HTTP = Counter "tessera:http_malformed_total", "Requests refused before routing.");
sensai_engine::metric!(CLIENT in HTTP = Counter "tessera:http_client_errors_total", "Responses with a 4xx status.");
sensai_engine::metric!(SERVER in HTTP = Counter "tessera:http_server_errors_total", "Responses with a 5xx status.");

/// A request as the routes see it.
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    pub body: Vec<u8>,
}

pub enum Body {
    Full {
        content_type: &'static str,
        bytes: Vec<u8>,
    },
    /// One `data:` line per item, until the sender drops.
    Events(Receiver<String>),
}

pub struct Response {
    pub status: u16,
    pub body: Body,
}

impl Response {
    pub fn json(status: u16, value: &serde_json::Value) -> Response {
        let bytes = value.to_string().into_bytes();
        let content_type = "application/json";
        Response {
            status,
            body: Body::Full {
                content_type,
                bytes,
            },
        }
    }
}

pub struct Limits {
    pub max_connections: usize,
    pub max_body: usize,
    pub max_head: usize,
    /// Time to read one request head, or to wait idle.
    pub head_timeout: Duration,
    pub body_timeout: Duration,
    pub write_timeout: Duration,
    /// Time between `: keep-alive` lines in a quiet stream.
    pub keep_alive: Duration,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            max_connections: 64,
            max_body: 32 << 20,
            max_head: 64 << 10,
            head_timeout: Duration::from_secs(60),
            body_timeout: Duration::from_secs(60),
            write_timeout: Duration::from_secs(60),
            keep_alive: Duration::from_secs(5),
        }
    }
}

/// A transport failure that ends the connection.
enum Fail {
    /// Answer with this status, then close.
    Status(u16),
    /// Close without an answer.
    Gone,
}

impl From<io::Error> for Fail {
    fn from(e: io::Error) -> Fail {
        match e.kind() {
            ErrorKind::WouldBlock | ErrorKind::TimedOut => Fail::Status(408),
            _ => Fail::Gone,
        }
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Content Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "",
    }
}

/// Lowers the count when its connection ends.
struct Held(Arc<AtomicUsize>);

impl Drop for Held {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
        OPEN.sub(1);
    }
}

/// Accepts connections until the listener fails.
pub fn run<F>(listener: TcpListener, limits: Limits, route: F)
where
    F: Fn(&Request) -> Response + Send + Sync + 'static,
{
    let limits = Arc::new(limits);
    let route = Arc::new(route);
    let open = Arc::new(AtomicUsize::new(0));
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(e) => {
                eprintln!("tessera-serve: accept: {e}");
                // Too many open files: give the others time to close.
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        let held = Held(open.clone());
        OPEN.add(1);
        if open.fetch_add(1, Ordering::Relaxed) >= limits.max_connections {
            REFUSED.add(1);
            let _ = stream.set_write_timeout(Some(limits.write_timeout));
            let _ = head(&mut stream, 503, "text/plain", Some(0), true);
            continue;
        }
        let (limits, route) = (limits.clone(), route.clone());
        std::thread::spawn(move || {
            let _held = held;
            connection(stream, &limits, &*route);
        });
    }
}

fn connection(mut stream: TcpStream, limits: &Limits, route: &dyn Fn(&Request) -> Response) {
    let _ = stream.set_write_timeout(Some(limits.write_timeout));
    let _ = stream.set_nodelay(true);
    let mut buffer = Vec::new();
    loop {
        let (request, close) = match read(&mut stream, &mut buffer, limits) {
            Ok(Some(read)) => read,
            Ok(None) | Err(Fail::Gone) => return,
            Err(Fail::Status(status)) => {
                MALFORMED.add(1);
                let _ = head(&mut stream, status, "text/plain", Some(0), true);
                return;
            }
        };
        let response = route(&request);
        REQUESTS.add(1);
        match response.status {
            400..=499 => CLIENT.add(1),
            500..=599 => SERVER.add(1),
            _ => {}
        }
        if answer(&mut stream, response, close, limits).is_err() || close {
            return;
        }
    }
}

/// Reads until `deadline`. `Ok(0)`: the peer closed.
fn fill(stream: &mut TcpStream, buffer: &mut Vec<u8>, deadline: Instant) -> io::Result<usize> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(ErrorKind::TimedOut.into());
    }
    stream.set_read_timeout(Some(left))?;
    let mut chunk = [0u8; 8192];
    let n = stream.read(&mut chunk)?;
    buffer.extend_from_slice(&chunk[..n]);
    Ok(n)
}

/// The next request, and whether to close after it.
fn read(
    stream: &mut TcpStream,
    buffer: &mut Vec<u8>,
    limits: &Limits,
) -> Result<Option<(Request, bool)>, Fail> {
    let deadline = Instant::now() + limits.head_timeout;
    loop {
        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut parsed = httparse::Request::new(&mut headers);
        let used = match parsed.parse(buffer) {
            Ok(httparse::Status::Complete(used)) => used,
            Ok(httparse::Status::Partial) if buffer.len() >= limits.max_head => {
                return Err(Fail::Status(431));
            }
            Ok(httparse::Status::Partial) => {
                let idle = buffer.is_empty();
                match fill(stream, buffer, deadline) {
                    Ok(0) if idle => return Ok(None),
                    Ok(0) => return Err(Fail::Gone),
                    Ok(_) => continue,
                    // An idle keep-alive connection timed out.
                    Err(_) if idle => return Ok(None),
                    Err(e) => return Err(e.into()),
                }
            }
            Err(_) => return Err(Fail::Status(400)),
        };
        let mut close = parsed.version != Some(1);
        let mut length: Option<usize> = None;
        let mut expect = false;
        for header in parsed.headers.iter() {
            let value = std::str::from_utf8(header.value).map_err(|_| Fail::Status(400))?;
            let value = value.trim();
            let name = header.name;
            if name.eq_ignore_ascii_case("transfer-encoding") {
                return Err(Fail::Status(501));
            } else if name.eq_ignore_ascii_case("content-length") {
                let n: usize = value.parse().map_err(|_| Fail::Status(400))?;
                if length.is_some_and(|m| m != n) {
                    return Err(Fail::Status(400));
                }
                length = Some(n);
            } else if name.eq_ignore_ascii_case("connection") {
                if value.eq_ignore_ascii_case("close") {
                    close = true;
                } else if value.eq_ignore_ascii_case("keep-alive") {
                    close = false;
                }
            } else if name.eq_ignore_ascii_case("expect") {
                expect = value.eq_ignore_ascii_case("100-continue");
            }
        }
        let length = length.unwrap_or(0);
        if length > limits.max_body {
            return Err(Fail::Status(413));
        }
        let target = parsed.path.unwrap_or("/");
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        let mut request = Request {
            method: parsed.method.unwrap_or("").to_string(),
            path: path.to_string(),
            query: query.to_string(),
            body: Vec::new(),
        };
        buffer.drain(..used);
        if expect && length > buffer.len() {
            head(stream, 100, "", None, false)?;
        }
        let deadline = Instant::now() + limits.body_timeout;
        while buffer.len() < length {
            if fill(stream, buffer, deadline)? == 0 {
                return Err(Fail::Gone);
            }
        }
        request.body = buffer.drain(..length).collect();
        return Ok(Some((request, close)));
    }
}

/// Writes a status line and headers. No length: chunked.
fn head(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    length: Option<usize>,
    close: bool,
) -> io::Result<()> {
    let mut out = format!("HTTP/1.1 {status} {}\r\n", reason(status));
    if status == 100 {
        out += "\r\n";
        return stream.write_all(out.as_bytes());
    }
    out += &format!("Content-Type: {content_type}\r\n");
    match length {
        Some(n) => out += &format!("Content-Length: {n}\r\n"),
        None => out += "Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n",
    }
    if close {
        out += "Connection: close\r\n";
    }
    out += "\r\n";
    stream.write_all(out.as_bytes())
}

fn chunk(stream: &mut TcpStream, data: &[u8]) -> io::Result<()> {
    let mut out = format!("{:x}\r\n", data.len()).into_bytes();
    out.extend_from_slice(data);
    out.extend_from_slice(b"\r\n");
    stream.write_all(&out)
}

/// Writes the response. A failed write drops the event receiver.
fn answer(
    stream: &mut TcpStream,
    response: Response,
    close: bool,
    limits: &Limits,
) -> io::Result<()> {
    let events = match response.body {
        Body::Full {
            content_type,
            bytes,
        } => {
            head(
                stream,
                response.status,
                content_type,
                Some(bytes.len()),
                close,
            )?;
            return stream.write_all(&bytes);
        }
        Body::Events(events) => events,
    };
    head(stream, response.status, "text/event-stream", None, close)?;
    loop {
        match events.recv_timeout(limits.keep_alive) {
            Ok(data) => chunk(stream, format!("data: {data}\n\n").as_bytes())?,
            Err(RecvTimeoutError::Timeout) => chunk(stream, b": keep-alive\n\n")?,
            Err(RecvTimeoutError::Disconnected) => return stream.write_all(b"0\r\n\r\n"),
        }
    }
}
