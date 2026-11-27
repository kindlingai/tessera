//! HTTP/1.1 on std threads: one thread per connection.

use crate::gate::{Capped, Gate};
use std::io::{self, ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

tessera_engine::metric_registry!(HTTP = "http");
tessera_engine::metric!(OPEN in HTTP = Gauge "tessera:http_connections_open", "Connections open now.");
tessera_engine::metric!(REFUSED in HTTP = Counter "tessera:http_refused_total", "Connections refused past the limit.");
tessera_engine::metric!(REQUESTS in HTTP = Counter "tessera:http_requests_total", "Requests read and routed.");
tessera_engine::metric!(MALFORMED in HTTP = Counter "tessera:http_malformed_total", "Requests refused before routing.");
tessera_engine::metric!(CLIENT in HTTP = Counter "tessera:http_client_errors_total", "Responses with a 4xx status.");
tessera_engine::metric!(SERVER in HTTP = Counter "tessera:http_server_errors_total", "Responses with a 5xx status.");

/// A request, in the routes' form.
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    /// Header names in lower case, and values.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    /// The value of header `name`, given in lower case.
    pub fn header(&self, name: &str) -> Option<&str> {
        let mut found = self.headers.iter().filter(|(n, _)| n == name);
        found.next().map(|(_, v)| v.as_str())
    }
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
    /// Chooses the connections to serve.
    pub gate: Arc<dyn Gate>,
    pub max_body: usize,
    pub max_head: usize,
    /// Time to read one request head, or to wait idle.
    pub head_timeout: Duration,
    pub body_timeout: Duration,
    pub write_timeout: Duration,
    /// Time between `: keep-alive` lines in a quiet stream.
    pub keep_alive: Duration,
    /// Optional `Access-Control-Allow-Origin` header value.
    pub cors: Option<String>,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            gate: Arc::new(Capped::new(64)),
            max_body: 32 << 20,
            max_head: 64 << 10,
            head_timeout: Duration::from_secs(60),
            body_timeout: Duration::from_secs(60),
            write_timeout: Duration::from_secs(60),
            keep_alive: Duration::from_secs(5),
            cors: None,
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
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
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

/// Tells the gate when its connection ends.
struct Held(Arc<dyn Gate>, Option<SocketAddr>);

impl Drop for Held {
    fn drop(&mut self) {
        self.0.leave(self.1);
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
        let peer = stream.peer_addr().ok();
        if !limits.gate.enter(peer) {
            REFUSED.add(1);
            let _ = stream.set_write_timeout(Some(limits.write_timeout));
            let _ = head(
                &mut stream,
                503,
                "text/plain",
                Some(0),
                true,
                limits.cors.as_deref(),
            );
            continue;
        }
        let held = Held(limits.gate.clone(), peer);
        OPEN.add(1);
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
                let _ = head(
                    &mut stream,
                    status,
                    "text/plain",
                    Some(0),
                    true,
                    limits.cors.as_deref(),
                );
                return;
            }
        };
        if request.method == "OPTIONS"
            && let Some(origin) = limits.cors.as_deref()
        {
            REQUESTS.add(1);
            if preflight(&mut stream, origin, close).is_err() || close {
                return;
            }
            continue;
        }
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
        let mut chunked = false;
        let mut expect = false;
        let mut headers = Vec::with_capacity(parsed.headers.len());
        for header in parsed.headers.iter() {
            let value = std::str::from_utf8(header.value).map_err(|_| Fail::Status(400))?;
            let value = value.trim();
            let name = header.name;
            headers.push((name.to_ascii_lowercase(), value.to_string()));
            if name.eq_ignore_ascii_case("transfer-encoding") {
                if !value.eq_ignore_ascii_case("chunked") || chunked {
                    return Err(Fail::Status(501));
                }
                chunked = true;
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
        if chunked && length.is_some() {
            return Err(Fail::Status(400));
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
            headers,
            body: Vec::new(),
        };
        buffer.drain(..used);
        if expect && (chunked || length > buffer.len()) {
            head(stream, 100, "", None, false, None)?;
        }
        let deadline = Instant::now() + limits.body_timeout;
        request.body = match chunked {
            true => read_chunked(stream, buffer, limits, deadline)?,
            false => {
                while buffer.len() < length {
                    if fill(stream, buffer, deadline)? == 0 {
                        return Err(Fail::Gone);
                    }
                }
                buffer.drain(..length).collect()
            }
        };
        return Ok(Some((request, close)));
    }
}

/// Reads one CRLF-terminated line from `buffer`, filling as needed.
fn read_line(
    stream: &mut TcpStream,
    buffer: &mut Vec<u8>,
    max: usize,
    deadline: Instant,
) -> Result<Vec<u8>, Fail> {
    loop {
        if let Some(at) = buffer.windows(2).position(|w| w == b"\r\n") {
            let line = buffer[..at].to_vec();
            buffer.drain(..at + 2);
            return Ok(line);
        }
        if buffer.len() >= max {
            return Err(Fail::Status(400));
        }
        if fill(stream, buffer, deadline)? == 0 {
            return Err(Fail::Gone);
        }
    }
}

/// Decodes an HTTP/1.1 chunked body up to `limits.max_body`.
fn read_chunked(
    stream: &mut TcpStream,
    buffer: &mut Vec<u8>,
    limits: &Limits,
    deadline: Instant,
) -> Result<Vec<u8>, Fail> {
    let mut body = Vec::new();
    loop {
        let line = read_line(stream, buffer, limits.max_head, deadline)?;
        let line = std::str::from_utf8(&line).map_err(|_| Fail::Status(400))?;
        let hex = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(hex, 16).map_err(|_| Fail::Status(400))?;
        if size == 0 {
            let mut trailer_bytes = 0usize;
            loop {
                let line = read_line(stream, buffer, limits.max_head, deadline)?;
                if line.is_empty() {
                    return Ok(body);
                }
                trailer_bytes = trailer_bytes.saturating_add(line.len() + 2);
                if trailer_bytes > limits.max_head {
                    return Err(Fail::Status(431));
                }
            }
        }
        let next = body.len().checked_add(size).ok_or(Fail::Status(413))?;
        if next > limits.max_body {
            return Err(Fail::Status(413));
        }
        let need = size.checked_add(2).ok_or(Fail::Status(413))?;
        while buffer.len() < need {
            if fill(stream, buffer, deadline)? == 0 {
                return Err(Fail::Gone);
            }
        }
        if &buffer[size..size + 2] != b"\r\n" {
            return Err(Fail::Status(400));
        }
        body.extend_from_slice(&buffer[..size]);
        buffer.drain(..size + 2);
    }
}

/// Appends CORS origin headers for `origin`.
fn write_cors(out: &mut String, origin: &str) {
    *out += &format!("Access-Control-Allow-Origin: {origin}\r\n");
    if origin != "*" {
        *out += "Vary: Origin\r\n";
    }
}

/// Answers a CORS preflight request with 204 No Content.
fn preflight(stream: &mut TcpStream, origin: &str, close: bool) -> io::Result<()> {
    let mut out = format!("HTTP/1.1 204 {}\r\n", reason(204));
    write_cors(&mut out, origin);
    out += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    out += "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key, anthropic-version\r\n";
    out += "Access-Control-Max-Age: 86400\r\nContent-Length: 0\r\n";
    if close {
        out += "Connection: close\r\n";
    }
    out += "\r\n";
    stream.write_all(out.as_bytes())
}

/// Writes a status line and headers. No length: chunked.
fn head(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    length: Option<usize>,
    close: bool,
    cors: Option<&str>,
) -> io::Result<()> {
    let mut out = format!("HTTP/1.1 {status} {}\r\n", reason(status));
    if status == 100 {
        out += "\r\n";
        return stream.write_all(out.as_bytes());
    }
    out += &format!("Content-Type: {content_type}\r\n");
    if let Some(origin) = cors {
        write_cors(&mut out, origin);
    }
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
                limits.cors.as_deref(),
            )?;
            return stream.write_all(&bytes);
        }
        Body::Events(events) => events,
    };
    head(
        stream,
        response.status,
        "text/event-stream",
        None,
        close,
        limits.cors.as_deref(),
    )?;
    loop {
        match events.recv_timeout(limits.keep_alive) {
            Ok(data) => {
                let frame = match data.starts_with("event: ") {
                    true => format!("{data}\n\n"),
                    false => format!("data: {data}\n\n"),
                };
                chunk(stream, frame.as_bytes())?;
            }
            Err(RecvTimeoutError::Timeout) => chunk(stream, b": keep-alive\n\n")?,
            Err(RecvTimeoutError::Disconnected) => return stream.write_all(b"0\r\n\r\n"),
        }
    }
}
