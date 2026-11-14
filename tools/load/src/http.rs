use crate::Failure;
use serde::Deserialize;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

#[derive(Clone)]
pub(crate) struct Endpoint {
    host: String,
    port: u16,
    authority: String,
}

impl Endpoint {
    pub fn parse(url: &str) -> Result<Self, Failure> {
        let authority = url
            .strip_prefix("http://")
            .ok_or("URL must start with http://")?
            .trim_end_matches('/');
        if authority.is_empty()
            || authority
                .chars()
                .any(|c| c.is_whitespace() || c.is_control())
            || authority.contains(['/', '?', '#', '@'])
        {
            return Err("invalid HTTP authority".into());
        }
        let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
            let (host, suffix) = rest.split_once(']').ok_or("invalid IPv6 address")?;
            host.parse::<std::net::Ipv6Addr>()?;
            let port = if suffix.is_empty() {
                80
            } else {
                suffix.strip_prefix(':').ok_or("invalid port")?.parse()?
            };
            (host, port)
        } else {
            let (host, port) = match authority.split_once(':') {
                Some((host, port)) => (host, port.parse()?),
                None => (authority, 80),
            };
            if host.is_empty() || host.contains(['[', ']']) {
                return Err("invalid host".into());
            }
            (host, port)
        };
        Ok(Self {
            host: host.into(),
            port,
            authority: authority.into(),
        })
    }

    async fn request(&self, path: &str, body: Option<&str>) -> Result<Response, Failure> {
        let mut socket = TcpStream::connect((self.host.as_str(), self.port)).await?;
        let method = if body.is_some() { "POST" } else { "GET" };
        let payload = body.unwrap_or("");
        let header = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            self.authority,
            payload.len()
        );
        socket.write_all(header.as_bytes()).await?;
        socket.write_all(payload.as_bytes()).await?;
        Response::open(socket).await
    }

    pub async fn health(&self) -> Result<(), Failure> {
        let mut response = self.request("/health", None).await?;
        response.check().await?;
        if response.collect().await? != b"ok" {
            return Err("health response is not ok".into());
        }
        Ok(())
    }

    pub async fn complete(&self, body: &str) -> Result<Sample, Failure> {
        let start = Instant::now();
        let mut response = self.request("/v1/completions", Some(body)).await?;
        response.check().await?;
        if !response.event_stream {
            return Err("expected text/event-stream".into());
        }
        let mut events = Events::new(start);
        while let Some(bytes) = response.read().await? {
            events.push(&bytes)?;
        }
        events.finish()
    }
}

struct Response {
    reader: BufReader<TcpStream>,
    status: u16,
    chunked: bool,
    remaining: Option<usize>,
    chunk_end: bool,
    ended: bool,
    event_stream: bool,
}

async fn line(reader: &mut BufReader<TcpStream>) -> Result<String, Failure> {
    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Err("unexpected HTTP EOF".into());
    }
    Ok(line.trim_end_matches(['\r', '\n']).into())
}

impl Response {
    async fn open(socket: TcpStream) -> Result<Self, Failure> {
        let mut reader = BufReader::new(socket);
        let status = line(&mut reader)
            .await?
            .split_whitespace()
            .nth(1)
            .ok_or("missing HTTP status")?
            .parse()?;
        let mut response = Self {
            reader,
            status,
            chunked: false,
            remaining: None,
            chunk_end: false,
            ended: false,
            event_stream: false,
        };
        loop {
            let header = line(&mut response.reader).await?;
            if header.is_empty() {
                break;
            }
            let (key, value) = header.split_once(':').ok_or("invalid HTTP header")?;
            let value = value.trim();
            match key.to_ascii_lowercase().as_str() {
                "content-length" => response.remaining = Some(value.parse()?),
                "transfer-encoding" => {
                    if !value.eq_ignore_ascii_case("chunked") {
                        return Err("unsupported transfer encoding".into());
                    }
                    response.chunked = true;
                }
                "content-type" => {
                    response.event_stream = value
                        .split(';')
                        .next()
                        .is_some_and(|v| v.trim().eq_ignore_ascii_case("text/event-stream"));
                }
                _ => {}
            }
        }
        if response.chunked {
            response.remaining = Some(0);
        }
        Ok(response)
    }

    async fn check(&mut self) -> Result<(), Failure> {
        if self.status != 200 {
            let body = self.collect().await?;
            return Err(format!("HTTP {}: {}", self.status, String::from_utf8_lossy(&body)).into());
        }
        Ok(())
    }

    async fn collect(&mut self) -> Result<Vec<u8>, Failure> {
        let mut body = Vec::new();
        while let Some(bytes) = self.read().await? {
            body.extend(bytes);
        }
        Ok(body)
    }

    async fn read(&mut self) -> Result<Option<Vec<u8>>, Failure> {
        if self.ended {
            return Ok(None);
        }
        if self.chunked && self.remaining == Some(0) {
            if self.chunk_end {
                let mut end = [0; 2];
                self.reader.read_exact(&mut end).await?;
                if end != *b"\r\n" {
                    return Err("invalid chunk ending".into());
                }
            }
            let size = line(&mut self.reader).await?;
            let size = size.split(';').next().ok_or("missing chunk size")?;
            self.remaining = Some(usize::from_str_radix(size.trim(), 16)?);
            self.chunk_end = true;
            if self.remaining == Some(0) {
                while !line(&mut self.reader).await?.is_empty() {}
                self.ended = true;
                return Ok(None);
            }
        }
        if self.remaining == Some(0) {
            self.ended = true;
            return Ok(None);
        }
        let mut bytes = vec![0; self.remaining.unwrap_or(8192).min(8192)];
        let count = self.reader.read(&mut bytes).await?;
        if count == 0 {
            if self.remaining.is_some() {
                return Err("unexpected HTTP body EOF".into());
            }
            self.ended = true;
            return Ok(None);
        }
        if let Some(remaining) = &mut self.remaining {
            *remaining -= count;
        }
        bytes.truncate(count);
        Ok(Some(bytes))
    }
}

pub(crate) struct Sample {
    pub tokens: usize,
    pub ttft_ms: f64,
    pub latency_ms: f64,
}

#[derive(Deserialize)]
struct Piece {
    #[serde(rename = "text")]
    _text: String,
    finish_reason: Option<String>,
}

struct Events {
    start: Instant,
    pending: Vec<u8>,
    tokens: usize,
    first: Option<f64>,
    done: Option<f64>,
}

impl Events {
    fn new(start: Instant) -> Self {
        Self {
            start,
            pending: Vec::new(),
            tokens: 0,
            first: None,
            done: None,
        }
    }

    fn push(&mut self, bytes: &[u8]) -> Result<(), Failure> {
        for byte in bytes {
            if *byte != b'\n' {
                self.pending.push(*byte);
                continue;
            }
            let line = std::str::from_utf8(&self.pending)?.trim_end_matches('\r');
            if let Some(data) = line.strip_prefix("data:") {
                if self.done.is_some() {
                    return Err("data after stream completion".into());
                }
                let piece: Piece = serde_json::from_str(data.trim_start())?;
                let elapsed = self.start.elapsed().as_secs_f64() * 1e3;
                if piece.finish_reason.is_some() {
                    self.done = Some(elapsed);
                } else {
                    self.tokens += 1;
                    self.first.get_or_insert(elapsed);
                }
            }
            self.pending.clear();
        }
        Ok(())
    }

    fn finish(self) -> Result<Sample, Failure> {
        let latency_ms = self.done.ok_or("stream ended without finish_reason")?;
        Ok(Sample {
            tokens: self.tokens,
            ttft_ms: self.first.unwrap_or(latency_ms),
            latency_ms,
        })
    }
}
