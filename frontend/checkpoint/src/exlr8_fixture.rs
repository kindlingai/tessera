//! Safetensors fixtures for the exlr8 tests.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use tessera_engine::{E4M3, F16};

/// One tensor: name, safetensors dtype, shape, bytes.
pub(crate) type Tensor<'a> = (&'a str, &'a str, Vec<usize>, Vec<u8>);

/// `tensors` as one safetensors file at `path`.
pub(crate) fn write(path: &Path, tensors: &[Tensor]) {
    write_meta(path, &[], tensors);
}

/// `tensors` and `meta` strings as one safetensors file at `path`.
pub(crate) fn write_meta(path: &Path, meta: &[(&str, &str)], tensors: &[Tensor]) {
    let (mut header, mut data): (Vec<String>, Vec<u8>) = (Vec::new(), Vec::new());
    if !meta.is_empty() {
        let map: serde_json::Map<_, _> = meta
            .iter()
            .map(|(k, v)| (k.to_string(), (*v).into()))
            .collect();
        header.push(format!(
            r#""__metadata__":{}"#,
            serde_json::Value::Object(map)
        ));
    }
    for (name, dtype, shape, bytes) in tensors {
        let at = data.len();
        data.extend(bytes);
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        let end = data.len();
        header.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{at},{end}]}}"#,
            shape.join(",")
        ));
    }
    let header = format!("{{{}}}", header.join(","));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// A record: format, out, in, K, side, then zeros.
pub(crate) fn record(format: i64, out: usize, inp: usize, side: i64) -> Vec<u8> {
    let fields = [format, out as i64, inp as i64, 0, side, 0, 0, 0];
    fields.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// `n` bytes below 0x7F, the E4M3 NaN.
pub(crate) fn ramp(n: usize, step: usize) -> Vec<u8> {
    (0..n).map(|i| (i * step % 120) as u8).collect()
}

pub(crate) fn halves(values: &[f32]) -> Vec<u8> {
    let half = |v: &f32| F16::from(*v).0.to_le_bytes();
    values.iter().flat_map(half).collect()
}

pub(crate) fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub(crate) fn e4m3(b: u8) -> f32 {
    f32::from(E4M3(b))
}

/// A trellis record at width `k`.
pub(crate) fn trellis_record(k: i64) -> Vec<u8> {
    let mut r = record(1, 128, 128, 1);
    r[24..32].copy_from_slice(&k.to_le_bytes());
    r
}

/// Serves files under `root` until `/stop`.
pub(crate) fn serve(root: PathBuf) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut s = stream;
            let mut buf = [0u8; 2048];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            let mut lines = req.lines();
            let path = lines
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or("/");
            if path == "/stop" {
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                break;
            }
            let range = lines.find_map(|l| {
                l.strip_prefix("Range: bytes=")
                    .or_else(|| l.strip_prefix("range: bytes="))
            });
            let full = root.join(path.trim_start_matches('/'));
            let Ok(bytes) = std::fs::read(&full) else {
                let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                continue;
            };
            if let Some(r) = range {
                let (a, b) = r.split_once('-').unwrap();
                let start: usize = a.parse().unwrap();
                let end = b.parse::<usize>().unwrap().min(bytes.len() - 1);
                let slice = &bytes[start..=end];
                let head = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\r\n",
                    slice.len()
                );
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(slice);
            } else {
                let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", bytes.len());
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(&bytes);
            }
        }
    });
    (format!("http://{addr}"), server)
}
