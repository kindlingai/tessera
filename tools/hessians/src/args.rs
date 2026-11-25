//! CLI argument parsing, layer-set ranges, and `.npy` token loader.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Parsed calibration CLI options matching `calib_forward.py`.
#[derive(Clone, Debug)]
pub struct Config {
    pub model: PathBuf,
    pub tokens: PathBuf,
    pub heldout: usize,
    pub layers: Vec<usize>,
    pub hess_layers: BTreeSet<usize>,
    pub save_inputs: BTreeSet<usize>,
    pub route_log: BTreeSet<usize>,
    pub hess_id: String,
    pub weights_id: Option<String>,
    pub batch: usize,
    pub experts_per_file: usize,
    pub dense_heldout: bool,
    pub no_dense: bool,
    pub no_experts: bool,
    pub work: PathBuf,
    pub out: PathBuf,
    pub resume: bool,
    pub state: Option<PathBuf>,
    pub keep_hidden: bool,
    pub gpu: bool,
    pub device: Option<usize>,
    pub tile: usize,
}

/// Top-level CLI mode: calibration, kernel bench, or parity check.
#[derive(Clone, Debug)]
pub enum Command {
    Calib(Box<Config>),
    BenchKernels {
        device: Option<usize>,
    },
    VerifyParity {
        got: PathBuf,
        want: PathBuf,
        layer: usize,
        max_rel: f64,
    },
}

/// Parses `'0-40,45,70-'` into a sorted deduplicated list.
pub(crate) fn parse_set(spec: &str, last: usize) -> Result<Vec<usize>, String> {
    let mut out = BTreeSet::new();
    for part in spec.split(',').filter(|s| !s.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.parse().map_err(|_| format!("bad range `{part}`"))?;
            let end: usize = if b.is_empty() {
                last
            } else {
                b.parse().map_err(|_| format!("bad range `{part}`"))?
            };
            if start > end {
                return Err(format!("inverted range `{part}`"));
            }
            for x in start..=end {
                out.insert(x);
            }
        } else {
            let v: usize = part.parse().map_err(|_| format!("bad index `{part}`"))?;
            out.insert(v);
        }
    }
    Ok(out.into_iter().collect())
}

/// Loads a 2D integer `.npy` matrix as `u32`.
pub(crate) fn read_npy(path: &Path) -> Result<(usize, usize, Vec<u32>), String> {
    let raw = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not a numpy .npy file", path.display()));
    }
    let major = raw[6];
    let (hdr_len, data_at) = if major == 1 {
        let n = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        (n, 10 + n)
    } else if major == 2 || major == 3 {
        if raw.len() < 12 {
            return Err("truncated npy header".into());
        }
        let n = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
        (n, 12 + n)
    } else {
        return Err(format!("unsupported npy version {major}"));
    };
    if data_at > raw.len() {
        return Err("truncated npy payload".into());
    }
    let hdr = std::str::from_utf8(&raw[data_at - hdr_len..data_at])
        .map_err(|e| format!("bad npy header utf8: {e}"))?;
    let descr = extract_quoted(hdr, "'descr'").ok_or("missing npy descr")?;
    let shape = extract_shape(hdr).ok_or("missing npy 2D shape")?;
    let (n, s) = (shape.0, shape.1);
    let data = &raw[data_at..];
    let total = n * s;
    let out = match descr.as_str() {
        "<i4" | "|i4" => decode_ints(data, total, 4, |b| {
            i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32
        })?,
        "<u4" | "|u4" => decode_ints(data, total, 4, |b| {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })?,
        "<i8" | "|i8" => decode_ints(data, total, 8, |b| {
            i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as u32
        })?,
        "<u8" | "|u8" => decode_ints(data, total, 8, |b| {
            u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as u32
        })?,
        other => return Err(format!("unsupported npy dtype `{other}`")),
    };
    Ok((n, s, out))
}

fn decode_ints(
    data: &[u8],
    count: usize,
    width: usize,
    parse: fn(&[u8]) -> u32,
) -> Result<Vec<u32>, String> {
    if data.len() < count * width {
        return Err("truncated npy token array".into());
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(parse(&data[i * width..(i + 1) * width]));
    }
    Ok(out)
}

fn extract_quoted(hdr: &str, key: &str) -> Option<String> {
    let at = hdr.find(key)? + key.len();
    let rest = &hdr[at..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let q = rest.chars().next()?;
    if q != '\'' && q != '"' {
        return None;
    }
    let inner = &rest[1..];
    let end = inner.find(q)?;
    Some(inner[..end].to_string())
}

fn extract_shape(hdr: &str) -> Option<(usize, usize)> {
    let at = hdr.find("'shape'")? + 7;
    let rest = &hdr[at..];
    let l = rest.find('(')?;
    let r = rest[l + 1..].find(')')?;
    let dims: Vec<usize> = rest[l + 1..l + 1 + r]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().ok())
        .collect::<Option<Vec<_>>>()?;
    if dims.len() != 2 {
        return None;
    }
    Some((dims[0], dims[1]))
}

/// Writes a 2D `<i4` `.npy` file for tests and tooling.
pub fn write_npy(path: &Path, rows: usize, cols: usize, tokens: &[u32]) -> Result<(), String> {
    assert_eq!(tokens.len(), rows * cols);
    let mut hdr =
        format!("{{'descr': '<i4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let total = 10 + hdr.len() + 1;
    let pad = (64 - total % 64) % 64;
    for _ in 0..pad {
        hdr.push(' ');
    }
    hdr.push('\n');
    let mut buf = Vec::with_capacity(10 + hdr.len() + tokens.len() * 4);
    buf.extend_from_slice(b"\x93NUMPY\x01\x00");
    buf.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
    buf.extend_from_slice(hdr.as_bytes());
    for &t in tokens {
        buf.extend_from_slice(&(t as i32).to_le_bytes());
    }
    fs::write(path, buf).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parses CLI arguments into a `Config`.
pub fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut model = None;
    let mut tokens = None;
    let mut work = None;
    let mut out = None;
    let mut hess_id = String::new();
    let mut weights_id = None;
    let mut heldout = 0usize;
    let mut layers_spec = "0-77".to_string();
    let mut hess_spec: Option<String> = None;
    let mut save_spec = String::new();
    let mut route_spec = String::new();
    let mut batch = 2usize;
    let mut experts_per_file = 32usize;
    let mut dense_heldout = false;
    let mut no_dense = false;
    let mut no_experts = false;
    let mut resume = false;
    let mut state = None;
    let mut keep_hidden = false;
    let mut gpu = false;
    let mut device = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let next = |idx: &mut usize| -> Result<String, String> {
            *idx += 1;
            args.get(*idx)
                .cloned()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag {
            "--model" => model = Some(PathBuf::from(next(&mut i)?)),
            "--tokens" => tokens = Some(PathBuf::from(next(&mut i)?)),
            "--work" => work = Some(PathBuf::from(next(&mut i)?)),
            "--out" => out = Some(PathBuf::from(next(&mut i)?)),
            "--hess-id" | "--hessian-id" => hess_id = next(&mut i)?,
            "--weights-id" => weights_id = Some(next(&mut i)?),
            "--heldout" => heldout = next(&mut i)?.parse().map_err(|_| "bad --heldout")?,
            "--layers" => layers_spec = next(&mut i)?,
            "--hess-layers" => hess_spec = Some(next(&mut i)?),
            "--save-inputs" => save_spec = next(&mut i)?,
            "--route-log" => route_spec = next(&mut i)?,
            "--batch" => batch = next(&mut i)?.parse().map_err(|_| "bad --batch")?,
            "--experts-per-file" => {
                experts_per_file = next(&mut i)?
                    .parse()
                    .map_err(|_| "bad --experts-per-file")?;
            }
            "--state" => state = Some(PathBuf::from(next(&mut i)?)),
            "--dense-heldout" => dense_heldout = true,
            "--no-dense" => no_dense = true,
            "--no-experts" => no_experts = true,
            "--resume" => resume = true,
            "--keep-hidden" => keep_hidden = true,
            "--gpu" => gpu = true,
            "--device" => device = Some(next(&mut i)?.parse().map_err(|_| "bad --device")?),
            other => return Err(format!("unknown flag `{other}`")),
        }
        i += 1;
    }
    let layers = parse_set(&layers_spec, 78)?;
    let last = layers.last().copied().unwrap_or(0);
    let hess_layers: BTreeSet<usize> = match hess_spec {
        Some(s) => parse_set(&s, last)?.into_iter().collect(),
        None => layers.iter().copied().collect(),
    };
    let save_inputs = parse_set(&save_spec, last)?.into_iter().collect();
    let route_log = parse_set(&route_spec, last)?.into_iter().collect();
    Ok(Config {
        model: model.ok_or("missing --model")?,
        tokens: tokens.ok_or("missing --tokens")?,
        heldout,
        layers,
        hess_layers,
        save_inputs,
        route_log,
        hess_id,
        weights_id,
        batch: batch.max(1),
        experts_per_file: experts_per_file.max(1),
        dense_heldout,
        no_dense,
        no_experts,
        work: work.ok_or("missing --work")?,
        out: out.ok_or("missing --out")?,
        resume,
        state,
        keep_hidden,
        gpu,
        device,
        tile: 128,
    })
}

/// Parses CLI flags into calibration, microbench, or parity mode.
pub fn parse_command(args: &[String]) -> Result<Command, String> {
    if args.iter().any(|a| a == "--bench-kernels") {
        let mut device = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--bench-kernels" => {}
                "--device" => {
                    i += 1;
                    let v = args.get(i).ok_or("missing value for --device")?;
                    device = Some(v.parse().map_err(|_| "bad --device")?);
                }
                other => return Err(format!("unknown flag `{other}`")),
            }
            i += 1;
        }
        return Ok(Command::BenchKernels { device });
    }
    if args.iter().any(|a| a == "--verify-parity") {
        let mut got = None;
        let mut want = None;
        let mut layer = None;
        let mut max_rel = 5e-3f64;
        let mut i = 0;
        while i < args.len() {
            let f = args[i].as_str();
            let next = |idx: &mut usize| -> Result<String, String> {
                *idx += 1;
                args.get(*idx)
                    .cloned()
                    .ok_or_else(|| format!("missing value for {f}"))
            };
            match f {
                "--verify-parity" => {
                    got = Some(PathBuf::from(next(&mut i)?));
                    want = Some(PathBuf::from(next(&mut i)?));
                }
                "--layer" => layer = Some(next(&mut i)?.parse().map_err(|_| "bad --layer")?),
                "--max-rel" | "--max-rel-frob" => {
                    max_rel = next(&mut i)?.parse().map_err(|_| "bad --max-rel")?;
                }
                other => return Err(format!("unknown flag `{other}`")),
            }
            i += 1;
        }
        return Ok(Command::VerifyParity {
            got: got.ok_or("missing --verify-parity <got> <want>")?,
            want: want.ok_or("missing --verify-parity <got> <want>")?,
            layer: layer.ok_or("missing --layer")?,
            max_rel,
        });
    }
    Ok(Command::Calib(Box::new(parse_args(args)?)))
}
