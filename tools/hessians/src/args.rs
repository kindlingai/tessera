//! Command line flags, layer sets and the `.npy` token reader.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tessera_checkpoint::hessian::sha256_hex;
use tessera_device::calib::TILE;

/// Layer sets as given, before the layer count is known.
#[derive(Clone, Debug, Default)]
pub struct LayerSpecs {
    pub layers: Option<String>,
    pub hess: Option<String>,
    pub save: String,
    pub route: String,
}

/// Calibration options.
#[derive(Clone, Debug)]
pub struct Config {
    pub model: PathBuf,
    pub tokens: PathBuf,
    pub heldout: usize,
    pub layers: Vec<usize>,
    pub hess_layers: BTreeSet<usize>,
    pub save_inputs: BTreeSet<usize>,
    pub route_log: BTreeSet<usize>,
    pub specs: LayerSpecs,
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
    pub tile: usize,
}

impl Config {
    /// Fills the layer sets for a model of `count` layers.
    pub fn resolve(&mut self, count: usize) -> Result<(), String> {
        let last = count.checked_sub(1).ok_or("the model has no layers")?;
        let set = |spec: &str| parse_set(spec, last);
        self.layers = match &self.specs.layers {
            Some(spec) => set(spec)?,
            None => (0..count).collect(),
        };
        self.hess_layers = match &self.specs.hess {
            Some(spec) => set(spec)?.into_iter().collect(),
            None => self.layers.iter().copied().collect(),
        };
        self.save_inputs = set(&self.specs.save)?.into_iter().collect();
        self.route_log = set(&self.specs.route)?.into_iter().collect();
        Ok(())
    }
}

/// A tokens file: `rows` by `seq_len` ids, and its hash.
pub struct Tokens {
    pub rows: usize,
    pub seq_len: usize,
    pub ids: Vec<u32>,
    pub hash: String,
}

/// Parses `'0-40,45,70-'` into sorted indices up to `last`.
pub(crate) fn parse_set(spec: &str, last: usize) -> Result<Vec<usize>, String> {
    let mut out = BTreeSet::new();
    for part in spec.split(',').filter(|s| !s.is_empty()) {
        let bad = || format!("bad layer set `{part}`");
        let (start, end) = match part.split_once('-') {
            Some((a, "")) => (a.parse().map_err(|_| bad())?, last),
            Some((a, b)) => (a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?),
            None => {
                let v = part.parse().map_err(|_| bad())?;
                (v, v)
            }
        };
        if start > end {
            return Err(format!("inverted range `{part}`"));
        }
        if end > last {
            return Err(format!("`{part}` is past the last layer {last}"));
        }
        out.extend(start..=end);
    }
    Ok(out.into_iter().collect())
}

/// Reads a 2D C-order integer `.npy` matrix of token ids.
pub(crate) fn read_npy(path: &Path) -> Result<Tokens, String> {
    let raw = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let fail = |what: &str| format!("{}: {what}", path.display());
    if raw.len() < 12 || &raw[..6] != b"\x93NUMPY" {
        return Err(fail("not a numpy .npy file"));
    }
    let (hdr_len, data_at) = match raw[6] {
        1 => {
            let n = u16::from_le_bytes([raw[8], raw[9]]) as usize;
            (n, 10 + n)
        }
        2 | 3 => {
            let n = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
            (n, 12 + n)
        }
        v => return Err(fail(&format!("unsupported npy version {v}"))),
    };
    if data_at > raw.len() {
        return Err(fail("truncated npy header"));
    }
    let hdr = std::str::from_utf8(&raw[data_at - hdr_len..data_at])
        .map_err(|_| fail("npy header is not UTF-8"))?;
    if hdr.contains("'fortran_order': True") {
        return Err(fail("Fortran-order arrays are not supported"));
    }
    let descr = extract_quoted(hdr, "'descr'").ok_or_else(|| fail("missing npy descr"))?;
    let (rows, seq_len) = extract_shape(hdr).ok_or_else(|| fail("missing npy 2D shape"))?;
    if rows == 0 || seq_len == 0 {
        return Err(fail("no tokens"));
    }
    let (width, signed) = match descr.trim_start_matches(['<', '|']) {
        "i4" => (4, true),
        "u4" => (4, false),
        "i8" => (8, true),
        "u8" => (8, false),
        other => return Err(fail(&format!("unsupported npy dtype `{other}`"))),
    };
    let data = &raw[data_at..];
    let total = rows * seq_len;
    if data.len() < total * width {
        return Err(fail("truncated npy token array"));
    }
    let mut ids = Vec::with_capacity(total);
    for c in data.chunks_exact(width).take(total) {
        let mut b = [0u8; 8];
        b[..width].copy_from_slice(c);
        let negative = signed && c[width - 1] & 0x80 != 0;
        let v = u64::from_le_bytes(b);
        let id = u32::try_from(v).ok().filter(|_| !negative);
        ids.push(id.ok_or_else(|| fail("token id outside u32"))?);
    }
    Ok(Tokens {
        rows,
        seq_len,
        ids,
        hash: sha256_hex(&[&raw]),
    })
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
    match dims[..] {
        [a, b] => Some((a, b)),
        _ => None,
    }
}

/// Writes a 2D `<i4` `.npy` file.
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

/// Calls `each` with every flag and a value reader.
fn walk(
    args: &[String],
    mut each: impl FnMut(&str, &mut dyn FnMut() -> Result<String, String>) -> Result<(), String>,
) -> Result<(), String> {
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let mut value = || {
            i += 1;
            (args.get(i).cloned()).ok_or_else(|| format!("missing value for {flag}"))
        };
        each(flag, &mut value)?;
        i += 1;
    }
    Ok(())
}

fn num<T: FromStr>(value: String, flag: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("bad {flag} `{value}`"))
}

fn unknown(flag: &str) -> Result<(), String> {
    Err(format!("unknown flag `{flag}`"))
}

/// Parses calibration flags into a `Config` with unresolved layer sets.
pub fn parse_args(args: &[String]) -> Result<Config, String> {
    let (mut model, mut tokens, mut work, mut out) = (None, None, None, None);
    let mut cfg = Config {
        model: PathBuf::new(),
        tokens: PathBuf::new(),
        heldout: 0,
        layers: Vec::new(),
        hess_layers: BTreeSet::new(),
        save_inputs: BTreeSet::new(),
        route_log: BTreeSet::new(),
        specs: LayerSpecs::default(),
        hess_id: String::new(),
        weights_id: None,
        batch: 2,
        experts_per_file: 32,
        dense_heldout: false,
        no_dense: false,
        no_experts: false,
        work: PathBuf::new(),
        out: PathBuf::new(),
        resume: false,
        state: None,
        keep_hidden: false,
        tile: TILE,
    };
    walk(args, |flag, value| {
        match flag {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--tokens" => tokens = Some(PathBuf::from(value()?)),
            "--work" => work = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--hess-id" | "--hessian-id" => cfg.hess_id = value()?,
            "--weights-id" => cfg.weights_id = Some(value()?),
            "--heldout" => cfg.heldout = num(value()?, flag)?,
            "--layers" => cfg.specs.layers = Some(value()?),
            "--hess-layers" => cfg.specs.hess = Some(value()?),
            "--save-inputs" => cfg.specs.save = value()?,
            "--route-log" => cfg.specs.route = value()?,
            "--batch" => cfg.batch = num::<usize>(value()?, flag)?.max(1),
            "--experts-per-file" => cfg.experts_per_file = num::<usize>(value()?, flag)?.max(1),
            "--state" => cfg.state = Some(PathBuf::from(value()?)),
            "--dense-heldout" => cfg.dense_heldout = true,
            "--no-dense" => cfg.no_dense = true,
            "--no-experts" => cfg.no_experts = true,
            "--resume" => cfg.resume = true,
            "--keep-hidden" => cfg.keep_hidden = true,
            other => return unknown(other),
        }
        Ok(())
    })?;
    cfg.model = model.ok_or("missing --model")?;
    cfg.tokens = tokens.ok_or("missing --tokens")?;
    cfg.work = work.ok_or("missing --work")?;
    cfg.out = out.ok_or("missing --out")?;
    Ok(cfg)
}
