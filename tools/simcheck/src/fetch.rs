//! Downloads EmbeddingGemma 2's text tower as a loadable snapshot.

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

const REPO: &str = "google/embeddinggemma-2";
/// The checkpoint revision. Cached embeddings depend on it.
pub const REVISION: &str = "914f7f89142e33e77833254d9c9b90c3cef7303b";
const KEEP: &str = "language_model.";
const SMALL: [&str; 3] = ["config.json", "tokenizer.json", "tokenizer_config.json"];
/// Tensors closer than this share one request.
const MERGE_GAP: u64 = 4 << 20;

fn url(file: &str) -> String {
    format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{file}")
}

/// Bytes `start..end` of a remote file, as a stream.
fn range(url: &str, start: u64, end: u64) -> Result<impl Read> {
    let resp = ureq::get(url)
        .set("Range", &format!("bytes={start}-{}", end - 1))
        .call()
        .with_context(|| format!("GET {url} bytes {start}..{end}"))?;
    ensure!(
        resp.status() == 206,
        "{url}: status {}, want 206",
        resp.status()
    );
    Ok(resp.into_reader().take(end - start))
}

struct Tensor {
    name: String,
    meta: Value,
    start: u64,
    end: u64,
}

/// The remote header's `language_model.*` tensors, by file offset.
fn text_tensors(model: &str) -> Result<(Map<String, Value>, Vec<Tensor>)> {
    let mut len = [0u8; 8];
    range(model, 0, 8)?.read_exact(&mut len)?;
    let header_len = u64::from_le_bytes(len);
    let mut text = Vec::new();
    range(model, 8, 8 + header_len)?.read_to_end(&mut text)?;
    let header: Map<String, Value> = serde_json::from_slice(&text)?;
    let base = 8 + header_len;
    let mut kept = Vec::new();
    for (name, v) in header.iter().filter(|(n, _)| n.starts_with(KEEP)) {
        let off = &v["data_offsets"];
        let (Some(a), Some(b)) = (off[0].as_u64(), off[1].as_u64()) else {
            bail!("{name}: bad data_offsets");
        };
        let mut meta = v.clone();
        meta.as_object_mut()
            .context("tensor entry")?
            .remove("data_offsets");
        kept.push(Tensor {
            name: name.clone(),
            meta,
            start: base + a,
            end: base + b,
        });
    }
    ensure!(!kept.is_empty(), "no {KEEP}* tensors in the header");
    kept.sort_by_key(|t| t.start);
    Ok((header, kept))
}

/// The kept tensors, packed tight in their old order.
fn packed_header(header: &Map<String, Value>, kept: &[Tensor]) -> Result<Vec<u8>> {
    let mut out = Map::new();
    if let Some(m) = header.get("__metadata__") {
        out.insert("__metadata__".into(), m.clone());
    }
    let mut at = 0u64;
    for t in kept {
        let mut meta = t.meta.clone();
        meta["data_offsets"] = json!([at, at + t.end - t.start]);
        out.insert(t.name.clone(), meta);
        at += t.end - t.start;
    }
    let mut bytes = serde_json::to_vec(&out)?;
    bytes.resize(bytes.len().next_multiple_of(8), b' ');
    Ok(bytes)
}

/// Writes config, tokenizer and text-only weights into `out`.
pub fn fetch(out: &Path) -> Result<()> {
    std::fs::create_dir_all(out)?;
    for f in SMALL {
        let mut body = ureq::get(&url(f)).call()?.into_reader();
        std::io::copy(&mut body, &mut File::create(out.join(f))?)?;
    }
    let model = url("model.safetensors");
    let (header, kept) = text_tensors(&model)?;
    let hdr = packed_header(&header, &kept)?;
    let file = File::create(out.join("model.safetensors"))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    w.write_all(&(hdr.len() as u64).to_le_bytes())?;
    w.write_all(&hdr)?;
    let mut i = 0;
    while i < kept.len() {
        let mut j = i + 1;
        while j < kept.len() && kept[j].start <= kept[j - 1].end + MERGE_GAP {
            j += 1;
        }
        let mut body = range(&model, kept[i].start, kept[j - 1].end)?;
        let mut pos = kept[i].start;
        for t in &kept[i..j] {
            std::io::copy(&mut (&mut body).take(t.start - pos), &mut std::io::sink())?;
            let n = std::io::copy(&mut (&mut body).take(t.end - t.start), &mut w)?;
            ensure!(n == t.end - t.start, "{}: short read", t.name);
            pos = t.end;
        }
        i = j;
    }
    w.flush()?;
    Ok(())
}
