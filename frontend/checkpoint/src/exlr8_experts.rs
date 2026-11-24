//! Routed and shared exlr8 expert storage and manifest widths.

use crate::exlr8::coded;
use crate::file::{Error, File};
use crate::offer::{Slice, Stored};
use std::collections::HashMap;
use tessera_engine::storage::Form;
use tessera_engine::trellis::{self, Codebook};

/// Channels one rotation block scale covers.
const BLOCK: usize = 128;

/// One expert MLP projection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Proj {
    Gate = 0,
    Up = 1,
    Down = 2,
}

impl Proj {
    pub(crate) const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    pub(crate) fn named(name: &str) -> Option<Self> {
        match name {
            "gate" => Some(Self::Gate),
            "up" => Some(Self::Up),
            "down" => Some(Self::Down),
            _ => None,
        }
    }
}

/// Manifest widths for routed and shared experts, by layer.
#[derive(Default)]
pub(crate) struct Widths {
    map: HashMap<(usize, bool, Proj), u32>,
}

/// Trellis forms for one layer's routed and shared projections.
pub(crate) struct Forms {
    pub(crate) routed: [&'static dyn Form; 3],
    pub(crate) shared: Option<[&'static dyn Form; 3]>,
}

impl Widths {
    pub(crate) fn clear(&mut self) {
        self.map.clear();
    }

    /// Width for `(layer, shared, proj)`, falling back to `default`.
    pub(crate) fn get(&self, layer: usize, shared: bool, proj: Proj, default: u32) -> u32 {
        if let Some(&k) = self.map.get(&(layer, shared, proj)) {
            return k;
        }
        if shared {
            let routed = self
                .map
                .get(&(layer, false, proj))
                .copied()
                .unwrap_or(default);
            return shared_width(routed);
        }
        default
    }

    /// Parses manifest `"layers"` entries into `self`.
    pub(crate) fn parse(&mut self, value: &serde_json::Value) -> Result<(), String> {
        if value.is_null() {
            return Ok(());
        }
        let layers = value.as_object().ok_or("layers must be an object")?;
        for (key, spec) in layers {
            let layer: usize = key.parse().map_err(|_| format!("layer {key}"))?;
            let spec = spec
                .as_object()
                .ok_or_else(|| format!("layer {layer} must be an object"))?;
            self.read_group(layer, false, spec)?;
            if let Some(shared) = spec.get("shared")
                && !shared.is_null()
            {
                let shared = shared
                    .as_object()
                    .ok_or_else(|| format!("layer {layer}.shared must be an object"))?;
                self.read_group(layer, true, shared)?;
            }
        }
        Ok(())
    }

    fn read_group(
        &mut self,
        layer: usize,
        shared: bool,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        for p in Proj::ALL {
            let Some(val) = obj.get(p.name()) else {
                continue;
            };
            let k = uniform_width(val).map_err(|why| {
                let kind = if shared { "shared." } else { "" };
                format!("layer {layer} {kind}{}: {why}", p.name())
            })?;
            self.map.insert((layer, shared, p), k);
        }
        Ok(())
    }
}

/// One uniform width from a scalar or nested array.
fn uniform_width(val: &serde_json::Value) -> Result<u32, String> {
    let mut seen = None;
    collect_width(val, &mut seen)?;
    seen.ok_or_else(|| "empty width list".into())
}

fn collect_width(val: &serde_json::Value, seen: &mut Option<u32>) -> Result<(), String> {
    if let Some(n) = val.as_u64() {
        let k = u32::try_from(n)
            .ok()
            .filter(|&k| k > 0)
            .ok_or_else(|| format!("invalid width {n}"))?;
        if let Some(prev) = *seen
            && prev != k
        {
            return Err(format!("mixed widths {prev} and {k} in one tensor"));
        }
        *seen = Some(k);
        return Ok(());
    }
    let items = val.as_array().ok_or("expected integer or array width")?;
    if items.is_empty() {
        return Err("empty width list".into());
    }
    for item in items {
        collect_width(item, seen)?;
    }
    Ok(())
}

/// A layer's index file: side data for every width.
pub(crate) fn index_file(layer: usize) -> String {
    format!("experts/exlr8-layer-{layer:03}.index")
}

/// A layer's trellis codes at width `k`.
pub(crate) fn codes_file(layer: usize, k: u32) -> String {
    format!("experts/exlr8-layer-{layer:03}.k{k}")
}

/// The shared expert has no K2 codes.
pub(crate) fn shared_width(k: u32) -> u32 {
    k.max(3)
}

/// A layer and its expert name, past `prefix`.
pub(crate) fn split<'a>(name: &'a str, prefix: &str) -> Option<(usize, &'a str)> {
    let rest = name.strip_prefix(prefix)?;
    let (layer, local) = rest.split_once('.')?;
    let local = local.strip_prefix("mlp.")?;
    let ours = local.starts_with("experts.") || local.starts_with("shared_experts.");
    ours.then_some((layer.parse().ok()?, local))
}

/// Whether `local` names a shared expert and which projection.
pub(crate) fn target(local: &str) -> Option<(bool, Proj)> {
    let (shared, rest) = match local.strip_prefix("shared_experts.") {
        Some(r) => (true, r),
        None => (false, local.strip_prefix("experts.")?),
    };
    let head = rest.split('.').next()?;
    let head = head.strip_suffix("_proj").unwrap_or(head);
    Some((shared, Proj::named(head)?))
}

/// Hidden width, expert count and expert width, from the index.
pub(crate) fn sizes(idx: &File, gate_k: u32) -> Option<[usize; 3]> {
    let hidden = idx.shape("signs.in")?[0];
    let &[experts, ffn] = idx.shape(&format!("k{gate_k}.gate.svh"))? else {
        return None;
    };
    Some([hidden, experts, ffn])
}

/// A projection's codebook, from its codes file tensor names.
pub(crate) fn codebook(codes: &File, shared: bool, proj: Proj) -> Result<Codebook, Error> {
    let head = if shared { "shared." } else { "" };
    Ok(coded(codes, &format!("{head}trellis.{}", proj.name()))?.1)
}

/// The layer's slices: codes as trellis sets, rotations as floats.
pub(crate) fn offered(
    idx: &File,
    prefix: &str,
    layer: usize,
    gate_k: u32,
    forms: Forms,
) -> Vec<Slice> {
    let Some([hidden, experts, ffn]) = sizes(idx, gate_k) else {
        return Vec::new();
    };
    let at = format!("{prefix}{layer}.mlp.");
    let slice = |name: String, shape: Vec<usize>, stored| Slice {
        name: format!("{at}{name}"),
        shape,
        stored,
    };
    let mut out = Vec::new();
    for (p, shape) in [
        (Proj::Gate, [ffn, hidden]),
        (Proj::Up, [ffn, hidden]),
        (Proj::Down, [hidden, ffn]),
    ] {
        let name = p.name();
        let [rows, cols] = shape;
        let set = vec![experts, rows, cols];
        out.push(slice(
            format!("experts.{name}.weight"),
            set,
            Stored::Form(forms.routed[p as usize]),
        ));
        let (into, from) = (format!("experts.{name}.su"), format!("experts.{name}.sv"));
        out.push(slice(into, vec![experts, cols], Stored::F32));
        out.push(slice(from, vec![experts, rows], Stored::F32));
        if let Some(shared) = forms.shared {
            let stem = format!("shared_experts.{name}_proj");
            out.push(slice(
                format!("{stem}.weight"),
                vec![rows, cols],
                Stored::Form(shared[p as usize]),
            ));
        }
    }
    for side in ["signs_in", "signs_out"] {
        out.push(slice(format!("experts.{side}"), vec![hidden], Stored::F32));
    }
    out
}

/// Each value repeated over its block of channels.
fn repeated(values: &[f32]) -> Vec<f32> {
    values.iter().flat_map(|v| [*v; BLOCK]).collect()
}

/// The layer's signs, as floats.
fn signs(idx: &File, which: &str) -> Result<Vec<f32>, Error> {
    let bytes = idx.bytes(&format!("signs.{which}"))?;
    Ok(bytes.iter().map(|b| f32::from(*b as i8)).collect())
}

/// A rotation table or vector, from the index file.
pub(crate) fn floats(idx: &File, local: &str, k: u32) -> Result<Vec<f32>, Error> {
    let half = |name: String| idx.floats(&name);
    let signed = |which: &str, v: Vec<f32>| -> Result<Vec<f32>, Error> {
        let s = signs(idx, which)?;
        Ok(v.iter().zip(s).map(|(a, b)| a * b).collect())
    };
    match local {
        "experts.signs_in" => signs(idx, "in"),
        "experts.signs_out" => signs(idx, "out"),
        "experts.gate.su" | "experts.up.su" => {
            let p = &local["experts.".len()..local.len() - 3];
            Ok(repeated(&half(format!("k{k}.{p}.su_block"))?))
        }
        "experts.gate.sv" | "experts.up.sv" => {
            let p = &local["experts.".len()..local.len() - 3];
            half(format!("k{k}.{p}.svh"))
        }
        "experts.down.su" => half(format!("k{k}.down.suh")),
        "experts.down.sv" => Ok(repeated(&half(format!("k{k}.down.sv_block"))?)),
        "shared_experts.gate_proj.suh" | "shared_experts.up_proj.suh" => {
            let p = &local["shared_experts.".len()..local.len() - "_proj.suh".len()];
            let block = repeated(&half(format!("shared.k{k}.{p}.su_block"))?);
            signed("in", block)
        }
        "shared_experts.gate_proj.svh" | "shared_experts.up_proj.svh" => {
            let p = &local["shared_experts.".len()..local.len() - "_proj.svh".len()];
            half(format!("shared.k{k}.{p}.svh"))
        }
        "shared_experts.down_proj.suh" => half(format!("shared.k{k}.down.suh")),
        "shared_experts.down_proj.svh" => {
            let block = repeated(&half(format!("shared.k{k}.down.sv_block"))?);
            signed("out", block)
        }
        _ => Err(Error::Missing(local.into())),
    }
}

/// A set or shared matrix as `k`-bit trellis codes.
pub(crate) fn packed(
    codes: &File,
    local: &str,
    [hidden, experts, ffn]: [usize; 3],
    k: u32,
    slice: Option<(u32, usize, usize)>,
) -> Result<Vec<u8>, Error> {
    let missing = || Error::Missing(local.into());
    let (tensor, p) = match local.strip_prefix("shared_experts.") {
        Some(rest) => (true, rest.strip_suffix("_proj.weight").ok_or_else(missing)?),
        None => {
            let rest = local.strip_prefix("experts.").ok_or_else(missing)?;
            (false, rest.strip_suffix(".weight").ok_or_else(missing)?)
        }
    };
    let head = if tensor { "shared." } else { "" };
    let name = coded(codes, &format!("{head}trellis.{p}"))?.0;
    let matrix = (hidden * ffn * k as usize / 8) as u64;
    let inner = if p == "down" { 2 } else { 1 };
    let (entries, ffn_out) = match slice {
        Some((0, f, c)) => {
            let (at, n) = (f as u64 * matrix, c as u64 * matrix);
            (codes.bytes_at(&name, at, Some(n))?, ffn)
        }
        Some((ax, f, c)) if ax == inner && f.is_multiple_of(BLOCK) && c.is_multiple_of(BLOCK) => {
            let entry = (hidden * BLOCK * k as usize / 8) as u64;
            let (off, take) = ((f / BLOCK) as u64 * entry, (c / BLOCK) as u64 * entry);
            let mut buf = Vec::with_capacity(experts * take as usize);
            for m in 0..experts {
                let at = m as u64 * matrix + off;
                buf.extend_from_slice(&codes.bytes_at(&name, at, Some(take))?);
            }
            (buf, c)
        }
        _ => (codes.bytes(&name)?, ffn),
    };
    let per = hidden * ffn_out * k as usize / 8;
    if !tensor && per >= 16 && entries.chunks_exact(per).any(|c| c[..16] == [0; 16]) {
        return Err(Error::Format(format!("{name}: zero-filled expert slice")));
    }
    Ok(match p {
        "down" => trellis::packed(ffn_out, hidden, trellis::INPUT_SIDE, &entries),
        _ => trellis::packed(hidden, ffn_out, 1, &entries),
    })
}
