// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! Thinc's saved models: msgpack nodes, attributes and float arrays.

use crate::tokenize::Error;
use rmpv::Value;
use std::collections::HashMap;

pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// One layer: its name, integer attributes and parameter arrays.
pub struct Node {
    pub name: String,
    pub attrs: HashMap<String, i64>,
    pub params: HashMap<String, Array>,
}

/// The value under `key`, a string or byte string.
fn get<'a>(map: &'a Value, key: &str) -> Option<&'a Value> {
    let pairs = map.as_map()?;
    pairs.iter().find_map(|(k, v)| {
        let name = k.as_str().map(str::as_bytes).or_else(|| k.as_slice())?;
        (name == key.as_bytes()).then_some(v)
    })
}

fn array(v: &Value) -> Result<Array, Error> {
    let kind = get(v, "type").and_then(Value::as_str);
    if kind != Some("<f4") {
        return Err(format!("array of {kind:?}, not little-endian f32").into());
    }
    let dims = get(v, "shape")
        .and_then(Value::as_array)
        .ok_or("array without a shape")?;
    let shape: Option<Vec<usize>> = dims
        .iter()
        .map(|d| d.as_u64().map(|n| n as usize))
        .collect();
    let bytes = get(v, "data")
        .and_then(Value::as_slice)
        .ok_or("array without data")?;
    let data = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok(Array {
        shape: shape.ok_or("bad array shape")?,
        data,
    })
}

/// Each layer of a saved Thinc model, in index order.
pub fn read(bytes: &[u8]) -> Result<Vec<Node>, Error> {
    let root = rmpv::decode::read_value(&mut &bytes[..])?;
    let list = |key| {
        get(&root, key)
            .and_then(Value::as_array)
            .ok_or(format!("model without {key}"))
    };
    let (nodes, attrs, params) = (list("nodes")?, list("attrs")?, list("params")?);
    let mut out = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        let name = get(node, "name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let mut ints = HashMap::new();
        for (k, v) in attrs.get(i).and_then(Value::as_map).into_iter().flatten() {
            // Each attribute is itself msgpack. Keep the integers.
            let Some(raw) = v.as_slice() else { continue };
            if let (Some(k), Ok(inner)) = (k.as_str(), rmpv::decode::read_value(&mut &raw[..]))
                && let Some(n) = inner.as_i64()
            {
                ints.insert(k.to_string(), n);
            }
        }
        let mut arrays = HashMap::new();
        for (k, v) in params.get(i).and_then(Value::as_map).into_iter().flatten() {
            if let Some(k) = k.as_str() {
                arrays.insert(k.to_string(), array(v)?);
            }
        }
        out.push(Node {
            name,
            attrs: ints,
            params: arrays,
        });
    }
    Ok(out)
}

/// The `lexeme_norm` table of a pipeline's `vocab/lookups.bin`.
pub fn lexeme_norms(bytes: &[u8]) -> Result<HashMap<u64, String>, Error> {
    let root = rmpv::decode::read_value(&mut &bytes[..])?;
    let mut out = HashMap::new();
    let Some(table) = get(&root, "lexeme_norm").and_then(Value::as_map) else {
        return Ok(out);
    };
    for (k, v) in table {
        if let (Some(k), Some(v)) = (k.as_u64(), v.as_str()) {
            out.insert(k, v.to_string());
        }
    }
    Ok(out)
}
