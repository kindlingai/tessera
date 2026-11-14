use crate::gguf::pins;
use crate::names_test::gguf_name as tensor_name;
use engine::{Engine, Row};
use graph::Graph;
use interp::Tensor;
use lower::form::{F32_FORM, Form, named};
use lower::{Pick, Recipe, Settings, Weight};
use models::qwen3::{Config, stepper};
use serde_json::{Value, json};
use std::path::PathBuf;

struct Fixture {
    meta: Vec<(String, Value)>,
    tensors: Vec<(String, u32, usize, Vec<u8>)>,
}

fn text(bytes: &mut Vec<u8>, s: &str) {
    bytes.extend((s.len() as u64).to_le_bytes());
    bytes.extend(s.bytes());
}

fn value(bytes: &mut Vec<u8>, v: &Value) {
    match v {
        Value::String(s) => text(bytes, s),
        Value::Number(n) if n.is_u64() => bytes.extend(n.as_u64().unwrap().to_le_bytes()),
        Value::Number(n) => bytes.extend((n.as_f64().unwrap() as f32).to_le_bytes()),
        Value::Array(a) => {
            bytes.extend(8u32.to_le_bytes());
            bytes.extend((a.len() as u64).to_le_bytes());
            for v in a {
                value(bytes, v);
            }
        }
        _ => panic!("fixture metadata type"),
    }
}

impl Fixture {
    fn new(form: &'static dyn Form) -> Self {
        let mut c = Config::toy(2, 8);
        c.stack.hidden = 32;
        let g = stepper(&c);
        let mut meta = vec![("general.architecture".into(), json!("qwen3"))];
        for (key, n) in [
            ("embedding_length", c.stack.hidden),
            ("block_count", c.stack.layers),
            ("attention.head_count", c.stack.heads),
            ("attention.head_count_kv", c.kv_heads),
            ("attention.key_length", c.stack.head_dim),
            ("feed_forward_length", c.stack.ffn),
        ] {
            meta.push((format!("qwen3.{key}"), json!(n)));
        }
        meta.extend([
            ("qwen3.rope.freq_base".into(), json!(c.rope_theta)),
            (
                "qwen3.attention.layer_norm_rms_epsilon".into(),
                json!(c.eps),
            ),
            ("tokenizer.ggml.tokens".into(), json!(["a", "b", "ab"])),
            ("tokenizer.ggml.merges".into(), json!(["a b"])),
            ("tokenizer.ggml.eos_token_id".into(), json!(2)),
            ("tokenizer.ggml.eot_token_id".into(), json!(2)),
            (
                "tokenizer.chat_template".into(),
                json!("{{ messages[0].content }}"),
            ),
        ]);
        let tensors = g
            .weights
            .iter()
            .enumerate()
            .map(|(index, w)| {
                let n =
                    w.ty.shape
                        .0
                        .iter()
                        .map(|e| e.poly(&g.dims).unwrap().as_const().unwrap() as usize)
                        .product();
                let name = tensor_name(&w.path).unwrap();
                let norm = name.contains("norm");
                let values: Vec<f32> = (0..n)
                    .map(|i| {
                        if norm {
                            1.
                        } else {
                            (((i * 17 + index * 23) % 127) as f32 - 63.) / 256.
                        }
                    })
                    .collect();
                let (ty, f) = if norm {
                    (0, F32_FORM)
                } else {
                    (if form.name() == "q8_0" { 8 } else { 2 }, form)
                };
                (name, ty, n, f.pack(&values))
            })
            .collect();
        Self { meta, tensors }
    }

    fn write(&self) -> PathBuf {
        let mut bytes = Vec::from(*b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend((self.tensors.len() as u64).to_le_bytes());
        bytes.extend((self.meta.len() as u64).to_le_bytes());
        for (key, v) in &self.meta {
            text(&mut bytes, key);
            let ty: u32 = match v {
                Value::String(_) => 8,
                Value::Array(_) => 9,
                Value::Number(n) if n.is_u64() => 10,
                _ => 6,
            };
            bytes.extend(ty.to_le_bytes());
            value(&mut bytes, v);
        }
        let mut data = Vec::new();
        for (name, ty, elements, raw) in &self.tensors {
            text(&mut bytes, name);
            bytes.extend(1u32.to_le_bytes());
            bytes.extend((*elements as u64).to_le_bytes());
            bytes.extend(ty.to_le_bytes());
            bytes.extend((data.len() as u64).to_le_bytes());
            data.extend(raw);
            data.resize(data.len().next_multiple_of(32), 0);
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.extend(data);
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/scratch"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("bundle-gguf-{}-{id}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn load(&self) -> Result<crate::Bundle, Box<dyn std::error::Error + Send + Sync>> {
        let path = self.write();
        let loaded = crate::load(&path, 2, 8);
        std::fs::remove_file(path).unwrap();
        loaded
    }
}

fn widened(weights: &[Weight]) -> Vec<Weight> {
    let wide = |w: &Weight| match w {
        Weight::Packed(f, bytes) => {
            let n = f.elements(bytes.len());
            Weight::Floats(Tensor::f32(
                vec![n],
                (0..n).map(|i| f.unpack(bytes, i)).collect(),
            ))
        }
        Weight::Floats(t) => Weight::Floats(t.clone()),
    };
    weights.iter().map(wide).collect()
}

fn engine(graph: Graph, recipe: &Recipe, weights: &[Weight]) -> Engine {
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let mut engine = Engine::open(graph, 8, recipe, &settings).unwrap();
    engine.load(weights).unwrap();
    engine
}

fn compare(form: &'static dyn Form) {
    let loaded = Fixture::new(form).load().unwrap();
    assert_eq!(loaded.stop, [2]);
    assert!(loaded.template.is_some());
    assert_eq!(loaded.tokenizer.encode("ab").unwrap(), [2]);
    let pins = loaded.pins.unwrap();
    assert_eq!(pins.weights.name(), form.name());
    assert_eq!(pins.named.len(), 5);
    for w in &loaded.graph.weights {
        let norm = tensor_name(&w.path).unwrap().contains("norm");
        assert_eq!(
            pins.form(&w.path).name(),
            if norm { "f32" } else { form.name() }
        );
    }
    let recipe = Recipe {
        pins,
        ..Recipe::default()
    };
    let wide = widened(&loaded.weights);
    let graph = &loaded.graph;
    let mut direct = engine(graph.clone(), &recipe, &loaded.weights);
    let mut roundtrip = engine(graph.clone(), &recipe, &wide);
    let mut override_direct = engine(graph.clone(), &Recipe::default(), &loaded.weights);
    let mut override_wide = engine(graph.clone(), &Recipe::default(), &wide);
    for position in 0..5 {
        let rows: Vec<_> = (0..2)
            .map(|slot| Row {
                slot,
                token: 3 + position + slot,
                position,
            })
            .collect();
        assert_eq!(direct.step(&rows).unwrap(), roundtrip.step(&rows).unwrap());
        assert_eq!(
            override_direct.step(&rows).unwrap(),
            override_wide.step(&rows).unwrap()
        );
    }
}

#[test]
fn packed_engine() {
    compare(named("q8_0").unwrap());
    compare(named("q4_0").unwrap());
}

#[test]
fn tied_output() {
    let mut fixture = Fixture::new(named("q8_0").unwrap());
    let mut output = fixture
        .tensors
        .iter()
        .find(|t| t.0 == "token_embd.weight")
        .unwrap()
        .clone();
    output.0 = "output.weight".into();
    fixture.tensors.push(output);
    assert!(fixture.load().is_ok());
    fixture.tensors.last_mut().unwrap().3[4] ^= 1;
    assert!(
        fixture
            .load()
            .err()
            .unwrap()
            .to_string()
            .contains("untied output.weight")
    );
    fixture.tensors.pop();
    fixture.meta[0].1 = json!("other");
    assert!(fixture.load().is_err());
}

#[test]
fn mixed_pins() {
    let graph = stepper(&Config::toy(2, 8));
    let base = named("q8_0").unwrap();
    let mut forms = vec![base; graph.weights.len()];
    forms[0] = F32_FORM;
    forms[3] = F32_FORM;
    let pins = pins(&graph, &forms, 2);
    assert!(pins.named.iter().all(|(name, _)| !name.contains('*')));
    for (w, f) in graph.weights.iter().zip(forms) {
        assert_eq!(pins.form(&w.path), f);
    }
}

/// Bytes of the wrong length are refused before any upload.
#[test]
fn short_bytes() {
    let loaded = Fixture::new(named("q8_0").unwrap()).load().unwrap();
    let recipe = Recipe {
        pins: loaded.pins.unwrap(),
        ..Recipe::default()
    };
    let mut weights = loaded.weights;
    let at = weights
        .iter()
        .position(|w| matches!(w, Weight::Packed(..)))
        .unwrap();
    let Weight::Packed(form, bytes) = &weights[at] else {
        unreachable!()
    };
    weights[at] = Weight::Packed(*form, bytes[..bytes.len() - 1].to_vec());
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let mut engine = Engine::open(loaded.graph, 8, &recipe, &settings).unwrap();
    assert!(matches!(engine.load(&weights), Err(lower::Error::Size(_))));
}
