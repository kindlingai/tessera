use crate::{Error, Recipe};
use sensai_engine::sprite;
use sensai_engine::wasm::Module;
use std::path::{Path, PathBuf};

const TEXT: &str = r#"
name = "small"
names = "names.json"

[weights]
path = "weights"
extra = { clip = "a/b@main" }

[lower]
favor = "flash"

[serve]
max_model_len = 2048
task = "generate"
reject_unknown_fields = true
"#;

#[test]
fn reads_a_recipe() {
    let mut recipe: Recipe = toml::from_str(TEXT).unwrap();
    recipe.dir = PathBuf::from("/r");
    assert_eq!(recipe.weights(None, None).unwrap(), Path::new("/r/weights"));
    assert_eq!(recipe.names(), Some(PathBuf::from("/r/names.json")));
    assert_eq!(recipe.weights.extra["clip"], "a/b@main");
    assert!(matches!(recipe.extras(None, None), Err(Error::NoCache)));
    let lower = recipe.lower.as_ref().unwrap();
    assert_eq!(lower.favor.as_deref(), Some("flash"));
    let flag = |f: &str, v: Option<&str>| (f.to_string(), v.map(String::from));
    let want = vec![
        flag("--max-model-len", Some("2048")),
        flag("--reject-unknown-fields", None),
        flag("--task", Some("generate")),
    ];
    assert_eq!(recipe.flags(), want);
    recipe.weights.hf = Some("a/b@main".into());
    assert!(matches!(recipe.weights(None, None), Err(Error::Weights)));
    assert!(toml::from_str::<Recipe>("name = \"x\"\nranks = 4\n[weights]").is_err());
}

/// tessera-small's `model_config.json`.
const SMALL: &str = r#"{"vocab_size": 512, "grid_size": 16, "dim": 512, "n_heads": 8,
  "depth_in": 2, "depth_core": 8, "depth_out": 2, "patch": 2, "mlp_ratio": 3.0,
  "n_registers": 4, "text_dim": 512, "text_len": 24, "rope_theta": 100.0,
  "conv_ffn": true, "color_freqs": 6, "self_cond": true}"#;

#[test]
fn module_graph_matches() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/recipes/tessera-small");
    let recipe = Recipe::read(&dir).unwrap();
    let cache = sensai_testkit::scratch("recipes");
    let bytes = recipe.module_bytes(Some(&cache)).unwrap().unwrap();
    let module = Module::open(&bytes).unwrap();
    // The module builds exactly the graph the engine would.
    let config = sprite::Config::upstream(SMALL, 4).unwrap();
    let graph = module.graph(SMALL, 4).unwrap();
    assert_eq!(graph, Some(sprite::program(&config)));
    // Tables match to rounding: wasm's sin and cos differ.
    let got = module.tables(SMALL).unwrap();
    let want = sprite::constants(&config);
    assert_eq!(got.len(), want.len());
    for ((path, values), c) in got.iter().zip(&want) {
        assert_eq!((path, values.len()), (&c.path, c.data.len()));
        let gap = |(a, b): (&f32, &f32)| (a - b).abs();
        let worst = values.iter().zip(&c.data).map(gap).fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "{path}: off by {worst:e}");
    }
}
