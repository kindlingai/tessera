use crate::{Error, Recipe};
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
