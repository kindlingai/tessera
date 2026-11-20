use crate::Error;
use crate::overrides::Overrides;
use crate::run::Outcome;
use sensai_codegen::Dialect;

fn check(files: &mut Overrides<String>) -> Option<Outcome> {
    let compile = |text: &str| match text.contains("bad") {
        true => Err(Error::Zones),
        false => Ok(text.to_string()),
    };
    let reload = files.check("k", || "planned".to_string(), compile);
    reload.map(|r| r.outcome)
}

#[test]
fn files_replace_kernels() {
    let dir = sensai_testkit::scratch(&format!("overrides-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut files = Overrides::new(dir.clone(), Dialect::Cuda);
    assert_eq!(check(&mut files), None);
    assert!(files.write("k", "planned").unwrap().is_some());
    assert_eq!(check(&mut files), None);
    // Each write changes the length, whatever the clock's grain.
    std::fs::write(files.path("k"), "edited").unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Loaded));
    std::fs::write(files.path("k"), "bad!!").unwrap();
    assert!(matches!(check(&mut files), Some(Outcome::Failed(_))));
    assert_eq!(files.kernel("k").map(String::as_str), Some("edited"));
    std::fs::remove_file(files.path("k")).unwrap();
    assert_eq!(check(&mut files), Some(Outcome::Planned));
    assert_eq!(files.kernel("k"), None);
    std::fs::remove_dir_all(&dir).unwrap();
}
