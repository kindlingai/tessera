use crate::{Grid, path};
use lower::{Device, rules};
use ops::testing::gpu_available;

/// A crate's rules register once this one links it.
#[test]
fn linked_rules_register() {
    assert!(rules().iter().any(|r| r.name == "lanes"));
}

#[test]
fn grid_reaches_rules() {
    if !gpu_available() {
        return eprintln!("no GPU here; skipped");
    }
    let mut device = Device::guess();
    let timed = Grid::tiny().run(&mut device).unwrap();
    assert!(device.samples() > 0 && !device.name.is_empty());
    for rule in rules() {
        let ran = timed.iter().any(|(step, _)| step.rule == rule.name);
        assert!(ran, "the grid never ran {}", rule.name);
    }
    // Everything is known now, so a second pass times nothing.
    assert!(Grid::tiny().run(&mut device).unwrap().is_empty());
    let file = std::env::temp_dir().join("tessera-calibrate-test.json");
    device.save(&file).unwrap();
    assert!(Device::load(&file).unwrap() == device, "a round trip");
    std::fs::remove_file(&file).unwrap();
    assert!(path(&device.name).ends_with(format!("{}.json", device.name.replace(' ', "-"))));
}
