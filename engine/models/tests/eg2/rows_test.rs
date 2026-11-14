use super::rotation_test::text;
use crate::common::slices;
use bundle::names::{repo, resolve};
use models::Rotated;
use models::eg2::{Config, program};

/// Each layer takes its rows of the stacked projection.
#[test]
fn stacked_rows() {
    let config = Config::release(1, 1024);
    let g = program::<Rotated>(&config);
    let slices = slices(include_str!("manifests/exlr8.json"));
    let claims = resolve(&g, &slices, &[&repo("eg2").unwrap()]).unwrap();
    let claim = |name: &str| {
        let at = g.weights.iter().position(|w| text(&w.path) == name);
        claims[at.unwrap()].clone()
    };
    let per = config.per_layer as usize;
    assert_eq!(claim("layers.1.ple_in.w").rows, Some((per, per)));
    assert_eq!(claim("layers.1.ple_in.svh").rows, Some((per, per)));
    assert_eq!(claim("layers.1.ple_in.suh").rows, None);
    assert!(
        claim("layers.1.scalar")
            .name
            .ends_with("layers.1.layer_scalar")
    );
}
