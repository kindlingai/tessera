use crate::surface::json;

#[test]
fn cli_json_current() {
    let want = format!("{:#}\n", json());
    let got = include_str!("../cli.json");
    let fix = "run `cargo run -p tessera-cli -- --cli-json > frontend/cli/cli.json`";
    assert!(got == want, "frontend/cli/cli.json is stale: {fix}");
}
