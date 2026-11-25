//! CLI entrypoint for layer-streaming Hessian calibration.

use std::process::ExitCode;
use tessera_hessians::args::parse_args;
use tessera_hessians::driver::run_dir;

const USAGE: &str = "tessera-hessians --model DIR --tokens FILE --work DIR --out DIR --hess-id ID
  [--heldout N] [--layers SET] [--hess-layers SET] [--save-inputs SET]
  [--route-log SET] [--batch N] [--experts-per-file 32] [--dense-heldout]
  [--no-dense] [--no-experts] [--weights-id HEX] [--resume] [--state FILE]
  [--keep-hidden]

A SET is like 0-40,45,70- and defaults to every layer.";

fn run_cmd(given: &[String]) -> Result<(), String> {
    if given.is_empty() || given.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    run_dir(parse_args(given)?)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run_cmd(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-hessians: {e}");
            ExitCode::FAILURE
        }
    }
}
