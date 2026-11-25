//! CLI entrypoint for layer-streaming Hessian calibration.

use std::process::ExitCode;
use tessera_hessians::args::{Command, parse_command};
use tessera_hessians::driver::run_dir;
use tessera_hessians::parity::{compare_dirs, print_report};

const USAGE: &str = "tessera-hessians --model DIR --tokens FILE --work DIR --out DIR --hess-id ID
  [--heldout N] [--layers SET] [--hess-layers SET] [--save-inputs SET]
  [--route-log SET] [--batch N] [--experts-per-file 32] [--dense-heldout]
  [--no-dense] [--no-experts] [--weights-id HEX] [--resume] [--state FILE]
  [--keep-hidden] [--gpu] [--device N]
tessera-hessians --bench-kernels [--device N] [--shape E,H,F,T,K,D]
tessera-hessians --verify-parity GOT_DIR REF_DIR --layer L [--max-rel 5e-3]

A SET is like 0-40,45,70- and defaults to every layer.
--gpu adds each batch's sums into f32 totals on the device.
Rounding error grows with the number of batches per run.";

fn run_cmd(given: &[String]) -> Result<(), String> {
    if given.is_empty() || given.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    match parse_command(given)? {
        Command::BenchKernels { device, shape } => {
            #[cfg(feature = "cuda")]
            return tessera_hessians::gpu::bench_kernels(device, shape);
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (device, shape);
                Err("rebuild with --features cuda for --bench-kernels".into())
            }
        }
        Command::VerifyParity {
            got,
            want,
            layer,
            max_rel,
        } => {
            let rep = compare_dirs(&got, &want, layer, max_rel)?;
            print_report(&rep, max_rel);
            match rep.passed {
                true => Ok(()),
                false => Err("parity failed".into()),
            }
        }
        Command::Calib(cfg) => run_dir(*cfg),
    }
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
