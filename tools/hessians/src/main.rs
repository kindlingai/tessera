//! CLI entrypoint for layer-streaming Hessian calibration.

use std::fs;
use std::process::ExitCode;
use tessera_bundle::mla::config as mla_config;
use tessera_hessians::args::{Command, parse_command};
use tessera_hessians::driver::run_dir;
use tessera_hessians::parity::{compare_dirs, print_report};

const USAGE: &str = "tessera-hessians --model DIR --tokens FILE --work DIR --out DIR --hess-id ID
  [--heldout N] [--layers 0-77] [--hess-layers SET] [--save-inputs SET]
  [--route-log SET] [--batch N] [--experts-per-file 32] [--dense-heldout]
  [--no-dense] [--no-experts] [--weights-id HEX] [--resume] [--state FILE]
  [--keep-hidden] [--gpu] [--device N]
tessera-hessians --bench-kernels [--device N]
tessera-hessians --verify-parity GOT_DIR REF_DIR --layer L [--max-rel 5e-3]";

fn run_cmd(given: &[String]) -> Result<(), String> {
    if given.is_empty() || given.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    match parse_command(given)? {
        Command::BenchKernels { device } => {
            #[cfg(feature = "cuda")]
            {
                tessera_hessians::gpu::bench_kernels(device)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device;
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
            if rep.passed {
                Ok(())
            } else {
                Err(format!(
                    "parity failed: worst_rel_frob={:.3e} > {:.3e}",
                    rep.worst_rel, max_rel
                ))
            }
        }
        Command::Calib(cfg) => {
            let json = fs::read_to_string(cfg.model.join("config.json"))
                .map_err(|e| format!("config.json: {e}"))?;
            let mla = mla_config(&json, cfg.batch as u64, 2048)
                .map_err(|e| format!("mla config: {e}"))?;
            if cfg.gpu {
                #[cfg(feature = "cuda")]
                {
                    tessera_hessians::gpu::run_gpu(&cfg, &mla)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("rebuild with --features cuda for --gpu".into())
                }
            } else {
                run_dir(&cfg, &mla)
            }
        }
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
