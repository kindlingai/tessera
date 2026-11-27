use std::path::Path;
use std::process::Command;

/// Runs `program` in `dir`: whether it succeeded, and its output.
pub fn run(program: &str, args: &[&str], dir: &Path) -> (bool, String) {
    let out = Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("{program}: {e}"));
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

/// Whether `backend`'s GPU toolkit is installed. Prints when not.
pub fn toolkit(backend: &str) -> bool {
    let (program, flag, want) = match backend {
        "cuda" => ("nvcc", "--version", "nvcc"),
        "rocm" => ("clang++", "--print-targets", "amdgcn"),
        _ => ("xcrun", "--find", "metal-tt"),
    };
    let found = Command::new(program)
        .args([flag, want])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(want));
    if !found {
        println!("{backend}: no toolkit here, skipped");
    }
    found
}
