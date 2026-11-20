//! Each toolkit compiles the tile text to check the table.

// The other matmul rules come from here.
use tessera_kernels as _;

use std::path::Path;
use std::process::Command;
use tessera_gpukit::platform::{backend_of, rows};
use tessera_linear::{PARTS, Product, SCHEDULES, Schedule, Tiled};
use tessera_lower::sites::After;
use tessera_lower::{Dialect, Kernel, Source};
use tessera_testkit::scratch;

/// The tiled kernel's text and entry name.
fn tiled_text(dialect: Dialect, (schedule, parts): (Schedule, u32)) -> (String, String) {
    let form = match parts {
        0 => tessera_lower::form::F32_FORM,
        _ => tessera_lower::form::BF16_FORM,
    };
    let tiled = Tiled {
        product: Product {
            m: 64,
            n: 64,
            k: 64,
            form,
            after: After::default(),
            picked: None,
            near: None,
        },
        schedule,
        parts,
    };
    let Source::Made(body) = tiled.source() else {
        panic!("tiled is generated");
    };
    let text = body.text();
    (text.render(dialect), text.entry.clone())
}

/// The metal-tt script for one compute pipeline.
fn pipeline_script(entry: &str) -> String {
    format!(
        r#"{{"libraries": {{"paths": [{{"label": "L", "path": "./tiled.metallib"}}]}},
 "pipelines": {{"compute_pipelines": [{{"compute_function": "alias:L#{entry}"}}]}}}}"#
    )
}

/// One tile multiply, for a bare amdgcn compile.
fn hip_probe() -> String {
    format!(
        "typedef unsigned u32;\n#define __device__\n#define __forceinline__ inline\n{}\n\
         void probe(tile_acc *c, tile_a a, tile_b b) {{ TILE_MMA(*c, a, b); }}\n",
        tessera_lower::hip_tiles()
    )
}

/// Runs `program`, giving its output and whether it succeeded.
fn run(program: &str, args: &[&str], dir: &Path) -> (bool, String) {
    let out = Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("{program}: {e}"));
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

/// A toolkit that is not installed is skipped.
#[test]
fn toolkits_agree() {
    let here = |backend: &str| {
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
    };
    let dir = scratch("platforms");
    std::fs::create_dir_all(&dir).unwrap();
    let (cuda, _) = tiled_text(Dialect::Cuda, (SCHEDULES[0], 0));
    let (parts, _) = tiled_text(Dialect::Cuda, (PARTS[1], 3));
    let (metal, entry) = tiled_text(Dialect::Metal, (SCHEDULES[0], 0));
    std::fs::write(dir.join("tiled.cu"), cuda).unwrap();
    std::fs::write(dir.join("parts.cu"), parts).unwrap();
    std::fs::write(dir.join("probe.cpp"), hip_probe()).unwrap();
    std::fs::write(dir.join("tiled.metal"), metal).unwrap();
    std::fs::write(dir.join("tiled.mtlp-json"), pipeline_script(&entry)).unwrap();
    for row in rows() {
        let backend = backend_of(&row).to_string();
        if row.check.is_empty() || !here(&backend) {
            continue;
        }
        for target in &row.check {
            match backend.as_str() {
                "cuda" => {
                    check_cuda(&row.tile, row.pdl, target, &dir);
                    check_parts(row.bf16_tile.is_some(), target, &dir);
                }
                "rocm" => check_rocm(&row.tile, target, &dir),
                "metal" => check_metal(&row.tile, target, &dir),
                other => panic!("no toolkit check for {other}"),
            }
            println!("{backend} {target}: agrees, tile {:?}", row.tile);
        }
    }
}

/// nvcc builds the text when the row has a unit.
fn check_cuda(tile: &Option<[usize; 3]>, pdl: bool, target: &str, dir: &Path) {
    let arch = format!("-arch={target}");
    let (built, log) = run(
        "nvcc",
        &[&arch, "-cubin", "-o", "tiled.cubin", "tiled.cu"],
        dir,
    );
    assert_eq!(built, tile.is_some(), "{target}: {log}");
    if tile.is_none() {
        return;
    }
    let (_, sass) = run("cuobjdump", &["-sass", "tiled.cubin"], dir);
    let ops: std::collections::BTreeSet<&str> = sass
        .split_whitespace()
        .filter(|w| {
            w.len() > 2
                && w.chars()
                    .all(|c| c.is_ascii_uppercase() || c == '.' || c.is_ascii_digit())
        })
        .collect();
    assert!(
        sass.contains("HMMA.1688"),
        "{target}: no 16x8x8 tf32 tile in the SASS. It has {ops:?}"
    );
    let (_, ptx) = run("nvcc", &[&arch, "-ptx", "-o", "-", "tiled.cu"], dir);
    assert_eq!(
        ptx.contains("griddepcontrol.wait"),
        pdl,
        "{target}: dependent launch"
    );
}

/// nvcc builds the bf16 parts text where listed.
fn check_parts(listed: bool, target: &str, dir: &Path) {
    if !listed {
        return;
    }
    let arch = format!("-arch={target}");
    let args = [&arch, "-cubin", "-o", "parts.cubin", "parts.cu"];
    let (built, log) = run("nvcc", &args, dir);
    assert!(built, "{target}: {log}");
    let (_, sass) = run("cuobjdump", &["-sass", "parts.cubin"], dir);
    assert!(
        sass.contains("HMMA.16816"),
        "{target}: no 16x8x16 bf16 tile"
    );
}

/// clang's amdgcn backend shows the f16 wmma for the target.
fn check_rocm(tile: &Option<[usize; 3]>, target: &str, dir: &Path) {
    let cpu = format!("-mcpu={target}");
    let args = [
        "-x",
        "c++",
        "--target=amdgcn-amd-amdhsa",
        &cpu,
        "-nogpulib",
        "-O2",
        "-S",
        "-o",
        "-",
        "probe.cpp",
    ];
    let (built, asm) = run("clang++", &args, dir);
    assert_eq!(built, tile.is_some(), "{target}: {asm}");
    if tile.is_some() {
        assert!(
            asm.contains("v_wmma_f32_16x16x16_f16"),
            "{target}: no wmma in the assembly"
        );
    }
}

/// metal-tt builds the GPU slice for the architecture.
fn check_metal(tile: &Option<[usize; 3]>, target: &str, dir: &Path) {
    let xcrun = |args: &[&str]| run("xcrun", &[&["-sdk", "macosx"], args].concat(), dir);
    let (ok, log) = xcrun(&["metal", "-c", "tiled.metal", "-o", "tiled.ir"]);
    assert!(ok, "metal: {log}");
    let (ok, log) = xcrun(&["metallib", "tiled.ir", "-o", "tiled.metallib"]);
    assert!(ok, "metallib: {log}");
    let (out, script) = ("tiled.binary.metallib", "tiled.mtlp-json");
    let (built, log) = xcrun(&[
        "metal-tt",
        "-arch",
        target,
        "tiled.metallib",
        script,
        "-o",
        out,
    ]);
    assert_eq!(built, tile.is_some(), "{target}: {log}");
    if tile.is_none() {
        return;
    }
    let (_, archs) = xcrun(&["metal-lipo", "-archs", out]);
    assert!(archs.contains(target), "{target}: no slice in {archs}");
}
