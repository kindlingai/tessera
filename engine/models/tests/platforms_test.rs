//! Each toolkit compiles the tile text to check the table.

// The other matmul rules come from here.
use kernels as _;

use gpukit::platform::{backend_of, row_for, rows};
use gpukit::testing::scratch;
use linear::{SCHEDULES, Tiled};
use lower::sites::After;
use lower::{Dialect, Kernel, Source};
use std::path::Path;
use std::process::Command;

/// The tiled kernel's text and entry name.
fn tiled_text(dialect: Dialect) -> (String, String) {
    let tiled = Tiled {
        m: 64,
        n: 64,
        k: 64,
        form: lower::form::F32_FORM,
        schedule: SCHEDULES[0],
        after: After::default(),
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
        lower::hip_tiles()
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

#[test]
fn table_resolves() {
    let tile = |id: &str| row_for(id).and_then(|r| r.tile);
    assert_eq!(tile("cuda sm_121"), Some([16, 8, 8]));
    assert_eq!(tile("cuda sm_90"), Some([16, 8, 8]));
    assert_eq!(tile("cuda sm_75"), None, "a row with no usable tile");
    assert!(row_for("cuda sm_75").is_some());
    assert!(row_for("cuda sm_61").is_none(), "no row at all");
    assert_eq!(tile("rocm gfx11_5"), Some([16, 16, 16]));
    assert_eq!(tile("metal applegpu_g15p"), Some([8, 8, 8]));
    assert_eq!(tile("metal apple9"), Some([8, 8, 8]));
    assert!(row_for("cuda sm_90").is_some_and(|r| r.pdl));
    assert!(row_for("cuda sm_89").is_some_and(|r| !r.pdl));
    for row in rows() {
        assert!(!row.matches.is_empty(), "{row:?}");
    }
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
    let (cuda, _) = tiled_text(Dialect::Cuda);
    let (metal, entry) = tiled_text(Dialect::Metal);
    std::fs::write(dir.join("tiled.cu"), cuda).unwrap();
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
                "cuda" => check_cuda(&row.tile, row.pdl, target, &dir),
                "rocm" => check_rocm(&row.tile, target, &dir),
                "metal" => check_metal(&row.tile, target, &dir),
                other => panic!("no toolkit check for {other}"),
            }
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
        "{target}: no 16x8x8 tf32 tile in the SASS, which has {ops:?}"
    );
    let (_, ptx) = run("nvcc", &[&arch, "-ptx", "-o", "-", "tiled.cu"], dir);
    assert_eq!(
        ptx.contains("griddepcontrol.wait"),
        pdl,
        "{target}: dependent launch"
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

/// metal-tt builds a GPU slice per architecture in the family.
fn check_metal(tile: &Option<[usize; 3]>, family: &str, dir: &Path) {
    let xcrun = |args: &[&str]| run("xcrun", &[&["-sdk", "macosx"], args].concat(), dir);
    let (ok, log) = xcrun(&["metal", "-c", "tiled.metal", "-o", "tiled.ir"]);
    assert!(ok, "metal: {log}");
    let (ok, log) = xcrun(&["metallib", "tiled.ir", "-o", "tiled.metallib"]);
    assert!(ok, "metallib: {log}");
    let (ok, flags) = xcrun(&["metal-config", "-gpu-family", family]);
    assert!(ok, "metal-config {family}: {flags}");
    let flags: Vec<&str> = flags.split_whitespace().collect();
    let out = "tiled.binary.metallib";
    let args = [
        &["metal-tt", "tiled.metallib", "tiled.mtlp-json", "-o", out],
        &flags[..],
    ]
    .concat();
    let (built, log) = xcrun(&args);
    assert_eq!(built, tile.is_some(), "{family} with {flags:?}: {log}");
    if tile.is_none() {
        return;
    }
    let (_, archs) = xcrun(&["metal-lipo", "-archs", out]);
    let gpus: Vec<&str> = archs
        .split_whitespace()
        .filter(|a| a.starts_with("applegpu_"))
        .collect();
    assert!(!gpus.is_empty(), "{family}: no GPU slice in {archs}");
    for gpu in gpus {
        let row = row_for(&format!("metal {gpu}")).unwrap_or_else(|| panic!("{gpu}: no row"));
        assert_eq!(&row.tile, tile, "{gpu}");
    }
}
