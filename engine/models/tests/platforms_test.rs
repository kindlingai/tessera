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

/// The tiled kernel's text. Every tile row must compile it.
fn tiled_text(dialect: Dialect) -> String {
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
    body.text().render(dialect)
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

/// The `toolkit-cuda` and `toolkit-rocm` features say which are here.
#[test]
fn toolkits_agree() {
    let here = |backend: &str| match backend {
        "cuda" => cfg!(feature = "toolkit-cuda"),
        "rocm" => cfg!(feature = "toolkit-rocm"),
        _ => false,
    };
    let dir = scratch("platforms");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tiled.cu"), tiled_text(Dialect::Cuda)).unwrap();
    std::fs::write(dir.join("probe.cpp"), hip_probe()).unwrap();
    for row in rows() {
        let backend = backend_of(&row).to_string();
        for target in row.check.iter().filter(|_| here(&backend)) {
            match backend.as_str() {
                "cuda" => check_cuda(&row.tile, row.pdl, target, &dir),
                "rocm" => check_rocm(&row.tile, target, &dir),
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
    assert!(
        sass.contains("HMMA.1688"),
        "{target}: no 16x8x8 tf32 tile in the SASS"
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
