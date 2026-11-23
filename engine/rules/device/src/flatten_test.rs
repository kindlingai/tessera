use crate::{Backend, Error, Flat, Import, Options, Problem, flatten};

const CUDA: Options = Options {
    backend: Backend::Cuda,
    remote_kv: false,
    marks: false,
};

fn run(name: &str, text: &str, files: &[(&str, &str)], backend: Backend) -> Result<Flat, Error> {
    let load = |p: &str| files.iter().find(|f| f.0 == p).map(|f| f.1.to_string());
    flatten(name, text, &load, &Options { backend, ..CUDA })
}

fn has(text: &str, backend: Backend, needle: &str) -> bool {
    run("k.kernel", text, &[], backend)
        .unwrap()
        .text
        .contains(needle)
}

#[test]
fn flattens_in_order() {
    let kernel = "#include <tessera/1/all_reduce>\n#include \"a.h\"\nint body;\n";
    let files = [
        ("a.h", "#include <tessera/1/kept>\n#include \"b.h\"\n"),
        ("b.h", "int b;\n"),
    ];
    let flat = run("k.cu", kernel, &files, Backend::Cuda).unwrap();
    assert_eq!(flat.imports, [Import::AllReduce, Import::Kept]);
    let order = [
        "TESSERA_CAP_KEPT",
        "tessera/1/collectives.h",
        "int b;",
        "\"a.h\"",
        "int body",
    ];
    let at: Vec<_> = order.iter().map(|w| flat.text.find(w).unwrap()).collect();
    assert!(at.is_sorted() && !flat.text.contains("#include"), "{at:?}");
    let core = "#include <tessera/1/core>\nint body;\n";
    let trellis = "#include <tessera/1/trellis>\nint body;\n";
    assert!(has(core, Backend::Metal, "core.h") && !has(core, Backend::Metal, "backend.h"));
    assert!(has(core, Backend::Cuda, "backend.h") && !has(core, Backend::Cuda, "grid.h"));
    for b in [Backend::Cuda, Backend::Rocm] {
        assert!(has(trellis, b, "TESSERA_CAP_TRELLIS") && has(trellis, b, "trellis.h"));
        assert!(!has(trellis, b, "grid.h"));
    }
    let sum = run("sum.cu", include_str!("sum.cu"), &[], Backend::Cuda).unwrap();
    assert!(sum.text.contains("stage_bf16") && sum.text.contains("cvt.rn.bf16x2.f32"));
}

#[test]
fn refuses_unbuildable() {
    let grid = "#include <tessera/1/grid>\n";
    let cycle = [("a.h", "#include \"a.h\"\n")];
    let prob = |n, t, b| run(n, t, &cycle, b).unwrap_err().problem;
    let metal = |s| Problem::Unsupported(Backend::Metal, s);
    let ext = Problem::Extension(Backend::Cuda);
    let cyc = Problem::Cycle("a.h".into());
    assert_eq!(prob("k.kernel", grid, Backend::Metal), metal("grid"));
    let trellis = "#include <tessera/1/trellis>\n";
    assert_eq!(prob("k.kernel", trellis, Backend::Metal), metal("trellis"));
    assert_eq!(prob("k.hip", grid, Backend::Cuda), ext);
    assert_eq!(prob("k.cu", "int x;\n", Backend::Cuda), Problem::NoImports);
    assert_eq!(prob("k.cu", "#include \"a.h\"\n", Backend::Cuda), cyc);
}
