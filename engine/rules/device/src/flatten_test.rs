use crate::flatten::scan;
use crate::{Backend, Error, Flat, Import, Options, Problem, flatten};

const CUDA: Options = Options {
    backend: Backend::Cuda,
    remote_kv: false,
    marks: false,
};

const WHOLE: &str = "#include <tessera/1/whole>\nint body;\n";

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
        "tessera/1/types.h",
        "tessera/1/abi.h",
        "struct kept {",
        "tessera/1/core.h",
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
        assert!(!has(trellis, b, "grid.h") && has(WHOLE, b, "TESSERA_WHOLE_H"));
    }
    let sum = run("sum.cu", include_str!("sum.cu"), &[], Backend::Cuda).unwrap();
    assert!(sum.text.contains("stage_bf16") && sum.text.contains("cvt.rn.bf16x2.f32"));
    let gqa = run("mla.cu", include_str!("attend/mla.cu"), &[], Backend::Cuda).unwrap();
    for entry in [
        "GQA_DEF(tessera_gqa_attend_fp8,",
        "GQA_DEF(tessera_gqa_attend_bf16,",
        "GQA_DEF(tessera_gqa_attend_f32,",
        "tessera_gqa_reduce(",
    ] {
        assert!(gqa.text.contains(entry), "missing {entry}");
    }
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

fn problem(text: &str) -> Problem {
    scan("k.cu", text).unwrap_err().problem
}

#[test]
fn refuses_bad_imports() {
    let bare = Problem::Bare("tessera/kept".into());
    assert_eq!(problem("#include <tessera/kept>\n"), bare);
    assert_eq!(
        problem("#include <tessera/2/kept>\n"),
        Problem::Edition("2".into())
    );
    assert_eq!(
        problem("#include <tessera/1/x>\n"),
        Problem::Unknown("x".into())
    );
    assert_eq!(
        problem("#include <stdio.h>\n"),
        Problem::Foreign("stdio.h".into())
    );
    assert_eq!(
        problem("#include \"../x.h\"\n"),
        Problem::Path("../x.h".into())
    );
    assert_eq!(problem("#include KERNEL_H\n"), Problem::Operand);
}

#[test]
fn includes_at_top() {
    let late = [
        "#if 1\n#include <tessera/1/llm>\n#endif\n",
        "#include <tessera/1/llm>\nint x;\n#include \"a.h\"\n",
        "#include <tessera/1/llm>\n#if __has_include(<x>)\n#endif\n",
        "%:include <tessera/1/llm>\n",
    ];
    for text in late {
        assert_eq!(problem(text), Problem::Late, "{text}");
    }
    let quiet = "/* a\n b */\n#include <tessera/1/llm>\n/* #include <x>\n */ int x; // #include\n";
    assert_eq!(scan("k.cu", quiet).unwrap().includes, [2]);
    let trellis = scan("k.cu", "#include <tessera/1/trellis>\n").unwrap();
    assert_eq!(trellis.imports, [Import::Trellis]);
}
