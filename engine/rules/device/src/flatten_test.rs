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
    // Core alone: types everywhere, primitives on devices.
    let core = "#include <tessera/1/core>\nint body;\n";
    let has = |b, h| run("k.kernel", core, &[], b).unwrap().text.contains(h);
    assert!(has(Backend::Metal, "tessera/1/core.h") && !has(Backend::Metal, "backend.h"));
    assert!(has(Backend::Cuda, "tessera/1/backend.h") && !has(Backend::Cuda, "grid.h"));
}

#[test]
fn refuses_unbuildable() {
    let grid = "#include <tessera/1/grid>\n";
    let cycle = [("a.h", "#include \"a.h\"\n")];
    let problem = |name, text, backend| run(name, text, &cycle, backend).unwrap_err().problem;
    let metal = Problem::Unsupported(Backend::Metal, "grid");
    assert_eq!(problem("k.kernel", grid, Backend::Metal), metal);
    assert_eq!(
        problem("k.hip", grid, Backend::Cuda),
        Problem::Extension(Backend::Cuda)
    );
    assert_eq!(
        problem("k.cu", "int x;\n", Backend::Cuda),
        Problem::NoImports
    );
    let looped = Problem::Cycle("a.h".into());
    assert_eq!(problem("k.cu", "#include \"a.h\"\n", Backend::Cuda), looped);
}
