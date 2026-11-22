use crate::scan::scan;
use crate::{Import, Problem};

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
