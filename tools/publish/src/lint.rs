//! Every crate in one binary, where `dead_code` flags unused `pub`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::Result;
use crate::meta::Meta;
use crate::plan::{Kind, Member, Out, Role, lib_name, module, single};

pub(crate) const NAME: &str = "tessera-lint";
/// The module that holds the members.
pub(crate) const LIBS: &str = "merged";

/// The macro crates, and the binary that holds the rest.
pub(crate) fn plan(meta: &Meta) -> Result<Vec<Out>> {
    let mut names = BTreeMap::new();
    let mut members = Vec::new();
    for (p, pkg) in meta.packages.iter().enumerate() {
        if pkg.is_macro() {
            continue;
        }
        for t in &pkg.targets {
            let (role, kind) = match t.kind[0].as_str() {
                "lib" | "cdylib" | "rlib" | "staticlib" => (Role::Lib, ""),
                "bin" | "example" => (Role::Main, t.kind[0].as_str()),
                "test" => (Role::Test, "test"),
                _ => continue,
            };
            let mut name = module(pkg);
            if role == Role::Lib {
                names.insert(lib_name(pkg), format!("crate::{LIBS}::{name}"));
            } else {
                name = format!("{name}_{kind}_{}", t.name.replace('-', "_"));
            }
            members.push(Member {
                pkg: p,
                to: PathBuf::from("src").join(LIBS).join(&name),
                module: name,
                root: t.src_path.strip_prefix(pkg.dir())?.to_path_buf(),
                role,
            });
        }
    }
    let mut outs = Vec::new();
    for (p, pkg) in meta.packages.iter().enumerate() {
        if pkg.is_macro() {
            outs.push(single(meta, p, Kind::Macro, names.clone()));
        }
    }
    outs.push(Out {
        name: NAME.into(),
        kind: Kind::Lint,
        description: "Every workspace crate, for dead code checks".into(),
        members,
        names,
        glob: None,
    });
    Ok(outs)
}

/// Makes a binary's or example's `main` a `dead_code` root.
pub(crate) fn root(file: &Path) -> Result<()> {
    let item = quote! {
        #[allow(dead_code)]
        fn lint_root() {
            let _ = main();
        }
    };
    append(file, item)
}

/// Writes `src/main.rs` with the member modules, and `build.rs`.
pub(crate) fn write(meta: &Meta, dir: &Path, items: &[TokenStream]) -> Result<()> {
    let libs = format_ident!("{LIBS}");
    let main = quote! {
        mod #libs {
            #(#items)*
        }
        fn main() {}
    };
    fs::write(dir.join("src/main.rs"), main.to_string())?;
    // Cargo sets these only for integration tests.
    let set = "cargo::rustc-env=";
    let mut vars = vec![format!("{set}CARGO_TARGET_TMPDIR=.")];
    for pkg in &meta.packages {
        for t in pkg.targets.iter().filter(|t| t.kind[0] == "bin") {
            vars.push(format!("{set}CARGO_BIN_EXE_{0}={0}", t.name));
        }
    }
    let build = quote! {
        fn main() {
            #(println!(#vars);)*
        }
    };
    fs::write(dir.join("build.rs"), build.to_string())?;
    Ok(())
}

fn append(file: &Path, item: TokenStream) -> Result<()> {
    let mut text = fs::read_to_string(file)?;
    text.push('\n');
    text.push_str(&item.to_string());
    text.push('\n');
    fs::write(file, text)?;
    Ok(())
}
