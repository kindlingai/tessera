//! Writes one published crate: files, root module and manifest.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::Result;
use crate::dist::{Release, pyproject};
use crate::manifest::{Owners, manifest};
use crate::meta::Meta;
use crate::plan::{Kind, Member, Out};
use crate::rewrite::{Moves, exports, rewrite};

/// Directories a published crate leaves out.
const SKIP: [&str; 4] = ["tests", "benches", "examples", "target"];

pub fn emit(meta: &Meta, out: &Out, owners: &Owners, rel: &Release, base: &Path) -> Result<()> {
    let dir = base.join(&out.name);
    let mut mods = Vec::new();
    for m in &out.members {
        let pkg = &meta.packages[m.pkg];
        let files = copy(meta, out, m, pkg.dir(), &dir)?;
        if out.members.len() > 1 {
            let lib = pkg.lib().ok_or(format!("{} has no library", pkg.name))?;
            let rel = lib.src_path.strip_prefix(pkg.dir())?;
            reexport(&dir.join(&m.to).join(rel), &files)?;
            mods.push((m.module.clone(), m.to.strip_prefix("src")?.join(rel)));
        }
    }
    if out.kind == Kind::Bin {
        let pkg = &meta.packages[out.members[0].pkg];
        let bin = pkg
            .targets
            .iter()
            .find(|t| t.kind.iter().any(|k| k == "bin"))
            .ok_or("no binary")?;
        let from = dir.join(bin.src_path.strip_prefix(pkg.dir())?);
        if from != dir.join("src/main.rs") {
            fs::rename(from, dir.join("src/main.rs"))?;
        }
    }
    if !mods.is_empty() {
        write_root(out, &mods, &dir.join("src/lib.rs"))?;
    }
    if out.kind == Kind::Bin {
        pyproject(out, rel, &dir)?;
    }
    let doc = manifest(meta, out, owners, rel)?;
    fs::write(dir.join("Cargo.toml"), toml::to_string(&doc)?)?;
    Ok(())
}

/// Copies one member, rewriting its Rust files. Returns those files.
fn copy(meta: &Meta, out: &Out, m: &Member, from: &Path, dir: &Path) -> Result<Vec<PathBuf>> {
    let mut rust = Vec::new();
    let mut todo = vec![from.to_path_buf()];
    while let Some(d) = todo.pop() {
        for e in fs::read_dir(&d)? {
            let path = e?.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let top = d == from;
            if path.is_dir() {
                if !(top && SKIP.contains(&name)) {
                    todo.push(path);
                }
                continue;
            }
            if top && name == "Cargo.toml" {
                continue;
            }
            let to = dir.join(&m.to).join(path.strip_prefix(from)?);
            fs::create_dir_all(to.parent().ok_or("no parent")?)?;
            if path.extension().is_some_and(|x| x == "rs") {
                let text = fs::read_to_string(&path)?;
                let include = |old: &str| moved(meta, out, &path, &to, old, dir);
                let rel = m.to.to_string_lossy();
                let moves = Moves {
                    module: &m.module,
                    names: &out.names,
                    dir: &rel,
                    include: &include,
                };
                let new = rewrite(&text, &moves).map_err(|e| format!("{}: {e}", path.display()))?;
                fs::write(&to, new)?;
                rust.push(to);
            } else {
                fs::copy(&path, &to)?;
            }
        }
    }
    Ok(rust)
}

/// A new `include!` path for a file outside the crate.
fn moved(meta: &Meta, out: &Out, file: &Path, to: &Path, old: &str, dir: &Path) -> Option<String> {
    let target = clean(&file.parent()?.join(old));
    if out
        .members
        .iter()
        .any(|m| target.starts_with(meta.packages[m.pkg].dir()))
    {
        return None;
    }
    let copy = dir.join("src/included").join(target.file_name()?);
    fs::create_dir_all(copy.parent()?).ok()?;
    let none = |_: &str| None;
    let moves = Moves {
        module: "",
        names: &out.names,
        dir: "",
        include: &none,
    };
    fs::write(
        &copy,
        rewrite(&fs::read_to_string(&target).ok()?, &moves).ok()?,
    )
    .ok()?;
    Some(relative(to.parent()?, &copy))
}

fn clean(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => drop(out.pop()),
            Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

fn relative(from: &Path, to: &Path) -> String {
    let common = from
        .components()
        .zip(to.components())
        .take_while(|(a, b)| a == b)
        .count();
    let ups = from.components().count() - common;
    let rest: PathBuf = to.components().skip(common).collect();
    let mut out = PathBuf::new();
    for _ in 0..ups {
        out.push("..");
    }
    out.join(rest).to_string_lossy().into_owned()
}

/// Exported macros stay reachable through their crate's module.
fn reexport(lib: &Path, files: &[PathBuf]) -> Result<()> {
    let mut names = Vec::new();
    for f in files {
        names.extend(exports(&fs::read_to_string(f)?)?);
    }
    if names.is_empty() {
        return Ok(());
    }
    let idents = names.iter().map(|n| format_ident!("{n}"));
    let item = quote! {
        #[doc(hidden)]
        pub use crate::{#(#idents),*};
    };
    let mut text = fs::read_to_string(lib)?;
    text.push('\n');
    text.push_str(&item.to_string());
    text.push('\n');
    fs::write(lib, text)?;
    Ok(())
}

fn write_root(out: &Out, mods: &[(String, PathBuf)], at: &Path) -> Result<()> {
    let doc = &out.description;
    let items: Vec<TokenStream> = mods
        .iter()
        .map(|(name, path)| {
            let ident = format_ident!("{name}");
            let path = path.to_string_lossy();
            quote! {
                #[doc(hidden)]
                #[path = #path]
                pub mod #ident;
            }
        })
        .collect();
    let crate_name = format_ident!("{}", out.name.replace('-', "_"));
    let glob = out.glob.as_ref().map(|g| {
        let g = format_ident!("{g}");
        quote! { #[doc(inline)] pub use #g::*; }
    });
    let root = quote! {
        #![doc = #doc]
        #![allow(clippy::module_inception)]
        extern crate self as #crate_name;
        #glob
        #(#items)*
    };
    fs::write(at, root.to_string())?;
    let status = Command::new("rustfmt")
        .args(["--edition", "2024"])
        .arg(at)
        .status()?;
    if !status.success() {
        return Err(format!("rustfmt failed on {}", at.display()).into());
    }
    Ok(())
}
