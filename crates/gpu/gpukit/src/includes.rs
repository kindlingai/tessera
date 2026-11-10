//! Link-time registry of shader include folders.

use crate::Error;

pub use include_dir;
pub use include_dir::Dir;
pub use scattered_collect;
pub use scattered_collect::declarative::{gather, scatter};

/// One registered folder of shader headers.
pub struct IncludeFolder(pub &'static Dir<'static>);

gather! {
    #[gather]
    pub static INCLUDE_DIRS: scattered_collect::ScatteredSlice<IncludeFolder>;
}

/// Embed and register a folder of shader headers.
#[macro_export]
macro_rules! register_includes {
    ($path:tt) => {
        const _: () = {
            // Spares the caller a direct `include_dir` dependency.
            use $crate::includes::include_dir;
            // Cargo misses embedded files: callers emit `rerun-if-changed`.
            static DIR: include_dir::Dir<'static> = include_dir::include_dir!($path);
            $crate::includes::scatter! {
                #[scatter($crate::includes::INCLUDE_DIRS)]
                static REG: $crate::includes::IncludeFolder =
                    $crate::includes::IncludeFolder(&DIR);
            }
        };
    };
}

/// Every registered header as (name, contents), failing on duplicates.
pub fn include_table() -> Result<Vec<(&'static str, &'static str)>, Error> {
    let mut table: Vec<(&'static str, &'static str)> = Vec::new();
    for folder in INCLUDE_DIRS.iter() {
        for file in folder.0.files() {
            let path = file.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(contents) = file.contents_utf8() else {
                return Err(Error::Compile(format!(
                    "registered include {name:?} is not UTF-8"
                )));
            };
            if table.iter().any(|(n, _)| *n == name) {
                return Err(Error::Compile(format!(
                    "include {name:?} is registered by two folders; bare-name \
                     #include resolution cannot pick one"
                )));
            }
            table.push((name, contents));
        }
    }
    Ok(table)
}
