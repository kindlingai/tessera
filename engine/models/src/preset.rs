//! Named model sizes. Each model file registers its own.

use tessera_trace::Graph;

pub(crate) use scattered_collect::declarative::{gather, scatter};

/// A model's graphs by name.
pub struct Preset {
    pub name: &'static str,
    /// Test sizes: the step graph, or the forward graph.
    pub small: Option<fn() -> Graph>,
    /// Released sizes, one sequence of `context` rows.
    pub step: Option<fn(u64) -> Graph>,
    pub forward: Option<fn(u64) -> Graph>,
}

impl Preset {
    pub const NONE: Self = Self {
        name: "",
        small: None,
        step: None,
        forward: None,
    };
}

gather! {
    #[gather]
    pub(crate) static PRESETS: scattered_collect::ScatteredSlice<Preset>;
}

/// Registers a preset: name, then `field: value` pairs.
macro_rules! preset {
    ($name:literal $(, $field:ident: $value:expr)* $(,)?) => {
        const _: () = {
            $crate::preset::scatter! {
                #[scatter($crate::preset::PRESETS)]
                static PRESET: $crate::preset::Preset = {
                    let mut p = $crate::preset::Preset::NONE;
                    p.name = $name;
                    $(p.$field = Some($value);)*
                    p
                };
            }
        };
    };
}
pub(crate) use preset;

/// Registers all three graphs from `stepper` and `program` in scope.
macro_rules! traced {
    ($name:literal, $small:path, $full:path, [$($kind:ty),+]) => {
        $crate::preset::preset!(
            $name,
            small: || stepper::<$($kind),+>(&$small()),
            step: |n| stepper::<$($kind),+>(&$full(1, n)),
            forward: |n| program::<$($kind),+>(&$full(1, n)),
        );
    };
}
pub(crate) use traced;

/// The `name` preset, or an error listing known names.
pub fn named(name: &str) -> Result<&'static Preset, String> {
    if let Some(p) = PRESETS.iter().find(|p| p.name == name) {
        return Ok(p);
    }
    let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
    names.sort_unstable();
    Err(format!("no model `{name}`. Known: {}", names.join(", ")))
}
