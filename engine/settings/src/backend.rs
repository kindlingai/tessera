//! GPU backends, and the facts each layer reads per backend.

/// A GPU that runs kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, derive_more::Display)]
#[serde(rename_all = "lowercase")]
#[display(rename_all = "lowercase")]
pub enum Backend {
    Metal,
    Cuda,
    Rocm,
}

/// One value per backend, indexed by it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByBackend<T>(pub [T; Backend::ALL.len()]);

impl<T> ByBackend<T> {
    pub fn from_fn(mut f: impl FnMut(Backend) -> T) -> Self {
        Self(std::array::from_fn(|i| f(Backend::ALL[i])))
    }
}

impl<T> std::ops::Index<Backend> for ByBackend<T> {
    type Output = T;

    fn index(&self, backend: Backend) -> &T {
        &self.0[backend as usize]
    }
}

struct Facts {
    /// The spelling platform ids and calibration files use.
    name: &'static str,
    /// Kernel files with this extension build for it alone.
    extension: &'static str,
    /// The macro its kernel text sees.
    define: &'static str,
    /// Grids that stay up across launches.
    persistent: bool,
    /// Host memory the device reads in place.
    pins: bool,
}

const FACTS: ByBackend<Facts> = ByBackend([
    Facts {
        name: "metal",
        extension: "metal",
        define: "TESSERA_METAL",
        persistent: false,
        pins: false,
    },
    Facts {
        name: "cuda",
        extension: "cu",
        define: "TESSERA_CUDA",
        persistent: true,
        pins: true,
    },
    Facts {
        name: "rocm",
        extension: "hip",
        define: "TESSERA_HIP",
        persistent: true,
        pins: true,
    },
]);

/// Kernel files with this extension build for every backend.
const SHARED: &str = "kernel";

impl Backend {
    /// In the order `ByBackend` holds them.
    pub const ALL: &'static [Backend] = &[Backend::Metal, Backend::Cuda, Backend::Rocm];

    pub fn name(self) -> &'static str {
        FACTS[self].name
    }

    /// The backend whose `name()` is `name`.
    pub fn named(name: &str) -> Option<Backend> {
        Self::ALL.iter().copied().find(|b| b.name() == name)
    }

    pub fn extension(self) -> &'static str {
        FACTS[self].extension
    }

    pub fn define(self) -> &'static str {
        FACTS[self].define
    }

    pub fn persistent(self) -> bool {
        FACTS[self].persistent
    }

    pub fn pins(self) -> bool {
        FACTS[self].pins
    }

    /// The backends a kernel file's extension allows.
    pub fn of(path: &str) -> &'static [Backend] {
        let Some((_, extension)) = path.rsplit_once('.') else {
            return &[];
        };
        if extension == SHARED {
            return Self::ALL;
        }
        match Self::ALL.iter().position(|b| b.extension() == extension) {
            Some(i) => &Self::ALL[i..=i],
            None => &[],
        }
    }
}
