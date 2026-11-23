//! The recipe module ABI: memory hand-off, the table format.

mod kit;
pub mod site;

#[cfg(test)]
mod kit_test;
#[cfg(test)]
mod lib_test;

/// The version of the JSON exchange with the host.
pub const ABI: u32 = 6;

/// Room for `len` bytes, kept until the module ends.
pub fn alloc(len: i32) -> i32 {
    let room = vec![0u8; len.max(0) as usize].leak();
    room.as_mut_ptr() as usize as i32
}

/// # Safety
///
/// `at` and `len` must come from `alloc`.
pub unsafe fn input(at: i32, len: i32) -> &'static [u8] {
    let at = at as usize as *const u8;
    // SAFETY: the caller passes back what `alloc` returned.
    unsafe { std::slice::from_raw_parts(at, len.max(0) as usize) }
}

/// `bytes` for the host, as `at << 32 | len`.
pub fn output(bytes: Vec<u8>) -> i64 {
    let bytes = bytes.leak();
    let at = bytes.as_ptr() as usize as u64;
    ((at << 32) | bytes.len() as u64) as i64
}

pub(crate) fn respond(
    bytes: &[u8],
    offer: impl Fn(&site::Ask) -> Option<Vec<site::Offer>>,
) -> Vec<u8> {
    let answer = match serde_json::from_slice::<site::Ask>(bytes) {
        Ok(ask) => serde_json::to_vec(&offer(&ask)),
        Err(e) => serde_json::to_vec(&site::Unread {
            unread: e.to_string(),
        }),
    };
    answer.unwrap_or_else(|_| b"null".to_vec())
}

/// # Safety
///
/// `at` and `len` must come from `alloc`.
pub unsafe fn answer_ask(
    at: i32,
    len: i32,
    offer: impl Fn(&site::Ask) -> Option<Vec<site::Offer>>,
) -> i64 {
    // SAFETY: the caller passes back what `alloc` returned.
    let bytes = unsafe { input(at, len) };
    output(respond(bytes, offer))
}

/// # Safety
///
/// `at` and `len` must come from `alloc`.
pub unsafe fn answer(
    at: i32,
    len: i32,
    offer: impl Fn(&site::Site) -> Option<Vec<site::Offer>>,
) -> i64 {
    // SAFETY: the caller passes back what `alloc` returned.
    unsafe { answer_ask(at, len, |ask| offer(&ask.site)) }
}

/// `manifest` for the host.
pub fn declare(manifest: &site::Manifest) -> i64 {
    output(serde_json::to_vec(manifest).unwrap_or_default())
}

/// The host's input to `graph` and `tables`.
#[derive(Debug, serde::Deserialize)]
pub struct Build {
    pub config: serde_json::Value,
    pub seqs: u64,
}

/// One table: its dotted path and values.
pub type Table = (String, Vec<f32>);

/// # Safety
///
/// `at` and `len` must come from `alloc`.
unsafe fn build(at: i32, len: i32) -> Result<Build, String> {
    // SAFETY: the caller passes back what `alloc` returned.
    let bytes = unsafe { input(at, len) };
    serde_json::from_slice(bytes).map_err(|e| e.to_string())
}

/// # Safety
///
/// `at` and `len` must come from `alloc`.
pub unsafe fn build_graph(
    at: i32,
    len: i32,
    graph: impl Fn(&Build) -> Result<String, String>,
) -> i64 {
    // SAFETY: the caller passes back what `alloc` returned.
    let text = unsafe { build(at, len) }.and_then(|b| graph(&b));
    output(text.unwrap_or_else(|e| e).into_bytes())
}

/// # Safety
///
/// `at` and `len` must come from `alloc`.
pub unsafe fn build_tables(
    at: i32,
    len: i32,
    tables: impl Fn(&Build) -> Result<Vec<Table>, String>,
) -> i64 {
    // SAFETY: the caller passes back what `alloc` returned.
    let made = unsafe { build(at, len) }.and_then(|b| tables(&b));
    output(made.map(|t| encode(&t)).unwrap_or_default())
}

/// A module's exports. Sections are optional, in pattern order.
#[macro_export]
macro_rules! exports {
    (
        $(rules: { $($rule:expr => $kind:expr),+ $(,)? },)?
        $(kernels: { $($file:literal: [$($kernel:expr),+ $(,)?]),+ $(,)? },)?
        $(files: [$($extra:literal),* $(,)?],)?
        $(propose: $propose:path,)?
        $(ask: $ask:path,)?
        $(graph: $graph:path, tables: $tables:path,)?
    ) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn alloc(len: i32) -> i32 {
            $crate::alloc(len)
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn manifest() -> i64 {
            $crate::declare(&$crate::site::Manifest {
                abi: $crate::ABI,
                kernels: vec![$($($(
                    $crate::site::Kernel::file($kernel, $kernel, $file),
                )+)+)?],
                rules: vec![$($(
                    $crate::site::Rule {
                        name: $rule,
                        site: $kind,
                    },
                )+)?],
                graph: $crate::exports!(@has $($graph)?),
                files: vec![
                    $($(
                        $crate::site::File {
                            path: $file,
                            text: include_str!($file),
                        },
                    )+)?
                    $($(
                        $crate::site::File {
                            path: $extra,
                            text: include_str!($extra),
                        },
                    )*)?
                ],
            })
        }

        $($crate::exports!(@propose answer, $propose);)?
        $($crate::exports!(@propose answer_ask, $ask);)?

        $(
            /// The graph JSON, else why the config is refused.
            #[unsafe(no_mangle)]
            pub extern "C" fn graph(at: i32, len: i32) -> i64 {
                // SAFETY: the host passes back what `alloc` returned.
                unsafe { $crate::build_graph(at, len, $graph) }
            }

            /// The tables the graph reads, else nothing.
            #[unsafe(no_mangle)]
            pub extern "C" fn tables(at: i32, len: i32) -> i64 {
                // SAFETY: the host passes back what `alloc` returned.
                unsafe { $crate::build_tables(at, len, $tables) }
            }
        )?
    };
    (@has) => {
        false
    };
    (@has $graph:path) => {
        true
    };
    (@propose $answer:ident, $offer:path) => {
        /// The offers for a site that fits, else `null`.
        #[unsafe(no_mangle)]
        pub extern "C" fn propose(at: i32, len: i32) -> i64 {
            // SAFETY: the host passes back what `alloc` returned.
            unsafe { $crate::$answer(at, len, $offer) }
        }
    };
}

/// Named tables of floats, as one buffer the host decodes.
pub fn encode(tables: &[Table]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend((tables.len() as u32).to_le_bytes());
    for (name, values) in tables {
        out.extend((name.len() as u32).to_le_bytes());
        out.extend(name.as_bytes());
        out.extend((values.len() as u32).to_le_bytes());
        out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    }
    out
}

/// The tables `encode` wrote, or `None` if bad.
pub fn decode(bytes: &[u8]) -> Option<Vec<Table>> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let part = bytes.get(at..at.checked_add(n)?)?;
        at += n;
        Some(part)
    };
    let word = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let count = word(take(4)?);
    let mut tables = Vec::new();
    for _ in 0..count {
        let len = word(take(4)?);
        let name = String::from_utf8(take(len)?.to_vec()).ok()?;
        let n = word(take(4)?);
        let raw = take(n.checked_mul(4)?)?;
        let values = raw
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        tables.push((name, values.collect()));
    }
    Some(tables)
}
