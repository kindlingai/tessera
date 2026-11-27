//! Generated kernels as WGSL modules, for WebGPU.

use crate::LANES;
use crate::dialect::{wgsl_entry, wgsl_function};
use crate::ir::{Body, Helper, Intrinsic, Param, Ty, gather};
use crate::print::{Kind, Printer};

/// One kernel as a WGSL module.
#[derive(Clone, Debug, PartialEq)]
pub struct Wgsl {
    pub entry: String,
    /// Bound in order from 0: name, element type, written.
    pub buffers: Vec<(String, &'static str, bool)>,
    /// The uniform after the buffers: field names and types.
    pub scalars: Vec<(String, &'static str)>,
    /// Shuffles lanes: browsers want `enable subgroups;`, naga refuses it.
    pub subgroups: bool,
    /// The `f_threads` override sets threads per group.
    pub text: String,
}

/// Names WGSL keeps, and names the printed text calls.
const RESERVED: &str = "alias break case const const_assert continue continuing default diagnostic discard else \
    enable false fn for if let loop override requires return struct switch true var while \
    NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto \
    await become cast catch class co_await co_return co_yield coherent column_major common \
    compile compile_fragment concept const_cast consteval constexpr constinit crate \
    debugger decltype delete demote demote_to_helper do dynamic_cast enum explicit export \
    extends extern external fallthrough filter final finally friend from fxgroup get goto \
    groupshared highp impl implements import inline instanceof interface layout lowp macro \
    macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept \
    noinline nointerpolation non_coherent noncoherent noperspective null nullptr of \
    operator package packoffset partition pass patch pixelfragment precise precision \
    premerge priv protected pub public readonly ref regardless register reinterpret_cast \
    require resource restrict self set shared signed sizeof smooth snorm static \
    static_assert static_cast std subroutine super target template this thread_local throw \
    trait try type typedef typeid typename typeof union unless unorm unsafe unsized use \
    using varying virtual volatile wgsl where with writeonly yield read write read_write \
    function private workgroup uniform storage f32 u32 i32 bool vec2 vec4 array select \
    bitcast workgroupBarrier atomic atomicAdd atomicLoad atomicStore";

/// Whether WGSL text cannot use `name` as it is.
pub(crate) fn reserved(name: &str) -> bool {
    RESERVED.split_whitespace().any(|w| w == name)
}

/// A type's WGSL name. Indices fit 32 bits.
pub(crate) fn ty(ty: Ty) -> &'static str {
    match ty {
        Ty::F32 => "f32",
        Ty::U32 | Ty::U64 | Ty::Bytes => "u32",
        Ty::Bool => "bool",
        Ty::F32x4 => "vec4<f32>",
    }
}

/// Four floats stored from `index` of `name`.
pub(crate) fn store4(name: &str, index: &str, value: &str) -> String {
    let at = |i: u32| format!("{name}[({index}) + {i}u]");
    format!(
        "{{ let f_v = {value}; {} = f_v.x; {} = f_v.y; {} = f_v.z; {} = f_v.w; }}",
        at(0),
        at(1),
        at(2),
        at(3)
    )
}

/// No definition: the printer spells the call.
fn bare() -> Option<String> {
    Some(String::new())
}

/// A definition: `fn sig -> ret { return value; }`.
fn call(sig: &str, ret: &str, value: &str) -> Option<String> {
    Some(wgsl_function(sig, ret, &format!("    return {value};")))
}

/// Each intrinsic's WGSL definition, from the table.
macro_rules! spellings {
    ($($(#[$doc:meta])* $var:ident = $kind:ident $($name:ident)::+ ($($arg:tt)*) $(-> $ret:ident)?
        { device: $dev:expr, text: $text:expr, wgsl: $wgsl:expr, $($rest:tt)* })*) => {
        /// None: WGSL has no spelling for it yet.
        fn spelling(i: Intrinsic) -> Option<String> {
            match i {
                $(Intrinsic::$var => $wgsl,)*
            }
        }
    };
}

crate::intrinsic::table!(spellings);

/// Definitions for `used`. Errs when WGSL has no spelling.
fn definitions(used: &[Intrinsic]) -> Result<String, String> {
    let mut text = String::new();
    for i in Intrinsic::ALL.iter().filter(|i| used.contains(i)) {
        text += &spelling(*i).ok_or_else(|| format!("`{i:?}`"))?;
    }
    Ok(text)
}

/// The printer's refusal, if any, as an error.
fn checked(p: &Printer) -> Result<(), String> {
    p.refused.clone().map_or(Ok(()), Err)
}

impl Helper {
    /// Returns the WGSL definition and its intrinsics.
    fn wgsl(&self) -> Result<(String, Vec<Intrinsic>), String> {
        let mut p = Printer::new(&self.body.params, true);
        let mut params = Vec::new();
        for param in p.params {
            match param {
                Param::Scalar { name, ty } => {
                    p.bind(name, Kind::Value(*ty));
                    params.push(format!("{}: {}", p.c_name(name), self::ty(*ty)));
                }
                _ => return Err(format!("`{}` takes `{}`", self.name, param.name())),
            }
        }
        p.stmts(&self.body.stmts);
        let value = self.body.ret.as_ref().expect("a helper has a value");
        let value = p.expr(value);
        p.line(format!("return {value};"));
        assert!(!p.lane, "`{}` reads the lane: pass it in", self.name);
        checked(&p)?;
        let sig = format!("{}({})", self.name, params.join(", "));
        let text = wgsl_function(&sig, ty(self.ret), &p.lines.join("\n"));
        Ok((text, p.used))
    }
}

impl Body {
    /// This kernel in WGSL. Errs on the first unspelled part.
    pub fn wgsl(&self) -> Result<Wgsl, String> {
        assert!(self.ret.is_none(), "`{}` has a value: call it", self.entry);
        let mut p = Printer::new(&self.params, true);
        let (mut buffers, mut scalars, mut helpers) = (Vec::new(), Vec::new(), Vec::new());
        for param in p.params {
            match param {
                Param::Buffer { name, form, writes } => {
                    let ty = form
                        .wgsl_ty()
                        .ok_or_else(|| format!("form `{}`", form.name()))?;
                    p.bind(name, Kind::Buffer(*form));
                    buffers.push((p.c_name(name), ty, *writes));
                    gather(&form.header(), &mut helpers);
                }
                Param::Scalar { ty: Ty::Bool, name } => return Err(format!("flag `{name}`")),
                Param::Scalar { name, ty } => {
                    p.bind(name, Kind::Value(*ty));
                    scalars.push((p.c_name(name), self::ty(*ty)));
                }
                _ => panic!("`{}` takes `{}` from a caller", self.entry, param.name()),
            }
        }
        p.stmts(&self.stmts);
        checked(&p)?;
        gather(&self.helpers, &mut helpers);
        let mut used = p.used.clone();
        let mut defined = String::new();
        for h in &helpers {
            let (text, calls) = h.wgsl()?;
            defined += &text;
            used.extend(calls);
        }
        let subgroups = used.contains(&Intrinsic::ShflXor);
        let mut text = String::new();
        // Groups of 32 lanes return together.
        if subgroups {
            text += "diagnostic(off, subgroup_uniformity);\n";
        }
        for (at, (name, ty, writes)) in buffers.iter().enumerate() {
            let access = if *writes { "read_write" } else { "read" };
            text +=
                &format!("@group(0) @binding({at}) var<storage, {access}> {name}: array<{ty}>;\n");
        }
        if !scalars.is_empty() {
            let fields: Vec<String> = scalars.iter().map(|(n, t)| format!("{n}: {t}")).collect();
            text += &format!("struct F_args {{ {} }}\n", fields.join(", "));
            let at = buffers.len();
            text += &format!("@group(0) @binding({at}) var<uniform> f_args: F_args;\n");
        }
        text += "override f_threads: u32 = 256u;\n";
        for decl in &p.module {
            text += &format!("{decl}\n");
        }
        text += &definitions(&used)?;
        text += &defined;
        let mut lines = Vec::new();
        if p.lane {
            lines.push(format!("    let lane = gid % {LANES}u;"));
        }
        for (name, _) in &scalars {
            lines.push(format!("    let {name} = f_args.{name};"));
        }
        lines.extend(p.lines);
        text += &wgsl_entry(&self.entry, &lines.join("\n"));
        Ok(Wgsl {
            entry: self.entry.clone(),
            buffers,
            scalars,
            subgroups,
            text,
        })
    }
}
