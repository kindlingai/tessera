//! Kernel bodies compiled for the host at plan time.

mod expr;
#[cfg(test)]
mod lib_test;
mod lower;
mod runtime;

use cranelift_codegen::ir::{AbiParam, Signature, Type, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tessera_codegen::ir::Body;
use tessera_codegen::{Dialect, hash};

pub use runtime::{Arg, Buf, Ctx, Entry, failure};

/// A helper's parameter or return type.
#[derive(Clone, Copy)]
enum K {
    /// A pointer, or a `usize`.
    P,
    I8,
    I32,
    F32,
}

impl K {
    fn ty(self, ptr: Type) -> Type {
        match self {
            K::P => ptr,
            K::I8 => types::I8,
            K::I32 => types::I32,
            K::F32 => types::F32,
        }
    }
}

/// A runtime function the compiled code may call.
struct Helper {
    name: &'static str,
    f: *const u8,
    params: &'static [K],
    ret: Option<K>,
}

// SAFETY: a function pointer and static tables.
unsafe impl Sync for Helper {}

macro_rules! ret {
    () => {
        None
    };
    ($r:ident) => {
        Some(K::$r)
    };
}

macro_rules! helpers {
    ($($name:ident($($p:ident),*) $(-> $r:ident)?;)*) => {
        const HELPERS: &[Helper] = &[$(Helper {
            name: stringify!($name),
            f: runtime::$name as *const u8,
            params: &[$(K::$p),*],
            ret: ret!($($r)?),
        }),*];
    };
}

helpers! {
    rd_f32(P, P, P) -> F32;
    rd_bf16(P, P, P) -> F32;
    rd_f16(P, P, P) -> F32;
    rd_fp8(P, P, P) -> F32;
    rd_u32(P, P, P) -> I32;
    rd_u8(P, P, P) -> I32;
    rd_form(P, P, P, P) -> F32;
    dot_f32(P, P, P, P, P, P, P) -> F32;
    dot_bf16(P, P, P, P, P, P, P) -> F32;
    dot_form(P, P, P, P, P, P, P, P) -> F32;
    mix_f32(P, P, P, F32, F32, P, P, P, P);
    attend_f32(P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, F32, P);
    stage_f32(P, P, P, P, I32, F32, P, P, P);
    dots_f32(P, P, P, P, P, P, P, P, P, P, P, P, P);
    dots_bf16(P, P, P, P, P, P, P, P, P, P, P, P, P);
    st_f32(P, P, P, F32);
    st_bf16(P, P, P, F32);
    st_fp8(P, P, P, F32);
    st_u32(P, P, P, I32);
    word(P, P, P) -> I32;
    half(P, P, P) -> I32;
    exp(F32) -> F32;
    tanh(F32) -> F32;
    erf(F32) -> F32;
    log(F32) -> F32;
    cos(F32) -> F32;
    sin(F32) -> F32;
    max(F32, F32) -> F32;
    pow(F32, F32) -> F32;
    clamp(F32, F32, F32) -> F32;
    f16(I32, I32) -> F32;
    e4m3(I32, I32) -> F32;
    e2m1(I32, I32) -> F32;
    bf16_pair(F32, F32) -> I32;
    round_f16(F32) -> F32;
    round_e4m3(F32) -> F32;
    shfl_xor(P, P, P, P, P);
    sum32(P, P, P, P);
    tiles_new(P, P, P, P) -> P;
    tiles_free(P);
    tiles_zero(P);
    tiles_per(P) -> P;
    tiles_row(P, P, P) -> I32;
    tiles_scale(P, P, P, F32);
    tiles_each(P, P, P, I8, P, P, P) -> I8;
    tiles_block(P, P, P, P, P, P, P, P, P, P, P, P, P, P);
    tiles_hold(P, P, P, P, P, P, P);
    tiles_block_held(P, P, P, P, P, P, P, P, P, P);
}

fn helper(name: &str) -> &'static Helper {
    HELPERS
        .iter()
        .find(|h| h.name == name)
        .unwrap_or_else(|| panic!("no helper `{name}`"))
}

/// The signature of helper `name`, with pointers `ptr` wide.
fn signature(name: &str, ptr: Type, call_conv: cranelift_codegen::isa::CallConv) -> Signature {
    let h = helper(name);
    let mut sig = Signature::new(call_conv);
    for p in h.params {
        sig.params.push(AbiParam::new(p.ty(ptr)));
    }
    if let Some(r) = h.ret {
        sig.returns.push(AbiParam::new(r.ty(ptr)));
    }
    sig
}

fn module() -> Result<JITModule, String> {
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
    let isa = cranelift_native::builder()
        .map_err(|e| e.to_string())?
        .finish(settings::Flags::new(flags))
        .map_err(|e| e.to_string())?;
    let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
    for h in HELPERS {
        builder.symbol(h.name, h.f);
    }
    Ok(JITModule::new(builder))
}

/// `body` as host code. Compiled once per distinct text.
pub fn compile(body: &Body) -> Result<Entry, String> {
    static CACHE: OnceLock<Mutex<HashMap<u64, Result<Entry, String>>>> = OnceLock::new();
    let key = hash(&body.text().render(Dialect::Cuda));
    let cache = CACHE.get_or_init(Mutex::default);
    if let Some(found) = cache.lock().expect("compile cache").get(&key) {
        return found.clone();
    }
    let made = build(body);
    cache
        .lock()
        .expect("compile cache")
        .insert(key, made.clone());
    made
}

fn build(body: &Body) -> Result<Entry, String> {
    let mut module = module()?;
    let ptr = module.target_config().pointer_type();
    let mut ctx = module.make_context();
    for _ in 0..3 {
        ctx.func.signature.params.push(AbiParam::new(ptr));
    }
    let mut fctx = cranelift_frontend::FunctionBuilderContext::new();
    lower::function(&mut module, &mut ctx.func, &mut fctx, body)?;
    let id = module
        .declare_function(&body.entry, Linkage::Export, &ctx.func.signature)
        .map_err(|e| e.to_string())?;
    module
        .define_function(id, &mut ctx)
        .map_err(|e| format!("{}: {e:?}", body.entry))?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let code = module.get_finalized_function(id);
    // The code stays mapped for the process's life.
    std::mem::forget(module);
    // SAFETY: declared above with a context and a group range.
    Ok(unsafe { std::mem::transmute::<*const u8, Entry>(code) })
}
