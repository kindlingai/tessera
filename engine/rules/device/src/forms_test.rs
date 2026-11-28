use crate::{Backend, Options, flatten};
use tessera_codegen::block::{LoadCode, LoadScale};
use tessera_codegen::prelude::*;
use tessera_codegen::trellis::{LoadBook, book};
use tessera_codegen::{Precision, form::F32_FORM};
use tessera_gpukit::backend::{OnGpu, dispatch, picked};
use tessera_gpukit::device::{Device, DeviceBuffer, cached_context};
use tessera_gpukit::launch::{KernelArgs, Launch};
use tessera_graph::{E2M1, E4M3, E8M0, F16};
use tessera_host::exec::{self, ArgVal, Grid};
use tessera_mem::block::{Code, Crumb, I8, Nibble, Scale};
use tessera_mem::trellis::Codebook;

const HAND: &str = r#"#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

// pmcgx pairs, then mcg and mul1 pairs `(s, 65535 - s)`.
extern "C" __global__ void trellis_states(u32* out) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    out[s] = trellis::decode_pmcgx(s);
    if (s < 65536u) {
        out[1048576u + s] = trellis::decode_pair<trellis::BOOK_MCG>(s, 65535u - s);
        out[1114112u + s] = trellis::decode_pair<trellis::BOOK_MUL1>(s, 65535u - s);
    }
}
"#;
/// pmcgx pairs, then mcg and mul1 pairs.
const WORDS: usize = (1 << 20) + 2 * (1 << 16);

/// Both weights of each of `n` states.
#[kernel]
fn weights<B: LoadBook>(mut out: BufMut<F32>, n: u32) {
    let s = gid();
    if s < n {
        out[2 * s as u64] = B::weight(s, 0);
        out[2 * s as u64 + 1] = B::weight(s, 1);
    }
}

/// The hand decode's words, on a CUDA device.
struct Hand(String);

impl OnGpu for Hand {
    type Out = Vec<u32>;

    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Vec<u32> {
        let ctx = cached_context::<D>().expect("a device");
        let kernel = ctx.compile(&self.0, "trellis_states").expect("compiles");
        let out = DeviceBuffer::alloc(ctx, 4 * WORDS).expect("allocates");
        let mut args = KernelArgs::new();
        args.device_ptr(out.device_ptr());
        let grid = ((1 << 20) / 256, 1, 1);
        ctx.launch(&kernel, grid, (256, 1, 1), 0, &mut args)
            .expect("launches");
        ctx.finish().expect("finishes");
        let mut bytes = vec![0; 4 * WORDS];
        out.read_bytes(&mut bytes).expect("reads");
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|w| u32::from_le_bytes(*w)).collect()
    }
}

/// Every state: kernel text and hand GPU match the host.
#[test]
fn trellis_states_match() {
    let cuda_text = flatten(
        "trellis_states.cu",
        HAND,
        &|_| None,
        &Options::plain(Backend::Cuda),
    )
    .expect("flattens cuda")
    .text;
    let rocm_text = flatten(
        "trellis_states.kernel",
        HAND,
        &|_| None,
        &Options::plain(Backend::Rocm),
    )
    .expect("flattens rocm")
    .text;
    tessera_gpukit::testing::require_gpu();
    let hand = match picked() {
        Some(Backend::Cuda) => dispatch(Some(Backend::Cuda), Hand(cuda_text)).ok(),
        Some(Backend::Rocm) => dispatch(Some(Backend::Rocm), Hand(rocm_text)).ok(),
        _ => None,
    };
    let half = |w: u32, high: bool| f32::from(F16((w >> (16 * u32::from(high))) as u16));
    macro_rules! each_book {
        ($(($b:ident, $name:literal, $suffix:literal, $bits:literal, $pairs:literal,
            $weight:ident),)*) => {$({
            let codebook = Codebook::$b;
            let n = 1u32 << $bits;
            let mut out = vec![0u8; 8 * n as usize];
            let args = [
                ArgVal::Buf { at: 0, skip: 0, form: F32_FORM },
                ArgVal::U32(n),
            ];
            let grid = Grid { groups: n as usize / 4096, threads: 4096 };
            let body = weights_ir::<book::$b>();
            exec::run(&body, &args, &mut [&mut out], grid, LANES, Precision::EXACT);
            let (got, _) = out.as_chunks::<4>();
            for s in 0..n {
                for odd in 0..2 {
                    let want = codebook.weight(s, odd == 1 && $pairs);
                    let kernel = f32::from_le_bytes(got[(2 * s + odd) as usize]);
                    assert_eq!(kernel.to_bits(), want.to_bits(), "{codebook:?} {s:#x} {odd}");
                }
                let Some(words) = &hand else { continue };
                let (at, other) = match codebook {
                    Codebook::Pmcgx => (s as usize, s),
                    Codebook::Mcg => (1 << 20 | s as usize, 65535 - s),
                    Codebook::Mul1 => ((1 << 20) + (1 << 16) + s as usize, 65535 - s),
                };
                let pair = [codebook.weight(s, false), codebook.weight(other, $pairs)];
                let cuda = [half(words[at], false), half(words[at], true)];
                let [cuda, pair] = [cuda, pair].map(|p| p.map(f32::to_bits));
                assert_eq!(cuda, pair, "CUDA {codebook:?} {s:#x}");
            }
        })*};
    }
    tessera_mem::codebooks!(each_book);
}

/// Each code at bit 4 of `s << 4 | 5`.
#[kernel]
fn codes<C: LoadCode>(mut out: BufMut<F32>, n: u32) {
    let s = gid();
    if s < n {
        out[s as u64] = C::field((s << 4) | 5, 4);
    }
}

/// Scale `s`, read from its bytes in `p`.
#[kernel]
fn scales<S: LoadScale>(mut out: BufMut<F32>, p: Buf<U8>, n: u32) {
    let s = gid();
    if s < n {
        let at = p + s as u64 * S::BYTES as u64;
        out[s as u64] = S::load(at);
    }
}

/// `n` floats from `body` on the host, after `bytes`.
fn host_run(body: &tessera_codegen::ir::Body, bytes: &mut [u8], n: u32) -> Vec<u32> {
    let mut out = vec![0u8; 4 * n as usize];
    let mut args = vec![
        ArgVal::Buf {
            at: 0,
            skip: 0,
            form: F32_FORM,
        },
        ArgVal::U32(n),
    ];
    if !bytes.is_empty() {
        args.insert(
            1,
            ArgVal::Buf {
                at: 1,
                skip: 0,
                form: <U8 as Form>::FORM,
            },
        );
    }
    let threads = (n as usize).min(4096);
    let grid = Grid {
        groups: n as usize / threads,
        threads,
    };
    exec::run(
        body,
        &args,
        &mut [&mut out, bytes],
        grid,
        LANES,
        Precision::EXACT,
    );
    let (got, _) = out.as_chunks::<4>();
    got.iter().map(|w| u32::from_le_bytes(*w)).collect()
}

/// Every code and scale word: kernel text matches the host.
#[test]
fn block_reads_match() {
    macro_rules! each_read {
        (scales: $(($s:ident, $repr:ty, $smax:literal, $get:ident)),*;
         codes: $(($c:ident, $bits:literal, $cmax:literal, $field:ident)),*;) => {
            $({
                let n = 1u32 << (8 * <$s as Scale>::BYTES);
                let mut bytes = Vec::new();
                for s in 0..n {
                    bytes.extend_from_slice(&s.to_le_bytes()[..<$s as Scale>::BYTES]);
                }
                let got = host_run(&scales_ir::<$s>(), &mut bytes, n);
                for s in 0..n as usize {
                    let at = s * <$s as Scale>::BYTES;
                    let want = <$s as Scale>::get(&bytes[at..]).to_bits();
                    assert_eq!(got[s], want, "{} {s:#x}", stringify!($s));
                }
            })*
            $({
                let got = host_run(&codes_ir::<$c>(), &mut [], 256);
                for s in 0..256u32 {
                    let want = <$c as Code>::decode((s << 4) | 5, 4).to_bits();
                    assert_eq!(got[s as usize], want, "{} {s:#x}", stringify!($c));
                }
            })*
        };
    }
    tessera_mem::block_reads!(each_read);
}
