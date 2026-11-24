use crate::{Backend, Options, flatten};
use tessera_codegen::prelude::*;
use tessera_codegen::trellis::{LoadBook, book};
use tessera_codegen::{Precision, form::F32_FORM};
use tessera_gpukit::backend::{OnGpu, available, dispatch};
use tessera_gpukit::device::{Device, DeviceBuffer, cached_context};
use tessera_gpukit::launch::{KernelArgs, Launch};
use tessera_graph::F16;
use tessera_host::exec::{self, ArgVal, Grid};
use tessera_mem::trellis::Codebook;

const HAND: &str = include_str!("trellis_states.cu");
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

/// Every state: kernel text and hand CUDA match the host.
#[test]
fn trellis_states_match() {
    let options = Options {
        backend: Backend::Cuda,
        remote_kv: false,
        marks: false,
    };
    let text = flatten("trellis_states.cu", HAND, &|_| None, &options)
        .expect("flattens")
        .text;
    let hand = match available() {
        Some(Backend::Cuda) => dispatch(Some(Backend::Cuda), Hand(text)).ok(),
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
