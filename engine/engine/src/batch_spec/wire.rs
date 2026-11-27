//! Wire codec for batched speculative round messages.

use super::Admitted;
use tessera_engine_comm::{Reader, Writer};
use tessera_engine_hooks::Request;
use tessera_engine_hooks::sample::Sampling;

/// Wire tag identifying a batched speculative round header.
const BATCH_STEP_TAG: u32 = 2;

pub(crate) fn encode_round(hash: u32, cancelled: &[u32], admitted: &[Admitted]) -> Vec<u8> {
    let mut w = Writer(Vec::new());
    w.u32(BATCH_STEP_TAG)
        .u32(hash)
        .u32s(cancelled)
        .u32(admitted.len() as u32);
    for a in admitted {
        w.u32(a.slot)
            .u32(a.skip as u32)
            .u32(a.request.max_tokens as u32)
            .u32s(&a.request.tokens)
            .u32s(&a.request.stop);
        a.request.sampling.encode(&mut w);
    }
    w.0
}

pub(crate) fn decode_round(bytes: &[u8]) -> Option<(u32, Vec<u32>, Vec<Admitted>)> {
    let mut r = Reader(bytes);
    if r.u32()? != BATCH_STEP_TAG {
        return None;
    }
    let hash = r.u32()?;
    let cancelled = r.u32s()?;
    let n = r.u32()? as usize;
    let mut admitted = Vec::with_capacity(n);
    for _ in 0..n {
        let slot = r.u32()?;
        let skip = r.u32()? as usize;
        let max_tokens = r.u32()? as usize;
        let tokens = r.u32s()?;
        let stop = r.u32s()?;
        let sampling = Sampling::decode(&mut r)?;
        admitted.push(Admitted {
            slot,
            skip,
            request: Request {
                tokens,
                max_tokens,
                stop,
                sampling,
                ..Request::default()
            },
            sink: None,
        });
    }
    Some((hash, cancelled, admitted))
}
