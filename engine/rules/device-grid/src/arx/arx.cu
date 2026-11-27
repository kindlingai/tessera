#include <tessera/1/core>

using namespace tessera;

namespace arx {

constexpr u64 SEND_BYTES = 32ull << 20;
constexpr u64 SLOT_BYTES = 16ull << 20;
constexpr u32 MAX_RANKS = 8;
constexpr u64 FAULT_BIT = 1ull << 63;
constexpr u32 PIECES = 8;
constexpr u64 ONE_PIECE_SLICE = 256ull << 10;
constexpr u64 FLAG_STRIDE = 16;
constexpr u32 RING_CAP = 64;
constexpr u64 FP8_CHUNK_BYTES = 1152ull;
constexpr u64 FP8_SCALE_OFF = 1024ull;

enum class Wire { F32, Bf16, Fp8 };

constexpr u64 MODE_FULL = 0;
constexpr u64 MODE_GATHER = 2;
constexpr u64 MODE_SCATTER_PIECE = 3;
constexpr u64 MODE_GATHER_PIECE = 4;

// State buffer slots: 0=seq_dev, 1=go, 2=pub, 3=work_next, 4..8=Timing.
constexpr u32 ST_SEQ = 0;
constexpr u32 ST_GO = 1;
constexpr u32 ST_PUB = 2;
constexpr u32 ST_WORK = 3;
constexpr u32 ST_TIMING = 4;

TESSERA_FN prim::Timing* timing_of(u64* state) {
    return reinterpret_cast<prim::Timing*>(state + ST_TIMING);
}

// Block counters: RS stage, AG publish, one-shot, finish.
constexpr u32 BLK_ONE = 16;
constexpr u32 BLK_DONE = 17;

struct work {
    u64 op;
    u64 seq;
    u64 piece;
    u64 slice;
    u64 lo;
    u64 hi;
    u64 tag;
};

struct ctl {
    u64 seq;
    u64 bytes[2];
    u64 mode[2];
    u64 stop;
    u64 done;
    u64 err;
    u64 gpu_err;
    u64 pad[7];
    work ring[RING_CAP];
};

#if defined(TESSERA_CUDA)
TESSERA_FN uint4 load_cv4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}
TESSERA_FN float rcp_f32(float x) {
    float y;
    asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
#else
TESSERA_FN uint4 load_cv4(const void* p) {
    const u16* q = reinterpret_cast<const u16*>(p);
    uint2 a = prim::load_fresh4(q);
    uint2 b = prim::load_fresh4(q + 4);
    return make_uint4(a.x, a.y, b.x, b.y);
}
TESSERA_FN float rcp_f32(float x) { return 1.0f / x; }
#endif

TESSERA_FN u64 recv_off(u64 par, u32 src, u32 nr) {
    return (par * nr + src) * SLOT_BYTES;
}

TESSERA_FN u64 flag_of(u64 seq, u32 piece) {
    return seq * FLAG_STRIDE + u64(piece) + 1ull;
}

TESSERA_FN void piece_bounds(u64 slice, u32 p, u64& lo, u64& hi) {
    u64 step = ((slice / PIECES) + 15ull) & ~15ull;
    lo = min(slice, step * u64(p));
    hi = p == PIECES - 1u ? slice : min(slice, step * u64(p + 1u));
}

TESSERA_FN void piece_bounds_fp8(u64 slice_pkts, u32 p, u64& p_lo, u64& p_hi) {
    u64 step = (slice_pkts + u64(PIECES) - 1ull) / u64(PIECES);
    p_lo = min(slice_pkts, step * u64(p));
    p_hi = p == PIECES - 1u ? slice_pkts : min(slice_pkts, step * u64(p + 1u));
}

TESSERA_FN bool check_abort(ctl* c, u64 deadline, u32 spin) {
    if ((spin & 1023u) != 0u) return false;
    if (prim::load_acquire_sys(&c->err) != 0ull) {
        prim::store_release_sys(&c->gpu_err, 1ull);
        return true;
    }
    if (prim::load_acquire_sys(&c->gpu_err) != 0ull) return true;
    if (deadline != 0ull && prim::clock_ticks() > deadline) {
        prim::store_release_sys(&c->gpu_err, 2ull);
        return true;
    }
    return false;
}

// Block 0 polls NIC flags, then releases peers via `go`.
TESSERA_FN bool wait_flags(ctl* c, const u64* flag, u64* state, u64 want, u32 me, u32 nr,
                           u32 lanes, u64 timeout) {
    __shared__ int bad;
    if (prim::leader_thread()) bad = 0;
    prim::block_sync();
    u64 deadline = timeout != 0ull ? prim::clock_ticks() + timeout : 0ull;
    if (prim::leader_block()) {
        u32 total = nr * lanes;
        if (threadIdx.x < total && (threadIdx.x / lanes) != me) {
            for (u32 spin = 1; prim::load_acquire_sys(flag + threadIdx.x) < want; spin++) {
                if (check_abort(c, deadline, spin)) {
                    atomicExch(&bad, 1);
                    break;
                }
                prim::pause();
            }
        }
        if (prim::leader_thread()) {
            for (u32 spin = 1; prim::load_acquire_gpu64(state + ST_PUB) < want; spin++) {
                if (check_abort(c, deadline, spin)) {
                    atomicExch(&bad, 1);
                    break;
                }
                prim::pause();
            }
        }
        prim::block_sync();
        if (prim::leader_thread()) {
            prim::timing_mark(timing_of(state));
            u64 val = bad != 0 ? (want | FAULT_BIT) : want;
            prim::store_release_gpu64(state + ST_GO, val);
        }
        return bad == 0;
    }
    if (prim::leader_thread()) {
        for (u32 spin = 1;; spin++) {
            u64 g = prim::load_acquire_gpu64(state + ST_GO);
            if ((g & ~FAULT_BIT) >= want) {
                if ((g & FAULT_BIT) != 0ull) bad = 1;
                break;
            }
            if (check_abort(c, deadline, spin)) {
                bad = 1;
                break;
            }
            prim::pause();
        }
    }
    prim::block_sync();
    // System-scope reads order recv after NIC writes.
    if (bad == 0 && threadIdx.x < nr * lanes && (threadIdx.x / lanes) != me) {
        for (u32 spin = 1; prim::load_acquire_sys(flag + threadIdx.x) < want; spin++) {
            if (check_abort(c, deadline, spin)) {
                atomicExch(&bad, 1);
                break;
            }
            prim::pause();
        }
    }
    prim::block_sync();
    return bad == 0;
}

TESSERA_FN void publish_piece(ctl* c, u32* blocks, u64* state, u32 slot, u64 op, u64 seq, u32 p,
                              u64 slice, u64 lo, u64 hi) {
    prim::fence_sys();
    prim::block_sync();
    if (prim::leader_thread() && prim::fetch_add_gpu(blocks + slot, 1u) == prim::blocks() - 1u) {
        prim::store_relaxed_gpu(blocks + slot, 0u);
        unsigned long long idx =
            atomicAdd(reinterpret_cast<unsigned long long*>(state + ST_WORK), 1ull);
        work& w = c->ring[idx % RING_CAP];
        w.op = op;
        w.seq = seq;
        w.piece = p;
        w.slice = slice;
        w.lo = lo;
        w.hi = hi;
        prim::store_release_sys(&w.tag, u64(idx) + 1ull);
        prim::store_release_gpu64(state + ST_PUB, flag_of(seq, p));
    }
}

TESSERA_FN void finish_seq(ctl* c, u32* blocks, u64* state, u64 seq) {
    prim::block_sync();
    if (prim::leader_thread() &&
        prim::fetch_add_gpu(blocks + BLK_DONE, 1u) == prim::blocks() - 1u) {
        prim::store_relaxed_gpu(blocks + BLK_DONE, 0u);
        prim::timing_end(timing_of(state));
        state[ST_SEQ] = seq;
        prim::store_release_sys(&c->done, seq);
    }
}

TESSERA_FN uint4 pack_bf16x8(float4 a, float4 b) {
    return make_uint4(prim::to_bf16x2(a.x, a.y), prim::to_bf16x2(a.z, a.w),
                      prim::to_bf16x2(b.x, b.y), prim::to_bf16x2(b.z, b.w));
}

TESSERA_FN void unpack_bf16x8(uint4 pk, float4& a, float4& b) {
    a = make_float4(prim::from_bits(pk.x << 16), prim::from_bits(pk.x & 0xFFFF0000u),
                    prim::from_bits(pk.y << 16), prim::from_bits(pk.y & 0xFFFF0000u));
    b = make_float4(prim::from_bits(pk.z << 16), prim::from_bits(pk.z & 0xFFFF0000u),
                    prim::from_bits(pk.w << 16), prim::from_bits(pk.w & 0xFFFF0000u));
}

TESSERA_FN void load_f32x16(const float* src, u64 blk, bool aligned, float4& v0, float4& v1,
                            float4& v2, float4& v3) {
    if (aligned) {
        const float4* p = reinterpret_cast<const float4*>(src) + (blk << 2);
        v0 = p[0];
        v1 = p[1];
        v2 = p[2];
        v3 = p[3];
    } else {
        const float* p = src + (blk << 4);
        v0 = make_float4(p[0], p[1], p[2], p[3]);
        v1 = make_float4(p[4], p[5], p[6], p[7]);
        v2 = make_float4(p[8], p[9], p[10], p[11]);
        v3 = make_float4(p[12], p[13], p[14], p[15]);
    }
}

TESSERA_FN void store_f32x16(float* dst, u64 blk, bool aligned, float4 v0, float4 v1, float4 v2,
                             float4 v3) {
    if (aligned) {
        float4* p = reinterpret_cast<float4*>(dst) + (blk << 2);
        p[0] = v0;
        p[1] = v1;
        p[2] = v2;
        p[3] = v3;
    } else {
        float* p = dst + (blk << 4);
        p[0] = v0.x; p[1] = v0.y; p[2] = v0.z; p[3] = v0.w;
        p[4] = v1.x; p[5] = v1.y; p[6] = v1.z; p[7] = v1.w;
        p[8] = v2.x; p[9] = v2.y; p[10] = v2.z; p[11] = v2.w;
        p[12] = v3.x; p[13] = v3.y; p[14] = v3.z; p[15] = v3.w;
    }
}

TESSERA_FN prim::Fp8Blk pack_fp8_16(float4 v0, float4 v1, float4 v2, float4 v3) {
    float amax = fmaxf(fmaxf(prim::max4_abs(v0), prim::max4_abs(v1)),
                       fmaxf(prim::max4_abs(v2), prim::max4_abs(v3)));
    u16 s = 0u;
    if (amax != 0.0f) {
        if (!(amax > 0.0f)) {
            s = 0x3F80u;
        } else {
            // 0x3B124925 is ceil_f32(1/448) so multiplication never clips.
            u32 bits = __float_as_uint(amax * __uint_as_float(0x3B124925u));
            if (bits >= 0x7F7F0000u) {
                s = 0x7F7Fu;
            } else {
                u32 up = (bits + 0xFFFFu) >> 16;
                s = static_cast<u16>(up < 0x0080u ? 0x0080u : up);
            }
        }
    }
    float inv = (s == 0u) ? 0.0f : rcp_f32(prim::from_bf16(s));
    uint4 q = make_uint4(prim::to_e4m3x4(prim::scale4(v0, inv)),
                         prim::to_e4m3x4(prim::scale4(v1, inv)),
                         prim::to_e4m3x4(prim::scale4(v2, inv)),
                         prim::to_e4m3x4(prim::scale4(v3, inv)));
    return {q, s};
}

// Writes one 16-float FP8 block to a wire chunk.
TESSERA_FN void store_fp8_blk(u8* base, u32 blk_idx, prim::Fp8Blk blk) {
    u8* p = base + u64(blk_idx >> 6) * FP8_CHUNK_BYTES;
    u32 b = blk_idx & 63u;
    reinterpret_cast<uint4*>(p)[b] = blk.q;
    reinterpret_cast<u16*>(p + FP8_SCALE_OFF)[b] = blk.s;
}

// Reads one 16-float FP8 block from a wire chunk.
TESSERA_FN prim::Fp8Blk load_fp8_blk(const u8* base, u32 blk_idx, bool is_me) {
    const u8* p = base + u64(blk_idx >> 6) * FP8_CHUNK_BYTES;
    u32 b = blk_idx & 63u;
    uint4 q = is_me ? reinterpret_cast<const uint4*>(p)[b] : load_cv4(p + u64(b) * 16ull);
    const u16* sp = reinterpret_cast<const u16*>(p + FP8_SCALE_OFF) + b;
    u16 s = is_me ? *sp : prim::load_fresh(sp);
    return {q, s};
}

// Sums one 16-float FP8 block across ranks in 4-rank batches.
TESSERA_FN void sum_fp8_blk(const u8* mine, const u8* recv, u64 par, u32 blk_idx, u32 me, u32 nr,
                            float4& a0, float4& a1, float4& a2, float4& a3) {
    a0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    a1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    a2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    a3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (u32 r0 = 0u; r0 < nr; r0 += 4u) {
        prim::Fp8Blk blk[4];
#pragma unroll
        for (u32 k = 0u; k < 4u; k++) {
            u32 r = r0 + k;
            if (r < nr) {
                const u8* p = r == me ? mine : recv + recv_off(par, r, nr);
                blk[k] = load_fp8_blk(p, blk_idx, r == me);
            }
        }
#pragma unroll
        for (u32 k = 0u; k < 4u; k++) {
            if (r0 + k < nr) {
                float4 v0, v1, v2, v3;
                prim::unpack_fp8_blk(blk[k], v0, v1, v2, v3);
                a0 = prim::add4(a0, v0);
                a1 = prim::add4(a1, v1);
                a2 = prim::add4(a2, v2);
                a3 = prim::add4(a3, v3);
            }
        }
    }
}

template <Wire W>
TESSERA_FN u64 stage_in(const float* src, u8* dst, u32 n) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    bool aligned = (reinterpret_cast<u64>(src) & 15ull) == 0ull;
    if constexpr (W == Wire::Fp8) {
        u32 n_chunks = n >> 10;
        u32 n_blks = n_chunks << 6;
        for (u32 i = tid; i < n_blks; i += stride) {
            float4 v0, v1, v2, v3;
            load_f32x16(src, i, aligned, v0, v1, v2, v3);
            store_fp8_blk(dst, i, pack_fp8_16(v0, v1, v2, v3));
        }
        u32 base = n_chunks << 10;
        float* tail = reinterpret_cast<float*>(dst + u64(n_chunks) * FP8_CHUNK_BYTES);
        for (u32 i = tid; i < n - base; i += stride) tail[i] = src[base + i];
        return u64(n_chunks) * FP8_CHUNK_BYTES + u64(n - base) * 4ull;
    } else if constexpr (W == Wire::Bf16) {
        u16* out = reinterpret_cast<u16*>(dst);
        u32 v8 = aligned ? (n >> 3) : 0u;
        for (u32 i = tid; i < v8; i += stride) {
            float4 a = reinterpret_cast<const float4*>(src)[i * 2];
            float4 b = reinterpret_cast<const float4*>(src)[i * 2 + 1];
            reinterpret_cast<uint4*>(out)[i] = pack_bf16x8(a, b);
        }
        for (u32 i = (v8 << 3) + tid; i < n; i += stride) {
            out[i] = prim::to_bf16(src[i]);
        }
        return u64(n) * 2ull;
    } else {
        float* out = reinterpret_cast<float*>(dst);
        u32 v4 = aligned ? (n >> 2) : 0u;
        for (u32 i = tid; i < v4; i += stride) {
            reinterpret_cast<float4*>(out)[i] = reinterpret_cast<const float4*>(src)[i];
        }
        for (u32 i = (v4 << 2) + tid; i < n; i += stride) {
            out[i] = src[i];
        }
        return u64(n) * 4ull;
    }
}

TESSERA_FN void add_bf16x8(float acc[8], uint4 pk) {
    u32 w[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
        acc[k * 2] += prim::from_bits(w[k] << 16);
        acc[k * 2 + 1] += prim::from_bits(w[k] & 0xFFFF0000u);
    }
}

TESSERA_FN void sum_bf16x8(const u8* mine, const u8* recv, u64 par, u32 idx, u32 me, u32 nr,
                           float acc[8]) {
    uint4 v[MAX_RANKS];
#pragma unroll
    for (u32 r = 0; r < MAX_RANKS; r++) {
        if (r < nr) {
            const u8* p = r == me ? mine : recv + recv_off(par, r, nr);
            v[r] = r == me ? reinterpret_cast<const uint4*>(p)[idx]
                           : load_cv4(p + u64(idx) * 16ull);
        }
    }
#pragma unroll
    for (int k = 0; k < 8; k++) acc[k] = 0.0f;
#pragma unroll
    for (u32 r = 0; r < MAX_RANKS; r++) {
        if (r < nr) add_bf16x8(acc, v[r]);
    }
}

TESSERA_FN float4 sum_f32x4(const u8* mine, const u8* recv, u64 par, u32 idx, u32 me, u32 nr) {
    float4 v[MAX_RANKS];
#pragma unroll
    for (u32 r = 0; r < MAX_RANKS; r++) {
        if (r < nr) {
            const float* p = reinterpret_cast<const float*>(
                r == me ? mine : recv + recv_off(par, r, nr));
            v[r] = r == me ? reinterpret_cast<const float4*>(p)[idx]
                           : prim::load_fresh4(p + u64(idx) * 4ull);
        }
    }
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (u32 r = 0; r < MAX_RANKS; r++) {
        if (r < nr) {
            acc.x += v[r].x;
            acc.y += v[r].y;
            acc.z += v[r].z;
            acc.w += v[r].w;
        }
    }
    return acc;
}

template <Wire W>
TESSERA_FN void reduce_range(const u8* mine, const u8* recv, u64 par, float* out, u32 n, u32 me,
                             u32 nr) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    bool aligned = (reinterpret_cast<u64>(out) & 15ull) == 0ull;
    if constexpr (W == Wire::Fp8) {
        u32 n_chunks = n >> 10;
        u32 n_blks = n_chunks << 6;
        for (u32 i = tid; i < n_blks; i += stride) {
            float4 a0, a1, a2, a3;
            sum_fp8_blk(mine, recv, par, i, me, nr, a0, a1, a2, a3);
            store_f32x16(out, i, aligned, a0, a1, a2, a3);
        }
        u32 base = n_chunks << 10;
        u64 tail_off = u64(n_chunks) * FP8_CHUNK_BYTES;
        for (u32 i = tid; i < n - base; i += stride) {
            float acc = 0.0f;
            for (u32 r = 0; r < nr; r++) {
                const float* p = reinterpret_cast<const float*>(
                    (r == me ? mine : recv + recv_off(par, r, nr)) + tail_off);
                acc += r == me ? p[i] : prim::load_fresh(p + i);
            }
            out[base + i] = acc;
        }
    } else if constexpr (W == Wire::Bf16) {
        u32 v8 = aligned ? (n >> 3) : 0u;
        for (u32 i = tid; i < v8; i += stride) {
            float acc[8];
            sum_bf16x8(mine, recv, par, i, me, nr, acc);
            reinterpret_cast<float4*>(out)[i * 2] = make_float4(acc[0], acc[1], acc[2], acc[3]);
            reinterpret_cast<float4*>(out)[i * 2 + 1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
        }
        for (u32 i = (v8 << 3) + tid; i < n; i += stride) {
            float acc = 0.0f;
            for (u32 r = 0; r < nr; r++) {
                const u16* p = reinterpret_cast<const u16*>(
                    r == me ? mine : recv + recv_off(par, r, nr));
                u16 h = r == me ? p[i] : prim::load_fresh(p + i);
                acc += prim::from_bits(u32(h) << 16);
            }
            out[i] = acc;
        }
    } else {
        u32 v4 = aligned ? (n >> 2) : 0u;
        for (u32 i = tid; i < v4; i += stride) {
            reinterpret_cast<float4*>(out)[i] = sum_f32x4(mine, recv, par, i, me, nr);
        }
        for (u32 i = (v4 << 2) + tid; i < n; i += stride) {
            float acc = 0.0f;
            for (u32 r = 0; r < nr; r++) {
                const float* p = reinterpret_cast<const float*>(
                    r == me ? mine : recv + recv_off(par, r, nr));
                acc += r == me ? p[i] : prim::load_fresh(p + i);
            }
            out[i] = acc;
        }
    }
}

template <Wire W>
TESSERA_FN void one_body(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p, u64* state, u32* blocks,
                         u64 from_p, u64 to_p, u32 n, u32 me, u32 nr, u32 lanes, u64 timeout) {
    if (prim::leader()) prim::timing_begin(timing_of(state));
    u64 seq = state[ST_SEQ] + 1ull;
    u64 par = seq & 1ull;
    u8* mine = reinterpret_cast<u8*>(send_p) + par * SEND_BYTES;
    const u8* recv = reinterpret_cast<const u8*>(recv_p);
    ctl* c = reinterpret_cast<ctl*>(ctl_p);
    u64 bytes = stage_in<W>(reinterpret_cast<const float*>(from_p), mine, n);
    publish_piece(c, blocks, state, BLK_ONE, MODE_FULL, seq, PIECES - 1u, 0ull, 0ull, bytes);
    if (!wait_flags(c, reinterpret_cast<const u64*>(flag_p), state, flag_of(seq, PIECES - 1u), me,
                    nr, lanes, timeout))
        return;
    reduce_range<W>(mine, recv, par, reinterpret_cast<float*>(to_p), n, me, nr);
    finish_seq(c, blocks, state, seq);
}

// Stages scatter pieces to send0 and publishes RS work.
template <Wire W>
TESSERA_FN void stage_scatter(ctl* c, u32* blocks, u64* state, u8* send0, const float* src,
                              u64 seq0, u64 slice, u64 slice_chunks, u32 v_per_chunk, u32 nr,
                              bool single_piece) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    u32 n_pieces = single_piece ? 1u : PIECES;
    for (u32 pi = 0u; pi < n_pieces; pi++) {
        u32 p = single_piece ? (PIECES - 1u) : pi;
        u64 lo, hi;
        if constexpr (W == Wire::Fp8) {
            u64 p_lo = 0ull, p_hi = slice_chunks;
            if (!single_piece) piece_bounds_fp8(slice_chunks, p, p_lo, p_hi);
            lo = p_lo * FP8_CHUNK_BYTES;
            hi = p_hi * FP8_CHUNK_BYTES;
            u32 b_lo = u32(p_lo << 6);
            u32 b_hi = u32(p_hi << 6);
            for (u32 r = 0u; r < nr; r++) {
                const float4* src_r = reinterpret_cast<const float4*>(src) + u64(r) * v_per_chunk;
                u8* send_r = send0 + u64(r) * slice;
                for (u32 j = b_lo + tid; j < b_hi; j += stride) {
                    const float4* sp = src_r + (u64(j) << 2);
                    store_fp8_blk(send_r, j, pack_fp8_16(sp[0], sp[1], sp[2], sp[3]));
                }
            }
        } else {
            lo = 0ull;
            hi = slice;
            if (!single_piece) piece_bounds(slice, p, lo, hi);
            u32 v_lo = u32(lo >> 4);
            u32 v_cnt = u32((hi - lo) >> 4);
            u32 total = v_cnt * nr;
            for (u32 q = tid; q < total; q += stride) {
                u32 r = q / v_cnt;
                u32 j = v_lo + (q - r * v_cnt);
                u32 idx = r * v_per_chunk + j;
                if constexpr (W == Wire::Bf16) {
                    float4 a = reinterpret_cast<const float4*>(src)[idx * 2];
                    float4 b = reinterpret_cast<const float4*>(src)[idx * 2 + 1];
                    *reinterpret_cast<uint4*>(send0 + u64(r) * slice + u64(j) * 16ull) =
                        pack_bf16x8(a, b);
                } else {
                    *reinterpret_cast<float4*>(send0 + u64(r) * slice + u64(j) * 16ull) =
                        reinterpret_cast<const float4*>(src)[idx];
                }
            }
        }
        publish_piece(c, blocks, state, p, MODE_SCATTER_PIECE, seq0, p, slice, lo, hi);
    }
}

// Unpacks arrived 8-piece gather slices into rank-strided output rows.
template <Wire W>
TESSERA_FN bool unpack_gather(ctl* c, const u64* flag, u64* state, const u8* send, const u8* recv,
                              float* dst, u64 seq, u64 par, u64 slice, u64 slice_chunks,
                              u32 dst_v_stride, u32 me, u32 nr, u32 lanes, u64 timeout) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    for (u32 p = 0u; p < PIECES; p++) {
        u64 lo, hi;
        if constexpr (W == Wire::Fp8) {
            u64 p_lo, p_hi;
            piece_bounds_fp8(slice_chunks, p, p_lo, p_hi);
            if (!wait_flags(c, flag, state, flag_of(seq, p), me, nr, lanes, timeout)) return false;
            u32 b_lo = u32(p_lo << 6);
            u32 b_hi = u32(p_hi << 6);
            for (u32 r = 0u; r < nr; r++) {
                bool is_me = (r == me);
                const u8* rp = is_me ? send : recv + recv_off(par, r, nr);
                float4* dst_r = reinterpret_cast<float4*>(dst) + u64(r) * dst_v_stride;
                for (u32 j = b_lo + tid; j < b_hi; j += stride) {
                    float4 v0, v1, v2, v3;
                    prim::unpack_fp8_blk(load_fp8_blk(rp, j, is_me), v0, v1, v2, v3);
                    float4* dp = dst_r + (u64(j) << 2);
                    dp[0] = v0;
                    dp[1] = v1;
                    dp[2] = v2;
                    dp[3] = v3;
                }
            }
        } else {
            piece_bounds(slice, p, lo, hi);
            if (!wait_flags(c, flag, state, flag_of(seq, p), me, nr, lanes, timeout)) return false;
            u32 v_lo = u32(lo >> 4);
            u32 v_cnt = u32((hi - lo) >> 4);
            u32 total = v_cnt * nr;
            for (u32 q = tid; q < total; q += stride) {
                u32 r = q / v_cnt;
                u32 j = v_lo + (q - r * v_cnt);
                u64 idx = u64(r) * dst_v_stride + j;
                const u8* rp = r == me ? send : recv + recv_off(par, r, nr);
                if constexpr (W == Wire::Bf16) {
                    uint4 pk = r == me ? reinterpret_cast<const uint4*>(rp)[j]
                                       : load_cv4(rp + u64(j) * 16ull);
                    float4 a, b;
                    unpack_bf16x8(pk, a, b);
                    reinterpret_cast<float4*>(dst)[idx * 2ull] = a;
                    reinterpret_cast<float4*>(dst)[idx * 2ull + 1ull] = b;
                } else {
                    const float* fp = reinterpret_cast<const float*>(rp);
                    float4 v = r == me ? reinterpret_cast<const float4*>(fp)[j]
                                       : prim::load_fresh4(fp + u64(j) * 4ull);
                    reinterpret_cast<float4*>(dst)[idx] = v;
                }
            }
        }
    }
    return true;
}

// Pipelined reduce-scatter, optionally followed by all-gather.
template <Wire W, bool GATHER>
TESSERA_FN void scatter_body(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p, u64* state,
                             u32* blocks, u64 from_p, u64 to_p, u32 n, u32 me, u32 nr, u32 lanes,
                             u64 timeout) {
    if (prim::leader()) prim::timing_begin(timing_of(state));
    u64 seq0 = state[ST_SEQ] + 1ull;
    u64 seq1 = seq0 + 1ull;
    u64 par0 = seq0 & 1ull;
    u64 par1 = seq1 & 1ull;
    u8* send0 = reinterpret_cast<u8*>(send_p) + par0 * SEND_BYTES;
    u8* send1 = reinterpret_cast<u8*>(send_p) + par1 * SEND_BYTES;
    const u8* recv = reinterpret_cast<const u8*>(recv_p);
    const u64* flag = reinterpret_cast<const u64*>(flag_p);
    ctl* c = reinterpret_cast<ctl*>(ctl_p);
    const float* src = reinterpret_cast<const float*>(from_p);
    float* dst = reinterpret_cast<float*>(to_p);
    u32 chunk = n / nr;
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    u64 slice_chunks = u64(chunk >> 10);
    u64 slice = (W == Wire::Fp8) ? (slice_chunks * FP8_CHUNK_BYTES)
                                 : (u64(chunk) * (W == Wire::Bf16 ? 2ull : 4ull));
    u32 v_per_chunk = (W == Wire::Bf16) ? (chunk >> 3) : (chunk >> 2);
    bool single_piece = !GATHER && slice <= ONE_PIECE_SLICE;
    stage_scatter<W>(c, blocks, state, send0, src, seq0, slice, slice_chunks, v_per_chunk, nr,
                     single_piece);

    const u8* own_rs = send0 + u64(me) * slice;
    u32 n_pieces = single_piece ? 1u : PIECES;
    for (u32 pi = 0u; pi < n_pieces; pi++) {
        u32 p = single_piece ? (PIECES - 1u) : pi;
        u64 lo = 0ull, hi = 0ull;
        if constexpr (W == Wire::Fp8) {
            u64 p_lo = 0ull, p_hi = slice_chunks;
            if (!single_piece) piece_bounds_fp8(slice_chunks, p, p_lo, p_hi);
            lo = p_lo * FP8_CHUNK_BYTES;
            hi = p_hi * FP8_CHUNK_BYTES;
            if (!wait_flags(c, flag, state, flag_of(seq0, p), me, nr, lanes, timeout)) return;
            u32 b_lo = u32(p_lo << 6);
            u32 b_hi = u32(p_hi << 6);
            for (u32 j = b_lo + tid; j < b_hi; j += stride) {
                float4 a0, a1, a2, a3;
                sum_fp8_blk(own_rs, recv, par0, j, me, nr, a0, a1, a2, a3);
                if constexpr (GATHER) {
                    store_fp8_blk(send1, j, pack_fp8_16(a0, a1, a2, a3));
                } else {
                    float4* dp = reinterpret_cast<float4*>(dst) + (u64(j) << 2);
                    dp[0] = a0;
                    dp[1] = a1;
                    dp[2] = a2;
                    dp[3] = a3;
                }
            }
        } else {
            lo = 0ull;
            hi = slice;
            if (!single_piece) piece_bounds(slice, p, lo, hi);
            if (!wait_flags(c, flag, state, flag_of(seq0, p), me, nr, lanes, timeout)) return;
            u32 v_lo = u32(lo >> 4);
            u32 v_hi = u32(hi >> 4);
            for (u32 j = v_lo + tid; j < v_hi; j += stride) {
                if constexpr (W == Wire::Bf16) {
                    float acc[8];
                    sum_bf16x8(own_rs, recv, par0, j, me, nr, acc);
                    float4 lo4 = make_float4(acc[0], acc[1], acc[2], acc[3]);
                    float4 hi4 = make_float4(acc[4], acc[5], acc[6], acc[7]);
                    if constexpr (GATHER) {
                        *reinterpret_cast<uint4*>(send1 + u64(j) * 16ull) = pack_bf16x8(lo4, hi4);
                    } else {
                        reinterpret_cast<float4*>(dst)[j * 2] = lo4;
                        reinterpret_cast<float4*>(dst)[j * 2 + 1] = hi4;
                    }
                } else {
                    float4 acc = sum_f32x4(own_rs, recv, par0, j, me, nr);
                    if constexpr (GATHER) {
                        *reinterpret_cast<float4*>(send1 + u64(j) * 16ull) = acc;
                    } else {
                        reinterpret_cast<float4*>(dst)[j] = acc;
                    }
                }
            }
        }
        if constexpr (GATHER) {
            publish_piece(c, blocks, state, PIECES + p, MODE_GATHER_PIECE, seq1, p, slice, lo, hi);
        }
    }

    if constexpr (!GATHER) {
        finish_seq(c, blocks, state, seq0);
        return;
    }

    if (!unpack_gather<W>(c, flag, state, send1, recv, dst, seq1, par1, slice, slice_chunks,
                          v_per_chunk, me, nr, lanes, timeout))
        return;
    finish_seq(c, blocks, state, seq1);
}

// Pipelined 8-piece contiguous row gather over any wire dtype.
template <Wire W>
TESSERA_FN void gather_rows_body(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p, u64* state,
                                 u32* blocks, u64 from_p, u64 to_p, u32 part_stride, u32 n,
                                 u32 me, u32 nr, u32 lanes, u64 timeout) {
    if (prim::leader()) prim::timing_begin(timing_of(state));
    u64 seq = state[ST_SEQ] + 1ull;
    u64 par = seq & 1ull;
    u8* send = reinterpret_cast<u8*>(send_p) + par * SEND_BYTES;
    const u8* recv = reinterpret_cast<const u8*>(recv_p);
    const u64* flag = reinterpret_cast<const u64*>(flag_p);
    ctl* c = reinterpret_cast<ctl*>(ctl_p);
    const float* src = reinterpret_cast<const float*>(from_p);
    float* dst = reinterpret_cast<float*>(to_p);
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    u64 slice_chunks = u64(n >> 10);
    u64 slice = (W == Wire::Fp8) ? (slice_chunks * FP8_CHUNK_BYTES)
                                 : (u64(n) * (W == Wire::Bf16 ? 2ull : 4ull));
    u32 dst_v_stride = (W == Wire::Bf16) ? (part_stride >> 3) : (part_stride >> 2);

    for (u32 p = 0u; p < PIECES; p++) {
        u64 lo = 0ull, hi = 0ull;
        if constexpr (W == Wire::Fp8) {
            u64 p_lo, p_hi;
            piece_bounds_fp8(slice_chunks, p, p_lo, p_hi);
            lo = p_lo * FP8_CHUNK_BYTES;
            hi = p_hi * FP8_CHUNK_BYTES;
            u32 b_lo = u32(p_lo << 6);
            u32 b_hi = u32(p_hi << 6);
            for (u32 j = b_lo + tid; j < b_hi; j += stride) {
                const float4* sp = reinterpret_cast<const float4*>(src) + (u64(j) << 2);
                store_fp8_blk(send, j, pack_fp8_16(sp[0], sp[1], sp[2], sp[3]));
            }
        } else {
            piece_bounds(slice, p, lo, hi);
            u32 v_lo = u32(lo >> 4);
            u32 v_hi = u32(hi >> 4);
            for (u32 j = v_lo + tid; j < v_hi; j += stride) {
                if constexpr (W == Wire::Bf16) {
                    float4 a = reinterpret_cast<const float4*>(src)[u64(j) * 2ull];
                    float4 b = reinterpret_cast<const float4*>(src)[u64(j) * 2ull + 1ull];
                    *reinterpret_cast<uint4*>(send + u64(j) * 16ull) = pack_bf16x8(a, b);
                } else {
                    *reinterpret_cast<float4*>(send + u64(j) * 16ull) =
                        reinterpret_cast<const float4*>(src)[j];
                }
            }
        }
        publish_piece(c, blocks, state, p, MODE_GATHER_PIECE, seq, p, slice, lo, hi);
    }

    if (!unpack_gather<W>(c, flag, state, send, recv, dst, seq, par, slice, slice_chunks,
                          dst_v_stride, me, nr, lanes, timeout))
        return;
    finish_seq(c, blocks, state, seq);
}

// Unpacks one rank's wire buffer into a contiguous float destination.
template <Wire W>
TESSERA_FN void unpack_one_rank(const u8* rp, float* dst_r, u32 n, bool is_me, bool aligned) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    if constexpr (W == Wire::Fp8) {
        u32 n_chunks = n >> 10;
        u32 n_blks = n_chunks << 6;
        for (u32 j = tid; j < n_blks; j += stride) {
            float4 v0, v1, v2, v3;
            prim::unpack_fp8_blk(load_fp8_blk(rp, j, is_me), v0, v1, v2, v3);
            store_f32x16(dst_r, j, aligned, v0, v1, v2, v3);
        }
        u32 base = n_chunks << 10;
        const float* tail = reinterpret_cast<const float*>(rp + u64(n_chunks) * FP8_CHUNK_BYTES);
        for (u32 i = tid; i < n - base; i += stride) {
            dst_r[base + i] = is_me ? tail[i] : prim::load_fresh(tail + i);
        }
    } else if constexpr (W == Wire::Bf16) {
        u32 v8 = aligned ? (n >> 3) : 0u;
        for (u32 j = tid; j < v8; j += stride) {
            uint4 pk = is_me ? reinterpret_cast<const uint4*>(rp)[j]
                             : load_cv4(rp + u64(j) * 16ull);
            float4 a, b;
            unpack_bf16x8(pk, a, b);
            reinterpret_cast<float4*>(dst_r)[u64(j) * 2ull] = a;
            reinterpret_cast<float4*>(dst_r)[u64(j) * 2ull + 1ull] = b;
        }
        const u16* hp = reinterpret_cast<const u16*>(rp);
        for (u32 i = (v8 << 3) + tid; i < n; i += stride) {
            u16 h = is_me ? hp[i] : prim::load_fresh(hp + i);
            dst_r[i] = prim::from_bits(u32(h) << 16);
        }
    } else {
        const float* sp = reinterpret_cast<const float*>(rp);
        u32 v4 = aligned ? (n >> 2) : 0u;
        for (u32 j = tid; j < v4; j += stride) {
            float4 v = is_me ? reinterpret_cast<const float4*>(sp)[j]
                             : prim::load_fresh4(sp + u64(j) * 4ull);
            reinterpret_cast<float4*>(dst_r)[j] = v;
        }
        for (u32 i = (v4 << 2) + tid; i < n; i += stride) {
            dst_r[i] = is_me ? sp[i] : prim::load_fresh(sp + i);
        }
    }
}

// One-shot contiguous row gather over any wire dtype.
template <Wire W>
TESSERA_FN void gather_rows_one_body(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p, u64* state,
                                     u32* blocks, u64 from_p, u64 to_p, u32 part_stride, u32 n,
                                     u32 me, u32 nr, u32 lanes, u64 timeout) {
    if (prim::leader()) prim::timing_begin(timing_of(state));
    u64 seq = state[ST_SEQ] + 1ull;
    u64 par = seq & 1ull;
    u8* mine_b = reinterpret_cast<u8*>(send_p) + par * SEND_BYTES;
    const u8* recv = reinterpret_cast<const u8*>(recv_p);
    ctl* c = reinterpret_cast<ctl*>(ctl_p);
    u64 bytes = stage_in<W>(reinterpret_cast<const float*>(from_p), mine_b, n);
    publish_piece(c, blocks, state, BLK_ONE, MODE_GATHER, seq, PIECES - 1u, 0ull, 0ull, bytes);
    if (!wait_flags(c, reinterpret_cast<const u64*>(flag_p), state, flag_of(seq, PIECES - 1u), me,
                    nr, lanes, timeout))
        return;
    float* out = reinterpret_cast<float*>(to_p);
    bool aligned = (part_stride & 3u) == 0u && (reinterpret_cast<u64>(out) & 15ull) == 0ull;
    for (u32 r = 0u; r < nr; r++) {
        bool is_me = (r == me);
        const u8* rp = is_me ? mine_b : recv + recv_off(par, r, nr);
        unpack_one_rank<W>(rp, out + u64(r) * part_stride, n, is_me, aligned);
    }
    finish_seq(c, blocks, state, seq);
}

template <Wire W>
TESSERA_FN void two_body(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p, u64* state, u32* blocks,
                         u64 from_p, u64 to_p, u32 n, u32 me, u32 nr, u32 lanes, u64 timeout) {
    scatter_body<W, true>(send_p, recv_p, flag_p, ctl_p, state, blocks, from_p, to_p, n, me, nr,
                          lanes, timeout);
}

template <Wire W>
TESSERA_FN void rs_body(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p, u64* state, u32* blocks,
                        u64 from_p, u64 to_p, u32 n, u32 me, u32 nr, u32 lanes, u64 timeout) {
    scatter_body<W, false>(send_p, recv_p, flag_p, ctl_p, state, blocks, from_p, to_p, n, me, nr,
                           lanes, timeout);
}

}  // namespace arx

#ifdef ARX_MIN_BLOCKS
#define ARX_BOUNDS __launch_bounds__(256, ARX_MIN_BLOCKS)
#else
#define ARX_BOUNDS
#endif

#define ARX_DEF(NAME, BODY, WIRE)                                                              \
    extern "C" __global__ ARX_BOUNDS void NAME(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p,  \
                                               u64* state, u32* blocks, u64 from_p, u64 to_p,  \
                                               u32 n, u32 me, u32 nr, u32 lanes, u64 timeout) { \
        arx::BODY<arx::Wire::WIRE>(send_p, recv_p, flag_p, ctl_p, state, blocks, from_p, to_p, \
                                   n, me, nr, lanes, timeout);                                 \
    }

ARX_DEF(tessera_arx_one_bf16, one_body, Bf16)
ARX_DEF(tessera_arx_one_f32, one_body, F32)
ARX_DEF(tessera_arx_one_fp8, one_body, Fp8)
ARX_DEF(tessera_arx_two_bf16, two_body, Bf16)
ARX_DEF(tessera_arx_two_f32, two_body, F32)
ARX_DEF(tessera_arx_two_fp8, two_body, Fp8)
ARX_DEF(tessera_arx_rs_bf16, rs_body, Bf16)
ARX_DEF(tessera_arx_rs_f32, rs_body, F32)
ARX_DEF(tessera_arx_rs_fp8, rs_body, Fp8)

#define ARX_GR_DEF(NAME, BODY, WIRE)                                                            \
    extern "C" __global__ ARX_BOUNDS void NAME(                                                 \
        u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p, u64* state, u32* blocks, u64 from_p,     \
        u64 to_p, u32 part_stride, u32 n, u32 me, u32 nr, u32 lanes, u64 timeout) {             \
        arx::BODY<arx::Wire::WIRE>(send_p, recv_p, flag_p, ctl_p, state, blocks, from_p, to_p,  \
                                   part_stride, n, me, nr, lanes, timeout);                     \
    }

ARX_GR_DEF(tessera_arx_gr_bf16, gather_rows_body, Bf16)
ARX_GR_DEF(tessera_arx_gr_f32, gather_rows_body, F32)
ARX_GR_DEF(tessera_arx_gr_fp8, gather_rows_body, Fp8)
ARX_GR_DEF(tessera_arx_gr1_bf16, gather_rows_one_body, Bf16)
ARX_GR_DEF(tessera_arx_gr1_f32, gather_rows_one_body, F32)
ARX_GR_DEF(tessera_arx_gr1_fp8, gather_rows_one_body, Fp8)

// Even-width row-major all-gather over F32.
extern "C" __global__ void tessera_arx_allgather(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p,
                                                 u64* state, u32* blocks, u64 from_p, u64 to_p,
                                                 u32 rows, u32 width, u32 me, u32 nr, u32 lanes,
                                                 u64 timeout) {
    if (prim::leader()) prim::timing_begin(arx::timing_of(state));
    u64 seq = state[arx::ST_SEQ] + 1ull;
    u64 par = seq & 1ull;
    u8* mine_b = reinterpret_cast<u8*>(send_p) + par * arx::SEND_BYTES;
    const u8* recv = reinterpret_cast<const u8*>(recv_p);
    arx::ctl* c = reinterpret_cast<arx::ctl*>(ctl_p);
    u32 n = rows * width;
    arx::stage_in<arx::Wire::F32>(reinterpret_cast<const float*>(from_p), mine_b, n);
    arx::publish_piece(c, blocks, state, arx::BLK_ONE, arx::MODE_GATHER, seq, arx::PIECES - 1u,
                       0ull, 0ull, u64(n) * 4ull);
    if (!arx::wait_flags(c, reinterpret_cast<const u64*>(flag_p), state,
                         arx::flag_of(seq, arx::PIECES - 1u), me, nr, lanes, timeout))
        return;
    const float* mine = reinterpret_cast<const float*>(mine_b);
    float* out = reinterpret_cast<float*>(to_p);
    u32 cols = width * nr;
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    bool aligned = (width & 3u) == 0u && (reinterpret_cast<u64>(out) & 15ull) == 0ull;
    if (aligned) {
        u32 w4 = width >> 2;
        u32 tot4 = rows * w4 * nr;
        for (u32 q = tid; q < tot4; q += stride) {
            u32 row = q / (w4 * nr);
            u32 rem = q - row * (w4 * nr);
            u32 src = rem / w4;
            u32 c4 = rem - src * w4;
            const float* sp = src == me ? mine
                                        : reinterpret_cast<const float*>(
                                              recv + arx::recv_off(par, src, nr));
            u64 off = u64(row) * width + u64(c4) * 4ull;
            float4 v = src == me ? *reinterpret_cast<const float4*>(sp + off)
                                 : prim::load_fresh4(sp + off);
            *reinterpret_cast<float4*>(out + u64(row) * cols + u64(src) * width + u64(c4) * 4ull) =
                v;
        }
    } else {
        u32 tot = rows * cols;
        for (u32 i = tid; i < tot; i += stride) {
            u32 row = i / cols;
            u32 col = i - row * cols;
            u32 src = col / width;
            u32 j = col - src * width;
            const float* sp = src == me ? mine
                                        : reinterpret_cast<const float*>(
                                              recv + arx::recv_off(par, src, nr));
            u64 off = u64(row) * width + j;
            out[i] = src == me ? sp[off] : prim::load_fresh(sp + off);
        }
    }
    arx::finish_seq(c, blocks, state, seq);
}
