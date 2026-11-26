#include <tessera/1/core>

using namespace tessera;

namespace arx {

constexpr u64 SEND_BYTES = 32ull << 20;
constexpr u64 SLOT_BYTES = 16ull << 20;
constexpr u32 MAX_RANKS = 8;
constexpr u64 FAULT_BIT = 1ull << 63;
constexpr u32 PIECES = 8;
constexpr u64 FLAG_STRIDE = 16;
constexpr u32 RING_CAP = 64;
constexpr u32 FP8_CHUNK_U4 = 72;
constexpr u64 FP8_CHUNK_BYTES = 1152ull;

enum class Wire { F32, Bf16, Fp8 };

union Fp8Chunk {
    uint4 u4[FP8_CHUNK_U4];
    struct {
        u32 q[256];
        u16 s[64];
    } f;
};

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
TESSERA_FN float shfl_xor_f32(float v, u32 mask) { return __shfl_xor_sync(0xFFFFFFFFu, v, mask); }
#else
TESSERA_FN uint4 load_cv4(const void* p) {
    const u16* q = reinterpret_cast<const u16*>(p);
    uint2 a = prim::load_fresh4(q);
    uint2 b = prim::load_fresh4(q + 4);
    return make_uint4(a.x, a.y, b.x, b.y);
}
TESSERA_FN float shfl_xor_f32(float v, u32 mask) { return __shfl_xor(v, mask); }
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
            }
        }
        if (prim::leader_thread()) {
            for (u32 spin = 1; prim::load_acquire_gpu64(state + ST_PUB) < want; spin++) {
                if (check_abort(c, deadline, spin)) {
                    atomicExch(&bad, 1);
                    break;
                }
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

TESSERA_FN float4 load_f32x4(const float* src, u32 j, bool aligned) {
    if (aligned) return reinterpret_cast<const float4*>(src)[j];
    const float* p = src + (j << 2);
    return make_float4(p[0], p[1], p[2], p[3]);
}

TESSERA_FN void store_f32x4(float* dst, u32 j, bool aligned, float4 v) {
    if (aligned) {
        reinterpret_cast<float4*>(dst)[j] = v;
    } else {
        float* p = dst + (j << 2);
        p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w;
    }
}

// Packs 1024 floats via shared memory into 72 uint4s.
TESSERA_FN void store_chunk_fp8(Fp8Chunk& sm, u8* base, u32 chunk_idx, float4 v) {
    float amax = prim::max4_abs(v);
    amax = fmaxf(amax, shfl_xor_f32(amax, 1u));
    amax = fmaxf(amax, shfl_xor_f32(amax, 2u));
    u16 s = prim::scale_bf16(amax);
    float inv = (s == 0u) ? 0.0f : (1.0f / prim::from_bf16(s));
    sm.f.q[threadIdx.x] = prim::to_e4m3x4(prim::scale4(v, inv));
    if ((threadIdx.x & 3u) == 0u) sm.f.s[threadIdx.x >> 2] = s;
    prim::block_sync();
    uint4* dst = reinterpret_cast<uint4*>(base + u64(chunk_idx) * FP8_CHUNK_BYTES);
    if (threadIdx.x < FP8_CHUNK_U4) dst[threadIdx.x] = sm.u4[threadIdx.x];
    prim::block_sync();
}

// Unpacks 1024 floats across 256 threads from 72 coalesced uint4s.
TESSERA_FN float4 load_chunk_fp8(Fp8Chunk& sm, const u8* base, u32 chunk_idx, bool is_me) {
    const uint4* src = reinterpret_cast<const uint4*>(base + u64(chunk_idx) * FP8_CHUNK_BYTES);
    if (threadIdx.x < FP8_CHUNK_U4) {
        sm.u4[threadIdx.x] = is_me ? src[threadIdx.x] : load_cv4(src + threadIdx.x);
    }
    prim::block_sync();
    float4 v = prim::scale4(prim::from_e4m3x4(sm.f.q[threadIdx.x]),
                            prim::from_bf16(sm.f.s[threadIdx.x >> 2]));
    prim::block_sync();
    return v;
}

// Loads all ranks' 1024-float chunks and sums to FP32.
TESSERA_FN float4 sum_chunk_fp8(Fp8Chunk sm[MAX_RANKS], const u8* mine, const u8* recv, u64 par,
                                u32 chunk_idx, u32 me, u32 nr) {
    u32 total = nr * FP8_CHUNK_U4;
    for (u32 k = threadIdx.x; k < total; k += blockDim.x) {
        u32 r = k / FP8_CHUNK_U4;
        u32 w = k - r * FP8_CHUNK_U4;
        const uint4* rp = reinterpret_cast<const uint4*>(
            (r == me ? mine : recv + recv_off(par, r, nr)) + u64(chunk_idx) * FP8_CHUNK_BYTES);
        sm[r].u4[w] = (r == me) ? rp[w] : load_cv4(rp + w);
    }
    prim::block_sync();
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (u32 r = 0; r < nr; r++) {
        float4 v = prim::scale4(prim::from_e4m3x4(sm[r].f.q[threadIdx.x]),
                                prim::from_bf16(sm[r].f.s[threadIdx.x >> 2]));
        acc = prim::add4(acc, v);
    }
    prim::block_sync();
    return acc;
}

template <Wire W>
TESSERA_FN u64 stage_in(const float* src, u8* dst, u32 n) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = gridDim.x * blockDim.x;
    bool aligned = (reinterpret_cast<u64>(src) & 15ull) == 0ull;
    if constexpr (W == Wire::Fp8) {
        __shared__ Fp8Chunk sm;
        u32 n_chunks = n >> 10;
        for (u32 c = blockIdx.x; c < n_chunks; c += gridDim.x) {
            float4 v = load_f32x4(src, (c << 8) + threadIdx.x, aligned);
            store_chunk_fp8(sm, dst, c, v);
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
        __shared__ Fp8Chunk sm[MAX_RANKS];
        u32 n_chunks = n >> 10;
        for (u32 c = blockIdx.x; c < n_chunks; c += gridDim.x) {
            float4 acc = sum_chunk_fp8(sm, mine, recv, par, c, me, nr);
            store_f32x4(out, (c << 8) + threadIdx.x, aligned, acc);
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

// Pipelined two-shot reduce-scatter + all-gather.
template <Wire W>
TESSERA_FN void two_body(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p, u64* state, u32* blocks,
                         u64 from_p, u64 to_p, u32 n, u32 me, u32 nr, u32 lanes, u64 timeout) {
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
    __shared__ Fp8Chunk sm[MAX_RANKS];

    // Stage pieces to send0 and publish RS work.
    for (u32 p = 0u; p < PIECES; p++) {
        u64 lo, hi;
        if constexpr (W == Wire::Fp8) {
            u64 p_lo, p_hi;
            piece_bounds_fp8(slice_chunks, p, p_lo, p_hi);
            lo = p_lo * FP8_CHUNK_BYTES;
            hi = p_hi * FP8_CHUNK_BYTES;
            u32 cnt = u32(p_hi - p_lo);
            u32 total = cnt * nr;
            for (u32 q = blockIdx.x; q < total; q += gridDim.x) {
                u32 r = q / cnt;
                u32 j = u32(p_lo) + (q - r * cnt);
                float4 v = reinterpret_cast<const float4*>(src)[r * v_per_chunk + (j << 8) +
                                                                threadIdx.x];
                store_chunk_fp8(sm[0], send0 + u64(r) * slice, j, v);
            }
        } else {
            piece_bounds(slice, p, lo, hi);
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

    // Reduce arrived RS pieces into send1 and publish AG.
    const u8* own_rs = send0 + u64(me) * slice;
    for (u32 p = 0u; p < PIECES; p++) {
        u64 lo, hi;
        if constexpr (W == Wire::Fp8) {
            u64 p_lo, p_hi;
            piece_bounds_fp8(slice_chunks, p, p_lo, p_hi);
            lo = p_lo * FP8_CHUNK_BYTES;
            hi = p_hi * FP8_CHUNK_BYTES;
            if (!wait_flags(c, flag, state, flag_of(seq0, p), me, nr, lanes, timeout)) return;
            for (u32 j = u32(p_lo) + blockIdx.x; j < u32(p_hi); j += gridDim.x) {
                float4 acc = sum_chunk_fp8(sm, own_rs, recv, par0, j, me, nr);
                store_chunk_fp8(sm[0], send1, j, acc);
            }
        } else {
            piece_bounds(slice, p, lo, hi);
            if (!wait_flags(c, flag, state, flag_of(seq0, p), me, nr, lanes, timeout)) return;
            u32 v_lo = u32(lo >> 4);
            u32 v_hi = u32(hi >> 4);
            for (u32 j = v_lo + tid; j < v_hi; j += stride) {
                if constexpr (W == Wire::Bf16) {
                    float acc[8];
                    sum_bf16x8(own_rs, recv, par0, j, me, nr, acc);
                    *reinterpret_cast<uint4*>(send1 + u64(j) * 16ull) =
                        pack_bf16x8(make_float4(acc[0], acc[1], acc[2], acc[3]),
                                    make_float4(acc[4], acc[5], acc[6], acc[7]));
                } else {
                    *reinterpret_cast<float4*>(send1 + u64(j) * 16ull) =
                        sum_f32x4(own_rs, recv, par0, j, me, nr);
                }
            }
        }
        publish_piece(c, blocks, state, PIECES + p, MODE_GATHER_PIECE, seq1, p, slice, lo, hi);
    }

    // Phase 3: unpack arrived AG pieces into output buffer.
    for (u32 p = 0u; p < PIECES; p++) {
        u64 lo, hi;
        if constexpr (W == Wire::Fp8) {
            u64 p_lo, p_hi;
            piece_bounds_fp8(slice_chunks, p, p_lo, p_hi);
            if (!wait_flags(c, flag, state, flag_of(seq1, p), me, nr, lanes, timeout)) return;
            u32 cnt = u32(p_hi - p_lo);
            u32 total = cnt * nr;
            for (u32 q = blockIdx.x; q < total; q += gridDim.x) {
                u32 r = q / cnt;
                u32 j = u32(p_lo) + (q - r * cnt);
                const u8* rp = r == me ? send1 : recv + recv_off(par1, r, nr);
                float4 v = load_chunk_fp8(sm[0], rp, j, r == me);
                reinterpret_cast<float4*>(dst)[r * v_per_chunk + (j << 8) + threadIdx.x] = v;
            }
        } else {
            piece_bounds(slice, p, lo, hi);
            if (!wait_flags(c, flag, state, flag_of(seq1, p), me, nr, lanes, timeout)) return;
            u32 v_lo = u32(lo >> 4);
            u32 v_cnt = u32((hi - lo) >> 4);
            u32 total = v_cnt * nr;
            for (u32 q = tid; q < total; q += stride) {
                u32 r = q / v_cnt;
                u32 j = v_lo + (q - r * v_cnt);
                u32 idx = r * v_per_chunk + j;
                const u8* rp = r == me ? send1 : recv + recv_off(par1, r, nr);
                if constexpr (W == Wire::Bf16) {
                    uint4 pk = r == me ? reinterpret_cast<const uint4*>(rp)[j]
                                       : load_cv4(rp + u64(j) * 16ull);
                    float4 a, b;
                    unpack_bf16x8(pk, a, b);
                    reinterpret_cast<float4*>(dst)[idx * 2] = a;
                    reinterpret_cast<float4*>(dst)[idx * 2 + 1] = b;
                } else {
                    const float* fp = reinterpret_cast<const float*>(rp);
                    float4 v = r == me ? reinterpret_cast<const float4*>(fp)[j]
                                       : prim::load_fresh4(fp + u64(j) * 4ull);
                    reinterpret_cast<float4*>(dst)[idx] = v;
                }
            }
        }
    }
    finish_seq(c, blocks, state, seq1);
}

}  // namespace arx

#define ARX_DEF(NAME, BODY, WIRE)                                                              \
    extern "C" __global__ void NAME(u64 send_p, u64 recv_p, u64 flag_p, u64 ctl_p, u64* state, \
                                    u32* blocks, u64 from_p, u64 to_p, u32 n, u32 me, u32 nr,  \
                                    u32 lanes, u64 timeout) {                                  \
        arx::BODY<arx::Wire::WIRE>(send_p, recv_p, flag_p, ctl_p, state, blocks, from_p, to_p, \
                                   n, me, nr, lanes, timeout);                                 \
    }

ARX_DEF(tessera_arx_one_bf16, one_body, Bf16)
ARX_DEF(tessera_arx_one_f32, one_body, F32)
ARX_DEF(tessera_arx_one_fp8, one_body, Fp8)
ARX_DEF(tessera_arx_two_bf16, two_body, Bf16)
ARX_DEF(tessera_arx_two_f32, two_body, F32)
ARX_DEF(tessera_arx_two_fp8, two_body, Fp8)

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
