#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

constant uint TM_HEAD = 16u;
constant uint TM_INPUT_SIDE = 2u;
constant uint TM_SIDE_MASK = 0xFFFFu;
constant uint TM_PAGED = 0x80000000u;
constant uint TM_TABLE = 24u;
constant uint TM_MIX_TABLE = 24u;
constant uint TM_MIX_ROW = 16u;
constant uint TM_MCG = 0xCBAC1FEDu;
constant uint TM_MUL1 = 0x83DCD12Du;
constant uint TM_PMCG2 = 0xBCA14C51u;
constant float TM_INV_SQRT128 = 0.08838834764831844f;
constant uint TM_XS_STRIDE = 132u;
constant uint TM_WS_STRIDE = 20u;

enum TmBook : uint {
    TM_BOOK_MCG = 0u,
    TM_BOOK_MUL1 = 1u,
    TM_BOOK_PMCGX = 2u,
};

// Rounds four half values (|v| < 448) to E4M3.
static inline half4 tm_round_e4m3_4(half4 v) {
    ushort4 bits = as_type<ushort4>(v);
    ushort4 sign = bits & ushort4(0x8000u);
    ushort4 ab = bits & ushort4(0x7FFFu);
    ushort4 nudge = ushort4(0x003Fu) + ((ab >> ushort4(7u)) & ushort4(1u));
    ushort4 norm = (ab + nudge) & ushort4(0xFF80u);
    half4 sub = (as_type<half4>(ab) + half4(2.0h)) - half4(2.0h);
    ushort4 out = select(norm, as_type<ushort4>(sub), ab < ushort4(0x2400u));
    return as_type<half4>(sign | out);
}

// Register 128-point Walsh-Hadamard transform across one SIMD group.
static inline void tm_wht128(thread float v[4], uint lane, float scale) {
    float a0 = v[0] + v[1], a1 = v[0] - v[1];
    float a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2;
    v[1] = a1 + a3;
    v[2] = a0 - a2;
    v[3] = a1 - a3;
    for (uint mask = 1u; mask < 32u; mask <<= 1u) {
        float sign = (lane & mask) ? -1.0f : 1.0f;
        for (uint i = 0u; i < 4u; i++) {
            float peer = simd_shuffle_xor(v[i], ushort(mask));
            v[i] = peer + sign * v[i];
        }
    }
    for (uint i = 0u; i < 4u; i++) v[i] *= scale;
}

template <uint BOOK>
static inline half4 tm_decode_quad(uint a, uint b, uint c, uint d) {
    if constexpr (BOOK == TM_BOOK_MUL1) {
        uint4 z = uint4(a, b, c, d) * TM_MUL1;
        uint4 s = (z & 255u) + ((z >> 8u) & 255u) + ((z >> 16u) & 255u) + (z >> 24u);
        half4 sc = half4(as_type<half>(ushort(0x1EEEu)));
        half4 of = half4(as_type<half>(ushort(0xC931u)));
        return tm_round_e4m3_4(fma(half4(1024u + s), sc, of));
    } else {
        uint4 h = ((uint4(a, b, c, d) * TM_MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        half2 h0 = as_type<half2>(h.x), h1 = as_type<half2>(h.y);
        half2 h2 = as_type<half2>(h.z), h3 = as_type<half2>(h.w);
        half4 lo = half4(h0.x, h1.x, h2.x, h3.x);
        half4 hi = half4(h0.y, h1.y, h2.y, h3.y);
        return tm_round_e4m3_4(lo + hi);
    }
}

static inline half4 tm_decode_pmcgx2(uint w0, uint w1) {
    uint2 w = uint2(w0, w1);
    uint2 x = ((w * TM_MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint2 y = ((w * TM_PMCG2) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    half2 hx0 = as_type<half2>(x.x), hy0 = as_type<half2>(y.x);
    half2 hx1 = as_type<half2>(x.y), hy1 = as_type<half2>(y.y);
    return tm_round_e4m3_4(half4(hx0.x + hy0.y, hx0.y + hy0.x, hx1.x + hy1.y, hx1.y + hy1.x));
}

struct TmWordMap {
    ushort i0;
    ushort i1;
    uint sh;
};

template <uint K, uint BOOK>
static inline TmWordMap tm_word_map(uint lane) {
    constexpr uint ring = 256u * K;
    constexpr uint words = 8u * K;
    uint b0 = (BOOK == TM_BOOK_PMCGX) ? (2u * K * (4u * lane + 1u) + ring - 20u) % ring
                                      : ((8u * lane + 1u) * K + ring - 16u) % ring;
    uint i0 = b0 >> 5u;
    return {ushort(i0), ushort((i0 + 1u) % words), b0 & 31u};
}

template <uint BITS, uint OFF>
static inline uint tm_extract(uint hi, uint lo) {
    constexpr uint mask = (1u << BITS) - 1u;
    if constexpr (OFF == 0u) {
        return hi >> (32u - BITS);
    } else if constexpr (OFF <= 32u - BITS) {
        return (hi >> (32u - BITS - OFF)) & mask;
    } else {
        return ((hi << (OFF - (32u - BITS))) | (lo >> (64u - BITS - OFF))) & mask;
    }
}

// Decodes one SIMD-held tile word with pure 32-bit shifts.
template <uint K, uint BOOK>
static inline void tm_decode_tile(uint raw, TmWordMap wm, thread half4 &wa, thread half4 &wb) {
    uint w0 = simd_shuffle(raw, wm.i0);
    uint w1 = (K == 4u) ? raw : simd_shuffle(raw, wm.i1);
    uint hi, lo;
    if constexpr (K == 4u) {
        hi = (w0 << 20u) | (w1 >> 12u);
        lo = w1 << 20u;
    } else {
        uint sh = wm.sh;
        hi = (sh == 0u) ? w0 : ((w0 << sh) | (w1 >> (32u - sh)));
        lo = w1 << sh;
    }
    if constexpr (BOOK == TM_BOOK_PMCGX) {
        wa = tm_decode_pmcgx2(tm_extract<20u, 0u>(hi, lo), tm_extract<20u, 2u * K>(hi, lo));
        wb = tm_decode_pmcgx2(tm_extract<20u, 4u * K>(hi, lo), tm_extract<20u, 6u * K>(hi, lo));
    } else {
        uint s0 = tm_extract<16u, 0u>(hi, lo);
        uint s1 = tm_extract<16u, K>(hi, lo);
        uint s2 = tm_extract<16u, 2u * K>(hi, lo);
        uint s3 = tm_extract<16u, 3u * K>(hi, lo);
        uint s4 = tm_extract<16u, 4u * K>(hi, lo);
        uint s5 = tm_extract<16u, 5u * K>(hi, lo);
        uint s6 = tm_extract<16u, 6u * K>(hi, lo);
        uint s7 = tm_extract<16u, 7u * K>(hi, lo);
        wa = tm_decode_quad<BOOK>(s0, s1, s2, s3);
        wb = tm_decode_quad<BOOK>(s4, s5, s6, s7);
    }
}

// Weight blocks split `p` of `ks` reads for output `ob`.
struct TmWalk {
    uint blocks;
    uint first;
    ulong stride;
    device const uchar *at;
};

template <uint K>
static inline TmWalk tm_walk(device const uchar *w, uint n, uint k, uint ks, uint ob, uint p) {
    uint blocks = (k >> 7u) / ks;
    uint first = p * blocks;
    uint side = *(device const uint *)(w + 8u);
    bool input_side = (side & TM_SIDE_MASK) == TM_INPUT_SIDE;
    device const uchar *base =
        (side & TM_PAGED) ? (w + *(device const ulong *)(w + TM_TABLE)) : (w + TM_HEAD);
    ulong local = input_side ? (ulong(first) * (n >> 1u) + ulong(ob) * 64u)
                             : (ulong(ob) * (k >> 1u) + ulong(first) * 64u);
    ulong stride = (input_side ? ulong(n >> 1u) : 64ull) * ulong(32u * K);
    return {blocks, first, stride, base + local * (32u * K)};
}

// Scales four inputs by `sc`, then rotates by Hadamard.
static inline float4 tm_turn_in(float4 x, float4 sc, uint lane) {
    float v[4] = {x.x * sc.x, x.y * sc.y, x.z * sc.z, x.w * sc.w};
    tm_wht128(v, lane, TM_INV_SQRT128);
    return float4(v[0], v[1], v[2], v[3]);
}

// Rotates four outputs by Hadamard, then scales by `sc`.
static inline float4 tm_turn_out(float4 y, float4 sc, uint lane) {
    float v[4] = {y.x, y.y, y.z, y.w};
    tm_wht128(v, lane, TM_INV_SQRT128);
    return float4(v[0] * sc.x, v[1] * sc.y, v[2] * sc.z, v[3] * sc.w);
}

static inline void tm_put4(threadgroup float *dst, float4 v) {
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = v.w;
}

static inline float4 tm_get4(threadgroup const float *src) {
    return float4(src[0], src[1], src[2], src[3]);
}

// Lane roles and weight walk for a `K`, `BOOK` body.
#define TM_START(KS, OB, P)                                                                  \
    constexpr uint tw = 8u * K;                                                              \
    uint simd = tid >> 5u;                                                                   \
    uint lane = tid & 31u;                                                                   \
    uint cc = lane >> 2u;                                                                    \
    uint rr = lane & 3u;                                                                     \
    TmWalk walk = tm_walk<K>(w, n, k, KS, OB, P);                                            \
    device const uchar *gb = walk.at;                                                        \
    TmWordMap wm = tm_word_map<K, BOOK>(lane);

// A decode group's output block and split, from `tg_x`.
#define TM_DEC_START                                                                         \
    uint nblk = n >> 7u;                                                                     \
    uint ob = tg_x % nblk;                                                                   \
    uint p = tg_x / nblk;                                                                    \
    if (p >= ks) return;                                                                     \
    TM_START(ks, ob, p)

// Sums lane partials of `m` rows and stores each row.
static inline void tm_dec_finish(
    thread float *acc0,
    thread float *acc1,
    uint m,
    uint n,
    uint ks,
    uint p,
    uint ob,
    uint simd,
    uint lane,
    uint cc,
    uint rr,
    uint turned,
    device const float *svh,
    device float *out,
    threadgroup float xs[4][128]) {
    for (uint r = 0u; r < m; r++) {
        acc0[r] += simd_shuffle_xor(acc0[r], ushort(1u));
        acc0[r] += simd_shuffle_xor(acc0[r], ushort(2u));
        acc1[r] += simd_shuffle_xor(acc1[r], ushort(1u));
        acc1[r] += simd_shuffle_xor(acc1[r], ushort(2u));
    }

    uint lcol = (simd << 4u) + cc;
    if (ks > 1u) {
        if (rr == 0u) {
            device float *pp = out + (ulong(ob) * ks + p) * m * 128u;
            for (uint r = 0u; r < m; r++) {
                pp[ulong(r) * 128u + lcol] = acc0[r];
                pp[ulong(r) * 128u + lcol + 8u] = acc1[r];
            }
        }
    } else if (!turned) {
        if (rr == 0u) {
            uint ob0 = ob << 7u;
            for (uint r = 0u; r < m; r++) {
                out[ulong(r) * n + ob0 + lcol] = acc0[r];
                out[ulong(r) * n + ob0 + lcol + 8u] = acc1[r];
            }
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rr == 0u) {
            for (uint r = 0u; r < m; r++) {
                xs[r][lcol] = acc0[r];
                xs[r][lcol + 8u] = acc1[r];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd < m) {
            uint base = (ob << 7u) + (lane << 2u);
            float4 sc = *(device const float4 *)(svh + base);
            float4 y = tm_get4(&xs[simd][lane << 2u]);
            *(device float4 *)(out + ulong(simd) * n + base) = tm_turn_out(y, sc, lane);
        }
    }
}

// Stages four input blocks per barrier, then streams their tiles.
template <uint K, uint BOOK>
static inline void tm_dec1_run(
    device const float *x,
    device const float *scale,
    bool turned,
    TmWalk walk,
    TmWordMap wm,
    uint simd,
    uint lane,
    threadgroup float xs[4][128],
    thread float &acc0,
    thread float &acc1) {
    constexpr uint tw = 8u * K;
    uint rr = lane & 3u;
    device const uchar *gb = walk.at;
    for (uint c0 = 0u; c0 < walk.blocks; c0 += 4u) {
        uint nb = min(4u, walk.blocks - c0);
        if (c0 > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd < nb) {
            uint col = ((c0 + simd) << 7u) + (lane << 2u);
            float4 v = *(device const float4 *)(x + col);
            if (turned) v = tm_turn_in(v, *(device const float4 *)(scale + col), lane);
            tm_put4(&xs[simd][lane << 2u], v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = 0u; b < nb; b++) {
            device const uint *tiles = (device const uint *)gb;
            uint raw = (K == 4u || lane < tw) ? tiles[simd * tw + lane] : 0u;
            for (uint u = 0u; u < 8u; u++) {
                uint next = (u + 1u < 8u && (K == 4u || lane < tw))
                                ? tiles[((u + 1u) * 8u + simd) * tw + lane]
                                : 0u;
                half4 wa, wb;
                tm_decode_tile<K, BOOK>(raw, wm, wa, wb);
                raw = next;
                uint xb = u * 16u + (rr << 1u);
                float4 xv = float4(xs[b][xb], xs[b][xb + 1u], xs[b][xb + 8u], xs[b][xb + 9u]);
                acc0 += dot(xv, float4(wa));
                acc1 += dot(xv, float4(wb));
            }
            gb += walk.stride;
        }
    }
}

// Single-row tile-streaming decode, staging four input blocks per barrier.
template <uint K, uint BOOK>
static inline void tm_dec1_body(
    device const float *x,
    device const uchar *w,
    device const float *suh,
    device const float *svh,
    device float *out,
    uint n,
    uint k,
    uint ks,
    uint turned,
    uint tg_x,
    uint tid,
    threadgroup float xs[4][128]) {
    TM_DEC_START

    (void)gb; (void)tw;
    float acc0[1] = {0.0f}, acc1[1] = {0.0f};
    uint at = walk.first << 7u;
    tm_dec1_run<K, BOOK>(x + at, suh + at, turned != 0u, walk, wm, simd, lane, xs, acc0[0], acc1[0]);

    tm_dec_finish(acc0, acc1, 1u, n, ks, p, ob, simd, lane, cc, rr, turned, svh, out, xs);
}

// Multi-row tile-streaming decode for 2 to 4 rows.
template <uint K, uint BOOK>
static void tm_dec4_body(
    device const float *x,
    device const uchar *w,
    device const float *suh,
    device const float *svh,
    device float *out,
    uint m,
    uint n,
    uint k,
    uint ks,
    uint turned,
    uint tg_x,
    uint tid,
    threadgroup float xs[4][128]) {
    TM_DEC_START

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint b = 0u; b < walk.blocks; b++) {
        if (b > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd < m) {
            uint col = ((walk.first + b) << 7u) + (lane << 2u);
            float4 v = *(device const float4 *)(x + ulong(simd) * k + col);
            if (turned) v = tm_turn_in(v, *(device const float4 *)(suh + col), lane);
            tm_put4(&xs[simd][lane << 2u], v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const uint *tiles = (device const uint *)gb;
        for (uint u = 0u; u < 8u; u++) {
            uint raw = (K == 4u || lane < tw) ? tiles[(u * 8u + simd) * tw + lane] : 0u;
            half4 wa, wb;
            tm_decode_tile<K, BOOK>(raw, wm, wa, wb);
            uint xb = u * 16u + (rr << 1u);
            for (uint r = 0u; r < m; r++) {
                float x0 = xs[r][xb];
                float x1 = xs[r][xb + 1u];
                float x2 = xs[r][xb + 8u];
                float x3 = xs[r][xb + 9u];
                acc0[r] = fma(x0, float(wa.x), fma(x1, float(wa.y), fma(x2, float(wa.z), fma(x3, float(wa.w), acc0[r]))));
                acc1[r] = fma(x0, float(wb.x), fma(x1, float(wb.y), fma(x2, float(wb.z), fma(x3, float(wb.w), acc1[r]))));
            }
        }
        gb += walk.stride;
    }

    tm_dec_finish(acc0, acc1, m, n, ks, p, ob, simd, lane, cc, rr, turned, svh, out, xs);
}

// Prefill tile-streaming matmul with 8x8 f32 simdgroup MMA.
template <uint K, uint BOOK, uint TM>
static inline void tm_pf_body(
    device const float *x,
    device const uchar *w,
    device const float *suh,
    device const float *svh,
    device float *out,
    uint m,
    uint n,
    uint k,
    uint turned,
    uint tg_x,
    uint tg_y,
    uint tid,
    threadgroup float xs[TM * 8u][TM_XS_STRIDE],
    threadgroup float ws[8][16 * TM_WS_STRIDE]) {
    constexpr uint bm = TM * 8u;
    uint nblk = n >> 7u;
    uint ob = tg_x;
    uint r0 = tg_y * bm;
    if (ob >= nblk || r0 >= m) return;
    uint nr = min(bm, m - r0);
    TM_START(1u, ob, 0u)

    simdgroup_float8x8 acc[TM][2];
    for (uint tm = 0u; tm < TM; tm++) {
        acc[tm][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc[tm][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    threadgroup float *wown = ws[simd];
    uint wr0 = (rr << 1u) * TM_WS_STRIDE + cc;
    uint wr1 = wr0 + TM_WS_STRIDE;
    uint wr8 = wr0 + 8u * TM_WS_STRIDE;
    uint wr9 = wr8 + TM_WS_STRIDE;

    for (uint b = 0u; b < walk.blocks; b++) {
        if (b > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        uint col = (b << 7u) + (lane << 2u);
        float4 sc = turned ? *(device const float4 *)(suh + col) : float4(1.0f);
        for (uint r = simd; r < bm; r += 8u) {
            float4 v = float4(0.0f);
            if (r < nr) {
                v = *(device const float4 *)(x + ulong(r0 + r) * k + col);
                if (turned) v = tm_turn_in(v, sc, lane);
            }
            tm_put4(&xs[r][lane << 2u], v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const uint *tiles = (device const uint *)gb;
        for (uint u = 0u; u < 8u; u++) {
            uint raw = (K == 4u || lane < tw) ? tiles[(u * 8u + simd) * tw + lane] : 0u;
            half4 wa, wb;
            tm_decode_tile<K, BOOK>(raw, wm, wa, wb);
            wown[wr0] = float(wa.x);
            wown[wr1] = float(wa.y);
            wown[wr8] = float(wa.z);
            wown[wr9] = float(wa.w);
            wown[wr0 + 8u] = float(wb.x);
            wown[wr1 + 8u] = float(wb.y);
            wown[wr8 + 8u] = float(wb.z);
            wown[wr9 + 8u] = float(wb.w);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k8 = 0u; k8 < 16u; k8 += 8u) {
                simdgroup_float8x8 tb0, tb1;
                simdgroup_load(tb0, wown + k8 * TM_WS_STRIDE, TM_WS_STRIDE);
                simdgroup_load(tb1, wown + k8 * TM_WS_STRIDE + 8u, TM_WS_STRIDE);
                for (uint tm = 0u; tm < TM; tm++) {
                    simdgroup_float8x8 ta;
                    simdgroup_load(ta, &xs[tm * 8u][u * 16u + k8], TM_XS_STRIDE);
                    simdgroup_multiply_accumulate(acc[tm][0], ta, tb0, acc[tm][0]);
                    simdgroup_multiply_accumulate(acc[tm][1], ta, tb1, acc[tm][1]);
                }
            }
        }
        gb += walk.stride;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tm = 0u; tm < TM; tm++) {
        simdgroup_store(acc[tm][0], &xs[tm * 8u][simd * 16u], TM_XS_STRIDE);
        simdgroup_store(acc[tm][1], &xs[tm * 8u][simd * 16u + 8u], TM_XS_STRIDE);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = (ob << 7u) + (lane << 2u);
    float4 sc = turned ? *(device const float4 *)(svh + base) : float4(1.0f);
    for (uint r = simd; r < nr; r += 8u) {
        float4 y = tm_get4(&xs[r][lane << 2u]);
        if (turned) y = tm_turn_out(y, sc, lane);
        *(device float4 *)(out + ulong(r0 + r) * n + base) = y;
    }
}

#define TM_ARGS                                                                              \
    device const float *x [[buffer(0)]],                                                     \
    device const uchar *w [[buffer(1)]],                                                     \
    device const float *suh [[buffer(2)]],                                                   \
    device const float *svh [[buffer(3)]],                                                   \
    device float *out [[buffer(4)]],                                                         \
    constant uint &m [[buffer(5)]],                                                          \
    constant uint &n [[buffer(6)]],                                                          \
    constant uint &k [[buffer(7)]],                                                          \
    constant uint &ks [[buffer(8)]],                                                         \
    constant uint &bits [[buffer(9)]],                                                       \
    constant uint &book [[buffer(10)]],                                                      \
    constant uint &turned [[buffer(11)]],                                                    \
    uint3 tg [[threadgroup_position_in_grid]],                                               \
    uint tid [[thread_index_in_threadgroup]]

#define TM_MCG_D1(NAME, K)                                                                   \
    kernel void NAME(TM_ARGS) {                                                              \
        (void)m; (void)bits; (void)book;                                                     \
        threadgroup float xs[4][128];                                                        \
        tm_dec1_body<K, TM_BOOK_MCG>(x, w, suh, svh, out, n, k, ks, turned, tg.x, tid, xs);  \
    }

TM_MCG_D1(tm_k2_d1, 2u)
TM_MCG_D1(tm_k3_d1, 3u)
TM_MCG_D1(tm_k4_d1, 4u)

#define TM_MCG_PF(NAME, K, TM)                                                               \
    kernel void NAME(TM_ARGS) {                                                              \
        (void)ks; (void)bits; (void)book;                                                    \
        threadgroup float xs[TM * 8u][TM_XS_STRIDE];                                         \
        threadgroup float ws[8][16 * TM_WS_STRIDE];                                          \
        tm_pf_body<K, TM_BOOK_MCG, TM>(x, w, suh, svh, out, m, n, k, turned, tg.x, tg.y, tid, xs, ws); \
    }

TM_MCG_PF(tm_k2_pf16, 2u, 2u)
TM_MCG_PF(tm_k3_pf16, 3u, 2u)
TM_MCG_PF(tm_k4_pf16, 4u, 2u)
TM_MCG_PF(tm_k2_pf32, 2u, 4u)
TM_MCG_PF(tm_k3_pf32, 3u, 4u)
TM_MCG_PF(tm_k4_pf32, 4u, 4u)

#define TM_DISPATCH_K(FN, BOOK, ...)                                                         \
    if (bits == 2u) FN<2u, BOOK>(__VA_ARGS__);                                               \
    else if (bits == 3u) FN<3u, BOOK>(__VA_ARGS__);                                          \
    else FN<4u, BOOK>(__VA_ARGS__);

kernel void tm_gen_d1(TM_ARGS) {
    (void)m;
    threadgroup float xs[4][128];
    if (book == TM_BOOK_PMCGX) {
        TM_DISPATCH_K(tm_dec1_body, TM_BOOK_PMCGX, x, w, suh, svh, out, n, k, ks, turned, tg.x, tid, xs)
    } else if (book == TM_BOOK_MUL1) {
        TM_DISPATCH_K(tm_dec1_body, TM_BOOK_MUL1, x, w, suh, svh, out, n, k, ks, turned, tg.x, tid, xs)
    } else {
        TM_DISPATCH_K(tm_dec1_body, TM_BOOK_MCG, x, w, suh, svh, out, n, k, ks, turned, tg.x, tid, xs)
    }
}

kernel void tm_dec4(TM_ARGS) {
    threadgroup float xs[4][128];
    if (book == TM_BOOK_PMCGX) {
        TM_DISPATCH_K(tm_dec4_body, TM_BOOK_PMCGX, x, w, suh, svh, out, m, n, k, ks, turned, tg.x, tid, xs)
    } else if (book == TM_BOOK_MUL1) {
        TM_DISPATCH_K(tm_dec4_body, TM_BOOK_MUL1, x, w, suh, svh, out, m, n, k, ks, turned, tg.x, tid, xs)
    } else {
        TM_DISPATCH_K(tm_dec4_body, TM_BOOK_MCG, x, w, suh, svh, out, m, n, k, ks, turned, tg.x, tid, xs)
    }
}

#define TM_DISPATCH_PF(BOOK, TM)                                                             \
    if (bits == 2u)                                                                          \
        tm_pf_body<2u, BOOK, TM>(x, w, suh, svh, out, m, n, k, turned, tg.x, tg.y, tid, xs, ws); \
    else if (bits == 3u)                                                                     \
        tm_pf_body<3u, BOOK, TM>(x, w, suh, svh, out, m, n, k, turned, tg.x, tg.y, tid, xs, ws); \
    else                                                                                     \
        tm_pf_body<4u, BOOK, TM>(x, w, suh, svh, out, m, n, k, turned, tg.x, tg.y, tid, xs, ws);

#define TM_GEN_PF(NAME, TM)                                                                  \
    kernel void NAME(TM_ARGS) {                                                              \
        (void)ks;                                                                            \
        threadgroup float xs[TM * 8u][TM_XS_STRIDE];                                         \
        threadgroup float ws[8][16 * TM_WS_STRIDE];                                          \
        if (book == TM_BOOK_PMCGX) {                                                         \
            TM_DISPATCH_PF(TM_BOOK_PMCGX, TM)                                                \
        } else if (book == TM_BOOK_MUL1) {                                                   \
            TM_DISPATCH_PF(TM_BOOK_MUL1, TM)                                                 \
        } else {                                                                             \
            TM_DISPATCH_PF(TM_BOOK_MCG, TM)                                                  \
        }                                                                                    \
    }

TM_GEN_PF(tm_gen_pf16, 2u)
TM_GEN_PF(tm_gen_pf32, 4u)

// Sums split-K partials in order and applies output rotation.
kernel void tm_reduce(
    device const float *part [[buffer(0)]],
    device const float *svh [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &ks [[buffer(5)]],
    constant uint &turned [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint ob = tg.x;
    if (ob >= (n >> 7u)) return;
    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    uint base = (ob << 7u) + (lane << 2u);
    float4 sc = turned ? *(device const float4 *)(svh + base) : float4(1.0f);
    for (uint r = simd; r < m; r += 8u) {
        device const float *pb = part + ulong(ob) * ks * m * 128u + ulong(r) * 128u + (lane << 2u);
        float4 sum = float4(0.0f);
        for (uint p = 0u; p < ks; p++) {
            sum += *(device const float4 *)(pb + ulong(p) * m * 128u);
        }
        if (turned) sum = tm_turn_out(sum, sc, lane);
        *(device float4 *)(out + ulong(r) * n + base) = sum;
    }
}

// Weight blocks of one picked matrix for output `ob`.
template <uint K>
static inline TmWalk tm_picked_walk(
    device const uchar *w, device const uchar *mat, uint n, uint k, uint ob) {
    uint side = *(device const uint *)(w + 8u);
    bool input_side = (side & TM_SIDE_MASK) == TM_INPUT_SIDE;
    ulong local = input_side ? (ulong(ob) * 64u) : (ulong(ob) * (k >> 1u));
    ulong stride = (input_side ? ulong(n >> 1u) : 64ull) * ulong(32u * K);
    return {k >> 7u, 0u, stride, mat + local * (32u * K)};
}

// Matrix `e` of a set, and its code width.
static inline device const uchar *tm_picked_mat(
    device const uchar *w, uint n, uint k, uint bits, uint e, thread uint &ek) {
    if (bits == 0u) {
        device const uchar *meta = w + TM_MIX_TABLE + ulong(e) * TM_MIX_ROW;
        ek = *(device const uint *)(meta + 8u);
        return w + *(device const ulong *)meta;
    }
    ek = bits;
    uint side = *(device const uint *)(w + 8u);
    if (side & TM_PAGED) return w + *(device const ulong *)(w + TM_TABLE + ulong(e) * 8u);
    return w + TM_HEAD + ulong(e) * ulong(k >> 4u) * ulong(n >> 4u) * ulong(32u * bits);
}

// One output row and block on its picked matrix.
template <uint K, uint BOOK, bool TURNED>
static inline void tm_picked_body(
    device const float *x,
    device const uchar *w,
    device const uchar *mat,
    device const float *su,
    device const float *sv,
    device const float *extra,
    device float *out,
    uint n,
    uint k,
    uint repeat,
    uint post,
    uint e,
    uint ob,
    uint r,
    uint tid,
    threadgroup float xs[4][128]) {
    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    uint cc = lane >> 2u;
    uint rr = lane & 3u;
    TmWalk walk = tm_picked_walk<K>(w, mat, n, k, ob);
    TmWordMap wm = tm_word_map<K, BOOK>(lane);
    device const float *xr = x + ulong(r / repeat) * k;

    float acc0 = 0.0f, acc1 = 0.0f;
    tm_dec1_run<K, BOOK>(xr, su + ulong(e) * k, TURNED, walk, wm, simd, lane, xs, acc0, acc1);

    acc0 += simd_shuffle_xor(acc0, ushort(1u));
    acc0 += simd_shuffle_xor(acc0, ushort(2u));
    acc1 += simd_shuffle_xor(acc1, ushort(1u));
    acc1 += simd_shuffle_xor(acc1, ushort(2u));
    uint lcol = (simd << 4u) + cc;
    if constexpr (!TURNED) {
        if (rr == 0u) {
            device float *row = out + ulong(r) * n + (ob << 7u);
            row[lcol] = acc0;
            row[lcol + 8u] = acc1;
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rr == 0u) {
            xs[0][lcol] = acc0;
            xs[0][lcol + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            uint base = (ob << 7u) + (lane << 2u);
            float4 sc = *(device const float4 *)(sv + ulong(e) * n + base);
            float4 y = tm_turn_out(tm_get4(&xs[0][lane << 2u]), sc, lane);
            if (post == 1u) {
                y *= *(device const float4 *)(extra + base);
            } else if (post == 2u) {
                float4 ex = *(device const float4 *)(extra + ulong(r) * n + base);
                y = (y / (1.0f + exp(-y))) * ex;
            }
            *(device float4 *)(out + ulong(r) * n + base) = y;
        }
    }
}

#define TM_PICKED_ARGS                                                                       \
    device const float *x [[buffer(0)]],                                                     \
    device const uchar *w [[buffer(1)]],                                                     \
    device const uint *ids [[buffer(2)]],                                                    \
    device const float *su [[buffer(3)]],                                                    \
    device const float *sv [[buffer(4)]],                                                    \
    device const float *extra [[buffer(5)]],                                                 \
    device float *out [[buffer(6)]],                                                         \
    constant uint &rows [[buffer(7)]],                                                       \
    constant uint &n [[buffer(8)]],                                                          \
    constant uint &k [[buffer(9)]],                                                          \
    constant uint &count [[buffer(10)]],                                                     \
    constant uint &repeat [[buffer(11)]],                                                    \
    constant uint &first [[buffer(12)]],                                                     \
    constant uint &bits [[buffer(13)]],                                                      \
    constant uint &book [[buffer(14)]],                                                      \
    constant uint &turned [[buffer(15)]],                                                    \
    constant uint &post [[buffer(16)]],                                                      \
    uint3 tg [[threadgroup_position_in_grid]],                                               \
    uint tid [[thread_index_in_threadgroup]]

// Finds the group's matrix. Rows off this rank store zeros.
#define TM_PICKED_START                                                                      \
    threadgroup float xs[4][128];                                                            \
    uint ob = tg.x;                                                                          \
    uint r = tg.y;                                                                           \
    if (ob >= (n >> 7u) || r >= rows) return;                                                \
    uint raw_id = ids[r];                                                                    \
    if (raw_id < first || raw_id - first >= count) {                                         \
        if (tid < 128u) out[ulong(r) * n + (ob << 7u) + tid] = 0.0f;                         \
        return;                                                                              \
    }                                                                                        \
    uint e = raw_id - first;                                                                 \
    uint ek;                                                                                 \
    device const uchar *mat = tm_picked_mat(w, n, k, bits, e, ek);

#define TM_PICKED_K(BOOK, TURNED, POST)                                                      \
    if (ek == 2u)                                                                            \
        tm_picked_body<2u, BOOK, TURNED>(x, w, mat, su, sv, extra, out, n, k, repeat, POST, e, ob, r, tid, xs); \
    else if (ek == 3u)                                                                       \
        tm_picked_body<3u, BOOK, TURNED>(x, w, mat, su, sv, extra, out, n, k, repeat, POST, e, ob, r, tid, xs); \
    else                                                                                     \
        tm_picked_body<4u, BOOK, TURNED>(x, w, mat, su, sv, extra, out, n, k, repeat, POST, e, ob, r, tid, xs);

#define TM_PICKED_BOOK(BOOK)                                                                 \
    if (turned) {                                                                            \
        TM_PICKED_K(BOOK, true, post)                                                        \
    } else {                                                                                 \
        TM_PICKED_K(BOOK, false, 0u)                                                         \
    }

// Rotated PMCGX experts, each at its own width.
kernel void tm_pickmix_pmcgx_t(TM_PICKED_ARGS) {
    (void)book; (void)turned;
    TM_PICKED_START
    TM_PICKED_K(TM_BOOK_PMCGX, true, post)
}

// Picked experts of any width, book and rotation.
kernel void tm_picked_gen(TM_PICKED_ARGS) {
    TM_PICKED_START
    if (book == TM_BOOK_PMCGX) {
        TM_PICKED_BOOK(TM_BOOK_PMCGX)
    } else if (book == TM_BOOK_MUL1) {
        TM_PICKED_BOOK(TM_BOOK_MUL1)
    } else {
        TM_PICKED_BOOK(TM_BOOK_MCG)
    }
}
