// Trellis tile loads, codebook decode, WHT128, and f16 MMA.

#ifndef TESSERA_CAP_TRELLIS
#error "import <tessera/1/trellis>"
#endif

#ifndef TESSERA_TRELLIS_H
#define TESSERA_TRELLIS_H

namespace tessera {
namespace trellis {

constexpr u32 HEAD = 16u;
constexpr u32 INPUT_SIDE = 2u;
constexpr u32 SIDE_MASK = 0xFFFFu;
constexpr u32 PAGED = 0x80000000u;
constexpr u32 TABLE = 24u;
constexpr u32 MCG = 0xCBAC1FEDu;
constexpr u32 MUL1 = 0x83DCD12Du;
constexpr u32 PMCG2 = 0xBCA14C51u;
constexpr float INV_SQRT128 = 0.08838834764831844f;
constexpr float MUL1_SCALE = 1774.0f / 262144.0f;
constexpr float MUL1_OFFSET = -1329.0f / 128.0f;

// The u32 at byte `at`, such as a head word.
TESSERA_FN u32 word(const u8* p, u32 at) {
    return *reinterpret_cast<const u32*>(p + at);
}

enum codebook : u32 {
    BOOK_MCG = 0u,
    BOOK_MUL1 = 1u,
    BOOK_PMCGX = 2u,
};

// Decodes a trellis `form_kind` into bit width and codebook.
TESSERA_FN bool from_form_kind(u32 form_kind, u32* k_out, u32* book_out) {
    if (form_kind < FORM_TRELLIS_K2 || form_kind > FORM_TRELLIS_K4_PMCGX) return false;
    u32 offset = form_kind - FORM_TRELLIS_K2;
    *k_out = 2u + offset % 3u;
    *book_out = offset / 3u;
    return true;
}

// Pointer to 8 tiles at input tile `ti`, block `ob`.
template <u32 K>
TESSERA_FN const u8* tile_ptr(const u8* base, u32 in_dim, u32 out_dim, u32 m, u32 ti, u32 ob,
                              bool input_side) {
    u64 local = input_side ? u64(ti / 8u) * (out_dim / 2u) + u64(ob) * 64u + (ti % 8u) * 8u
                           : u64(ob) * (in_dim / 2u) + u64(ti) * 8u;
    u32 side = *reinterpret_cast<const u32*>(base + 8u);
    if (side & PAGED) {
        u64 delta = *reinterpret_cast<const u64*>(base + TABLE + u64(m) * 8u);
        u64 addr = reinterpret_cast<u64>(base) + delta;
        return reinterpret_cast<const u8*>(addr) + local * (32u * K);
    }
    u64 first = u64(m) * (in_dim / 16u) * (out_dim / 16u);
    return base + HEAD + (first + local) * (32u * K);
}

// Records one activation of expert `m` when `base` is paged.
TESSERA_FN void note_hit(const u8* base, u32 m) {
    u32 side = *reinterpret_cast<const u32*>(base + 8u);
    if (side & PAGED) {
        u64 hits = *reinterpret_cast<const u64*>(base + HEAD);
        if (hits != 0ull) {
            atomicAdd(reinterpret_cast<u32*>(hits) + m, 1u);
        }
    }
}

#if defined(TESSERA_CUDA)

TESSERA_FN float f16_to_f32(u32 h) {
    float out;
    u16 bits = static_cast<u16>(h);
    asm("{.reg .b16 tmp; mov.b16 tmp, %1; cvt.f32.f16 %0, tmp;}" : "=f"(out) : "h"(bits));
    return out;
}

TESSERA_FN u32 f32x2_to_f16x2(float lo, float hi) {
    u32 packed;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(hi), "f"(lo));
    return packed;
}

// Rounds two f16 values to E4M3, saturated at 448.
TESSERA_FN u32 narrow_f16x2_e4m3(u32 h) {
    u32 out;
    asm("{.reg .b16 q; cvt.rn.satfinite.e4m3x2.f16x2 q, %1; cvt.rn.f16x2.e4m3x2 %0, q;}"
        : "=r"(out)
        : "r"(h));
    return out;
}

TESSERA_FN float2 e4m3x2_to_f32x2(u32 packed) {
    u32 h;
    u16 q = static_cast<u16>(packed);
    asm("{.reg .b16 tmp; mov.b16 tmp, %1; cvt.rn.f16x2.e4m3x2 %0, tmp;}" : "=r"(h) : "h"(q));
    return make_float2(f16_to_f32(h & 0xFFFFu), f16_to_f32(h >> 16u));
}

TESSERA_FN float e4m3_to_f32(u32 byte) {
    return e4m3x2_to_f32x2(byte & 0xFFu).x;
}

TESSERA_FN u8 f32_to_e4m3(float v) {
    u16 out;
    asm("{.reg .b16 tmp; cvt.rn.satfinite.e4m3x2.f32 tmp, %1, %1; mov.b16 %0, tmp;}"
        : "=h"(out)
        : "f"(v));
    return static_cast<u8>(out & 0xFFu);
}

TESSERA_FN u16 f32_to_bf16(float v) {
    u16 out;
    asm("{.reg .b16 tmp; cvt.rn.bf16.f32 tmp, %1; mov.b16 %0, tmp;}" : "=h"(out) : "f"(v));
    return out;
}

TESSERA_FN u32 shf_l(u32 lo, u32 hi, u32 shift) {
    u32 out;
    asm("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(lo), "r"(hi), "r"(shift));
    return out;
}

TESSERA_FN u32 prmt(u32 a, u32 b, u32 sel) {
    u32 out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
    return out;
}

TESSERA_FN u32 ld_nc_u32(const u8* p) {
    u32 out;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}

TESSERA_FN void ld_nc_v4(const u8* p, u32 dst[4]) {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                 : "l"(p));
}

TESSERA_FN float4 ld_cg_f4(const float* p) {
    float4 out;
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(out.x), "=f"(out.y), "=f"(out.z), "=f"(out.w)
                 : "l"(p));
    return out;
}

// Register-only 128-point Walsh-Hadamard transform over one warp.
TESSERA_FN void wht128(float v[4], float scale) {
    u32 lane = threadIdx.x & 31u;
    float a0 = v[0] + v[1], a1 = v[0] - v[1];
    float a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2;
    v[1] = a1 + a3;
    v[2] = a0 - a2;
    v[3] = a1 - a3;
#pragma unroll
    for (u32 mask = 1u; mask < 32u; mask <<= 1u) {
        float sign = (lane & mask) ? -1.0f : 1.0f;
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
            float peer = __shfl_xor_sync(0xFFFFFFFFu, v[i], mask);
            v[i] = peer + sign * v[i];
        }
    }
#pragma unroll
    for (u32 i = 0u; i < 4u; i++) v[i] *= scale;
}

template <u32 K>
TESSERA_FN void tile_load(const u8* p, u32 lane, bool wide, u32 reg[8]) {
    if (wide) {
        ld_nc_v4(p + 16u * lane, reg);
        if constexpr (K > 2u) {
            if (lane < 16u * K - 32u) ld_nc_v4(p + 512u + 16u * lane, reg + 4u);
        }
        return;
    }
#pragma unroll
    for (u32 mt = 0u; mt < 8u; mt++) {
        reg[mt] = lane < 8u * K ? ld_nc_u32(p + 4u * (mt * 8u * K + lane)) : 0u;
    }
}

template <u32 K>
TESSERA_FN void tile_store(const u32 reg[8], u32 lane, bool wide, u32* smem) {
    if (wide) {
        uint4* dst = reinterpret_cast<uint4*>(smem);
        dst[lane] = make_uint4(reg[0], reg[1], reg[2], reg[3]);
        if constexpr (K > 2u) {
            if (lane < 16u * K - 32u) {
                dst[32u + lane] = make_uint4(reg[4], reg[5], reg[6], reg[7]);
            }
        }
        return;
    }
    if (lane >= 8u * K) return;
#pragma unroll
    for (u32 mt = 0u; mt < 8u; mt++) smem[mt * 8u * K + lane] = reg[mt];
}

// Word offsets and bit shift for one warp lane.
struct lane_map {
    u32 q0, q1, q2, shift;
};

template <u32 K, u32 BOOK>
TESSERA_FN lane_map map_lane(u32 lane) {
    constexpr u32 words = 8u * K;
    constexpr u32 ring = 256u * K;
    u32 start = (BOOK == BOOK_PMCGX) ? (2u * K * (4u * lane + 1u) + ring - 20u) % ring
                                     : (K * (8u * lane + 1u) + ring - 16u) % ring;
    u32 q = start >> 5u;
    return {q, (q + 1u) % words, (q + 2u) % words, start & 31u};
}

template <u32 BOOK>
TESSERA_FN u32 decode_pair_f16x2(u32 a, u32 b) {
    if constexpr (BOOK == BOOK_MUL1) {
        u32 sx, sy, v;
        asm("dp4a.u32.u32 %0, %1, %2, %3;"
            : "=r"(sx)
            : "r"(a * MUL1), "r"(0x01010101u), "r"(0x6400u));
        asm("dp4a.u32.u32 %0, %1, %2, %3;"
            : "=r"(sy)
            : "r"(b * MUL1), "r"(0x01010101u), "r"(0x6400u));
        asm("fma.rn.f16x2 %0, %1, %2, %3;"
            : "=r"(v)
            : "r"(prmt(sx, sy, 0x5410u)), "r"(0x1EEE1EEEu), "r"(0xC931C931u));
        return v;
    } else {
        u32 x = a * MCG, y = b * MCG, sum;
        asm("lop3.b32 %0, %1, %2, %3, 0x6A;"
            : "=r"(x)
            : "r"(x), "r"(0x8FFF8FFFu), "r"(0x3B603B60u));
        asm("lop3.b32 %0, %1, %2, %3, 0x6A;"
            : "=r"(y)
            : "r"(y), "r"(0x8FFF8FFFu), "r"(0x3B603B60u));
        // Pair sums use FMA, not ALU: this decode is ALU-bound.
        asm("{.reg .b16 xl, xh, yl, yh; .reg .b32 xhh, yll, yhh;\n"
            " mov.b32 {xl, xh}, %1; mov.b32 {yl, yh}, %2;\n"
            " mov.b32 xhh, {xh, xh}; mov.b32 yll, {yl, yl}; mov.b32 yhh, {yh, yh};\n"
            " fma.rn.f16x2 %0, xhh, %3, %1;\n"
            " fma.rn.f16x2 %0, yll, %4, %0;\n"
            " fma.rn.f16x2 %0, yhh, %4, %0;}"
            : "=r"(sum)
            : "r"(x), "r"(y), "r"(0xBC003C00u), "r"(0x3C000000u));
        return sum;
    }
}

template <u32 BOOK>
TESSERA_FN u32 decode_pair(u32 a, u32 b) {
    return narrow_f16x2_e4m3(decode_pair_f16x2<BOOK>(a, b));
}

TESSERA_FN u32 decode_pmcgx_f16x2(u32 w) {
    u32 x = w * MCG, y = w * PMCG2, sum;
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;"
        : "=r"(x)
        : "r"(x), "r"(0x8FFF8FFFu), "r"(0x3B603B60u));
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;"
        : "=r"(y)
        : "r"(y), "r"(0x8FFF8FFFu), "r"(0x3B603B60u));
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(sum) : "r"(x), "r"(prmt(y, y, 0x1032u)));
    return sum;
}

TESSERA_FN u32 decode_pmcgx(u32 w) {
    return narrow_f16x2_e4m3(decode_pmcgx_f16x2(w));
}

// Shuffle indices and bit shift for direct tile decode.
struct word_map {
    u32 i0, i1, sh;
};

template <u32 K>
TESSERA_FN word_map map_word(u32 lane) {
    u32 b0 = (lane * 8u + 1u) * K + 256u * K - 16u;
    u32 i0 = (b0 >> 5u) % (8u * K);
    return {i0, (i0 + 1u) % (8u * K), b0 & 31u};
}

// Loads one 64-tile block (2048*K bytes) cooperatively with 128-bit loads.
template <u32 K>
TESSERA_FN void blk_load(const u8* blk, u32 tid, bool aligned, u32 reg[8]) {
    if (aligned) {
        ld_nc_v4(blk + (tid << 4u), reg);
        if constexpr (K == 3u) {
            if (tid < 128u) ld_nc_v4(blk + 4096u + (tid << 4u), reg + 4u);
        } else if constexpr (K == 4u) {
            ld_nc_v4(blk + 4096u + (tid << 4u), reg + 4u);
        }
        return;
    }
#pragma unroll
    for (u32 i = 0u; i < 4u; i++) reg[i] = ld_nc_u32(blk + (tid << 4u) + (i << 2u));
    if constexpr (K == 3u) {
        if (tid < 128u) {
#pragma unroll
            for (u32 i = 0u; i < 4u; i++) {
                reg[4u + i] = ld_nc_u32(blk + 4096u + (tid << 4u) + (i << 2u));
            }
        }
    } else if constexpr (K == 4u) {
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
            reg[4u + i] = ld_nc_u32(blk + 4096u + (tid << 4u) + (i << 2u));
        }
    }
}

// Stores one cooperatively loaded 64-tile block into shared memory.
template <u32 K>
TESSERA_FN void blk_store(const u32 reg[8], u32 tid, u32* smem) {
    uint4* dst = reinterpret_cast<uint4*>(smem);
    dst[tid] = make_uint4(reg[0], reg[1], reg[2], reg[3]);
    if constexpr (K == 3u) {
        if (tid < 128u) dst[256u + tid] = make_uint4(reg[4], reg[5], reg[6], reg[7]);
    } else if constexpr (K == 4u) {
        dst[256u + tid] = make_uint4(reg[4], reg[5], reg[6], reg[7]);
    }
}

// Decodes one warp-held tile word into an MMA A fragment.
template <u32 K, u32 BOOK>
TESSERA_FN void decode_word(u32 raw, const word_map& wm, u32 frag[4]) {
    u32 a = __shfl_sync(0xFFFFFFFFu, raw, wm.i0);
    u32 b = (K == 4u) ? raw : __shfl_sync(0xFFFFFFFFu, raw, wm.i1);
    u32 s0 = (K == 4u) ? 20u : wm.sh;
    if constexpr (BOOK == BOOK_PMCGX) {
        u32 sp = s0 - (4u - K);
        u32 hi = shf_l(b, a, sp);
        u32 lo = b << sp;
        frag[0] = decode_pmcgx(shf_l(lo, hi, 0u) >> 12u);
        frag[2] = decode_pmcgx(shf_l(lo, hi, 2u * K) >> 12u);
        frag[1] = decode_pmcgx(shf_l(lo, hi, 4u * K) >> 12u);
        frag[3] = decode_pmcgx(shf_l(lo, hi, 6u * K) >> 12u);
    } else {
        u32 hi = shf_l(b, a, s0);
        u32 lo = b << s0;
        u32 st[8];
#pragma unroll
        for (u32 s = 0u; s < 8u; s++) st[s] = shf_l(lo, hi, K * s) >> 16u;
        frag[0] = decode_pair<BOOK>(st[0], st[1]);
        frag[1] = decode_pair<BOOK>(st[4], st[5]);
        frag[2] = decode_pair<BOOK>(st[2], st[3]);
        frag[3] = decode_pair<BOOK>(st[6], st[7]);
    }
}

// Stores one 128-vector into an MMA B fragment table.
TESSERA_FN void store_bfrag(u32* tab, u32 row, u32 lane, const float v[4]) {
    u32 ks = lane >> 2u;
    u32 q = lane & 3u;
    u32 reg = q >> 1u;
    u32 c = (q & 1u) << 1u;
    u32* p = tab + ((ks * 32u + row * 4u + c) * 2u + reg);
    p[0] = f32x2_to_f16x2(v[0], v[1]);
    p[2] = f32x2_to_f16x2(v[2], v[3]);
}

// Decodes one 16x16 tile into an MMA A fragment.
template <u32 K, u32 BOOK>
TESSERA_FN void decode_frag(const u32* words, const lane_map& lm, u32 frag[4]) {
    u32 w0 = words[lm.q0], w1 = words[lm.q1], w2 = words[lm.q2];
    u32 hi = shf_l(w1, w0, lm.shift);
    u32 lo = shf_l(w2, w1, lm.shift);
    if constexpr (BOOK == BOOK_PMCGX) {
        u32 p[4];
#pragma unroll
        for (u32 q = 0u; q < 4u; q++) p[q] = decode_pmcgx(shf_l(lo, hi, 2u * K * q) >> 12u);
        frag[0] = p[0];
        frag[1] = p[2];
        frag[2] = p[1];
        frag[3] = p[3];
    } else {
        u32 st[8];
#pragma unroll
        for (u32 s = 0u; s < 8u; s++) st[s] = shf_l(lo, hi, K * s) >> 16u;
        frag[0] = decode_pair<BOOK>(st[0], st[1]);
        frag[1] = decode_pair<BOOK>(st[4], st[5]);
        frag[2] = decode_pair<BOOK>(st[2], st[3]);
        frag[3] = decode_pair<BOOK>(st[6], st[7]);
    }
}

// Loads hi+lo f16 MMA B fragment for step `ks`.
TESSERA_FN void act_frag(const float* buf, u32 n, u32 ks, u32 lane, u32 b[2]) {
    u32 col = lane >> 2u;
    u32 row = col >> 1u;
    if (row >= n) {
        b[0] = 0u;
        b[1] = 0u;
        return;
    }
    const float* src = buf + row * 128u + ks * 16u + 2u * (lane & 3u);
    bool is_lo = (col & 1u) != 0u;
#pragma unroll
    for (u32 h = 0u; h < 2u; h++) {
        float x0 = src[8u * h], x1 = src[8u * h + 1u];
        u32 hi = f32x2_to_f16x2(x0, x1);
        b[h] = is_lo ? f32x2_to_f16x2(x0 - f16_to_f32(hi & 0xFFFFu), x1 - f16_to_f32(hi >> 16u))
                     : hi;
    }
}

TESSERA_FN void mma_m16n8k16(float d[4], const u32 a[4], const u32 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Decodes one staged 64-tile block into `NT` MMA tiles.
template <u32 K, u32 BOOK, u32 NT>
TESSERA_FN void blk_mma(const u32* wtiles, const u32* xb, u32 xts, u32 j, u32 lane,
                        const word_map& wm, float acc[NT][4]) {
    constexpr u32 TW = 8u * K;
#pragma unroll
    for (u32 u = 0u; u < 8u; u++) {
        u32 raw = (K == 4u || lane < TW) ? wtiles[(u * 8u + j) * TW + lane] : 0u;
        u32 fa[4];
        decode_word<K, BOOK>(raw, wm, fa);
#pragma unroll
        for (u32 ti = 0u; ti < NT; ti++) {
            uint2 b2 = *reinterpret_cast<const uint2*>(xb + ti * xts + u * 64u);
            u32 fb[2] = {b2.x, b2.y};
            mma_m16n8k16(acc[ti], fa, fb);
        }
    }
}

// Streams `nb` 64-tile blocks through shared memory into MMA.
template <u32 K, u32 BOOK, u32 NT>
TESSERA_FN void blk_stream(const u8*& gb, u64 sG, u32 c0, u32 nb, u32 total, u32 tid, u32 j,
                           u32 lane, bool aligned, const word_map& wm, u32 nw[8], u32* wtiles,
                           const u32* xb, u32 xts, float acc[NT][4]) {
    for (u32 g = 0u; g < nb; g++) {
        if (g > 0u) prim::block_sync();
        blk_store<K>(nw, tid, wtiles);
        prim::block_sync();
        if (c0 + g + 1u < total) {
            gb += sG;
            blk_load<K>(gb, tid, aligned, nw);
        }
        blk_mma<K, BOOK, NT>(wtiles, xb, xts, j, lane, wm, acc);
        xb += 512u;
    }
}

// Sums hi+lo accumulators for `n` rows into `buf`.
TESSERA_FN void store_acc(const float acc[8][4], u32 n, u32 lane, float* buf) {
    u32 r = lane & 3u;
    u32 g = lane >> 2u;
    if (r >= n) return;
    float* dst = buf + r * 128u;
#pragma unroll
    for (u32 mt = 0u; mt < 8u; mt++) {
        dst[mt * 16u + g] = acc[mt][0] + acc[mt][1];
        dst[mt * 16u + g + 8u] = acc[mt][2] + acc[mt][3];
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
constexpr u32 MX8_TILE = 272u;

TESSERA_FN u32 mx_scale(float amax, float& inv) {
    u32 u = __float_as_uint(__fmul_rn(amax, 0.002232142857f));
    int e = int(u >> 23u) - 127 + ((u & 0x7FFFFFu) != 0u ? 1 : 0);
    if (amax == 0.0f) e = 0;
    if (e < -126) e = -126;
    if (e > 127) e = 127;
    inv = __uint_as_float(u32(127 - e) << 23u);
    return u32(e + 127);
}

TESSERA_FN u16 f2_e4m3x2(float a, float b) {
    u16 r;
    asm("{.reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; mov.b16 %0, t;}"
        : "=h"(r)
        : "f"(a), "f"(b));
    return r;
}

TESSERA_FN u32 h2_e4m3x2(u32 h2) {
    u32 r;
    asm("{.reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.u32.u16 %0, t;}"
        : "=r"(r)
        : "r"(h2));
    return r;
}

TESSERA_FN void mma_mx8(float d[4], const u32 a[4], const u32 b[2], u32 sa, u32 sb) {
    const u16 z = 0u;
    asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
}

template <u32 BOOK>
TESSERA_FN u32 decode_pair_e4m3(u32 a, u32 b) {
    return h2_e4m3x2(decode_pair_f16x2<BOOK>(a, b));
}

TESSERA_FN u32 decode_pmcgx_e4m3(u32 w) {
    return h2_e4m3x2(decode_pmcgx_f16x2(w));
}

template <u32 K, u32 BOOK>
TESSERA_FN void decode_word_e4m3(u32 raw, const word_map& wm, u32& r_lo, u32& r_hi) {
    u32 a = __shfl_sync(0xFFFFFFFFu, raw, wm.i0);
    u32 b = (K == 4u) ? raw : __shfl_sync(0xFFFFFFFFu, raw, wm.i1);
    u32 s0 = (K == 4u) ? 20u : wm.sh;
    if constexpr (BOOK == BOOK_PMCGX) {
        u32 sp = s0 - (4u - K), hi = shf_l(b, a, sp), lo = b << sp;
        r_lo = decode_pmcgx_e4m3(shf_l(lo, hi, 0u) >> 12u) |
               (decode_pmcgx_e4m3(shf_l(lo, hi, 2u * K) >> 12u) << 16u);
        r_hi = decode_pmcgx_e4m3(shf_l(lo, hi, 4u * K) >> 12u) |
               (decode_pmcgx_e4m3(shf_l(lo, hi, 6u * K) >> 12u) << 16u);
    } else {
        u32 hi = shf_l(b, a, s0), lo = b << s0, st[8];
#pragma unroll
        for (u32 s = 0u; s < 8u; s++) st[s] = shf_l(lo, hi, K * s) >> 16u;
        r_lo = decode_pair_e4m3<BOOK>(st[0], st[1]) | (decode_pair_e4m3<BOOK>(st[2], st[3]) << 16u);
        r_hi = decode_pair_e4m3<BOOK>(st[4], st[5]) | (decode_pair_e4m3<BOOK>(st[6], st[7]) << 16u);
    }
}

// Quantizes one 128-vector into permuted-K MXFP8 B-fragments.
TESSERA_FN void store_bfrag_mx8(u32* tile, u32 row, u32 lane, const float v[4]) {
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (u32 off = 4u; off > 0u; off >>= 1u) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off, 8));
    }
    float inv;
    u32 sb = mx_scale(amax, inv);
    u16 q0 = f2_e4m3x2(v[0] * inv, v[1] * inv);
    u16 q1 = f2_e4m3x2(v[2] * inv, v[3] * inv);
    u32 ks32 = lane >> 3u, sub = lane & 7u;
    u32 reg = sub >> 2u, hi_half = (sub >> 1u) & 1u, t0 = (sub & 1u) << 1u;
    u16* p = reinterpret_cast<u16*>(tile + ((ks32 * 32u + row * 4u + t0) * 2u + reg));
    p[hi_half] = q0;
    p[4u + hi_half] = q1;
    if (sub == 0u) reinterpret_cast<u8*>(tile + 256u)[ks32 * 8u + row] = u8(sb);
}

// Decodes one staged 64-tile block with Blackwell m16n8k32 MXFP8 MMA.
template <u32 K, u32 BOOK, u32 NT>
TESSERA_FN void blk_mma_mx8(const u32* wtiles, const u32* btab, u32 j, u32 lane,
                            const word_map& wm, float acc[NT][4]) {
    constexpr u32 TW = 8u * K;
    u32 g = lane >> 2u;
#pragma unroll
    for (u32 p = 0u; p < 4u; p++) {
        u32 u0 = p << 1u, u1 = u0 + 1u;
        u32 raw0 = (K == 4u || lane < TW) ? wtiles[(u0 * 8u + j) * TW + lane] : 0u;
        u32 raw1 = (K == 4u || lane < TW) ? wtiles[(u1 * 8u + j) * TW + lane] : 0u;
        u32 fa[4];
        decode_word_e4m3<K, BOOK>(raw0, wm, fa[0], fa[1]);
        decode_word_e4m3<K, BOOK>(raw1, wm, fa[2], fa[3]);
#pragma unroll
        for (u32 ti = 0u; ti < NT; ti++) {
            const u32* tile = btab + ti * MX8_TILE;
            uint2 b2 = *reinterpret_cast<const uint2*>(tile + p * 64u + (lane << 1u));
            u32 fb[2] = {b2.x, b2.y};
            u32 sb = reinterpret_cast<const u8*>(tile + 256u)[p * 8u + g];
            mma_mx8(acc[ti], fa, fb, 127u, sb);
        }
    }
}
#endif

// Staged codebook and weight tiles, or the output `Out`.
template <typename Out>
union alignas(16) stage_smem {
    struct {
        u32 btab[2048];
        u32 wtiles[2048];
    } k;
    Out out;
};

// Reads four floats from 16-byte aligned `p`.
TESSERA_FN void load4(const float* p, float v[4]) {
    float4 f = *reinterpret_cast<const float4*>(p);
    v[0] = f.x;
    v[1] = f.y;
    v[2] = f.z;
    v[3] = f.w;
}

#if defined(X_BF16)
using x_t = u16;
#else
using x_t = float;
#endif

// Reads two activation elements as float2.
TESSERA_FN float2 load_x2(const x_t* p) {
#if defined(X_BF16)
    u32 h = *reinterpret_cast<const u32*>(p);
    return make_float2(__uint_as_float(h << 16u), __uint_as_float(h & 0xFFFF0000u));
#else
    return *reinterpret_cast<const float2*>(p);
#endif
}

// Reads four activation elements as float4.
TESSERA_FN float4 load_x4(const x_t* p) {
#if defined(X_BF16)
    uint2 h = *reinterpret_cast<const uint2*>(p);
    return make_float4(__uint_as_float(h.x << 16u), __uint_as_float(h.x & 0xFFFF0000u),
                       __uint_as_float(h.y << 16u), __uint_as_float(h.y & 0xFFFF0000u));
#else
    return *reinterpret_cast<const float4*>(p);
#endif
}

// Reads four activation elements into `v`.
TESSERA_FN void load_x4(const x_t* p, float v[4]) {
    float4 f = load_x4(p);
    v[0] = f.x;
    v[1] = f.y;
    v[2] = f.z;
    v[3] = f.w;
}

TESSERA_FN float half_to_f32(u32 h) {
    return f16_to_f32(h);
}

TESSERA_FN float narrow_f32(float v) {
    return f16_to_f32(narrow_f16x2_e4m3(f32x2_to_f16x2(v, v)) & 0xFFFFu);
}

#else

TESSERA_FN float half_to_f32(u32 h) {
    u32 sign = (h & 0x8000u) << 16u;
    u32 exp = (h >> 10u) & 31u;
    u32 mant = h & 1023u;
    if (exp == 0u) {
        float v = float(mant) * (1.0f / 16777216.0f);
        return sign ? -v : v;
    }
    if (exp == 31u) return __uint_as_float(sign | 0x7F800000u | (mant << 13u));
    return __uint_as_float(sign | ((exp + 112u) << 23u) | (mant << 13u));
}

TESSERA_FN float round_narrow(float x, u32 e, u32 m, float top) {
    float a = x < 0.0f ? -x : x;
    float sign = x < 0.0f ? -1.0f : 1.0f;
    u32 half_bias = 1u << (e - 1u);
    float steps = __uint_as_float((125u + half_bias + m) << 23u);
    float least = __uint_as_float((129u - half_bias) << 23u);
    float small = (a * steps + 8388608.0f) - 8388608.0f;
    u32 bits = __float_as_uint(a);
    u32 shift = 23u - m;
    u32 nudge = ((1u << (shift - 1u)) - 1u) + ((bits >> shift) & 1u);
    float r = __uint_as_float((bits + nudge) & ~((1u << shift) - 1u));
    float big = r > top ? top : r;
    return sign * (a < least ? small / steps : big);
}

TESSERA_FN float narrow_f32(float v) {
    return round_narrow(round_narrow(v, 5u, 10u, 65504.0f), 4u, 3u, 448.0f);
}

#endif

// Portable scalar decode of element `(row, col)` in matrix `m`.
TESSERA_FN float scalar_elem(const u8* base, u32 in_dim, u32 out_dim, u32 k, u32 book, u64 m,
                             u32 row, u32 col) {
    bool input_side = (*reinterpret_cast<const u32*>(base + 8u) & SIDE_MASK) == INPUT_SIDE;
    u32 ti = row >> 4u, to = col >> 4u;
    u64 local = input_side
                    ? u64(ti >> 3u) * (out_dim >> 1u) + u64(to >> 3u) * 64u + (ti & 7u) * 8u +
                          (to & 7u)
                    : u64(to >> 3u) * (in_dim >> 1u) + u64(ti) * 8u + (to & 7u);
    u64 first = m * (u64(in_dim >> 4u) * (out_dim >> 4u));
    const u32* words = reinterpret_cast<const u32*>(base + HEAD + (first + local) * (32u * k));
    u32 r = row & 15u, c = col & 15u;
    u32 j = 8u * (4u * (c & 7u) + ((r & 7u) >> 1u)) + 4u * (c >= 8u) + 2u * (r >= 8u) + (r & 1u);
    u32 ring = 256u * k, wcount = 8u * k;
    u32 start = (book == BOOK_PMCGX) ? (2u * k * ((j >> 1u) + 1u) + ring - 20u) % ring
                                     : (k * (j + 1u) + ring - 16u) % ring;
    u32 q = start >> 5u, sh = start & 31u;
    u32 w0 = words[q], w1 = words[(q + 1u) % wcount];
    u32 win = sh == 0u ? w0 : (w0 << sh) | (w1 >> (32u - sh));
    if (book == BOOK_PMCGX) {
        u32 st = win >> 12u;
        u32 x = ((st * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        u32 y = ((st * PMCG2) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        float raw = (j & 1u) ? (half_to_f32(x >> 16u) + half_to_f32(y & 0xFFFFu))
                             : (half_to_f32(x & 0xFFFFu) + half_to_f32(y >> 16u));
        return narrow_f32(raw);
    }
    u32 st = win >> 16u;
    if (book == BOOK_MUL1) {
        u32 z = st * MUL1;
        u32 sum = (z & 255u) + ((z >> 8u) & 255u) + ((z >> 16u) & 255u) + (z >> 24u);
        return narrow_f32(float(1024u + sum) * MUL1_SCALE + MUL1_OFFSET);
    }
    u32 x = ((st * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    return narrow_f32(half_to_f32(x & 0xFFFFu) + half_to_f32(x >> 16u));
}

}  // namespace trellis
}  // namespace tessera

#endif  // TESSERA_TRELLIS_H
