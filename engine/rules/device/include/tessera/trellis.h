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
constexpr u32 MCG = 0xCBAC1FEDu;
constexpr u32 MUL1 = 0x83DCD12Du;
constexpr u32 PMCG2 = 0xBCA14C51u;
constexpr float INV_SQRT128 = 0.08838834764831844f;
constexpr float MUL1_SCALE = 1774.0f / 262144.0f;
constexpr float MUL1_OFFSET = -1329.0f / 128.0f;

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
    u64 first = u64(m) * (in_dim / 16u) * (out_dim / 16u);
    return base + HEAD + (first + local) * (32u * K);
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
TESSERA_FN u32 decode_pair(u32 a, u32 b) {
    if constexpr (BOOK == BOOK_MUL1) {
        u32 sx, sy;
        asm("dp4a.u32.u32 %0, %1, %2, %3;"
            : "=r"(sx)
            : "r"(a * MUL1), "r"(0x01010101u), "r"(0x4B000400u));
        asm("dp4a.u32.u32 %0, %1, %2, %3;"
            : "=r"(sy)
            : "r"(b * MUL1), "r"(0x01010101u), "r"(0x4B000400u));
        constexpr float base = MUL1_OFFSET - 8388608.0f * MUL1_SCALE;
        float lo = fmaf(__uint_as_float(sx), MUL1_SCALE, base);
        float hi = fmaf(__uint_as_float(sy), MUL1_SCALE, base);
        return narrow_f16x2_e4m3(f32x2_to_f16x2(lo, hi));
    } else {
        u32 x = ((a * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        u32 y = ((b * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        u32 sum;
        asm("add.rn.f16x2 %0, %1, %2;"
            : "=r"(sum)
            : "r"(prmt(x, y, 0x5410u)), "r"(prmt(x, y, 0x7632u)));
        return narrow_f16x2_e4m3(sum);
    }
}

TESSERA_FN u32 decode_pmcgx(u32 w) {
    u32 x = ((w * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    u32 y = ((w * PMCG2) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    u32 sum;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(sum) : "r"(x), "r"(prmt(y, y, 0x1032u)));
    return narrow_f16x2_e4m3(sum);
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
    bool input_side = *reinterpret_cast<const u32*>(base + 8u) == INPUT_SIDE;
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
