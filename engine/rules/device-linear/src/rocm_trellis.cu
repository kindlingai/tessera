// RDNA 3/3.5 HIP trellis WMMA kernels (small-M split-K and prefill).

#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_HIP)

namespace {

using trellis::f16x16;
using trellis::v8f;

// Stores one 128-row split into f16 hi/lo tiles.
TESSERA_FN void store_x_split(const float v[4], u32 lane, u32 r, _Float16* hi_base,
                              _Float16* lo_base) {
    u32 u = lane >> 2u;
    u32 kk = (lane & 3u) << 2u;
    u32 off = ((r >> 4u) * 8u + u) * 256u + (r & 15u) * 16u + kk;
#pragma unroll
    for (u32 i = 0u; i < 4u; i++) {
        _Float16 h = static_cast<_Float16>(v[i]);
        hi_base[off + i] = h;
        lo_base[off + i] = static_cast<_Float16>(v[i] - static_cast<float>(h));
    }
}

TESSERA_FN float tr_silu(float x) { return x / (1.0f + expf(-x)); }

TESSERA_FN float tr_gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

TESSERA_FN float tr_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}

TESSERA_FN void load_bias4(const u8* p, u32 col, bool bf16, float b[4]) {
    if (bf16) {
        uint2 w = *reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(p) + col);
        b[0] = prim::from_bf16(static_cast<u16>(w.x));
        b[1] = prim::from_bf16(static_cast<u16>(w.x >> 16u));
        b[2] = prim::from_bf16(static_cast<u16>(w.y));
        b[3] = prim::from_bf16(static_cast<u16>(w.y >> 16u));
    } else {
        trellis::load4(reinterpret_cast<const float*>(p) + col, b);
    }
}

__device__ __attribute__((noinline)) float4 apply_epi4(const float* other, u64 off, float4 b,
                                                       float4 v, u32 mode) {
    u32 base = mode & 15u;
    if (base == 1u || base == 3u || base == 4u || base == 8u) {
        float4 o = *reinterpret_cast<const float4*>(other + off);
        if (base == 1u) {
            v.x += o.x;
            v.y += o.y;
            v.z += o.z;
            v.w += o.w;
        } else if (base == 8u) {
            v.x += b.x + o.x;
            v.y += b.y + o.y;
            v.z += b.z + o.z;
            v.w += b.w + o.w;
        } else if (base == 3u) {
            v.x = tr_silu(v.x) * o.x;
            v.y = tr_silu(v.y) * o.y;
            v.z = tr_silu(v.z) * o.z;
            v.w = tr_silu(v.w) * o.w;
        } else {
            v.x *= o.x;
            v.y *= o.y;
            v.z *= o.z;
            v.w *= o.w;
        }
    } else if (base == 2u) {
        v.x += b.x;
        v.y += b.y;
        v.z += b.z;
        v.w += b.w;
    } else if (base == 5u) {
        v.x = tr_silu(v.x);
        v.y = tr_silu(v.y);
        v.z = tr_silu(v.z);
        v.w = tr_silu(v.w);
    } else if (base == 6u) {
        v.x = tr_gelu_tanh(v.x + b.x);
        v.y = tr_gelu_tanh(v.y + b.y);
        v.z = tr_gelu_tanh(v.z + b.z);
        v.w = tr_gelu_tanh(v.w + b.w);
    } else if (base == 7u) {
        v.x = tr_gelu(v.x + b.x);
        v.y = tr_gelu(v.y + b.y);
        v.z = tr_gelu(v.z + b.z);
        v.w = tr_gelu(v.w + b.w);
    }
    return v;
}

// Writes accumulator rows through LDS with optional WHT128 and epilogue.
template <u32 ROWS, bool TURNED, bool SPLIT, bool ROT_OUT = false>
TESSERA_FN void store_out_block(const v8f* acc, float out[ROWS][128], const float* svh,
                                const float* other, const u8* bias_p, float* y, u32 r0, u32 m,
                                u32 n, u32 ob, u32 split, u32 mode, u32 j, u32 lane) {
    constexpr u32 TILES = ROWS / 16u;
#pragma unroll
    for (u32 ti = 0u; ti < TILES; ti++) {
#pragma unroll
        for (u32 e = 0u; e < 8u; e++) {
            u32 row = ti * 16u + 2u * e + (lane >> 4u);
            out[row][j * 16u + (lane & 15u)] = acc[ti][e];
        }
    }
    prim::block_sync();
    u32 c_base = ob * 128u + lane * 4u;
    float sv[4];
    float bias[4] = {};
    if constexpr (TURNED && !SPLIT) trellis::load4(svh + c_base, sv);
    u32 eb = mode & 15u;
    if constexpr (!SPLIT) {
        bool bf16 = (mode & 16u) != 0u;
        if (eb == 2u || eb == 6u || eb == 7u) {
            load_bias4(reinterpret_cast<const u8*>(other), c_base, bf16, bias);
        } else if constexpr (!ROT_OUT) {
            if (eb == 8u) load_bias4(bias_p, c_base, bf16, bias);
        }
    }
    if constexpr (ROT_OUT) {
        typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
        float suh_n[4];
        trellis::load4(reinterpret_cast<const float*>(bias_p) + c_base, suh_n);
        f16x4 pack[4];
#pragma unroll
        for (u32 k = 0u; k < 4u; k++) {
            u32 r = j + k * 8u;
            u32 row = r0 + r;
            if (row < m) {
                float v[4];
                trellis::load4(&out[r][lane * 4u], v);
                trellis::wht128(v, trellis::INV_SQRT128);
#pragma unroll
                for (u32 i = 0u; i < 4u; i++) v[i] *= sv[i];
                if (eb != 0u) {
                    float4 ev = apply_epi4(other, u64(row) * n + c_base,
                                           make_float4(bias[0], bias[1], bias[2], bias[3]),
                                           make_float4(v[0], v[1], v[2], v[3]), mode);
                    v[0] = ev.x;
                    v[1] = ev.y;
                    v[2] = ev.z;
                    v[3] = ev.w;
                }
#pragma unroll
                for (u32 i = 0u; i < 4u; i++) v[i] *= suh_n[i];
                trellis::wht128(v, trellis::INV_SQRT128);
                pack[k] = (f16x4){static_cast<_Float16>(v[0]), static_cast<_Float16>(v[1]),
                                  static_cast<_Float16>(v[2]), static_cast<_Float16>(v[3])};
            } else {
                pack[k] = (f16x4){static_cast<_Float16>(0.0f), static_cast<_Float16>(0.0f),
                                  static_cast<_Float16>(0.0f), static_cast<_Float16>(0.0f)};
            }
        }
        prim::block_sync();
        _Float16(*sh)[32][16] = reinterpret_cast<_Float16(*)[32][16]>(out);
        u32 u = lane >> 2u;
        u32 kk = (lane & 3u) << 2u;
#pragma unroll
        for (u32 k = 0u; k < 4u; k++) {
            *reinterpret_cast<f16x4*>(&sh[u][j + k * 8u][kk]) = pack[k];
        }
        prim::block_sync();
        u32 tid = (j << 5u) | lane;
        u32 mb = r0 >> 6u;
        u32 half = (r0 >> 5u) & 1u;
        u16* y16 = reinterpret_cast<u16*>(y);
        u64 blk_off = (u64(mb) * (n >> 7u) + ob) * 8192u + u64(half) * 512u;
        const uint4* src = reinterpret_cast<const uint4*>(sh);
#pragma unroll
        for (u32 s = 0u; s < 2u; s++) {
            u32 idx = s * 256u + tid;
            u32 su = idx >> 6u;
            u32 rem = idx & 63u;
            reinterpret_cast<uint4*>(y16 + blk_off + u64(su) * 1024u)[rem] = src[idx];
        }
        return;
    }
    for (u32 r = j; r < ROWS; r += 8u) {
        u32 row = r0 + r;
        if (row >= m) continue;
        float v[4];
        trellis::load4(&out[r][lane * 4u], v);
        if constexpr (TURNED && !SPLIT) {
            trellis::wht128(v, trellis::INV_SQRT128);
#pragma unroll
            for (u32 i = 0u; i < 4u; i++) v[i] *= sv[i];
        }
        if constexpr (SPLIT) {
            *reinterpret_cast<float4*>(y + (u64(split) * m + row) * n + c_base) =
                make_float4(v[0], v[1], v[2], v[3]);
        } else if (eb == 0u) {
            *reinterpret_cast<float4*>(y + u64(row) * n + c_base) =
                make_float4(v[0], v[1], v[2], v[3]);
        } else {
            *reinterpret_cast<float4*>(y + u64(row) * n + c_base) = apply_epi4(
                other, u64(row) * n + c_base, make_float4(bias[0], bias[1], bias[2], bias[3]),
                make_float4(v[0], v[1], v[2], v[3]), mode);
        }
    }
}

template <u32 NT>
union alignas(16) dtr_smem {
    struct {
        u32 wtiles[2048];
        u32 w_lds[8][192];
        _Float16 x_hi[NT * 2048u];
        _Float16 x_lo[NT * 2048u];
    } k;
    float out[16u * NT][128];
};

// Prefill LDS: double-buffered wtiles plus 32-row output staging.
union alignas(16) pre_smem {
    struct {
        u32 wtiles[2][2048];
        u32 w_lds[8][160];
    } k;
    float out[32][128];
};

// Runtime-width 64-tile block load into registers.
TESSERA_FN void blk_load_dyn(const u8* blk, u32 tid, u32 bits, bool aligned, u32 reg[8]) {
    if (bits == 4u) {
        trellis::blk_load<4u>(blk, tid, aligned, reg);
    } else if (bits == 3u) {
        trellis::blk_load<3u>(blk, tid, aligned, reg);
    } else {
        trellis::blk_load<2u>(blk, tid, aligned, reg);
    }
}

// Runtime-width 64-tile block store into LDS.
TESSERA_FN void blk_store_dyn(const u32 reg[8], u32 tid, u32 bits, u32* smem) {
    if (bits == 4u) {
        trellis::blk_store<4u>(reg, tid, smem);
    } else if (bits == 3u) {
        trellis::blk_store<3u>(reg, tid, smem);
    } else {
        trellis::blk_store<2u>(reg, tid, smem);
    }
}

// Runtime-width tile pointer helper.
TESSERA_FN const u8* tile_ptr_dyn(const u8* w, u32 k, u32 n, u32 ti, u32 ob, u32 bits,
                                  bool in_side) {
    if (bits == 4u) return trellis::tile_ptr<4u>(w, k, n, 0u, ti, ob, in_side);
    if (bits == 3u) return trellis::tile_ptr<3u>(w, k, n, 0u, ti, ob, in_side);
    return trellis::tile_ptr<2u>(w, k, n, 0u, ti, ob, in_side);
}

struct dtr_cta {
    u32 tid, j, lane, ob, split, r0, ibs, c0, nb;
    const u8* gb;
    u64 sG;
    bool aligned;
};

template <u32 K, bool SPLIT, u32 NT>
TESSERA_FN dtr_cta init_dtr_cta(const u8* w, u32 n, u32 k, u32 ks, u32 bits) {
    dtr_cta c;
    c.tid = threadIdx.x;
    c.j = c.tid >> 5u;
    c.lane = c.tid & 31u;
    bool split_on = (K != 0u) ? SPLIT : (ks > 1u);
    u32 bx = blockIdx.x;
    c.ob = split_on ? bx / ks : bx;
    c.split = split_on ? bx - c.ob * ks : 0u;
    c.r0 = blockIdx.y * (16u * NT);
    c.ibs = k >> 7u;
    c.c0 = split_on ? (c.split * c.ibs) / ks : 0u;
    u32 c1 = split_on ? ((c.split + 1u) * c.ibs) / ks : c.ibs;
    c.nb = c1 - c.c0;
    bool in_side = (trellis::word(w, 8u) & trellis::SIDE_MASK) == trellis::INPUT_SIDE;
    if constexpr (K != 0u) {
        c.gb = trellis::tile_ptr<K>(w, k, n, 0u, c.c0 * 8u, c.ob, in_side);
        c.sG = in_side ? u64(n) * (16u * K) : u64(2048u * K);
    } else {
        c.gb = tile_ptr_dyn(w, k, n, c.c0 * 8u, c.ob, bits, in_side);
        c.sG = in_side ? u64(n) * (16u * bits) : u64(2048u * bits);
    }
    c.aligned = (reinterpret_cast<u64>(c.gb) & 15u) == 0u;
    return c;
}

template <u32 NT>
TESSERA_FN void stage_dtr_x(dtr_smem<NT>& sm, const float* x, const float* suh, u32 r0, u32 m,
                            u32 k, u32 k_off, u32 j, u32 lane, bool rot) {
    float sv[4];
    if (rot) trellis::load4(suh + k_off, sv);
#pragma unroll
    for (u32 r = j; r < 16u * NT; r += 8u) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (r0 + r < m) {
            trellis::load4(x + u64(r0 + r) * k + k_off, v);
            if (rot) {
#pragma unroll
                for (u32 i = 0u; i < 4u; i++) v[i] *= sv[i];
                trellis::wht128(v, trellis::INV_SQRT128);
            }
        }
        store_x_split(v, lane, r, sm.k.x_hi, sm.k.x_lo);
    }
    prim::block_sync();
}

template <u32 NT>
TESSERA_FN void wmma_dtr_x(const dtr_smem<NT>& sm, f16x16 wf, v8f (&acc)[NT], u32 u, u32 lane) {
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) {
        u32 x_off = (ti * 8u + u) * 256u + (lane & 15u) * 16u;
        f16x16 xh = *reinterpret_cast<const f16x16*>(sm.k.x_hi + x_off);
        f16x16 xl = *reinterpret_cast<const f16x16*>(sm.k.x_lo + x_off);
        acc[ti] = trellis::wmma_f16(acc[ti], xh, wf);
        acc[ti] = trellis::wmma_f16(acc[ti], xl, wf);
    }
}

template <u32 K, u32 BOOK, bool TURNED, bool SPLIT, u32 NT>
TESSERA_FN void rocm_dtr_run(dtr_smem<NT>& sm, const float* x, const u8* w, const float* suh,
                             const float* svh, float* y, u32 m, u32 n, u32 k, u32 ks) {
    dtr_cta c = init_dtr_cta<K, SPLIT, NT>(w, n, k, ks, K);
    v8f acc[NT];
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) acc[ti] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

    constexpr u32 TW = 8u * K;
    trellis::word_map wm = trellis::map_word<K>(c.lane);
    u32 nw[8];
    if (c.nb > 0u) trellis::blk_load<K>(c.gb, c.tid, c.aligned, nw);

    for (u32 g = 0u; g < c.nb; g++) {
        if (g > 0u) prim::block_sync();
        trellis::blk_store<K>(nw, c.tid, sm.k.wtiles);
        stage_dtr_x<NT>(sm, x, suh, c.r0, m, k, (c.c0 + g) * 128u + c.lane * 4u, c.j, c.lane,
                        TURNED);
        if (g + 1u < c.nb) {
            c.gb += c.sG;
            trellis::blk_load<K>(c.gb, c.tid, c.aligned, nw);
        }
        u32* w_buf = sm.k.w_lds[c.j];
#pragma clang loop unroll(disable)
        for (u32 u = 0u; u < 8u; u++) {
            u32 raw = (K == 4u || c.lane < TW) ? sm.k.wtiles[(u * 8u + c.j) * TW + c.lane] : 0u;
            f16x16 wf = trellis::decode_tile_hip<K, BOOK>(raw, wm, c.lane, w_buf);
            wmma_dtr_x<NT>(sm, wf, acc, u, c.lane);
        }
    }

    prim::block_sync();
    store_out_block<16u * NT, TURNED, SPLIT>(acc, sm.out, svh, nullptr, nullptr, y, c.r0, m, n,
                                             c.ob, c.split, 0u, c.j, c.lane);
}

// Runtime-dispatched trellis tile decode for non-K4 codebooks.
template <u32 STRIDE>
TESSERA_FN f16x16 decode_tile_dyn(u32 raw, u32 lane, u32 bits, u32 book, u32* w_lds) {
    u32 b0 = (lane * 8u + 1u) * bits + 256u * bits - 16u;
    u32 tw = 8u * bits;
    u32 i0 = (b0 >> 5u) % tw;
    u32 i1 = (i0 + 1u) % tw;
    u32 s0 = (bits == 4u) ? 20u : (b0 & 31u);
    u32 a = trellis::shfl_u32(raw, i0);
    u32 b = (bits == 4u) ? raw : trellis::shfl_u32(raw, i1);
    u32 fa[4];
    if (book == trellis::BOOK_PMCGX) {
        u32 sp = s0 - (4u - bits);
        u32 hi = trellis::shf_l(b, a, sp);
        u32 lo = b << sp;
        fa[0] = trellis::decode_pmcgx(trellis::shf_l(lo, hi, 0u) >> 12u);
        fa[2] = trellis::decode_pmcgx(trellis::shf_l(lo, hi, 2u * bits) >> 12u);
        fa[1] = trellis::decode_pmcgx(trellis::shf_l(lo, hi, 4u * bits) >> 12u);
        fa[3] = trellis::decode_pmcgx(trellis::shf_l(lo, hi, 6u * bits) >> 12u);
    } else {
        u32 hi = trellis::shf_l(b, a, s0);
        u32 lo = b << s0;
        u32 st[8];
#pragma unroll
        for (u32 s = 0u; s < 8u; s++) st[s] = trellis::shf_l(lo, hi, bits * s) >> 16u;
        if (book == trellis::BOOK_MUL1) {
            fa[0] = trellis::decode_pair<trellis::BOOK_MUL1>(st[0], st[1]);
            fa[1] = trellis::decode_pair<trellis::BOOK_MUL1>(st[4], st[5]);
            fa[2] = trellis::decode_pair<trellis::BOOK_MUL1>(st[2], st[3]);
            fa[3] = trellis::decode_pair<trellis::BOOK_MUL1>(st[6], st[7]);
        } else {
            fa[0] = trellis::decode_pair<trellis::BOOK_MCG>(st[0], st[1]);
            fa[1] = trellis::decode_pair<trellis::BOOK_MCG>(st[4], st[5]);
            fa[2] = trellis::decode_pair<trellis::BOOK_MCG>(st[2], st[3]);
            fa[3] = trellis::decode_pair<trellis::BOOK_MCG>(st[6], st[7]);
        }
    }
    u32 g = lane >> 2u;
    u32 p = lane & 3u;
    w_lds[g * STRIDE + p] = fa[0];
    w_lds[(g + 8u) * STRIDE + p] = fa[1];
    w_lds[g * STRIDE + p + 4u] = fa[2];
    w_lds[(g + 8u) * STRIDE + p + 4u] = fa[3];
    asm volatile("" ::: "memory");
    if constexpr (STRIDE == 10u) {
        const u32* wp = w_lds + (lane & 15u) * 10u;
        uint2 v0 = *reinterpret_cast<const uint2*>(wp);
        uint2 v1 = *reinterpret_cast<const uint2*>(wp + 2u);
        uint2 v2 = *reinterpret_cast<const uint2*>(wp + 4u);
        uint2 v3 = *reinterpret_cast<const uint2*>(wp + 6u);
        u32 out[8] = {v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};
        return __builtin_bit_cast(f16x16, out);
    }
    return *reinterpret_cast<const f16x16*>(w_lds + (lane & 15u) * STRIDE);
}

template <u32 NT>
TESSERA_FN void rocm_dtr_gen_run(dtr_smem<NT>& sm, const float* x, const u8* w, const float* suh,
                                 const float* svh, float* y, u32 m, u32 n, u32 k, u32 ks, u32 bits,
                                 u32 book, u32 turned) {
    dtr_cta c = init_dtr_cta<0u, false, NT>(w, n, k, ks, bits);
    bool split_on = ks > 1u;
    bool rot = turned != 0u;
    v8f acc[NT];
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) acc[ti] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

    u32 tw = 8u * bits;
    u32 nw[8];
    if (c.nb > 0u) blk_load_dyn(c.gb, c.tid, bits, c.aligned, nw);

    for (u32 g = 0u; g < c.nb; g++) {
        if (g > 0u) prim::block_sync();
        blk_store_dyn(nw, c.tid, bits, sm.k.wtiles);
        stage_dtr_x<NT>(sm, x, suh, c.r0, m, k, (c.c0 + g) * 128u + c.lane * 4u, c.j, c.lane, rot);
        if (g + 1u < c.nb) {
            c.gb += c.sG;
            blk_load_dyn(c.gb, c.tid, bits, c.aligned, nw);
        }
        u32* w_buf = sm.k.w_lds[c.j];
#pragma clang loop unroll(disable)
        for (u32 u = 0u; u < 8u; u++) {
            u32 raw = (c.lane < tw) ? sm.k.wtiles[(u * 8u + c.j) * tw + c.lane] : 0u;
            f16x16 wf = decode_tile_dyn<12u>(raw, c.lane, bits, book, w_buf);
            wmma_dtr_x<NT>(sm, wf, acc, u, c.lane);
        }
    }

    prim::block_sync();
    if (rot && !split_on) {
        store_out_block<16u * NT, true, false>(acc, sm.out, svh, nullptr, nullptr, y, c.r0, m, n,
                                               c.ob, 0u, 0u, c.j, c.lane);
    } else if (split_on) {
        store_out_block<16u * NT, false, true>(acc, sm.out, svh, nullptr, nullptr, y, c.r0, m, n,
                                               c.ob, c.split, 0u, c.j, c.lane);
    } else {
        store_out_block<16u * NT, false, false>(acc, sm.out, svh, nullptr, nullptr, y, c.r0, m, n,
                                                c.ob, 0u, 0u, c.j, c.lane);
    }
}

// Branchless SWAR packed f16x2-to-E4M3 RNE for bounded codebook pairs.
TESSERA_FN u32 narrow_f16x2_swar(u32 h) {
    u32 sign = h & 0x80008000u;
    u32 mag = h & 0x7FFF7FFFu;
    trellis::f16x2 av = __builtin_bit_cast(trellis::f16x2, mag);
    const trellis::f16x2 two = {static_cast<_Float16>(2.0f), static_cast<_Float16>(2.0f)};
    u32 sub = __builtin_bit_cast(u32, (av + two) - two);
    u32 norm = (mag + 0x003F003Fu + ((mag >> 7u) & 0x00010001u)) & 0xFF80FF80u;
    u32 ge = (mag + 0x5C005C00u) & 0x80008000u;
    u32 mask = ge - (ge >> 15u);
    return sign | ((norm & mask) | (sub & ~mask));
}

// Full-rate 24-bit multiply by the MCG codebook constant.
TESSERA_FN u32 mul_mcg_u24(u32 a) {
    return __umul24(a, 0x00AC1FEDu) + (__umul24(a, 0xCBu) << 24u);
}

// Decodes one MCG pair into E4M3-narrowed f16x2 using 24-bit multiplies.
TESSERA_FN u32 decode_mcg_pair(u32 a, u32 b) {
    u32 x = (mul_mcg_u24(a) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    u32 y = (mul_mcg_u24(b) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    u32 lo_pair = __builtin_amdgcn_perm(y, x, 0x05040100u);
    u32 hi_pair = __builtin_amdgcn_perm(y, x, 0x07060302u);
    trellis::f16x2 sum =
        __builtin_bit_cast(trellis::f16x2, lo_pair) + __builtin_bit_cast(trellis::f16x2, hi_pair);
    return narrow_f16x2_swar(__builtin_bit_cast(u32, sum));
}

// Decodes one K4 MCG tile into an RDNA WMMA fragment.
TESSERA_FN f16x16 decode_k4_mcg_hip(u32 raw, const trellis::word_map& wm, u32 lane, u32* w_lds) {
    u32 a = trellis::shfl_u32(raw, wm.i0);
    u32 hi = __builtin_amdgcn_alignbit(a, raw, 12u);
    u32 lo = raw << 20u;
    u32 mid = __builtin_amdgcn_alignbit(hi, lo, 16u);
    u32 fa0 = decode_mcg_pair(__builtin_amdgcn_ubfe(hi, 16u, 16u),
                              __builtin_amdgcn_ubfe(hi, 12u, 16u));
    u32 fa2 = decode_mcg_pair(__builtin_amdgcn_ubfe(hi, 8u, 16u),
                              __builtin_amdgcn_ubfe(hi, 4u, 16u));
    u32 fa1 = decode_mcg_pair(__builtin_amdgcn_ubfe(hi, 0u, 16u),
                              __builtin_amdgcn_ubfe(mid, 12u, 16u));
    u32 fa3 = decode_mcg_pair(__builtin_amdgcn_ubfe(mid, 8u, 16u),
                              __builtin_amdgcn_ubfe(mid, 4u, 16u));
    u32 g = lane >> 2u;
    u32 p = lane & 3u;
    w_lds[g * 10u + p] = fa0;
    w_lds[(g + 8u) * 10u + p] = fa1;
    w_lds[g * 10u + p + 4u] = fa2;
    w_lds[(g + 8u) * 10u + p + 4u] = fa3;
    asm volatile("" ::: "memory");
    const u32* wp = w_lds + (lane & 15u) * 10u;
    uint2 v0 = *reinterpret_cast<const uint2*>(wp);
    uint2 v1 = *reinterpret_cast<const uint2*>(wp + 2u);
    uint2 v2 = *reinterpret_cast<const uint2*>(wp + 4u);
    uint2 v3 = *reinterpret_cast<const uint2*>(wp + 6u);
    u32 out[8] = {v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};
    return __builtin_bit_cast(f16x16, out);
}

template <u32 K, u32 BOOK>
TESSERA_FN f16x16 decode_pre_tile(u32 raw, const trellis::word_map& wm, u32 lane, u32* w_buf) {
    if constexpr (K == 4u && BOOK == trellis::BOOK_MCG) {
        return decode_k4_mcg_hip(raw, wm, lane, w_buf);
    }
    return decode_tile_dyn<10u>(raw, lane, K, BOOK, w_buf);
}

template <u32 NT>
TESSERA_FN void wmma_pre_tiles(const _Float16* const pre_p[NT / 4u], u32 u_off, f16x16 wf,
                               v8f (&acc)[NT], u32 rem_t, bool full_m) {
#pragma unroll
    for (u32 t = 0u; t < NT; t++) {
        if (full_m || t < rem_t) {
            u32 h = t >> 2u;
            u32 s = t & 3u;
            f16x16 xh = *reinterpret_cast<const f16x16*>(pre_p[h] + u_off + s * 256u);
            acc[t] = trellis::wmma_f16(acc[t], xh, wf);
        }
    }
}

template <u32 NT, bool TURNED, bool SPLIT, bool ROT_OUT = false>
TESSERA_FN void store_pre_tiles(const v8f (&acc)[NT], float sm_out[32][128], const float* svh,
                                const float* other, const u8* bias_p, float* y, const dtr_cta& c,
                                u32 m, u32 n, u32 mode) {
#pragma unroll
    for (u32 q = 0u; q < NT / 2u; q++) {
        if (c.r0 + q * 32u >= m) break;
        prim::block_sync();
        store_out_block<32u, TURNED, SPLIT, ROT_OUT>(&acc[q * 2u], sm_out, svh, other, bias_p, y,
                                                     c.r0 + q * 32u, m, n, c.ob, c.split, mode, c.j,
                                                     c.lane);
    }
}

template <u32 NT>
TESSERA_FN void init_pre_ptrs(const u16* pre, const dtr_cta& c, u32 m, u32 mode,
                              const _Float16* (&pre_p)[NT / 4u], v8f (&acc)[NT]) {
    u32 n_mb = (m + 63u) >> 6u;
    pre += u64((mode >> 6u) & 3u) * u64(n_mb) * c.ibs * 8192u;
    u32 mb0 = blockIdx.y * (NT / 4u);
    u32 lane16 = c.lane & 15u;
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) acc[ti] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (u32 h = 0u; h < NT / 4u; h++) {
        u32 mb = mb0 + h;
        if (mb >= n_mb) mb = n_mb - 1u;
        pre_p[h] = reinterpret_cast<const _Float16*>(pre + (u64(mb) * c.ibs + c.c0) * 8192u) +
                   lane16 * 16u;
    }
}

// Prefill WMMA reading pre-rotated f16 directly with ping-pong wtiles.
template <u32 K, u32 BOOK, bool TURNED, bool SPLIT, u32 NT, bool ROT_OUT = false>
TESSERA_FN void rocm_dtr_pre_run(pre_smem& sm, const u16* pre, const u8* w, const float* svh,
                                 const float* other, const u8* bias_p, float* y, u32 m, u32 n,
                                 u32 k, u32 mode, u32 ks) {
    dtr_cta c = init_dtr_cta<K, SPLIT, NT>(w, n, k, ks, K);
    v8f acc[NT];
    const _Float16* pre_p[NT / 4u];
    init_pre_ptrs<NT>(pre, c, m, mode, pre_p, acc);

    constexpr u32 TW = 8u * K;
    constexpr u32 HALVES_64 = 8192u;
    trellis::word_map wm = trellis::map_word<K>(c.lane);
    u32 nw[8];
    if (c.nb > 0u) {
        trellis::blk_load<K>(c.gb, c.tid, c.aligned, nw);
        trellis::blk_store<K>(nw, c.tid, sm.k.wtiles[0]);
        prim::block_sync();
    }

    u32* w_buf = sm.k.w_lds[c.j];
    u32 rem_t = (c.r0 < m) ? ((m - c.r0 + 15u) >> 4u) : 0u;
    bool full_m = rem_t >= NT;

    for (u32 g = 0u; g < c.nb; g++) {
        u32 cur = g & 1u;
        u32 nxt = cur ^ 1u;
        if (g + 1u < c.nb) {
            c.gb += c.sG;
            trellis::blk_load<K>(c.gb, c.tid, c.aligned, nw);
        }

#pragma clang loop unroll(disable)
        for (u32 u = 0u; u < 8u; u++) {
            u32 u_off = u * 1024u;
            u32 raw =
                (K == 4u || c.lane < TW) ? sm.k.wtiles[cur][(u * 8u + c.j) * TW + c.lane] : 0u;
            if constexpr (NT == 16u) {
                const _Float16* xp0 = pre_p[0] + u_off;
                f16x16 x0 = *reinterpret_cast<const f16x16*>(xp0);
                f16x16 x1 = *reinterpret_cast<const f16x16*>(xp0 + 256u);
                f16x16 x2 = *reinterpret_cast<const f16x16*>(xp0 + 512u);
                f16x16 x3 = *reinterpret_cast<const f16x16*>(xp0 + 768u);
                f16x16 wf = decode_pre_tile<K, BOOK>(raw, wm, c.lane, w_buf);
#pragma unroll
                for (u32 h = 0u; h < 4u; h++) {
                    f16x16 nx0, nx1, nx2, nx3;
                    if (h + 1u < 4u) {
                        const _Float16* nxp = pre_p[h + 1u] + u_off;
                        nx0 = *reinterpret_cast<const f16x16*>(nxp);
                        nx1 = *reinterpret_cast<const f16x16*>(nxp + 256u);
                        nx2 = *reinterpret_cast<const f16x16*>(nxp + 512u);
                        nx3 = *reinterpret_cast<const f16x16*>(nxp + 768u);
                    }
                    acc[h * 4u + 0u] = trellis::wmma_f16(acc[h * 4u + 0u], x0, wf);
                    acc[h * 4u + 1u] = trellis::wmma_f16(acc[h * 4u + 1u], x1, wf);
                    acc[h * 4u + 2u] = trellis::wmma_f16(acc[h * 4u + 2u], x2, wf);
                    acc[h * 4u + 3u] = trellis::wmma_f16(acc[h * 4u + 3u], x3, wf);
                    if (h + 1u < 4u) {
                        x0 = nx0;
                        x1 = nx1;
                        x2 = nx2;
                        x3 = nx3;
                    }
                }
            } else if constexpr (NT == 8u && SPLIT) {
                f16x16 x_cur = *reinterpret_cast<const f16x16*>(pre_p[0] + u_off);
                f16x16 wf = decode_pre_tile<K, BOOK>(raw, wm, c.lane, w_buf);
#pragma unroll
                for (u32 t = 0u; t < NT; t++) {
                    if (full_m || t < rem_t) {
                        f16x16 x_nxt;
                        if (t + 1u < NT && (full_m || t + 1u < rem_t)) {
                            u32 nt = t + 1u;
                            x_nxt = *reinterpret_cast<const f16x16*>(pre_p[nt >> 2u] + u_off +
                                                                     (nt & 3u) * 256u);
                        }
                        acc[t] = trellis::wmma_f16(acc[t], x_cur, wf);
                        x_cur = x_nxt;
                    }
                }
            } else {
                f16x16 wf = decode_pre_tile<K, BOOK>(raw, wm, c.lane, w_buf);
                wmma_pre_tiles<NT>(pre_p, u_off, wf, acc, rem_t, full_m);
            }
        }
#pragma unroll
        for (u32 h = 0u; h < NT / 4u; h++) pre_p[h] += HALVES_64;

        if (g + 1u < c.nb) {
            trellis::blk_store<K>(nw, c.tid, sm.k.wtiles[nxt]);
            prim::block_sync();
        }
    }

    store_pre_tiles<NT, TURNED, SPLIT, ROT_OUT>(acc, sm.out, svh, other, bias_p, y, c, m, n, mode);
}

template <u32 NT>
TESSERA_FN void rocm_dtr_pre_gen_run(pre_smem& sm, const u16* pre, const u8* w, const float* svh,
                                     const float* other, const u8* bias_p, float* y, u32 m, u32 n,
                                     u32 k, u32 mode, u32 ks, u32 bits, u32 book, u32 turned) {
    dtr_cta c = init_dtr_cta<0u, false, NT>(w, n, k, ks, bits);
    v8f acc[NT];
    const _Float16* pre_p[NT / 4u];
    init_pre_ptrs<NT>(pre, c, m, mode, pre_p, acc);

    u32 tw = 8u * bits;
    u32 nw[8];
    if (c.nb > 0u) {
        blk_load_dyn(c.gb, c.tid, bits, c.aligned, nw);
        blk_store_dyn(nw, c.tid, bits, sm.k.wtiles[0]);
        prim::block_sync();
    }

    u32 rem_t = (c.r0 < m) ? ((m - c.r0 + 15u) >> 4u) : 0u;
    for (u32 g = 0u; g < c.nb; g++) {
        u32 cur = g & 1u;
        if (g + 1u < c.nb) {
            c.gb += c.sG;
            blk_load_dyn(c.gb, c.tid, bits, c.aligned, nw);
        }
#pragma clang loop unroll(disable)
        for (u32 u = 0u; u < 8u; u++) {
            u32 raw = (c.lane < tw) ? sm.k.wtiles[cur][(u * 8u + c.j) * tw + c.lane] : 0u;
            f16x16 wf = decode_tile_dyn<10u>(raw, c.lane, bits, book, sm.k.w_lds[c.j]);
            wmma_pre_tiles<NT>(pre_p, u * 1024u, wf, acc, rem_t, false);
        }
#pragma unroll
        for (u32 h = 0u; h < NT / 4u; h++) pre_p[h] += 8192u;
        if (g + 1u < c.nb) {
            blk_store_dyn(nw, c.tid, bits, sm.k.wtiles[cur ^ 1u]);
            prim::block_sync();
        }
    }

    if (turned != 0u && ks <= 1u) {
        store_pre_tiles<NT, true, false>(acc, sm.out, svh, other, bias_p, y, c, m, n, mode);
    } else if (ks > 1u) {
        store_pre_tiles<NT, false, true>(acc, sm.out, svh, other, bias_p, y, c, m, n, 0u);
    } else {
        store_pre_tiles<NT, false, false>(acc, sm.out, svh, other, bias_p, y, c, m, n, mode);
    }
}

}  // namespace

#define ROCM_DTR_DEF(NAME, K, B, T, S, NT)                                                  \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                 \
        const float* x, const u8* w, const float* suh, const float* svh, float* y, u32 m,   \
        u32 n, u32 k, u32 ks, u32, u32, u32) {                                              \
        __shared__ dtr_smem<NT> sm;                                                         \
        rocm_dtr_run<K, B, T, S, NT>(sm, x, w, suh, svh, y, m, n, k, ks);                  \
    }

ROCM_DTR_DEF(rocm_dtr_k4_t1, 4u, trellis::BOOK_MCG, true, false, 1u)
ROCM_DTR_DEF(rocm_dtr_k4_t1s, 4u, trellis::BOOK_MCG, true, true, 1u)
ROCM_DTR_DEF(rocm_dtr_k4_p1, 4u, trellis::BOOK_MCG, false, false, 1u)
ROCM_DTR_DEF(rocm_dtr_k4_p1s, 4u, trellis::BOOK_MCG, false, true, 1u)
ROCM_DTR_DEF(rocm_dtr_k4_t2, 4u, trellis::BOOK_MCG, true, false, 2u)
ROCM_DTR_DEF(rocm_dtr_k4_t2s, 4u, trellis::BOOK_MCG, true, true, 2u)
#undef ROCM_DTR_DEF

extern "C" __global__ __launch_bounds__(256) void rocm_dtr_gen1(
    const float* x, const u8* w, const float* suh, const float* svh, float* y, u32 m, u32 n, u32 k,
    u32 ks, u32 bits, u32 book, u32 turned) {
    __shared__ dtr_smem<1u> sm;
    rocm_dtr_gen_run<1u>(sm, x, w, suh, svh, y, m, n, k, ks, bits, book, turned);
}

extern "C" __global__ __launch_bounds__(256) void rocm_dtr_gen2(
    const float* x, const u8* w, const float* suh, const float* svh, float* y, u32 m, u32 n, u32 k,
    u32 ks, u32 bits, u32 book, u32 turned) {
    __shared__ dtr_smem<2u> sm;
    rocm_dtr_gen_run<2u>(sm, x, w, suh, svh, y, m, n, k, ks, bits, book, turned);
}

extern "C" __global__ __launch_bounds__(256) void rocm_dtr_reduce(
    const float* part, const float* svh, float* y, u32 m, u32 n, u32 ks, u32 turned) {
    u32 j = threadIdx.x >> 5u;
    u32 lane = threadIdx.x & 31u;
    u32 ob = blockIdx.x;
    u32 c_base = ob * 128u + lane * 4u;
    float sv[4];
    if (turned != 0u) trellis::load4(svh + c_base, sv);
    u64 stride = u64(m) * n;
    for (u32 r = j; r < m; r += 8u) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (u32 s = 0u; s < ks; s++) {
            float p[4];
            trellis::load4(part + u64(s) * stride + u64(r) * n + c_base, p);
#pragma unroll
            for (u32 i = 0u; i < 4u; i++) acc[i] += p[i];
        }
        if (turned != 0u) {
            trellis::wht128(acc, trellis::INV_SQRT128);
#pragma unroll
            for (u32 i = 0u; i < 4u; i++) acc[i] *= sv[i];
        }
        *reinterpret_cast<float4*>(y + u64(r) * n + c_base) =
            make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
}

extern "C" __global__ __launch_bounds__(256) void rocm_dtr_reduce_epi(
    const float* part, const float* svh, const float* other, const u8* bias_p, float* y, u32 m,
    u32 n, u32 ks, u32 turned, u32 mode) {
    u32 tid = threadIdx.x;
    u32 j = tid >> 5u;
    u32 lane = tid & 31u;
    u32 r = (blockIdx.y << 3u) + j;
    u32 ob = blockIdx.x;
    u32 nblk = n >> 7u;
    if (ob >= nblk) return;
    bool rot_out = (mode & 32u) != 0u;
    if (!rot_out && r >= m) return;
    u32 c_base = ob * 128u + lane * 4u;
    u32 eb = mode & 15u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (r < m) {
        bool bf16 = (mode & 16u) != 0u;
        float sv[4];
        float bias[4] = {};
        if (turned != 0u) trellis::load4(svh + c_base, sv);
        if (eb == 2u || eb == 6u || eb == 7u) {
            load_bias4(reinterpret_cast<const u8*>(other), c_base, bf16, bias);
        } else if (!rot_out && eb == 8u) {
            load_bias4(bias_p, c_base, bf16, bias);
        }
        u64 stride = u64(m) * n;
        for (u32 s = 0u; s < ks; s++) {
            float p[4];
            trellis::load4(part + u64(s) * stride + u64(r) * n + c_base, p);
#pragma unroll
            for (u32 i = 0u; i < 4u; i++) acc[i] += p[i];
        }
        if (turned != 0u) {
            trellis::wht128(acc, trellis::INV_SQRT128);
#pragma unroll
            for (u32 i = 0u; i < 4u; i++) acc[i] *= sv[i];
        }
        if (eb != 0u) {
            float4 ev = apply_epi4(other, u64(r) * n + c_base,
                                   make_float4(bias[0], bias[1], bias[2], bias[3]),
                                   make_float4(acc[0], acc[1], acc[2], acc[3]), mode);
            acc[0] = ev.x;
            acc[1] = ev.y;
            acc[2] = ev.z;
            acc[3] = ev.w;
        }
        if (rot_out) {
            float suh_n[4];
            trellis::load4(reinterpret_cast<const float*>(bias_p) + c_base, suh_n);
#pragma unroll
            for (u32 i = 0u; i < 4u; i++) acc[i] *= suh_n[i];
            trellis::wht128(acc, trellis::INV_SQRT128);
        }
    }
    if (rot_out) {
        typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
        __shared__ alignas(16) _Float16 sh[8][8][16];
        u32 u = lane >> 2u;
        u32 kk = (lane & 3u) << 2u;
        *reinterpret_cast<f16x4*>(&sh[u][j][kk]) =
            (f16x4){static_cast<_Float16>(acc[0]), static_cast<_Float16>(acc[1]),
                    static_cast<_Float16>(acc[2]), static_cast<_Float16>(acc[3])};
        prim::block_sync();
        if (tid < 128u) {
            u32 su = tid >> 4u;
            u32 sq = tid & 15u;
            u32 mb = blockIdx.y >> 3u;
            u32 ti = blockIdx.y & 7u;
            u16* y16 = reinterpret_cast<u16*>(y);
            u64 off = (u64(mb) * nblk + ob) * 8192u + u64(su) * 1024u + u64(ti) * 128u;
            reinterpret_cast<uint4*>(y16 + off)[sq] = reinterpret_cast<const uint4*>(sh[su])[sq];
        }
    } else {
        *reinterpret_cast<float4*>(y + u64(r) * n + c_base) =
            make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
}

extern "C" __global__ __launch_bounds__(256) void rocm_dtr_prep(
    const float* x, const float* suh, u16* out, u32 m, u32 k, u32 turned) {
    u32 tid = threadIdx.x;
    u32 j = tid >> 5u;
    u32 lane = tid & 31u;
    u32 ib = blockIdx.x;
    u32 mb = blockIdx.y;
    u32 ibs = k >> 7u;
    u32 r0 = mb * 64u;
    u32 k_off = ib * 128u + lane * 4u;

    __shared__ alignas(16) _Float16 sh[8][64][16];
    float sv[4];
    if (turned != 0u) trellis::load4(suh + k_off, sv);

    u32 u = lane >> 2u;
    u32 kk = (lane & 3u) << 2u;
    for (u32 r = j; r < 64u; r += 8u) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (r0 + r < m) {
            trellis::load4(x + u64(r0 + r) * k + k_off, v);
            if (turned != 0u) {
#pragma unroll
                for (u32 i = 0u; i < 4u; i++) v[i] *= sv[i];
                trellis::wht128(v, trellis::INV_SQRT128);
            }
        }
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) sh[u][r][kk + i] = static_cast<_Float16>(v[i]);
    }
    prim::block_sync();

    const uint4* src = reinterpret_cast<const uint4*>(sh);
    uint4* dst = reinterpret_cast<uint4*>(out + (u64(mb) * ibs + ib) * 8192u);
#pragma unroll
    for (u32 v = 0u; v < 4u; v++) dst[v * 256u + tid] = src[v * 256u + tid];
}

#define ROCM_DTR_PRE_DEF(NAME, NT, SPLIT, ROT)                                                    \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                       \
        const u16* pre, const u8* w, const float* svh, const float* other, const u8* bias_p,      \
        float* y, u32 m, u32 n, u32 k, u32, u32, u32, u32 mode, u32 ks) {                         \
        __shared__ pre_smem sm;                                                                   \
        rocm_dtr_pre_run<4u, trellis::BOOK_MCG, true, SPLIT, NT, ROT>(sm, pre, w, svh, other,     \
                                                                      bias_p, y, m, n, k, mode,   \
                                                                      ks);                        \
    }

ROCM_DTR_PRE_DEF(rocm_dtr_pre_k4_t4, 4u, false, false)
ROCM_DTR_PRE_DEF(rocm_dtr_pre_k4_t4s, 4u, true, false)
ROCM_DTR_PRE_DEF(rocm_dtr_pre_k4_t4r, 4u, false, true)
ROCM_DTR_PRE_DEF(rocm_dtr_pre_k4_t8, 8u, false, false)
ROCM_DTR_PRE_DEF(rocm_dtr_pre_k4_t8s, 8u, true, false)
ROCM_DTR_PRE_DEF(rocm_dtr_pre_k4_t8r, 8u, false, true)
ROCM_DTR_PRE_DEF(rocm_dtr_pre_k4_t16, 16u, false, false)
ROCM_DTR_PRE_DEF(rocm_dtr_pre_k4_t16r, 16u, false, true)
#undef ROCM_DTR_PRE_DEF

#define ROCM_DTR_PGEN_DEF(NAME, NT)                                                             \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                     \
        const u16* pre, const u8* w, const float* svh, const float* other, const u8* bias_p,    \
        float* y, u32 m, u32 n, u32 k, u32 bits, u32 book, u32 turned, u32 mode, u32 ks) {      \
        __shared__ pre_smem sm;                                                                 \
        rocm_dtr_pre_gen_run<NT>(sm, pre, w, svh, other, bias_p, y, m, n, k, mode, ks, bits,    \
                                 book, turned);                                                 \
    }

ROCM_DTR_PGEN_DEF(rocm_dtr_pre_gen4, 4u)
ROCM_DTR_PGEN_DEF(rocm_dtr_pre_gen8, 8u)
ROCM_DTR_PGEN_DEF(rocm_dtr_pre_gen16, 16u)
#undef ROCM_DTR_PGEN_DEF

#endif
