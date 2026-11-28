#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_CUDA)

#if !defined(DTR_FORCE_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define DTR_MX8 1
constexpr u32 DTR_TILE = trellis::MX8_TILE;
#else
#define DTR_MX8 0
constexpr u32 DTR_TILE = 512u;
#endif

#ifndef DTR_T8_BLK
#define DTR_T8_BLK 2
#endif
#ifndef DTR_PREP_BLK
#define DTR_PREP_BLK 3
#endif
#ifndef DTR_T16_BLK
#define DTR_T16_BLK 2
#endif
#ifndef DTR_GEN8_BLK
#define DTR_GEN8_BLK 2
#endif
#ifndef DTR_GEN16_BLK
#define DTR_GEN16_BLK 2
#endif
#ifndef MM_PREP_BLK
#define MM_PREP_BLK 4
#endif
#ifndef MM_MX8_BLK
#define MM_MX8_BLK 1
#endif
#ifndef MM_MX8_SK_BLK
#define MM_MX8_SK_BLK 1
#endif

constexpr u32 DTR_STRIDE = 8u * DTR_TILE;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void pre_cp16(void* smem, const void* gmem, bool pred) {
    u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem));
    if (pred) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem));
    else *reinterpret_cast<uint4*>(smem) = make_uint4(0u, 0u, 0u, 0u);
}
static __device__ __forceinline__ void pre_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
static __device__ __forceinline__ void pre_cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
static __device__ __forceinline__ void pre_cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
#else
static __device__ __forceinline__ void pre_cp16(void* smem, const void* gmem, bool pred) {
    *reinterpret_cast<uint4*>(smem) = pred ? *reinterpret_cast<const uint4*>(gmem) : make_uint4(0u, 0u, 0u, 0u);
}
static __device__ __forceinline__ void pre_cp_commit() {}
static __device__ __forceinline__ void pre_cp_wait1() {}
static __device__ __forceinline__ void pre_cp_wait0() {}
#endif

static __device__ __forceinline__ float mx8_silu(float v) { return v / (1.0f + expf(-v)); }
static __device__ __forceinline__ float pre_gelu_tanh(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
static __device__ __forceinline__ float pre_gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f)); }
static __device__ __forceinline__ float pre_load_b1(const u8* p, u32 col, bool bf16) {
    return bf16 ? __uint_as_float(u32(reinterpret_cast<const u16*>(p)[col]) << 16u)
                : reinterpret_cast<const float*>(p)[col];
}
static __device__ __forceinline__ void pre_load_b4(const u8* p, u32 col, bool bf16, float b[4]) {
    if (bf16) {
        uint2 w = *reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(p) + col);
        b[0] = __uint_as_float(w.x << 16u); b[1] = __uint_as_float(w.x & 0xFFFF0000u);
        b[2] = __uint_as_float(w.y << 16u); b[3] = __uint_as_float(w.y & 0xFFFF0000u);
    } else {
        trellis::load4(reinterpret_cast<const float*>(p) + col, b);
    }
}

// Reads a bias from `other` as bf16 or `o_t`.
static __device__ __forceinline__ float pre_bias1(const trellis::o_t* other, u32 col, bool bf16) {
    return bf16 ? pre_load_b1(reinterpret_cast<const u8*>(other), col, true)
                : trellis::load_o(other, col);
}
static __device__ __forceinline__ void pre_bias4(const trellis::o_t* other, u32 col, bool bf16,
                                                 float b[4]) {
    if (bf16) pre_load_b4(reinterpret_cast<const u8*>(other), col, true, b);
    else trellis::load_o4(other + col, b);
}

static __device__ __forceinline__ float mx8_epilogue(float v, const trellis::o_t* other, u64 off,
                                                     u32 col, u32 mode,
                                                     const u8* bias_p = nullptr) {
    u32 base = mode & 15u;
    bool bf16 = (mode & 16u) != 0u;
    if (base == 1u) return v + trellis::load_o(other, off);
    if (base == 2u) return v + pre_bias1(other, col, bf16);
    if (base == 3u) return mx8_silu(v) * trellis::load_o(other, off);
    if (base == 4u) return v * trellis::load_o(other, off);
    if (base == 5u) return mx8_silu(v);
    if (base == 6u) return pre_gelu_tanh(v + pre_bias1(other, col, bf16));
    if (base == 7u) return pre_gelu(v + pre_bias1(other, col, bf16));
    if (base == 8u) return v + pre_load_b1(bias_p, col, bf16) + trellis::load_o(other, off);
    return v;
}

static __device__ __forceinline__ void pre_apply_epi4(const trellis::o_t* other, u64 off,
                                                      const float b[4], float v[4], u32 mode) {
    u32 base = mode & 15u;
    if (base == 1u || base == 3u || base == 4u || base == 8u) {
        float o[4];
        trellis::load_o4(other + off, o);
        if (base == 1u) { v[0] += o[0]; v[1] += o[1]; v[2] += o[2]; v[3] += o[3]; }
        else if (base == 8u) { v[0] += b[0] + o[0]; v[1] += b[1] + o[1]; v[2] += b[2] + o[2]; v[3] += b[3] + o[3]; }
        else if (base == 3u) { v[0] = mx8_silu(v[0]) * o[0]; v[1] = mx8_silu(v[1]) * o[1]; v[2] = mx8_silu(v[2]) * o[2]; v[3] = mx8_silu(v[3]) * o[3]; }
        else { v[0] *= o[0]; v[1] *= o[1]; v[2] *= o[2]; v[3] *= o[3]; }
    } else if (base == 2u) { v[0] += b[0]; v[1] += b[1]; v[2] += b[2]; v[3] += b[3]; }
    else if (base == 5u) { v[0] = mx8_silu(v[0]); v[1] = mx8_silu(v[1]); v[2] = mx8_silu(v[2]); v[3] = mx8_silu(v[3]); }
    else if (base == 6u) { v[0] = pre_gelu_tanh(v[0] + b[0]); v[1] = pre_gelu_tanh(v[1] + b[1]); v[2] = pre_gelu_tanh(v[2] + b[2]); v[3] = pre_gelu_tanh(v[3] + b[3]); }
    else if (base == 7u) { v[0] = pre_gelu(v[0] + b[0]); v[1] = pre_gelu(v[1] + b[1]); v[2] = pre_gelu(v[2] + b[2]); v[3] = pre_gelu(v[3] + b[3]); }
}
static __device__ __forceinline__ void pre_store_epi4(trellis::y_t* out,
                                                      const trellis::o_t* other, u64 off,
                                                      const float b[4], float v[4], u32 mode) {
    pre_apply_epi4(other, off, b, v, mode);
    trellis::store_y4(out + off, make_float4(v[0], v[1], v[2], v[3]));
}

template <bool SPLIT>
static __device__ __forceinline__ void pre_store_r64(const float acc[4][4],
                                                     const trellis::o_t* other, const u8* bias_p,
                                                     trellis::y_t* out, u32 m, u32 n,
                                                     u32 r0, u32 n0, u32 m_base, u32 n_base, u32 p,
                                                     u32 g, u32 t, u32 mode) {
    u32 col0 = n0 + n_base + g, col1 = col0 + 8u;
#pragma unroll
    for (u32 ti = 0u; ti < 4u; ti++) {
        u32 row0 = r0 + m_base + (ti << 3u) + (t << 1u);
        if (row0 < m) {
            u64 off0 = u64(SPLIT ? (p * m + row0) : row0) * n;
            if (col0 < n) trellis::store_y(out, off0 + col0, SPLIT ? acc[ti][0] : mx8_epilogue(acc[ti][0], other, off0 + col0, col0, mode, bias_p));
            if (col1 < n) trellis::store_y(out, off0 + col1, SPLIT ? acc[ti][2] : mx8_epilogue(acc[ti][2], other, off0 + col1, col1, mode, bias_p));
        }
        if (row0 + 1u < m) {
            u64 off1 = u64(SPLIT ? (p * m + row0 + 1u) : (row0 + 1u)) * n;
            if (col0 < n) trellis::store_y(out, off1 + col0, SPLIT ? acc[ti][1] : mx8_epilogue(acc[ti][1], other, off1 + col0, col0, mode, bias_p));
            if (col1 < n) trellis::store_y(out, off1 + col1, SPLIT ? acc[ti][3] : mx8_epilogue(acc[ti][3], other, off1 + col1, col1, mode, bias_p));
        }
    }
}

static __device__ __forceinline__ void pre_swizzle(u32 nb, u32 mb, u32 ob_in, u32 by_in, u32& bx,
                                                   u32& by, u32 sw = 8u) {
    u32 lin = by_in * nb + ob_in, grp_sz = sw * mb;
    u32 gid = lin / grp_sz, rem = lin - gid * grp_sz;
    u32 first_n = gid * sw, width = min(sw, nb - first_n);
    by = rem / width;
    bx = first_n + (rem - by * width);
}

#if !defined(PRE_ONLY_MM) && !defined(PRE_ONLY_MX8)

template <u32 NT>
union alignas(16) dtr_pre_smem {
    static constexpr u32 BS = (NT == 8u) ? 2u : 1u;
    struct { u32 btab[BS][NT * DTR_TILE]; u32 wtiles[2][2048]; } k;
    struct { float out[32][128]; u32 rot[4 * DTR_TILE]; } e;
};

static __device__ __forceinline__ void dtr_store_frag(u32* tile, u32 row, u32 lane, bool valid,
                                                      const float v[4]) {
#if DTR_MX8
    (void)valid;
    trellis::store_bfrag_mx8(tile, row, lane, v);
#else
    if (valid) trellis::store_bfrag(tile, row, lane, v);
    else {
        float z[4] = {};
        trellis::store_bfrag(tile, row, lane, z);
    }
#endif
}

template <u32 K, u32 BOOK, u32 NT>
static __device__ __forceinline__ void dtr_run_mma(const u32* wtiles, const u32* btab, u32 j,
                                                   u32 lane, u32 active_ti,
                                                   const trellis::word_map& wm,
                                                   float acc[NT][4]) {
#if DTR_MX8
    (void)active_ti;
    trellis::blk_mma_mx8<K, BOOK, NT>(wtiles, btab, j, lane, wm, acc);
#else
    if constexpr (NT == 16u) {
        if (active_ti < 16u) {
            constexpr u32 TW = 8u * K;
            const u32* xb = btab + (lane << 1u);
#pragma unroll
            for (u32 u = 0u; u < 8u; u++) {
                u32 raw = (K == 4u || lane < TW) ? wtiles[(u * 8u + j) * TW + lane] : 0u;
                u32 fa[4];
                trellis::decode_word<K, BOOK>(raw, wm, fa);
#pragma unroll
                for (u32 ti = 0u; ti < 16u; ti++) {
                    if (ti < active_ti) {
                        uint2 b2 = *reinterpret_cast<const uint2*>(xb + ti * 512u + u * 64u);
                        u32 fb[2] = {b2.x, b2.y};
                        trellis::mma_m16n8k16(acc[ti], fa, fb);
                    }
                }
            }
            return;
        }
    }
    (void)active_ti;
    trellis::blk_mma<K, BOOK, NT>(wtiles, btab + (lane << 1u), 512u, j, lane, wm, acc);
#endif
}

// Rotates and quantizes dense inputs once per 64x128 block.
extern "C" __global__ __launch_bounds__(256, DTR_PREP_BLK) void dtr_prep(
    const trellis::x_t* x, const float* suh, float* out, u32 m, u32 k, u32 turned) {
    u32 c0 = blockIdx.x, rt = blockIdx.y, r0 = rt << 6u;
    if (r0 >= m) return;
    u32 nr = min(64u, m - r0), active_ti = (nr + 7u) >> 3u;
    u32 tid = threadIdx.x, lane = tid & 31u, j = tid >> 5u;
    __shared__ u32 btab[DTR_STRIDE];
    u32 col = (c0 << 7u) + (lane << 2u);
    float sc[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (turned != 0u) trellis::load4(suh + col, sc);
#pragma unroll
    for (u32 ti = 0u; ti < 8u; ti++) {
        if (ti >= active_ti) break;
        u32 r = (ti << 3u) + j;
        float v[4] = {};
        if (r < nr) {
            trellis::load_x4(x + u64(r0 + r) * k + col, v);
            if (turned != 0u) {
                v[0] *= sc[0]; v[1] *= sc[1]; v[2] *= sc[2]; v[3] *= sc[3];
                trellis::wht128(v, trellis::INV_SQRT128);
            }
        }
        dtr_store_frag(btab + ti * DTR_TILE, j, lane, r < nr, v);
    }
    prim::block_sync();
    u32 nblk = k >> 7u, n_v4 = active_ti * (DTR_TILE >> 2u);
    uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<u32*>(out) +
                                          (u64(rt) * nblk + c0) * DTR_STRIDE);
    const uint4* src = reinterpret_cast<const uint4*>(btab);
    for (u32 i = tid; i < n_v4; i += 256u) dst[i] = src[i];
}

template <u32 K>
static __device__ __forceinline__ void dtr_load_w(u32* dst, const u8* gb, bool aligned, u32 tid) {
    if (aligned) {
        pre_cp16(dst + (tid << 2u), gb + (tid << 4u), true);
        if constexpr (K == 3u) {
            if (tid < 128u) pre_cp16(dst + 1024u + (tid << 2u), gb + 4096u + (tid << 4u), true);
        } else if constexpr (K == 4u) {
            pre_cp16(dst + 1024u + (tid << 2u), gb + 4096u + (tid << 4u), true);
        }
        return;
    }
    u32 reg[8];
    trellis::blk_load<K>(gb, tid, false, reg);
    trellis::blk_store<K>(reg, tid, dst);
}

template <u32 NT>
static __device__ __forceinline__ void dtr_load_x(u32* dst_base, const u32* xpre, u32 c0,
                                                  u32 blk_total, u32 rblk, u32 active_ti,
                                                  u32 tid) {
    constexpr u32 SUBS = NT / 8u;
#pragma unroll
    for (u32 s = 0u; s < SUBS; s++) {
        u32 want = (active_ti > s * 8u) ? min(active_ti - s * 8u, 8u) : 0u;
        if (want > 0u) {
            u32 rt = rblk * SUBS + s, n_v4 = want * (DTR_TILE >> 2u);
            const u32* src = xpre + (u64(rt) * blk_total + c0) * DTR_STRIDE;
            u32* dst = dst_base + s * DTR_STRIDE;
            for (u32 i = tid; i < n_v4; i += 256u) {
                pre_cp16(dst + (i << 2u), src + (i << 2u), true);
            }
        }
    }
}

template <u32 K, u32 BOOK, u32 NT, bool TURNED, bool SPLIT, bool ROT_OUT = false>
static __device__ __forceinline__ void dtr_pre_body(const float* xpre_f, const u8* w,
                                                    const float* svh, const trellis::o_t* other,
                                                    const u8* bias_p, trellis::y_t* out, u32 m, u32 n,
                                                    u32 k, u32 mode, u32 ks,
                                                    dtr_pre_smem<NT>& sm) {
    constexpr u32 TB = 32u * K, BS = dtr_pre_smem<NT>::BS;
    u32 blk_total = k >> 7u, xslot = (mode >> 6u) & 3u;
    const u32* xpre = reinterpret_cast<const u32*>(xpre_f) +
                      u64(xslot) * u64((m + 63u) >> 6u) * blk_total * DTR_STRIDE;
    u32 nblk = n >> 7u, mb = (m + 8u * NT - 1u) / (8u * NT);
    u32 ob_in = SPLIT ? (blockIdx.x % nblk) : blockIdx.x;
    u32 p = SPLIT ? (blockIdx.x / nblk) : 0u;
    if (ob_in >= nblk || p >= ks || blockIdx.y >= mb) return;
    u32 ob = ob_in, rblk = blockIdx.y;
    if constexpr (!SPLIT) pre_swizzle(nblk, mb, ob_in, blockIdx.y, ob, rblk, 16u);
    u32 r0 = rblk * (8u * NT);
    if (r0 >= m) return;
    u32 nr = min(8u * NT, m - r0), active_ti = (nr + 7u) >> 3u;
    u32 tid = threadIdx.x, lane = tid & 31u, j = tid >> 5u;
    u32 blk_per = SPLIT ? (blk_total / ks) : blk_total, ib0 = SPLIT ? (p * blk_per) : 0u;
    bool aligned = (reinterpret_cast<u64>(w + trellis::HEAD) & 15ull) == 0ull;
    bool input_side = (*reinterpret_cast<const u32*>(w + 8u) & trellis::SIDE_MASK) == trellis::INPUT_SIDE;
    u64 sG = (input_side ? u64(n >> 1u) : 64ull) * u64(TB);
    const u8* gb = trellis::tile_ptr<K>(w, k, n, 0u, ib0 << 3u, ob, input_side);
    trellis::word_map wm = trellis::map_word<K>(lane);
    float acc[NT][4];
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) acc[ti][0] = acc[ti][1] = acc[ti][2] = acc[ti][3] = 0.0f;
    dtr_load_w<K>(sm.k.wtiles[0], gb, aligned, tid);
    if constexpr (BS == 2u) dtr_load_x<NT>(sm.k.btab[0], xpre, ib0, blk_total, rblk, active_ti, tid);
    pre_cp_commit();
    for (u32 c0 = 0u; c0 < blk_per; c0++) {
        u32 st = c0 & 1u, bst = (BS == 2u) ? st : 0u;
        bool has_nxt = (c0 + 1u < blk_per);
        if constexpr (BS == 1u) {
            dtr_load_x<NT>(sm.k.btab[0], xpre, ib0 + c0, blk_total, rblk, active_ti, tid);
            pre_cp_commit();
        }
        if (has_nxt) {
            gb += sG;
            dtr_load_w<K>(sm.k.wtiles[st ^ 1u], gb, aligned, tid);
            if constexpr (BS == 2u) {
                dtr_load_x<NT>(sm.k.btab[st ^ 1u], xpre, ib0 + c0 + 1u, blk_total, rblk, active_ti, tid);
            }
            pre_cp_commit();
            pre_cp_wait1();
        } else {
            pre_cp_wait0();
        }
        prim::block_sync();
        dtr_run_mma<K, BOOK, NT>(sm.k.wtiles[st], sm.k.btab[bst], j, lane, active_ti, wm, acc);
        if (has_nxt) prim::block_sync();
    }
    u32 col = (j << 4u) + (lane >> 2u), rc = (lane & 3u) << 1u;
    if constexpr (SPLIT) {
        float* pp = reinterpret_cast<float*>(out) + (u64(ob) * ks + p) * m * 128u;
#pragma unroll
        for (u32 ti = 0u; ti < NT; ti++) {
            if (ti >= active_ti) break;
            u32 r = r0 + (ti << 3u) + rc;
            if (r < m) { pp[u64(r) * 128u + col] = acc[ti][0]; pp[u64(r) * 128u + col + 8u] = acc[ti][2]; }
            if (r + 1u < m) { pp[u64(r + 1u) * 128u + col] = acc[ti][1]; pp[u64(r + 1u) * 128u + col + 8u] = acc[ti][3]; }
        }
        return;
    }
    if constexpr (!TURNED) {
        u32 ob0 = ob << 7u, c0 = ob0 + col, c1 = c0 + 8u;
#pragma unroll
        for (u32 ti = 0u; ti < NT; ti++) {
            if (ti >= active_ti) break;
            u32 r = r0 + (ti << 3u) + rc;
            if (r < m) {
                u64 off0 = u64(r) * n;
                trellis::store_y(out, off0 + c0, mx8_epilogue(acc[ti][0], other, off0 + c0, c0, mode, bias_p));
                trellis::store_y(out, off0 + c1, mx8_epilogue(acc[ti][2], other, off0 + c1, c1, mode, bias_p));
            }
            if (r + 1u < m) {
                u64 off1 = u64(r + 1u) * n;
                trellis::store_y(out, off1 + c0, mx8_epilogue(acc[ti][1], other, off1 + c0, c0, mode, bias_p));
                trellis::store_y(out, off1 + c1, mx8_epilogue(acc[ti][3], other, off1 + c1, c1, mode, bias_p));
            }
        }
        return;
    }
    u32 base = (ob << 7u) + (lane << 2u), eb = mode & 15u;
    bool bf16 = (mode & 16u) != 0u;
    float sc[4], bias[4] = {}, suh_n[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    trellis::load4(svh + base, sc);
    if (eb == 2u || eb == 6u || eb == 7u) pre_bias4(other, base, bf16, bias);
    else if (eb == 8u) pre_load_b4(bias_p, base, bf16, bias);
    if constexpr (ROT_OUT) trellis::load4(reinterpret_cast<const float*>(bias_p) + base, suh_n);
#pragma unroll
    for (u32 h = 0u; h < NT; h += 4u) {
        if (h >= active_ti) break;
        prim::block_sync();
#pragma unroll
        for (u32 ti = 0u; ti < 4u; ti++) {
            u32 r = ((h + ti) << 3u) + rc, lr = (ti << 3u) + rc;
            if (r < nr) { sm.e.out[lr][col] = acc[h + ti][0]; sm.e.out[lr][col + 8u] = acc[h + ti][2]; }
            if (r + 1u < nr) { sm.e.out[lr + 1u][col] = acc[h + ti][1]; sm.e.out[lr + 1u][col + 8u] = acc[h + ti][3]; }
        }
        prim::block_sync();
        if constexpr (ROT_OUT) {
            u32 n_ti = min(4u, active_ti - h);
#pragma unroll
            for (u32 ti = 0u; ti < 4u; ti++) {
                if (ti >= n_ti) break;
                u32 r = ((h + ti) << 3u) + j;
                float v[4] = {};
                if (r < nr) {
                    trellis::load4(&sm.e.out[(ti << 3u) + j][lane << 2u], v);
                    trellis::wht128(v, trellis::INV_SQRT128);
                    v[0] *= sc[0]; v[1] *= sc[1]; v[2] *= sc[2]; v[3] *= sc[3];
                    pre_apply_epi4(other, u64(r0 + r) * n + base, bias, v, mode);
                    v[0] *= suh_n[0]; v[1] *= suh_n[1]; v[2] *= suh_n[2]; v[3] *= suh_n[3];
                    trellis::wht128(v, trellis::INV_SQRT128);
                }
                dtr_store_frag(sm.e.rot + ti * DTR_TILE, j, lane, r < nr, v);
            }
            prim::block_sync();
            u32 rt = (r0 >> 6u) + (h >> 3u), n_v4 = n_ti * (DTR_TILE >> 2u);
            uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<u32*>(out) +
                                                  (u64(rt) * nblk + ob) * DTR_STRIDE + (h & 4u) * DTR_TILE);
            const uint4* src = reinterpret_cast<const uint4*>(sm.e.rot);
            for (u32 i = tid; i < n_v4; i += 256u) dst[i] = src[i];
        } else {
#pragma unroll
            for (u32 ti = 0u; ti < 4u; ti++) {
                u32 r = ((h + ti) << 3u) + j;
                if (r < nr) {
                    float v[4];
                    trellis::load4(&sm.e.out[(ti << 3u) + j][lane << 2u], v);
                    trellis::wht128(v, trellis::INV_SQRT128);
                    v[0] *= sc[0]; v[1] *= sc[1]; v[2] *= sc[2]; v[3] *= sc[3];
                    pre_store_epi4(out, other, u64(r0 + r) * n + base, bias, v, mode);
                }
            }
        }
    }
}

template <u32 K, u32 BOOK, u32 NT, bool SPLIT>
static __device__ __noinline__ void dtr_pre_k(const float* xpre, const u8* w, const float* svh, const trellis::o_t* other, const u8* bias_p, trellis::y_t* out, u32 m, u32 n, u32 k, u32 turned, u32 mode, u32 ks, dtr_pre_smem<NT>& sm) {
    if (turned != 0u) dtr_pre_body<K, BOOK, NT, true, SPLIT>(xpre, w, svh, other, bias_p, out, m, n, k, mode, ks, sm);
    else dtr_pre_body<K, BOOK, NT, false, SPLIT>(xpre, w, svh, other, bias_p, out, m, n, k, mode, ks, sm);
}
template <u32 BOOK, u32 NT, bool SPLIT>
static __device__ void dtr_pre_book(const float* xpre, const u8* w, const float* svh, const trellis::o_t* other, const u8* bias_p, trellis::y_t* out, u32 m, u32 n, u32 k, u32 bits, u32 turned, u32 mode, u32 ks, dtr_pre_smem<NT>& sm) {
    if (bits == 2u) dtr_pre_k<2u, BOOK, NT, SPLIT>(xpre, w, svh, other, bias_p, out, m, n, k, turned, mode, ks, sm);
    else if (bits == 3u) dtr_pre_k<3u, BOOK, NT, SPLIT>(xpre, w, svh, other, bias_p, out, m, n, k, turned, mode, ks, sm);
    else dtr_pre_k<4u, BOOK, NT, SPLIT>(xpre, w, svh, other, bias_p, out, m, n, k, turned, mode, ks, sm);
}
template <u32 NT, bool SPLIT>
static __device__ void dtr_pre_gen(const float* xpre, const u8* w, const float* svh, const trellis::o_t* other, const u8* bias_p, trellis::y_t* out, u32 m, u32 n, u32 k, u32 bits, u32 book, u32 turned, u32 mode, u32 ks, dtr_pre_smem<NT>& sm) {
    if (book == trellis::BOOK_PMCGX) dtr_pre_book<trellis::BOOK_PMCGX, NT, SPLIT>(xpre, w, svh, other, bias_p, out, m, n, k, bits, turned, mode, ks, sm);
    else if (book == trellis::BOOK_MUL1) dtr_pre_book<trellis::BOOK_MUL1, NT, SPLIT>(xpre, w, svh, other, bias_p, out, m, n, k, bits, turned, mode, ks, sm);
    else dtr_pre_book<trellis::BOOK_MCG, NT, SPLIT>(xpre, w, svh, other, bias_p, out, m, n, k, bits, turned, mode, ks, sm);
}

#define DTR_PRE_ARGS                                                                               \
    const float* xpre, const u8* w, const float* svh, const trellis::o_t* other, const u8* bias_p, \
        trellis::y_t* out, u32 m, u32 n, u32 k, u32 bits, u32 book, u32 turned, u32 mode, u32 ks
#define DTR_PRE_DEF(NAME, NT, SPLIT, ROT, MIN_BLK)                                                  \
    extern "C" __global__ __launch_bounds__(256, MIN_BLK) void NAME(DTR_PRE_ARGS) {                 \
        (void)bits; (void)book; (void)turned;                                                       \
        __shared__ dtr_pre_smem<NT> sm;                                                             \
        dtr_pre_body<4u, trellis::BOOK_MCG, NT, true, SPLIT, ROT>(xpre, w, svh, other, bias_p, out, m, n, k, mode, ks, sm); \
    }
DTR_PRE_DEF(dtr_pre_k4_t8, 8u, false, false, DTR_T8_BLK)
DTR_PRE_DEF(dtr_pre_k4_t8s, 8u, true, false, DTR_T8_BLK)
DTR_PRE_DEF(dtr_pre_k4_t8r, 8u, false, true, DTR_T8_BLK)
DTR_PRE_DEF(dtr_pre_k4_t16, 16u, false, false, DTR_T16_BLK)
DTR_PRE_DEF(dtr_pre_k4_t16s, 16u, true, false, DTR_T16_BLK)
DTR_PRE_DEF(dtr_pre_k4_t16r, 16u, false, true, DTR_T16_BLK)
#if !defined(PRE_ONLY_DTR_K4)
#define DTR_PGEN_DEF(NAME, NT, SPLIT, MIN_BLK)                                                     \
    extern "C" __global__ __launch_bounds__(256, MIN_BLK) void NAME(DTR_PRE_ARGS) {                \
        __shared__ dtr_pre_smem<NT> sm;                                                            \
        dtr_pre_gen<NT, SPLIT>(xpre, w, svh, other, bias_p, out, m, n, k, bits, book, turned, mode, ks, sm); \
    }
DTR_PGEN_DEF(dtr_pre_gen8, 8u, false, DTR_GEN8_BLK)
DTR_PGEN_DEF(dtr_pre_gen16, 16u, false, DTR_GEN16_BLK)
#endif

// Sums split-K partials with output rotation and fused epilogue.
extern "C" __global__ __launch_bounds__(256, 4) void dtr_reduce_epi(
    const float* part, const float* svh, const trellis::o_t* other, const u8* bias_p,
    trellis::y_t* out, u32 m, u32 n, u32 ks, u32 turned, u32 mode) {
    u32 ob = blockIdx.x, nblk = n >> 7u;
    if (ob >= nblk) return;
    u32 tid = threadIdx.x, lane = tid & 31u, j = tid >> 5u, r = (blockIdx.y << 3u) + j;
    bool rot_out = (mode & 32u) != 0u;
    if (!rot_out && r >= m) return;
    u32 base = (ob << 7u) + (lane << 2u), eb = mode & 15u;
    bool bf16 = (mode & 16u) != 0u;
    float v[4] = {};
    if (r < m) {
        float sc[4] = {1.0f, 1.0f, 1.0f, 1.0f}, bias[4] = {};
        if (turned != 0u) trellis::load4(svh + base, sc);
        if (eb == 2u || eb == 6u || eb == 7u) pre_bias4(other, base, bf16, bias);
        else if (eb == 8u) pre_load_b4(bias_p, base, bf16, bias);
        const float* pb = part + u64(ob) * ks * m * 128u + u64(r) * 128u + (lane << 2u);
        for (u32 p = 0u; p < ks; p++) {
            float4 w = trellis::ld_cg_f4(pb + u64(p) * m * 128u);
            v[0] += w.x; v[1] += w.y; v[2] += w.z; v[3] += w.w;
        }
        if (turned != 0u) {
            trellis::wht128(v, trellis::INV_SQRT128);
            v[0] *= sc[0]; v[1] *= sc[1]; v[2] *= sc[2]; v[3] *= sc[3];
        }
        pre_apply_epi4(other, u64(r) * n + base, bias, v, mode);
        if (!rot_out) {
            trellis::store_y4(out + u64(r) * n + base, make_float4(v[0], v[1], v[2], v[3]));
            return;
        }
        float suh_n[4];
        trellis::load4(reinterpret_cast<const float*>(bias_p) + base, suh_n);
        v[0] *= suh_n[0]; v[1] *= suh_n[1]; v[2] *= suh_n[2]; v[3] *= suh_n[3];
        trellis::wht128(v, trellis::INV_SQRT128);
    }
    __shared__ u32 sm_rot[DTR_TILE];
    dtr_store_frag(sm_rot, j, lane, r < m, v);
    prim::block_sync();
    constexpr u32 NV4 = DTR_TILE >> 2u;
    if (tid < NV4) {
        u32 rt = blockIdx.y >> 3u, ti = blockIdx.y & 7u;
        uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<u32*>(out) +
                                              (u64(rt) * nblk + ob) * DTR_STRIDE + ti * DTR_TILE);
        dst[tid] = reinterpret_cast<const uint4*>(sm_rot)[tid];
    }
}

#endif

#if !defined(DTR_FORCE_FP16) && !defined(PRE_ONLY_DTR_K4) && !defined(PRE_ONLY_MX8)

static __device__ __forceinline__ u32 pre_bf16x2(float lo, float hi) {
    u32 out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
    return out;
}

static __device__ __forceinline__ void pre_mma_bf16(float d[4], const u32 a[4], const u32 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ u32 pre_bf16x2_f16x2(u32 bw) {
    return trellis::f32x2_to_f16x2(__uint_as_float(bw << 16u), __uint_as_float(bw & 0xFFFF0000u));
}

struct alignas(16) pre_bf64_smem {
    u32 sw[2][64][20];
    u32 sxh[64][20];
    u32 sxl[64][20];
};

template <bool SPLIT>
static __device__ __forceinline__ void fg_bf16_r64(const trellis::x_t* x, const u16* w,
                                                   const trellis::o_t* other, const u8* bias_p,
                                                   trellis::y_t* out, u32 m, u32 n,
                                                   u32 k, u32 mode, u32 ks, pre_bf64_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    u32 wm = warp >> 2u, wn = warp & 3u, nb = (n + 63u) >> 6u, mb = (m + 63u) >> 6u;
    u32 ob_in = SPLIT ? (blockIdx.x % nb) : blockIdx.x, p = SPLIT ? (blockIdx.x / nb) : 0u;
    if (ob_in >= nb || p >= ks || blockIdx.y >= mb) return;
    u32 bx = ob_in, by = blockIdx.y;
    if constexpr (!SPLIT) pre_swizzle(nb, mb, ob_in, blockIdx.y, bx, by);
    u32 n0 = bx << 6u, r0 = by << 6u, n_base = wn << 4u, m_base = wm << 5u;
    u32 steps = (k + 31u) >> 5u, st_per = SPLIT ? ((steps + ks - 1u) / ks) : steps;
    u32 s0 = SPLIT ? (p * st_per) : 0u, s1 = SPLIT ? min(steps, s0 + st_per) : steps;
    u32 wr = tid >> 2u, wc = tid & 3u, xr0 = tid >> 3u, xc = (tid & 7u) << 2u, xw = (tid & 7u) << 1u;
    float acc[4][4];
#pragma unroll
    for (u32 ti = 0u; ti < 4u; ti++) acc[ti][0] = acc[ti][1] = acc[ti][2] = acc[ti][3] = 0.0f;
    if (s0 < s1) {
        u32 k0 = s0 << 5u;
        pre_cp16(&sm.sw[0][wr][wc << 2u], w + u64(n0 + wr) * k + k0 + (wc << 3u),
                 (n0 + wr < n) && (k0 + (wc << 3u) < k));
        pre_cp_commit();
    }
    for (u32 s = s0; s < s1; s++) {
        u32 st = (s - s0) & 1u, k0 = s << 5u;
        bool has_nxt = (s + 1u < s1);
        if (has_nxt) {
            u32 kn = (s + 1u) << 5u;
            pre_cp16(&sm.sw[st ^ 1u][wr][wc << 2u], w + u64(n0 + wr) * k + kn + (wc << 3u),
                     (n0 + wr < n) && (kn + (wc << 3u) < k));
            pre_cp_commit();
        }
#pragma unroll
        for (u32 dr = 0u; dr < 64u; dr += 32u) {
            u32 r = xr0 + dr;
            if (r0 + r < m && k0 + xc < k) {
                float4 xv = trellis::load_x4(x + u64(r0 + r) * k + k0 + xc);
                u32 h0 = pre_bf16x2(xv.x, xv.y), h1 = pre_bf16x2(xv.z, xv.w);
                u32 l0 = pre_bf16x2(xv.x - __uint_as_float(h0 << 16u), xv.y - __uint_as_float(h0 & 0xFFFF0000u));
                u32 l1 = pre_bf16x2(xv.z - __uint_as_float(h1 << 16u), xv.w - __uint_as_float(h1 & 0xFFFF0000u));
                *reinterpret_cast<uint2*>(&sm.sxh[r][xw]) = make_uint2(h0, h1);
                *reinterpret_cast<uint2*>(&sm.sxl[r][xw]) = make_uint2(l0, l1);
            } else {
                *reinterpret_cast<uint2*>(&sm.sxh[r][xw]) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(&sm.sxl[r][xw]) = make_uint2(0u, 0u);
            }
        }
        if (has_nxt) pre_cp_wait1(); else pre_cp_wait0();
        prim::block_sync();
#pragma unroll
        for (u32 ks_i = 0u; ks_i < 2u; ks_i++) {
            u32 off = (ks_i << 3u) + t;
            u32 a[4] = {sm.sw[st][n_base + g][off], sm.sw[st][n_base + g + 8u][off],
                        sm.sw[st][n_base + g][off + 4u], sm.sw[st][n_base + g + 8u][off + 4u]};
#pragma unroll
            for (u32 ti = 0u; ti < 4u; ti++) {
                u32 xr = m_base + (ti << 3u) + g;
                u32 bh[2] = {sm.sxh[xr][off], sm.sxh[xr][off + 4u]};
                u32 bl[2] = {sm.sxl[xr][off], sm.sxl[xr][off + 4u]};
                pre_mma_bf16(acc[ti], a, bh);
                pre_mma_bf16(acc[ti], a, bl);
            }
        }
        if (has_nxt) prim::block_sync();
    }
    pre_store_r64<SPLIT>(acc, other, bias_p, out, m, n, r0, n0, m_base, n_base, p, g, t, mode);
}

struct alignas(16) pre_bf128_smem {
    u32 sw[2][128][20];
    u32 sx[2][128][20];
};

// Staged BF16->FP16 Tensor-Core matmul, 128x128 per block.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_mm_bf128(
    const trellis::x_t* x, const u16* w, const trellis::o_t* other, const u8* bias_p,
    trellis::y_t* out, u32 m, u32 n, u32 k, u32 mode) {
    __shared__ pre_bf128_smem sm;
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    u32 wm = warp >> 1u, wn = warp & 1u, nb = (n + 127u) >> 7u, mb = (m + 127u) >> 7u;
    if (blockIdx.x >= nb || blockIdx.y >= mb) return;
    u32 bx, by;
    pre_swizzle(nb, mb, blockIdx.x, blockIdx.y, bx, by);
    u32 n0 = bx << 7u, r0 = by << 7u, n_base = wn << 6u, m_base = wm << 5u;
    u32 steps = (k + 31u) >> 5u, wr0 = tid >> 2u, wc = tid & 3u;
    u32 xr0 = tid >> 3u, xc = (tid & 7u) << 2u, xw = (tid & 7u) << 1u;
    float acc[4][4][4];
#pragma unroll
    for (u32 ni = 0u; ni < 4u; ni++)
#pragma unroll
        for (u32 mi = 0u; mi < 4u; mi++)
            acc[ni][mi][0] = acc[ni][mi][1] = acc[ni][mi][2] = acc[ni][mi][3] = 0.0f;
    if (steps > 0u) {
        pre_cp16(&sm.sw[0][wr0][wc << 2u], w + u64(n0 + wr0) * k + (wc << 3u), (n0 + wr0 < n) && ((wc << 3u) < k));
        pre_cp16(&sm.sw[0][wr0 + 64u][wc << 2u], w + u64(n0 + wr0 + 64u) * k + (wc << 3u), (n0 + wr0 + 64u < n) && ((wc << 3u) < k));
        pre_cp_commit();
    }
    for (u32 s = 0u; s < steps; s++) {
        u32 st = s & 1u, k0 = s << 5u;
        bool has_nxt = (s + 1u < steps);
        if (has_nxt) {
            u32 kn = (s + 1u) << 5u;
            pre_cp16(&sm.sw[st ^ 1u][wr0][wc << 2u], w + u64(n0 + wr0) * k + kn + (wc << 3u), (n0 + wr0 < n) && (kn + (wc << 3u) < k));
            pre_cp16(&sm.sw[st ^ 1u][wr0 + 64u][wc << 2u], w + u64(n0 + wr0 + 64u) * k + kn + (wc << 3u), (n0 + wr0 + 64u < n) && (kn + (wc << 3u) < k));
            pre_cp_commit();
        }
#pragma unroll
        for (u32 dr = 0u; dr < 128u; dr += 32u) {
            u32 r = xr0 + dr;
            if (r0 + r < m && k0 + xc < k) {
                float4 xv = trellis::load_x4(x + u64(r0 + r) * k + k0 + xc);
                *reinterpret_cast<uint2*>(&sm.sx[st][r][xw]) = make_uint2(
                    trellis::f32x2_to_f16x2(xv.x, xv.y), trellis::f32x2_to_f16x2(xv.z, xv.w));
            } else {
                *reinterpret_cast<uint2*>(&sm.sx[st][r][xw]) = make_uint2(0u, 0u);
            }
        }
        if (has_nxt) pre_cp_wait1(); else pre_cp_wait0();
        prim::block_sync();
#pragma unroll
        for (u32 ks_i = 0u; ks_i < 2u; ks_i++) {
            u32 off = (ks_i << 3u) + t;
            u32 b[4][2];
#pragma unroll
            for (u32 mi = 0u; mi < 4u; mi++) {
                u32 xr = m_base + (mi << 3u) + g;
                b[mi][0] = sm.sx[st][xr][off]; b[mi][1] = sm.sx[st][xr][off + 4u];
            }
#pragma unroll
            for (u32 ni = 0u; ni < 4u; ni++) {
                u32 nr = n_base + (ni << 4u) + g;
                u32 a[4] = {pre_bf16x2_f16x2(sm.sw[st][nr][off]), pre_bf16x2_f16x2(sm.sw[st][nr + 8u][off]),
                            pre_bf16x2_f16x2(sm.sw[st][nr][off + 4u]), pre_bf16x2_f16x2(sm.sw[st][nr + 8u][off + 4u])};
#pragma unroll
                for (u32 mi = 0u; mi < 4u; mi++) trellis::mma_m16n8k16(acc[ni][mi], a, b[mi]);
            }
        }
        if (has_nxt) prim::block_sync();
    }
#pragma unroll
    for (u32 ni = 0u; ni < 4u; ni++) {
        pre_store_r64<false>(acc[ni], other, bias_p, out, m, n, r0, n0, m_base, n_base + (ni << 4u),
                             0u, g, t, mode);
    }
}

extern "C" __global__ __launch_bounds__(256, 4) void tessera_mm_bf64(
    const trellis::x_t* x, const u16* w, const trellis::o_t* other, const u8* bias_p,
    trellis::y_t* out, u32 m, u32 n, u32 k, u32 mode) {
    __shared__ pre_bf64_smem sm; fg_bf16_r64<false>(x, w, other, bias_p, out, m, n, k, mode, 1u, sm);
}
extern "C" __global__ __launch_bounds__(256, 4) void tessera_mm_bf64_sk(
    const trellis::x_t* x, const u16* w, const trellis::o_t* other, const u8* bias_p,
    trellis::y_t* out, u32 m, u32 n, u32 k, u32 mode, u32 ks) {
    __shared__ pre_bf64_smem sm; fg_bf16_r64<true>(x, w, other, bias_p, out, m, n, k, mode, ks, sm);
}

#endif

#if !defined(DTR_FORCE_FP16) && !defined(PRE_ONLY_DTR_K4) && !defined(PRE_ONLY_MM)

// Quantizes X[m, k] once to packed E4M3 and UE8M0 scales.
extern "C" __global__ __launch_bounds__(256, MM_PREP_BLK) void tessera_mm_mxfp8_prep(
    const trellis::x_t* x, float* out, u32 m, u32 k) {
#if DTR_MX8
    u32 blk = blockIdx.x * 32u + (threadIdx.x >> 3u);
    u32 row = blockIdx.y;
    u32 nblk = k >> 5u;
    if (row >= m || blk >= nblk) return;
    u32 sub = threadIdx.x & 7u;
    float4 v = trellis::load_x4(x + u64(row) * k + (blk << 5u) + (sub << 2u));
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4u));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2u));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1u));
    float inv;
    u32 e = trellis::mx_scale(amax, inv);
    u16 p0 = trellis::f2_e4m3x2(v.x * inv, v.y * inv);
    u16 p1 = trellis::f2_e4m3x2(v.z * inv, v.w * inv);
    u8* xq = reinterpret_cast<u8*>(out);
    u8* xs = xq + u64(m) * k;
    *reinterpret_cast<u32*>(xq + u64(row) * k + (blk << 5u) + (sub << 2u)) =
        u32(p0) | (u32(p1) << 16u);
    if (sub == 0u) xs[u64(row) * nblk + blk] = u8(e);
#else
    (void)x; (void)out; (void)m; (void)k;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void mx8_cp16(void* smem, const void* gmem, bool pred) {
    u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem));
    if (pred) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem));
    else *reinterpret_cast<uint4*>(smem) = make_uint4(0u, 0u, 0u, 0u);
}

static __device__ __forceinline__ void mx8_ldm4(u32 (&r)[4], const void* smem_ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

template <bool SPLIT>
static __device__ __forceinline__ void mx8_store_vec4(
    float4 v, const trellis::o_t* other, trellis::y_t* out, u64 off, u32 col, u32 n, u32 mode,
    bool vec4) {
    if (vec4 && col + 3u < n) {
        if (!SPLIT && mode != 0u) {
            v.x = mx8_epilogue(v.x, other, off, col, mode);
            v.y = mx8_epilogue(v.y, other, off + 1u, col + 1u, mode);
            v.z = mx8_epilogue(v.z, other, off + 2u, col + 2u, mode);
            v.w = mx8_epilogue(v.w, other, off + 3u, col + 3u, mode);
        }
        trellis::store_y4(out + off, v);
    } else {
        float vals[4] = {v.x, v.y, v.z, v.w};
        for (u32 j = 0u; j < 4u && col + j < n; j++) {
            trellis::store_y(out, off + j, SPLIT ? vals[j] : mx8_epilogue(vals[j], other, off + j, col + j, mode));
        }
    }
}

template <u32 MI, bool SPLIT>
static __device__ __forceinline__ void mx8_epilogue_smem(
    float (*out_sm)[132], const float (&acc)[2][MI][4], const trellis::o_t* other,
    trellis::y_t* out, u32 m, u32 n, u32 r0, u32 n0, u32 p, u32 mode, u32 tid, u32 wm, u32 n_base,
    u32 g, u32 t) {
    constexpr u32 Q_ROWS = MI << 3u;
    constexpr u32 Q_SHIFT = (MI == 8u) ? 6u : 5u;
    u32 sr = tid >> 5u, sc4 = (tid & 31u) << 2u;
    bool vec4 = (n & 3u) == 0u;
#pragma unroll
    for (u32 q = 0u; q < 2u; q++) {
        if (wm == q) {
#pragma unroll
            for (u32 ni = 0u; ni < 2u; ni++) {
                u32 lc0 = n_base + (ni << 4u) + g, lc1 = lc0 + 8u;
#pragma unroll
                for (u32 mi = 0u; mi < MI; mi++) {
                    u32 lr = (mi << 3u) + (t << 1u);
                    out_sm[lr][lc0] = acc[ni][mi][0];
                    out_sm[lr][lc1] = acc[ni][mi][2];
                    out_sm[lr + 1u][lc0] = acc[ni][mi][1];
                    out_sm[lr + 1u][lc1] = acc[ni][mi][3];
                }
            }
        }
        prim::block_sync();
        u32 col = n0 + sc4;
#pragma unroll
        for (u32 dr = 0u; dr < Q_ROWS; dr += 8u) {
            u32 row = r0 + (q << Q_SHIFT) + sr + dr;
            if (row < m && col < n) {
                float4 v = *reinterpret_cast<const float4*>(&out_sm[sr + dr][sc4]);
                u64 off = u64(SPLIT ? (p * m + row) : row) * n + col;
                mx8_store_vec4<SPLIT>(v, other, out, off, col, n, mode, vec4);
            }
        }
        if (q == 0u) prim::block_sync();
    }
}
#endif

#if DTR_MX8
union alignas(16) mx8_128_smem {
    struct {
        u32 sw[2][128][36];
        u32 sx[2][128][32];
        u32 sxs[2][128];
    } k;
    float out[64][132];
};

static __device__ __forceinline__ void mx8_cp4(void* smem, const void* gmem, bool pred) {
    u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem));
    if (pred) asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(gmem));
    else *reinterpret_cast<u32*>(smem) = 0u;
}

template <u32 SH>
static __device__ __forceinline__ void mx8_load_fa(
    const u32* sw0, const u32* sw1, u32 w_base, u32 t, u32 (&fa)[4]) {
    u32 w0 = w_base + t;
    if constexpr (SH == 0u) {
        fa[0] = sw0[w0]; fa[1] = sw1[w0]; fa[2] = sw0[w0 + 4u]; fa[3] = sw1[w0 + 4u];
    } else {
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fa[0]) : "r"(sw0[w0]), "r"(sw0[w0 + 1u]), "n"(SH));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fa[1]) : "r"(sw1[w0]), "r"(sw1[w0 + 1u]), "n"(SH));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fa[2]) : "r"(sw0[w0 + 4u]), "r"(sw0[w0 + 5u]), "n"(SH));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fa[3]) : "r"(sw1[w0 + 4u]), "r"(sw1[w0 + 5u]), "n"(SH));
    }
}

static __device__ __forceinline__ void mx8_load_stage(
    mx8_128_smem& sm, u32 st, u32 ch128, const u8* xq, const u8* xs, const u8* w,
    u32 m, u32 n, u32 k, u32 nblk, u64 row_bytes, u32 n0, u32 r0, u32 tid, u32 r0_t, u32 col16) {
    u64 aligned_off = (u64(ch128) * 132ull) & ~15ull;
#pragma unroll
    for (u32 dr = 0u; dr < 128u; dr += 32u) {
        u32 nr = r0_t + dr;
        mx8_cp16(&sm.k.sw[st][nr][col16 << 2u],
                 w + u64(n0 + nr) * row_bytes + aligned_off + (col16 << 4u), n0 + nr < n);
    }
    if (tid < 128u) {
        mx8_cp16(&sm.k.sw[st][tid][32u], w + u64(n0 + tid) * row_bytes + aligned_off + 128u, n0 + tid < n);
    }
    u32 k_off = ch128 << 7u;
#pragma unroll
    for (u32 dr = 0u; dr < 128u; dr += 32u) {
        u32 mr = r0_t + dr;
        mx8_cp16(&sm.k.sx[st][mr][(col16 ^ (mr & 7u)) << 2u],
                 xq + u64(r0 + mr) * k + k_off + (col16 << 4u), r0 + mr < m);
    }
    if (tid < 128u) {
        mx8_cp4(&sm.k.sxs[st][tid], xs + u64(r0 + tid) * nblk + (ch128 << 2u), r0 + tid < m);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <u32 PH, u32 BLK>
static __device__ __forceinline__ void mx8_run_blk(
    const u32 (*sw_st)[36], const u32 (*sx_st)[32], const u32 sb4[8],
    u32 n_base, u32 m_base, u32 g, u32 t, u32 lb_row, u32 lb_half, float acc[2][8][4]) {
    constexpr u32 B_OFF = PH * 4u + BLK * 33u;
    constexpr u32 Q_BASE = B_OFF + 1u;
    constexpr u32 SH = (Q_BASE & 3u) << 3u;
    constexpr u32 W_BASE = Q_BASE >> 2u;
    constexpr u32 S_SH = BLK << 3u;
    u32 fb[8][2], sb[8];
#pragma unroll
    for (u32 q = 0u; q < 4u; q++) {
        u32 mr = m_base + (q << 4u) + lb_row, tmp[4];
        mx8_ldm4(tmp, &sx_st[mr][(((BLK << 1u) | lb_half) ^ (mr & 7u)) << 2u]);
        fb[q << 1u][0] = tmp[0]; fb[q << 1u][1] = tmp[1];
        fb[(q << 1u) + 1u][0] = tmp[2]; fb[(q << 1u) + 1u][1] = tmp[3];
    }
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) sb[mi] = (sb4[mi] >> S_SH) & 0xFFu;

    u32 fa0[4], fa1[4], sa0, sa1;
    {
        u32 nr0 = n_base + g, nr1 = nr0 + 8u;
        const u32 *sw0 = sw_st[nr0], *sw1 = sw_st[nr1];
        sa0 = u32(reinterpret_cast<const u8*>((t & 1u) ? sw1 : sw0)[B_OFF]);
        mx8_load_fa<SH>(sw0, sw1, W_BASE, t, fa0);
    }
    {
        u32 nr0 = n_base + 16u + g, nr1 = nr0 + 8u;
        const u32 *sw0 = sw_st[nr0], *sw1 = sw_st[nr1];
        sa1 = u32(reinterpret_cast<const u8*>((t & 1u) ? sw1 : sw0)[B_OFF]);
        mx8_load_fa<SH>(sw0, sw1, W_BASE, t, fa1);
    }
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) trellis::mma_mx8(acc[0][mi], fa0, fb[mi], sa0, sb[mi]);
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) trellis::mma_mx8(acc[1][mi], fa1, fb[mi], sa1, sb[mi]);
}

template <u32 PH>
static __device__ __forceinline__ void mx8_phase_step(
    mx8_128_smem& sm, u32 st, u32 nxt_ch, bool has_nxt,
    const u8* xq, const u8* xs, const u8* w, u32 m, u32 n, u32 k, u32 nblk, u64 row_bytes,
    u32 n0, u32 r0, u32 tid, u32 r0_t, u32 col16,
    u32 n_base, u32 m_base, u32 g, u32 t, u32 lb_row, u32 lb_half, float acc[2][8][4]) {
    if (has_nxt) {
        mx8_load_stage(sm, st ^ 1u, nxt_ch, xq, xs, w, m, n, k, nblk, row_bytes, n0, r0, tid, r0_t, col16);
        pre_cp_wait1();
    } else {
        pre_cp_wait0();
    }
    prim::block_sync();
    u32 sb4[8];
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) sb4[mi] = sm.k.sxs[st][m_base + (mi << 3u) + g];
    mx8_run_blk<PH, 0u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 1u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 2u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 3u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    if (has_nxt) prim::block_sync();
}

template <bool SPLIT>
static __device__ __forceinline__ void fg_mxfp8_mx8(
    const float* xpre, const u8* w, const trellis::o_t* other, trellis::y_t* out,
    u32 m, u32 n, u32 k, u32 mode, u32 ks) {
    extern __shared__ u8 raw_smem[];
    mx8_128_smem& sm = *reinterpret_cast<mx8_128_smem*>(raw_smem);
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    u32 g = lane >> 2u, t = lane & 3u, wn = warp & 3u, wm = warp >> 2u;
    u32 nb = (n + 127u) >> 7u, mb = (m + 127u) >> 7u;
    u32 ob = SPLIT ? (blockIdx.x % nb) : blockIdx.x;
    u32 p = SPLIT ? (blockIdx.x / nb) : 0u;
    u32 bx, by;
    pre_swizzle(nb, mb, ob, blockIdx.y, bx, by);
    u32 n0 = bx << 7u, r0 = by << 7u;

    u32 chunks = k >> 9u, ch_per = SPLIT ? (chunks / ks) : chunks;
    u32 c5_0 = SPLIT ? (p * ch_per) : 0u, c5_1 = c5_0 + ch_per;
    u32 nblk = k >> 5u;
    u64 row_bytes = u64(nblk) * 33ull;
    const u8* xq = reinterpret_cast<const u8*>(xpre);
    const u8* xs = xq + u64(m) * k;
    u32 r0_t = tid >> 3u, col16 = tid & 7u;
    u32 n_base = wn << 5u, m_base = wm << 6u;
    u32 lb_row = ((lane >> 4u) << 3u) | (lane & 7u), lb_half = (lane >> 3u) & 1u;

    float acc[2][8][4];
#pragma unroll
    for (u32 ni = 0u; ni < 2u; ni++)
#pragma unroll
        for (u32 mi = 0u; mi < 8u; mi++)
            acc[ni][mi][0] = acc[ni][mi][1] = acc[ni][mi][2] = acc[ni][mi][3] = 0.0f;

    mx8_load_stage(sm, 0u, c5_0 << 2u, xq, xs, w, m, n, k, nblk, row_bytes, n0, r0, tid, r0_t, col16);
    for (u32 c5 = c5_0; c5 < c5_1; c5++) {
        u32 base = c5 << 2u;
        mx8_phase_step<0u>(sm, 0u, base + 1u, true, xq, xs, w, m, n, k, nblk, row_bytes,
                           n0, r0, tid, r0_t, col16, n_base, m_base, g, t, lb_row, lb_half, acc);
        mx8_phase_step<1u>(sm, 1u, base + 2u, true, xq, xs, w, m, n, k, nblk, row_bytes,
                           n0, r0, tid, r0_t, col16, n_base, m_base, g, t, lb_row, lb_half, acc);
        mx8_phase_step<2u>(sm, 0u, base + 3u, true, xq, xs, w, m, n, k, nblk, row_bytes,
                           n0, r0, tid, r0_t, col16, n_base, m_base, g, t, lb_row, lb_half, acc);
        mx8_phase_step<3u>(sm, 1u, base + 4u, c5 + 1u < c5_1, xq, xs, w, m, n, k, nblk, row_bytes,
                           n0, r0, tid, r0_t, col16, n_base, m_base, g, t, lb_row, lb_half, acc);
    }
    prim::block_sync();
    mx8_epilogue_smem<8u, SPLIT>(sm.out, acc, other, out, m, n, r0, n0, p, mode, tid, wm, n_base, g, t);
}
#endif

// Blackwell native MXFP8 MMA matmul, 128x128 per block.
extern "C" __global__ __launch_bounds__(256, MM_MX8_BLK) void tessera_mm_mxfp8_mx8(
    const float* xpre, const u8* w, const trellis::o_t* other, trellis::y_t* out, u32 m, u32 n,
    u32 k, u32 mode) {
#if DTR_MX8
    fg_mxfp8_mx8<false>(xpre, w, other, out, m, n, k, mode, 1u);
#else
    (void)xpre; (void)w; (void)other; (void)out; (void)m; (void)n; (void)k; (void)mode;
#endif
}

// Split-K Blackwell native MXFP8 MMA matmul, 128x128 per block.
extern "C" __global__ __launch_bounds__(256, MM_MX8_SK_BLK) void tessera_mm_mxfp8_mx8_sk(
    const float* xpre, const u8* w, const trellis::o_t* other, trellis::y_t* out, u32 m, u32 n,
    u32 k, u32 mode, u32 ks) {
#if DTR_MX8
    fg_mxfp8_mx8<true>(xpre, w, other, out, m, n, k, mode, ks);
#else
    (void)xpre; (void)w; (void)other; (void)out; (void)m; (void)n; (void)k; (void)mode; (void)ks;
#endif
}
#endif
#endif
