#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_CUDA)

#ifndef MLA_MIN_BLOCKS
#define MLA_MIN_BLOCKS 2
#endif
#ifndef MLA_REDUCE_THREADS
#define MLA_REDUCE_THREADS 160
#endif
#ifndef MLA_REDUCE_MIN_BLOCKS
#define MLA_REDUCE_MIN_BLOCKS 4
#endif
#ifndef GQA_MIN_BLOCKS
#define GQA_MIN_BLOCKS 4
#endif
#ifndef GQA_REDUCE_MIN_BLOCKS
#define GQA_REDUCE_MIN_BLOCKS 4
#endif
#ifdef IDX_MIN_BLOCKS
#define IDX_BOUNDS __launch_bounds__(128, IDX_MIN_BLOCKS)
#else
#define IDX_BOUNDS __launch_bounds__(128)
#endif
#ifndef IDX_QT_MIN_BLOCKS
#define IDX_QT_MIN_BLOCKS 2
#endif

static __device__ __forceinline__ float attn_neg_inf() {
    return __int_as_float(0xff800000u);
}

static __device__ __forceinline__ void attn_fma4(float4& s, const float* p, float w) {
    float4 v = *reinterpret_cast<const float4*>(p);
    s.x = fmaf(v.x, w, s.x); s.y = fmaf(v.y, w, s.y);
    s.z = fmaf(v.z, w, s.z); s.w = fmaf(v.w, w, s.w);
}

static __device__ __forceinline__ u32 attn_e4m3_f16x2(u16 q) {
    u32 h;
    asm("{.reg .b16 t; mov.b16 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(h) : "h"(q));
    return h;
}

static __device__ __forceinline__ void attn_unpack_fp8(uint4 r, u32* d) {
    *reinterpret_cast<uint4*>(d) = make_uint4(
        attn_e4m3_f16x2(u16(r.x)), attn_e4m3_f16x2(u16(r.x >> 16)),
        attn_e4m3_f16x2(u16(r.y)), attn_e4m3_f16x2(u16(r.y >> 16)));
    *reinterpret_cast<uint4*>(d + 4u) = make_uint4(
        attn_e4m3_f16x2(u16(r.z)), attn_e4m3_f16x2(u16(r.z >> 16)),
        attn_e4m3_f16x2(u16(r.w)), attn_e4m3_f16x2(u16(r.w >> 16)));
}

template <u32 STEPS, u32 STRIDE>
static __device__ __forceinline__ void attn_qk_tile(const u32 (*q)[STRIDE], const u32 (*k)[STRIDE],
                                                    float red[4][16][16], u32 warp, u32 g, u32 t) {
    float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (u32 step = 0u; step < STEPS; step++) {
        u32 k0 = (warp * STEPS + step) * 8u;
        u32 a[4] = {q[g][k0 + t], q[g + 8u][k0 + t], q[g][k0 + t + 4u], q[g + 8u][k0 + t + 4u]};
        u32 b0[2] = {k[g][k0 + t], k[g][k0 + t + 4u]};
        u32 b1[2] = {k[g + 8u][k0 + t], k[g + 8u][k0 + t + 4u]};
        trellis::mma_m16n8k16(s0, a, b0);
        trellis::mma_m16n8k16(s1, a, b1);
    }
    red[warp][g][t * 2u] = s0[0]; red[warp][g][t * 2u + 1u] = s0[1];
    red[warp][g + 8u][t * 2u] = s0[2]; red[warp][g + 8u][t * 2u + 1u] = s0[3];
    red[warp][g][8u + t * 2u] = s1[0]; red[warp][g][8u + t * 2u + 1u] = s1[1];
    red[warp][g + 8u][8u + t * 2u] = s1[2]; red[warp][g + 8u][8u + t * 2u + 1u] = s1[3];
}

static __device__ __forceinline__ void attn_softmax_row(const float red[4][16][16], u32 valid_mask,
                                                        u32 tid, float scale, float m[16],
                                                        float l[16], float corr[16],
                                                        u32 p16[16][8]) {
    float sc[16], tmax = attn_neg_inf();
#pragma unroll
    for (u32 c = 0u; c < 16u; c++) {
        float v = ((valid_mask >> c) & 1u)
            ? (red[0][tid][c] + red[1][tid][c] + red[2][tid][c] + red[3][tid][c]) * scale
            : attn_neg_inf();
        sc[c] = v; if (v > tmax) tmax = v;
    }
    float old_m = m[tid], old_l = l[tid], next_m = (old_m > tmax) ? old_m : tmax;
    float cval = (old_l > 0.0f) ? __expf(old_m - next_m) : 0.0f, psum = 0.0f;
#pragma unroll
    for (u32 c = 0u; c < 16u; c += 2u) {
        float p0v = ((valid_mask >> c) & 1u) ? __expf(sc[c] - next_m) : 0.0f;
        float p1v = ((valid_mask >> (c + 1u)) & 1u) ? __expf(sc[c + 1u] - next_m) : 0.0f;
        psum += p0v + p1v;
        p16[tid][c >> 1] = trellis::f32x2_to_f16x2(p0v, p1v);
    }
    m[tid] = next_m; l[tid] = old_l * cval + psum; corr[tid] = cval;
}

template <u32 NT, u32 V_STRIDE16>
static __device__ __forceinline__ void attn_pv_tile(float o[NT][4], const u32 p16[16][8],
                                                    const float corr[16], const u16* v16,
                                                    u32 d_base, u32 g, u32 t) {
    u32 a_p[4] = {p16[g][t], p16[g + 8u][t], p16[g][t + 4u], p16[g + 8u][t + 4u]};
    float c0 = corr[g], c1 = corr[g + 8u];
    u32 r0 = t << 1, r1 = r0 + 8u;
#pragma unroll
    for (u32 nt = 0u; nt < NT; nt++) {
        o[nt][0] *= c0; o[nt][1] *= c0; o[nt][2] *= c1; o[nt][3] *= c1;
        u32 col = d_base + nt * 8u + g;
        u32 b_v[2] = {
            u32(v16[r0 * V_STRIDE16 + col]) | (u32(v16[(r0 + 1u) * V_STRIDE16 + col]) << 16),
            u32(v16[r1 * V_STRIDE16 + col]) | (u32(v16[(r1 + 1u) * V_STRIDE16 + col]) << 16),
        };
        trellis::mma_m16n8k16(o[nt], a_p, b_v);
    }
}

template <bool BF16>
static __device__ __forceinline__ u32 attn_pack2(float lo, float hi) {
    if constexpr (BF16) { u32 o; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o) : "f"(lo), "f"(hi)); return o; }
    return trellis::f32x2_to_f16x2(lo, hi);
}
template <bool BF16>
static __device__ __forceinline__ void attn_mma(float d[4], const u32 a[4], const u32 b[2]) {
    if constexpr (BF16) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else { trellis::mma_m16n8k16(d, a, b); }
}
template <u32 NT, u32 WIDTH, u32 STRIDE>
static __device__ __forceinline__ void attn_store_partials(
    float* part, const float o[NT][4], const float m[16], const float l[16], u32 row, u32 heads,
    u32 h0, u32 nh, u32 split, u32 splits, u32 d_base, u32 g, u32 t, u32 tid) {
#pragma unroll
    for (u32 h = 0u; h < 2u; h++) {
        u32 gh = g + (h << 3u);
        if (gh >= nh) continue;
        float sc = (splits == 1u) ? ((l[gh] > 0.0f) ? (1.0f / l[gh]) : 0.0f) : 1.0f;
        float* pg = part + ((splits == 1u) ? u64(row * heads + h0 + gh) * WIDTH
                                           : (u64(row * heads + h0 + gh) * splits + split) * STRIDE);
#pragma unroll
        for (u32 nt = 0u; nt < NT; nt++)
            *reinterpret_cast<float2*>(pg + d_base + nt * 8u + (t << 1)) = make_float2(o[nt][h * 2u] * sc, o[nt][h * 2u + 1u] * sc);
    }
    if (splits > 1u && tid < nh) {
        u64 ob = (u64(row * heads + h0 + tid) * splits + split) * STRIDE;
        part[ob + WIDTH] = m[tid]; part[ob + WIDTH + 1u] = l[tid];
    }
}
static __device__ __forceinline__ void attn_init_head_state(
    float& m, float& l, const float* sinks, u32 has_sinks, u32 split, u32 h) {
    if (has_sinks != 0u && split == 0u) {
        float sv = sinks[h];
        m = sv;
        l = (sv > attn_neg_inf()) ? 1.0f : 0.0f;
    } else {
        m = attn_neg_inf();
        l = 0.0f;
    }
}
template <u32 WIDTH, u32 STRIDE, u32 NTH>
static __device__ __forceinline__ void attn_store_empty(
    float* part, const float* sinks, u32 has_sinks, u32 row, u32 heads, u32 h0, u32 nh, u32 split,
    u32 splits, u32 tid) {
    if (splits == 1u) {
        float* base = part + u64(row * heads + h0) * WIDTH;
        for (u32 q = tid; q < nh * (WIDTH >> 2); q += NTH) reinterpret_cast<float4*>(base)[q] = make_float4(0.f, 0.f, 0.f, 0.f);
    } else if (tid < nh) {
        u64 ob = (u64(row * heads + h0 + tid) * splits + split) * STRIDE;
        float m, l;
        attn_init_head_state(m, l, sinks, has_sinks, split, h0 + tid);
        part[ob + WIDTH] = m; part[ob + WIDTH + 1u] = l;
    }
}

static __device__ __forceinline__ void attn_ldm_x2(u32 d[2], const void* s) {
    u32 a = u32(__cvta_generic_to_shared(s));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(a));
}
static __device__ __forceinline__ void attn_ldm_x4(u32 d[4], const void* s) {
    u32 a = u32(__cvta_generic_to_shared(s));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}
static __device__ __forceinline__ void attn_ldm_x2_t(u32 d[2], const void* s) {
    u32 a = u32(__cvta_generic_to_shared(s));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(a));
}
static __device__ __forceinline__ void attn_ldm_x4_t(u32 d[4], const void* s) {
    u32 a = u32(__cvta_generic_to_shared(s));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}
static __device__ __forceinline__ void attn_cp_async16(void* dst, const void* src, u32 bytes) {
    u32 a = u32(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(a), "l"(src), "r"(bytes) : "memory");
}
static __device__ __forceinline__ void attn_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
static __device__ __forceinline__ void attn_cp_wait1() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}
static __device__ __forceinline__ void attn_cp_wait0() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

struct alignas(16) mla_smem {
    u16 kv[2][16][584];
    float red[4][16][32];
    u32 p16[16][16];
    float m[16], l[16], corr[16];
    u32 stop, valid_mask;
};

static __device__ __forceinline__ u32 mla_pick_id(const u32* ids, u32 row, u32 picks, u32 fl,
                                                  u32 from, u32 to, u32 p, u32 p_end) {
    if (p >= p_end) return 0xFFFFFFFFu;
    u32 c = (fl == 1u) ? ids[u64(row) * picks + p] : p;
    return (c >= from && c < to) ? c : 0xFFFFFFFFu;
}

static __device__ __forceinline__ u64 kv_row_at(const u32* pt, u32 slot, u32 pos, u32 ctx, u32 sh) {
    if (sh == 0u) return u64(slot) * ctx + pos;
    if (sh == 32u) return u64(slot) * ctx + (pos % ctx);
    return (u64(pt[u64(slot) * ctx + (pos >> sh)]) << sh) | u64(pos & ((1u << sh) - 1u));
}

static __device__ __forceinline__ const u8* kv_row_ptr(
    const void* dev, const void* host, const u32* pt, u32 slot, u32 pos, u32 ctx, u32 sh,
    u64 row_bytes, bool& is_host) {
    if (sh == 0u || sh == 32u) {
        is_host = false;
        return static_cast<const u8*>(dev) + kv_row_at(pt, slot, pos, ctx, sh) * row_bytes;
    }
    u32 e = pt[u64(slot) * ctx + (pos >> sh)];
    is_host = (e >> 30) != 0u;
    const u8* base = static_cast<const u8*>(is_host ? host : dev);
    u64 row = (u64(e & 0x3FFFFFFFu) << sh) | u64(pos & ((1u << sh) - 1u));
    return base + row * row_bytes;
}

static __device__ __forceinline__ void mla_prefetch_fp8(const u8* kv, const u8* kv_host,
                                                        const u32* pt, u32 slot, u32 j, u32 ctx,
                                                        u32 sh, u32 l16, uint4 p[3]) {
    if (j != 0xFFFFFFFFu) {
        bool is_host = false;
        const u8* r = kv_row_ptr(kv, kv_host, pt, slot, j, ctx, sh, 576ull, is_host);
        p[0] = *reinterpret_cast<const uint4*>(r + u64(l16) * 16ull);
        p[1] = *reinterpret_cast<const uint4*>(r + u64(l16 + 16u) * 16ull);
        if (l16 < 4u) p[2] = *reinterpret_cast<const uint4*>(r + u64(l16 + 32u) * 16ull);
    } else {
        p[0] = p[1] = p[2] = make_uint4(0u, 0u, 0u, 0u);
    }
}

static __device__ __forceinline__ void mla_unpack_fp8_row(const uint4 p[3], u32 l16, u32* dst32) {
    attn_unpack_fp8(p[0], dst32 + l16 * 8u);
    attn_unpack_fp8(p[1], dst32 + (l16 + 16u) * 8u);
    if (l16 < 4u) attn_unpack_fp8(p[2], dst32 + (l16 + 32u) * 8u);
}

static __device__ __forceinline__ void mla_stage_bf16(const u16* kv, const u16* kv_host,
                                                      const u32* pt, u32 slot, u32 j, u32 ctx,
                                                      u32 sh, u32 l16, u16* dst) {
    bool is_host = false;
    const u16* row_p = (j != 0xFFFFFFFFu)
        ? reinterpret_cast<const u16*>(kv_row_ptr(kv, kv_host, pt, slot, j, ctx, sh, 576ull * 2ull, is_host))
        : kv;
    if (is_host) {
#pragma unroll
        for (u32 q8 = l16; q8 < 72u; q8 += 16u) {
            *reinterpret_cast<uint4*>(dst + q8 * 8u) = *reinterpret_cast<const uint4*>(row_p + u64(q8) * 8ull);
        }
    } else {
        u32 bytes = (j != 0xFFFFFFFFu) ? 16u : 0u;
#pragma unroll
        for (u32 q8 = l16; q8 < 72u; q8 += 16u) {
            attn_cp_async16(dst + q8 * 8u, row_p + u64(q8) * 8ull, bytes);
        }
    }
}

static __device__ __forceinline__ void mla_stage_f32(const float* kv, const float* kv_host,
                                                     const u32* pt, u32 slot, u32 j, u32 ctx,
                                                     u32 sh, u32 l16, u32* dst32) {
    bool is_host = false;
    const float* rf = (j != 0xFFFFFFFFu)
        ? reinterpret_cast<const float*>(kv_row_ptr(kv, kv_host, pt, slot, j, ctx, sh, 576ull * 4ull, is_host))
        : nullptr;
    for (u32 q4 = l16; q4 < 144u; q4 += 16u) {
        uint2 o2 = make_uint2(0u, 0u);
        if (rf != nullptr) {
            float4 r = *reinterpret_cast<const float4*>(rf + u64(q4) * 4ull);
            o2 = make_uint2(trellis::f32x2_to_f16x2(r.x, r.y), trellis::f32x2_to_f16x2(r.z, r.w));
        }
        *reinterpret_cast<uint2*>(dst32 + q4 * 2u) = o2;
    }
}

static __device__ __forceinline__ void mla_set_mask(
    const u32* ids, u32 row, u32 picks, u32 fl, u32 from, u32 to, u32 p0, u32 p_end, u32 tid,
    u32* out_mask) {
    if (tid < 32u) {
        bool ok = mla_pick_id(ids, row, picks, fl, from, to, p0 + tid, p_end) != 0xFFFFFFFFu;
        u32 m = __ballot_sync(0xFFFFFFFFu, ok);
        if (tid == 0u) *out_mask = m;
    }
}

template <bool USE_BF16>
static __device__ __forceinline__ void mla_qk_half(
    const u16 (*kv)[584], const u32 q_reg[9][4], u32 krow, u32 kcol, u32 wl, float s0[4]) {
#pragma unroll
    for (u32 s = 0u; s < 9u; s++) {
        u32 b0[2];
        attn_ldm_x2(b0, &kv[krow][(wl * 9u + s) * 16u + kcol]);
        attn_mma<USE_BF16>(s0, q_reg[s], b0);
    }
}

template <bool USE_BF16, u32 NT>
static __device__ __forceinline__ void mla_pv_half(
    const u16 (*kv)[584], const u32 (*p16)[16], u32 p_off, u32 d_base, u32 vrow, u32 vcol,
    u32 g, u32 t, float o[NT][4]) {
    u32 ap[4] = {p16[g][p_off + t], p16[g + 8u][p_off + t],
                 p16[g][p_off + t + 4u], p16[g + 8u][p_off + t + 4u]};
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt += 2u) {
        u32 bv[4];
        attn_ldm_x4_t(bv, &kv[vrow][d_base + nt * 8u + vcol]);
        u32 b0[2] = {bv[0], bv[1]}, b1[2] = {bv[2], bv[3]};
        attn_mma<USE_BF16>(o[nt], ap, b0);
        attn_mma<USE_BF16>(o[nt + 1u], ap, b1);
    }
    if constexpr (NT == 9u) {
        u32 b8[2];
        attn_ldm_x2_t(b8, &kv[vrow][d_base + 64u]);
        attn_mma<USE_BF16>(o[8], ap, b8);
    }
}

template <u32 KV_FMT, u32 OUT_W>
static __device__ __forceinline__ void mla_attend_body(
    const float* q, const void* kv_ptr, const void* kv_host, const u32* ids, const u32* slots,
    const u32* positions, const float* sinks, float* part, float scale, u32 rows, u32 keys,
    u32 context, u32 heads, u32 picks, u32 from_list, u32 before, u32 after, u32 splits,
    const u32* page_table, u32 page_shift, u32 has_sinks, mla_smem& sm) {
    constexpr u32 NT = OUT_W >> 6u;
    constexpr u32 OUT_STRIDE = OUT_W + 4u;
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    u32 wg = warp >> 2, wl = warp & 3u, hgroups = heads >> 4;
    u32 split = blockIdx.x, rb = blockIdx.y / hgroups, hg = blockIdx.y % hgroups;
    if (rb >= rows || split >= splits) return;
    u32 row = (rows - 1u) - rb, h0 = hg << 4, slot = slots[row], pos = positions[row];
    u32 from = (pos > before) ? (pos - before) : 0u;
    u32 to = (pos + after < keys) ? (pos + after + 1u) : keys;
    u32 start = (from_list == 0u) ? from : 0u;
    if (tid == 0u) {
        u32 stop = (from_list == 0u && to < picks) ? to : picks;
        if (from_list == 1u && picks > 0u && ids[u64(row) * picks + picks - 1u] >= keys) {
            u32 lo = 0u, hi = picks;
            for (u32 s = 0u; s < 16u; s++) {
                u32 mid = (lo + hi) >> 1;
                if (mid < hi) { if (ids[u64(row) * picks + mid] < keys) lo = mid + 1u; else hi = mid; }
            }
            stop = lo;
        }
        sm.stop = stop;
    }
    if (tid < 16u) { attn_init_head_state(sm.m[tid], sm.l[tid], sinks, has_sinks, split, h0 + tid); }
    prim::block_sync();
    u32 stop = sm.stop, total_p = (stop > start) ? (stop - start) : 0u;
    u32 per = ((((total_p + 31u) >> 5) + splits - 1u) / splits) << 5;
    u32 p_begin = start + split * per, p_end = p_begin + per;
    if (p_begin > stop) p_begin = stop;
    if (p_end > stop) p_end = stop;
    if (p_begin >= p_end) {
        attn_store_empty<OUT_W, OUT_STRIDE, 256u>(part, sinks, has_sinks, row, heads, h0, 16u, split, splits, tid);
        return;
    }
    constexpr bool USE_BF16 = (KV_FMT == 1u);
    // Keep Q[16, 144] per warp in registers.
    u32 q_reg[9][4];
    const float* q0 = q + u64(row * heads + h0 + g) * 576ull;
    const float* q8 = q + u64(row * heads + h0 + g + 8u) * 576ull;
#pragma unroll
    for (u32 s = 0u; s < 9u; s++) {
        u32 k0 = (wl * 9u + s) * 16u + (t << 1);
        float2 a0 = *reinterpret_cast<const float2*>(q0 + k0);
        float2 a1 = *reinterpret_cast<const float2*>(q8 + k0);
        float2 a2 = *reinterpret_cast<const float2*>(q0 + k0 + 8u);
        float2 a3 = *reinterpret_cast<const float2*>(q8 + k0 + 8u);
        q_reg[s][0] = attn_pack2<USE_BF16>(a0.x, a0.y);
        q_reg[s][1] = attn_pack2<USE_BF16>(a1.x, a1.y);
        q_reg[s][2] = attn_pack2<USE_BF16>(a2.x, a2.y);
        q_reg[s][3] = attn_pack2<USE_BF16>(a3.x, a3.y);
    }
    float o[NT][4];
#pragma unroll
    for (u32 nt = 0u; nt < NT; nt++) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
    const u8* kv_u8 = static_cast<const u8*>(kv_ptr);
    const u8* kv_host_u8 = static_cast<const u8*>(kv_host);
    const u16* kv_u16 = static_cast<const u16*>(kv_ptr);
    const u16* kv_host_u16 = static_cast<const u16*>(kv_host);
    const float* kv_f32 = static_cast<const float*>(kv_ptr);
    const float* kv_host_f32 = static_cast<const float*>(kv_host);
    u32 tk = tid >> 4u, l16 = tid & 15u, d_base = warp * (NT * 8u);
    u32 krow = wg * 8u + (lane & 7u), kcol = ((lane >> 3u) & 1u) * 8u;
    u32 vrow = lane & 15u, vcol = (lane >> 4u) << 3u;
    u32 cb = wg * 8u, h = tid >> 4u, c = tid & 15u;
    uint4 pref[3];
    u32 j0_init = mla_pick_id(ids, row, picks, from_list, from, to, p_begin + tk, p_end);
    if constexpr (KV_FMT == 1u) {
        mla_stage_bf16(kv_u16, kv_host_u16, page_table, slot, j0_init, context, page_shift, l16, sm.kv[0][tk]);
        attn_cp_commit();
    } else if constexpr (KV_FMT == 0u) {
        mla_prefetch_fp8(kv_u8, kv_host_u8, page_table, slot, j0_init, context, page_shift, l16, pref);
    }
    for (u32 p0 = p_begin; p0 < p_end; p0 += 32u) {
        bool has_hi = (p0 + 16u < p_end), has_next = (p0 + 32u < p_end);
        u32 j1 = has_hi ? mla_pick_id(ids, row, picks, from_list, from, to, p0 + 16u + tk, p_end) : 0xFFFFFFFFu;
        mla_set_mask(ids, row, picks, from_list, from, to, p0, p_end, tid, &sm.valid_mask);
        float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
        if constexpr (KV_FMT == 1u) {
            if (has_hi) {
                mla_stage_bf16(kv_u16, kv_host_u16, page_table, slot, j1, context, page_shift, l16, sm.kv[1][tk]);
                attn_cp_commit();
                attn_cp_wait1();
            } else {
                attn_cp_wait0();
            }
        } else if constexpr (KV_FMT == 0u) {
            mla_unpack_fp8_row(pref, l16, reinterpret_cast<u32*>(sm.kv[0][tk]));
            if (has_hi) mla_prefetch_fp8(kv_u8, kv_host_u8, page_table, slot, j1, context, page_shift, l16, pref);
        } else {
            u32 j0 = mla_pick_id(ids, row, picks, from_list, from, to, p0 + tk, p_end);
            mla_stage_f32(kv_f32, kv_host_f32, page_table, slot, j0, context, page_shift, l16, reinterpret_cast<u32*>(sm.kv[0][tk]));
            if (has_hi) mla_stage_f32(kv_f32, kv_host_f32, page_table, slot, j1, context, page_shift, l16, reinterpret_cast<u32*>(sm.kv[1][tk]));
        }
        prim::block_sync();
        u32 vmask = sm.valid_mask;
        if (vmask == 0u) {
            if constexpr (KV_FMT == 1u) { if (has_hi) attn_cp_wait0(); }
            if (has_next) {
                u32 nj0 = mla_pick_id(ids, row, picks, from_list, from, to, p0 + 32u + tk, p_end);
                if constexpr (KV_FMT == 1u) {
                    prim::block_sync();
                    mla_stage_bf16(kv_u16, kv_host_u16, page_table, slot, nj0, context, page_shift, l16, sm.kv[0][tk]);
                    attn_cp_commit();
                } else {
                    if constexpr (KV_FMT == 0u) mla_prefetch_fp8(kv_u8, kv_host_u8, page_table, slot, nj0, context, page_shift, l16, pref);
                    prim::block_sync();
                }
            }
            continue;
        }
        if ((vmask & 0xFFFFu) != 0u) mla_qk_half<USE_BF16>(sm.kv[0], q_reg, krow, kcol, wl, s0);
        if (has_hi) {
            if constexpr (KV_FMT == 1u) {
                attn_cp_wait0();
                prim::block_sync();
            } else if constexpr (KV_FMT == 0u) {
                mla_unpack_fp8_row(pref, l16, reinterpret_cast<u32*>(sm.kv[1][tk]));
                prim::block_sync();
            }
            if ((vmask >> 16u) != 0u) mla_qk_half<USE_BF16>(sm.kv[1], q_reg, krow, kcol, wl, s1);
        }
        if constexpr (KV_FMT == 0u) {
            if (has_next) {
                u32 nj0 = mla_pick_id(ids, row, picks, from_list, from, to, p0 + 32u + tk, p_end);
                mla_prefetch_fp8(kv_u8, kv_host_u8, page_table, slot, nj0, context, page_shift, l16, pref);
            }
        }
        *reinterpret_cast<float2*>(&sm.red[wl][g][cb + t * 2u]) = make_float2(s0[0], s0[1]);
        *reinterpret_cast<float2*>(&sm.red[wl][g + 8u][cb + t * 2u]) = make_float2(s0[2], s0[3]);
        if (has_hi) {
            *reinterpret_cast<float2*>(&sm.red[wl][g][16u + cb + t * 2u]) = make_float2(s1[0], s1[1]);
            *reinterpret_cast<float2*>(&sm.red[wl][g + 8u][16u + cb + t * 2u]) = make_float2(s1[2], s1[3]);
        }
        prim::block_sync();
        bool ok0 = ((vmask >> c) & 1u) != 0u;
        bool ok1 = ((vmask >> (c + 16u)) & 1u) != 0u;
        float v0 = ok0
            ? (sm.red[0][h][c] + sm.red[1][h][c] + sm.red[2][h][c] + sm.red[3][h][c]) * scale
            : attn_neg_inf();
        float v1 = ok1
            ? (sm.red[0][h][c + 16u] + sm.red[1][h][c + 16u] + sm.red[2][h][c + 16u] + sm.red[3][h][c + 16u]) * scale
            : attn_neg_inf();
        float tmax = fmaxf(v0, v1);
        tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFu, tmax, 1u));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFu, tmax, 2u));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFu, tmax, 4u));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFFu, tmax, 8u));
        float om = sm.m[h], ol = sm.l[h], nm = (om > tmax) ? om : tmax;
        float cv = (ol > 0.0f) ? __expf(om - nm) : 0.0f;
        float pv0 = ok0 ? __expf(v0 - nm) : 0.0f;
        float pv1 = ok1 ? __expf(v1 - nm) : 0.0f;
        float pv0n = __shfl_xor_sync(0xFFFFFFFFu, pv0, 1u);
        float pv1n = __shfl_xor_sync(0xFFFFFFFFu, pv1, 1u);
        if ((c & 1u) == 0u) {
            sm.p16[h][c >> 1u] = attn_pack2<USE_BF16>(pv0, pv0n);
            sm.p16[h][8u + (c >> 1u)] = attn_pack2<USE_BF16>(pv1, pv1n);
        }
        float ps = pv0 + pv1;
        ps += __shfl_xor_sync(0xFFFFFFFFu, ps, 1u);
        ps += __shfl_xor_sync(0xFFFFFFFFu, ps, 2u);
        ps += __shfl_xor_sync(0xFFFFFFFFu, ps, 4u);
        ps += __shfl_xor_sync(0xFFFFFFFFu, ps, 8u);
        if (c == 0u) { sm.m[h] = nm; sm.l[h] = ol * cv + ps; sm.corr[h] = cv; }
        prim::block_sync();
        float c0 = sm.corr[g], c1 = sm.corr[g + 8u];
#pragma unroll
        for (u32 nt = 0u; nt < NT; nt++) {
            o[nt][0] *= c0; o[nt][1] *= c0; o[nt][2] *= c1; o[nt][3] *= c1;
        }
        if ((vmask & 0xFFFFu) != 0u) mla_pv_half<USE_BF16, NT>(sm.kv[0], sm.p16, 0u, d_base, vrow, vcol, g, t, o);
        if constexpr (KV_FMT == 1u) {
            if (has_next) {
                prim::block_sync();
                u32 nj0 = mla_pick_id(ids, row, picks, from_list, from, to, p0 + 32u + tk, p_end);
                mla_stage_bf16(kv_u16, kv_host_u16, page_table, slot, nj0, context, page_shift, l16, sm.kv[0][tk]);
                attn_cp_commit();
            }
        }
        if ((vmask >> 16u) != 0u) mla_pv_half<USE_BF16, NT>(sm.kv[1], sm.p16, 8u, d_base, vrow, vcol, g, t, o);
        if (has_next) prim::block_sync();
    }
    attn_store_partials<NT, OUT_W, OUT_STRIDE>(part, o, sm.m, sm.l, row, heads, h0, 16u, split, splits,
                                               d_base, g, t, tid);
}

#define MLA_DEF(NAME, NAME512, T, FMT) \
    extern "C" __global__ __launch_bounds__(256, MLA_MIN_BLOCKS) void NAME(const float* q, const T* kv, const u32* ids, const u32* slots, const u32* positions, const float* sinks, float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads, u32 picks, u32 from_list, u32 before, u32 after, u32 splits, const u32* page_table, u32 page_shift, u32 has_sinks, const T* kv_host) { \
        __shared__ mla_smem sm; \
        mla_attend_body<FMT, 576u>(q, kv, kv_host, ids, slots, positions, sinks, part, scale, rows, keys, context, heads, picks, from_list, before, after, splits, page_table, page_shift, has_sinks, sm); \
    } \
    extern "C" __global__ __launch_bounds__(256, MLA_MIN_BLOCKS) void NAME512(const float* q, const T* kv, const u32* ids, const u32* slots, const u32* positions, const float* sinks, float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads, u32 picks, u32 from_list, u32 before, u32 after, u32 splits, const u32* page_table, u32 page_shift, u32 has_sinks, const T* kv_host) { \
        __shared__ mla_smem sm; \
        mla_attend_body<FMT, 512u>(q, kv, kv_host, ids, slots, positions, sinks, part, scale, rows, keys, context, heads, picks, from_list, before, after, splits, page_table, page_shift, has_sinks, sm); \
    }

MLA_DEF(tessera_mla_attend_fp8, tessera_mla_attend_512_fp8, u8, 0u)
MLA_DEF(tessera_mla_attend_bf16, tessera_mla_attend_512_bf16, u16, 1u)
MLA_DEF(tessera_mla_attend_f32, tessera_mla_attend_512_f32, float, 2u)

// Combines split-K MLA partials for one row and head.
extern "C" __global__ __launch_bounds__(MLA_REDUCE_THREADS, MLA_REDUCE_MIN_BLOCKS) void tessera_mla_reduce(
    const float* part, float* out, u32 lists, u32 splits, u32 out_w) {
    u32 list = blockIdx.x, tid = threadIdx.x, stride = out_w + 4u;
    if (list >= lists) return;
    __shared__ float s_m[32], s_l[32], s_w[32], s_inv;
    const float* base = part + u64(list) * splits * stride;
    if (tid < splits) { s_m[tid] = base[u64(tid) * stride + out_w]; s_l[tid] = base[u64(tid) * stride + out_w + 1u]; }
    prim::block_sync();
    if (tid == 0u) {
        float best = attn_neg_inf(), denom = 0.0f;
        for (u32 s = 0u; s < splits; s++) if (s_l[s] > 0.0f && s_m[s] > best) best = s_m[s];
        for (u32 s = 0u; s < splits; s++) {
            float w = (s_l[s] > 0.0f) ? __expf(s_m[s] - best) : 0.0f;
            s_w[s] = w; denom += s_l[s] * w;
        }
        s_inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    }
    prim::block_sync();
    for (u32 t4 = tid; t4 < (out_w >> 2u); t4 += blockDim.x) {
        float inv = s_inv;
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        u32 d4 = t4 << 2;
        for (u32 s = 0u; s < splits; s++) if (s_w[s] > 0.0f) attn_fma4(sum, base + u64(s) * stride + d4, s_w[s]);
        *reinterpret_cast<float4*>(out + u64(list) * out_w + d4) = make_float4(sum.x * inv, sum.y * inv, sum.z * inv, sum.w * inv);
    }
}

struct alignas(16) gqa_smem {
    u32 q[16][68], k[16][68], v[16][68];
    float red[4][16][16];
    u32 p16[16][8];
    float m[16], l[16], corr[16];
};

template <u32 KV_FMT>
static __device__ __forceinline__ void gqa_stage_vec(const void* ptr, u64 rb, u32 l8, bool valid, u32* dst) {
    if constexpr (KV_FMT == 0u) {
        if (valid) attn_unpack_fp8(*reinterpret_cast<const uint4*>(static_cast<const u8*>(ptr) + rb + l8 * 16u), dst + l8 * 8u);
        else { *reinterpret_cast<uint4*>(dst + l8 * 8u) = make_uint4(0u, 0u, 0u, 0u); *reinterpret_cast<uint4*>(dst + l8 * 8u + 4u) = make_uint4(0u, 0u, 0u, 0u); }
    } else {
#pragma unroll
        for (u32 q4 = l8; q4 < 32u; q4 += 8u) {
            u32 h0 = 0u, h1 = 0u;
            if (valid) {
                if constexpr (KV_FMT == 1u) {
                    uint2 r = *reinterpret_cast<const uint2*>(static_cast<const u16*>(ptr) + rb + u64(q4) * 4ull);
                    h0 = trellis::f32x2_to_f16x2(__uint_as_float(r.x << 16), __uint_as_float(r.x & 0xFFFF0000u));
                    h1 = trellis::f32x2_to_f16x2(__uint_as_float(r.y << 16), __uint_as_float(r.y & 0xFFFF0000u));
                } else {
                    float4 r = *reinterpret_cast<const float4*>(static_cast<const float*>(ptr) + rb + u64(q4) * 4ull);
                    h0 = trellis::f32x2_to_f16x2(r.x, r.y); h1 = trellis::f32x2_to_f16x2(r.z, r.w);
                }
            }
            *reinterpret_cast<uint2*>(dst + q4 * 2u) = make_uint2(h0, h1);
        }
    }
}

template <u32 KV_FMT>
static __device__ __forceinline__ void gqa_attend_body(
    const float* q, const void* k_ptr, const void* v_ptr, const u32* slots, const u32* positions,
    const float* sinks, float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads,
    u32 kv_heads, u32 before, u32 after, u32 splits, const u32* page_table, u32 page_shift,
    u32 has_sinks, gqa_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    u32 gqa = heads / kv_heads, nh = (gqa < 16u) ? gqa : 16u, per_kv = (gqa + 15u) >> 4;
    u32 kg_groups = kv_heads * per_kv, split = blockIdx.x, rb = blockIdx.y / kg_groups, kg = blockIdx.y % kg_groups;
    if (rb >= rows || split >= splits) return;
    u32 row = (rows - 1u) - rb, kv_head = kg / per_kv, h0 = kv_head * gqa + ((kg % per_kv) << 4);
    u32 slot = slots[row], pos = positions[row];
    u32 from = (pos > before) ? (pos - before) : 0u;
    u32 to = (pos + after < keys) ? (pos + after + 1u) : keys;
    u32 total_p = (to > from) ? (to - from) : 0u;
    u32 per = ((((total_p + 15u) >> 4) + splits - 1u) / splits) << 4;
    u32 p_begin = from + split * per, p_end = p_begin + per;
    if (p_begin > to) p_begin = to;
    if (p_end > to) p_end = to;
    if (p_begin >= p_end) {
        attn_store_empty<128u, 132u, 128u>(part, sinks, has_sinks, row, heads, h0, nh, split, splits, tid);
        return;
    }
    u32 tk = tid >> 3, lane8 = tid & 7u;
    const float* q_row = (tk < nh) ? (q + u64(row * heads + h0 + tk) * 128ull) : nullptr;
#pragma unroll
    for (u32 q4 = lane8; q4 < 32u; q4 += 8u) {
        uint2 v = make_uint2(0u, 0u);
        if (tk < nh) {
            float4 qv = *reinterpret_cast<const float4*>(q_row + u64(q4) * 4ull);
            v = make_uint2(trellis::f32x2_to_f16x2(qv.x, qv.y), trellis::f32x2_to_f16x2(qv.z, qv.w));
        }
        *reinterpret_cast<uint2*>(&sm.q[tk][q4 * 2u]) = v;
    }
    if (tid < 16u) {
        if (tid < nh) { attn_init_head_state(sm.m[tid], sm.l[tid], sinks, has_sinks, split, h0 + tid); }
        else { sm.m[tid] = attn_neg_inf(); sm.l[tid] = 0.0f; }
        sm.corr[tid] = 0.0f;
#pragma unroll
        for (u32 c = 0u; c < 8u; c++) sm.p16[tid][c] = 0u;
    }
    prim::block_sync();
    float o[4][4];
#pragma unroll
    for (u32 nt = 0u; nt < 4u; nt++) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
    const u16* v16 = reinterpret_cast<const u16*>(sm.v);
    u32 d_base = warp << 5;
    for (u32 j0 = p_begin; j0 < p_end; j0 += 16u) {
        u32 j = j0 + tk;
        bool valid = j < p_end;
        u64 kb = valid ? (kv_row_at(page_table, slot, j, context, page_shift) * kv_heads + kv_head) * 128ull : 0ull;
        gqa_stage_vec<KV_FMT>(k_ptr, kb, lane8, valid, sm.k[tk]);
        gqa_stage_vec<KV_FMT>(v_ptr, kb, lane8, valid, sm.v[tk]);
        prim::block_sync();
        attn_qk_tile<2u, 68u>(sm.q, sm.k, sm.red, warp, g, t);
        prim::block_sync();
        u32 vmask = (1u << ((p_end - j0 < 16u) ? (p_end - j0) : 16u)) - 1u;
        if (tid < nh) attn_softmax_row(sm.red, vmask, tid, scale, sm.m, sm.l, sm.corr, sm.p16);
        prim::block_sync();
        attn_pv_tile<4u, 136u>(o, sm.p16, sm.corr, v16, d_base, g, t);
        if (j0 + 16u < p_end) prim::block_sync();
    }
    attn_store_partials<4u, 128u, 132u>(part, o, sm.m, sm.l, row, heads, h0, nh, split, splits,
                                        d_base, g, t, tid);
}

#define GQA_DEF(NAME, T, FMT) \
    extern "C" __global__ __launch_bounds__(128, GQA_MIN_BLOCKS) void NAME(const float* q, const T* k, const T* v, const u32* slots, const u32* positions, const float* sinks, float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads, u32 kv_heads, u32 before, u32 after, u32 splits, const u32* page_table, u32 page_shift, u32 has_sinks) { \
        __shared__ gqa_smem sm; \
        gqa_attend_body<FMT>(q, k, v, slots, positions, sinks, part, scale, rows, keys, context, heads, kv_heads, before, after, splits, page_table, page_shift, has_sinks, sm); \
    }

GQA_DEF(tessera_gqa_attend_fp8, u8, 0u)
GQA_DEF(tessera_gqa_attend_bf16, u16, 1u)
GQA_DEF(tessera_gqa_attend_f32, float, 2u)

// Warp-level reduction of split-K GQA partials for head_dim 128.
extern "C" __global__ __launch_bounds__(128, GQA_REDUCE_MIN_BLOCKS) void tessera_gqa_reduce(
    const float* part, float* out, u32 lists, u32 splits) {
    u32 list = blockIdx.x * 4u + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (list >= lists) return;
    const float* base = part + u64(list) * splits * 132ull;
    float my_m = (lane < splits) ? base[u64(lane) * 132ull + 128ull] : attn_neg_inf();
    float my_l = (lane < splits) ? base[u64(lane) * 132ull + 129ull] : 0.0f;
    float best = (my_l > 0.0f) ? my_m : attn_neg_inf();
#pragma unroll
    for (u32 off = 16u; off > 0u; off >>= 1) best = fmaxf(best, __shfl_xor_sync(0xFFFFFFFFu, best, off));
    float my_w = (lane < splits && my_l > 0.0f) ? __expf(my_m - best) : 0.0f, denom = my_l * my_w;
#pragma unroll
    for (u32 off = 16u; off > 0u; off >>= 1) denom += __shfl_xor_sync(0xFFFFFFFFu, denom, off);
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    u32 d4 = lane << 2;
    for (u32 s = 0u; s < splits; s++) {
        float w = __shfl_sync(0xFFFFFFFFu, my_w, s);
        if (w > 0.0f) attn_fma4(sum, base + u64(s) * 132ull + d4, w);
    }
    *reinterpret_cast<float4*>(out + u64(list) * 128ull + d4) = make_float4(sum.x * inv, sum.y * inv, sum.z * inv, sum.w * inv);
}

constexpr u32 IDX_SM_WORDS = 4608u;
struct alignas(16) idx_smem { u32 q[IDX_SM_WORDS], k[IDX_SM_WORDS]; float w[128]; };
static __device__ __forceinline__ u32 idx_e4m3x2_bf16x2(u16 q) {
    u32 h = attn_e4m3_f16x2(q);
    return attn_pack2<true>(trellis::f16_to_f32(h & 0xFFFFu), trellis::f16_to_f32(h >> 16u));
}

template <u32 KV_FMT>
static __device__ __forceinline__ void idx_load_k16(
    const void* kp, u32* sk, u32 slot, u32 ctx, u32 dim, u32 stride, u32 j0, u32 to,
    const u32* pt, u32 sh, u32 lane) {
    u32 qw = dim >> 2, tot = qw << 4;
    for (u32 idx = lane; idx < tot; idx += 32u) {
        u32 kr = idx / qw, q4 = idx % qw, j = j0 + kr;
        uint2 pk = make_uint2(0u, 0u);
        if (j < to) {
            u64 b = kv_row_at(pt, slot, j, ctx, sh) * dim + (u64(q4) << 2);
            if constexpr (KV_FMT == 0u) {
                u32 r = *reinterpret_cast<const u32*>(static_cast<const u8*>(kp) + b);
                pk = make_uint2(idx_e4m3x2_bf16x2(u16(r)), idx_e4m3x2_bf16x2(u16(r >> 16)));
            } else if constexpr (KV_FMT == 1u) {
                pk = *reinterpret_cast<const uint2*>(static_cast<const u16*>(kp) + b);
            } else {
                float4 v = *reinterpret_cast<const float4*>(static_cast<const float*>(kp) + b);
                pk = make_uint2(attn_pack2<true>(v.x, v.y), attn_pack2<true>(v.z, v.w));
            }
        }
        *reinterpret_cast<uint2*>(sk + kr * stride + (q4 << 1)) = pk;
    }
}

template <u32 KV_FMT>
static __device__ __forceinline__ void idx_body(
    const float* q, const void* k_ptr, const float* weights, const u32* slots,
    const u32* positions, float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context,
    u32 heads, u32 dim, u32 masked, u32 tile_keys, const u32* page_table, u32 page_shift,
    idx_smem& sm) {
    u32 row = blockIdx.y, j_begin = blockIdx.x * tile_keys, j_end = j_begin + tile_keys;
    if (j_end > out_keys) j_end = out_keys;
    if (row >= rows || j_begin >= j_end) return;
    u32 tid = threadIdx.x, slot = slots[row], to = live_keys;
    if (masked == 1u && positions[row] < to) to = positions[row] + 1u;
    float* out_row = out + u64(row) * out_keys;
    float neg = attn_neg_inf();
    if (j_begin >= to) {
        for (u32 j = j_begin + tid; j < j_end; j += 128u) out_row[j] = neg;
        return;
    }
    if (tid < heads) sm.w[tid] = weights[u64(row) * heads + tid];
    u32 stride = (dim >> 1) + 2u, q_words = dim >> 2, total_q4 = heads * q_words;
    const float* q_base = q + u64(row) * heads * dim;
    for (u32 idx = tid; idx < total_q4; idx += 128u) {
        u32 h = idx / q_words, q4 = idx % q_words;
        float4 v = *reinterpret_cast<const float4*>(q_base + u64(h) * dim + (u64(q4) << 2));
        *reinterpret_cast<uint2*>(sm.q + h * stride + (q4 << 1)) =
            make_uint2(attn_pack2<true>(v.x, v.y), attn_pack2<true>(v.z, v.w));
    }
    prim::block_sync();
    u32 warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u, d_steps = dim >> 4;
    u32* sk = sm.k + warp * 16u * stride;
    for (u32 j0 = j_begin + (warp << 4); j0 < j_end; j0 += 64u) {
        if (j0 >= to) { if (lane < 16u && j0 + lane < j_end) out_row[j0 + lane] = neg; continue; }
        idx_load_k16<KV_FMT>(k_ptr, sk, slot, context, dim, stride, j0, to, page_table, page_shift, lane);
        __syncwarp();
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (u32 h0 = 0u; h0 < heads; h0 += 16u) {
            float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
            for (u32 ds = 0u; ds < d_steps; ds++) {
                u32 k0 = ds << 3;
                u32 a[4] = {sm.q[(h0 + g) * stride + k0 + t], sm.q[(h0 + g + 8u) * stride + k0 + t],
                            sm.q[(h0 + g) * stride + k0 + t + 4u], sm.q[(h0 + g + 8u) * stride + k0 + t + 4u]};
                u32 b0[2] = {sk[g * stride + k0 + t], sk[g * stride + k0 + t + 4u]};
                u32 b1[2] = {sk[(g + 8u) * stride + k0 + t], sk[(g + 8u) * stride + k0 + t + 4u]};
                attn_mma<true>(s0, a, b0); attn_mma<true>(s1, a, b1);
            }
            float w0 = sm.w[h0 + g], w1 = sm.w[h0 + g + 8u];
            acc[0] += fmaxf(s0[0], 0.f) * w0 + fmaxf(s0[2], 0.f) * w1;
            acc[1] += fmaxf(s0[1], 0.f) * w0 + fmaxf(s0[3], 0.f) * w1;
            acc[2] += fmaxf(s1[0], 0.f) * w0 + fmaxf(s1[2], 0.f) * w1;
            acc[3] += fmaxf(s1[1], 0.f) * w0 + fmaxf(s1[3], 0.f) * w1;
        }
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
            acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 4u);
            acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 8u);
            acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 16u);
        }
        if (g == 0u) {
            u32 k0 = j0 + (t << 1), k2 = j0 + 8u + (t << 1);
            if (k0 < j_end) out_row[k0] = (k0 < to) ? acc[0] : neg;
            if (k0 + 1u < j_end) out_row[k0 + 1u] = (k0 + 1u < to) ? acc[1] : neg;
            if (k2 < j_end) out_row[k2] = (k2 < to) ? acc[2] : neg;
            if (k2 + 1u < j_end) out_row[k2 + 1u] = (k2 + 1u < to) ? acc[3] : neg;
        }
    }
}

struct alignas(16) idx_qt_smem {
    u16 k[2][32][136];
    u32 max_to, same_slot;
};

template <u32 KV_FMT>
static __device__ __forceinline__ void idx_qt_load_k(
    const void* kp, u16 dst[32][136], u32 slot, u32 ctx, u32 j0, u32 lim,
    const u32* pt, u32 sh, u32 tid) {
    u32 kr = tid >> 3, c = tid & 7u, j = j0 + kr;
    u32* d = reinterpret_cast<u32*>(dst[kr]) + c * 8u;
    if (j >= lim) {
        *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(d + 4u) = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    u64 b = kv_row_at(pt, slot, j, ctx, sh) * 128ull + u64(c) * 16ull;
    if constexpr (KV_FMT == 0u) {
        uint4 r = *reinterpret_cast<const uint4*>(static_cast<const u8*>(kp) + b);
        *reinterpret_cast<uint4*>(d) = make_uint4(idx_e4m3x2_bf16x2(u16(r.x)), idx_e4m3x2_bf16x2(u16(r.x >> 16)), idx_e4m3x2_bf16x2(u16(r.y)), idx_e4m3x2_bf16x2(u16(r.y >> 16)));
        *reinterpret_cast<uint4*>(d + 4u) = make_uint4(idx_e4m3x2_bf16x2(u16(r.z)), idx_e4m3x2_bf16x2(u16(r.z >> 16)), idx_e4m3x2_bf16x2(u16(r.w)), idx_e4m3x2_bf16x2(u16(r.w >> 16)));
    } else if constexpr (KV_FMT == 1u) {
        const u16* p = static_cast<const u16*>(kp) + b;
        *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(p);
        *reinterpret_cast<uint4*>(d + 4u) = *reinterpret_cast<const uint4*>(p + 8u);
    } else {
        const float* p = static_cast<const float*>(kp) + b;
        float4 v0 = *reinterpret_cast<const float4*>(p), v1 = *reinterpret_cast<const float4*>(p + 4u);
        float4 v2 = *reinterpret_cast<const float4*>(p + 8u), v3 = *reinterpret_cast<const float4*>(p + 12u);
        *reinterpret_cast<uint4*>(d) = make_uint4(attn_pack2<true>(v0.x, v0.y), attn_pack2<true>(v0.z, v0.w), attn_pack2<true>(v1.x, v1.y), attn_pack2<true>(v1.z, v1.w));
        *reinterpret_cast<uint4*>(d + 4u) = make_uint4(attn_pack2<true>(v2.x, v2.y), attn_pack2<true>(v2.z, v2.w), attn_pack2<true>(v3.x, v3.y), attn_pack2<true>(v3.z, v3.w));
    }
}

template <u32 KV_FMT>
static __device__ __forceinline__ void idx_qtile_body(
    const float* q, const void* k_ptr, const float* weights, const u32* slots,
    const u32* positions, float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context,
    u32 masked, u32 tile_keys, const u32* page_table, u32 page_shift, idx_qt_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    u32 r0 = blockIdx.y << 3, row = r0 + warp;
    u32 j_begin = blockIdx.x * tile_keys, j_end = j_begin + tile_keys;
    if (j_end > out_keys) j_end = out_keys;
    if (r0 >= rows || j_begin >= j_end) return;
    u32 nr = (rows - r0 < 8u) ? (rows - r0) : 8u;
    if (tid < 8u) {
        u32 r = r0 + tid, ok = tid < nr;
        u32 to = ok ? ((masked == 1u && positions[r] < live_keys) ? (positions[r] + 1u) : live_keys) : 0u;
        u32 s0 = slots[r0], same = ok ? (slots[r] == s0 ? 1u : 0u) : 1u;
        to = max(to, __shfl_xor_sync(0xFFu, to, 1u));
        to = max(to, __shfl_xor_sync(0xFFu, to, 2u));
        to = max(to, __shfl_xor_sync(0xFFu, to, 4u));
        same &= __shfl_xor_sync(0xFFu, same, 1u);
        same &= __shfl_xor_sync(0xFFu, same, 2u);
        same &= __shfl_xor_sync(0xFFu, same, 4u);
        if (tid == 0u) { sm.max_to = to; sm.same_slot = same; }
    }
    prim::block_sync();
    u32 max_to = sm.max_to, same_slot = sm.same_slot;
    u32 my_to = (row < rows) ? ((masked == 1u && positions[row] < live_keys) ? (positions[row] + 1u) : live_keys) : 0u;
    float* out_row = (row < rows) ? (out + u64(row) * out_keys) : nullptr;
    float neg = attn_neg_inf();
    if (j_begin >= max_to) {
        if (row < rows) for (u32 j = j_begin + lane; j < j_end; j += 32u) out_row[j] = neg;
        return;
    }
    u32 q_reg[2][8][4];
    float w_reg[2][2];
    if (row < rows) {
        const float* qb = q + u64(row) * 4096ull;
        const float* wb = weights + u64(row) * 32ull;
#pragma unroll
        for (u32 hi = 0u; hi < 2u; hi++) {
            u32 h0 = hi * 16u;
            w_reg[hi][0] = wb[h0 + g]; w_reg[hi][1] = wb[h0 + g + 8u];
            const float* q0 = qb + u64(h0 + g) * 128ull;
            const float* q8 = qb + u64(h0 + g + 8u) * 128ull;
#pragma unroll
            for (u32 ds = 0u; ds < 8u; ds++) {
                u32 k0 = ds * 16u + (t << 1);
                float2 a0 = *reinterpret_cast<const float2*>(q0 + k0), a1 = *reinterpret_cast<const float2*>(q8 + k0);
                float2 a2 = *reinterpret_cast<const float2*>(q0 + k0 + 8u), a3 = *reinterpret_cast<const float2*>(q8 + k0 + 8u);
                q_reg[hi][ds][0] = attn_pack2<true>(a0.x, a0.y); q_reg[hi][ds][1] = attn_pack2<true>(a1.x, a1.y);
                q_reg[hi][ds][2] = attn_pack2<true>(a2.x, a2.y); q_reg[hi][ds][3] = attn_pack2<true>(a3.x, a3.y);
            }
        }
    }
    u32 krow0 = ((lane >> 4) << 3) + (lane & 7u), krow1 = 16u + krow0, kcol = ((lane >> 3) & 1u) * 8u;
    u32 passes = same_slot ? 1u : nr;
    for (u32 pass = 0u; pass < passes; pass++) {
        u32 cur_slot = slots[r0 + pass];
        bool active = same_slot ? (row < rows) : (warp == pass);
        u32 lim = same_slot ? max_to : ((masked == 1u && positions[r0 + pass] < live_keys) ? (positions[r0 + pass] + 1u) : live_keys);
        u32 st = 0u;
        idx_qt_load_k<KV_FMT>(k_ptr, sm.k[0], cur_slot, context, j_begin, lim, page_table, page_shift, tid);
        prim::block_sync();
        for (u32 j0 = j_begin; j0 < j_end; j0 += 32u) {
            u32 nst = st ^ 1u;
            bool has_next = (j0 + 32u < j_end && j0 + 32u < lim);
            if (has_next) idx_qt_load_k<KV_FMT>(k_ptr, sm.k[nst], cur_slot, context, j0 + 32u, lim, page_table, page_shift, tid);
            if (active) {
                if (j0 >= my_to) {
                    if (j0 + lane < j_end) out_row[j0 + lane] = neg;
                } else {
                    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (u32 hi = 0u; hi < 2u; hi++) {
                        float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
                        float s2[4] = {0.f, 0.f, 0.f, 0.f}, s3[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                        for (u32 ds = 0u; ds < 8u; ds++) {
                            u32 b01[4], b23[4];
                            attn_ldm_x4(b01, &sm.k[st][krow0][ds * 16u + kcol]);
                            attn_ldm_x4(b23, &sm.k[st][krow1][ds * 16u + kcol]);
                            u32 b0[2] = {b01[0], b01[1]}, b1[2] = {b01[2], b01[3]};
                            u32 b2[2] = {b23[0], b23[1]}, b3[2] = {b23[2], b23[3]};
                            attn_mma<true>(s0, q_reg[hi][ds], b0); attn_mma<true>(s1, q_reg[hi][ds], b1);
                            attn_mma<true>(s2, q_reg[hi][ds], b2); attn_mma<true>(s3, q_reg[hi][ds], b3);
                        }
                        float w0 = w_reg[hi][0], w1 = w_reg[hi][1];
                        acc[0] += fmaxf(s0[0], 0.f) * w0 + fmaxf(s0[2], 0.f) * w1; acc[1] += fmaxf(s0[1], 0.f) * w0 + fmaxf(s0[3], 0.f) * w1;
                        acc[2] += fmaxf(s1[0], 0.f) * w0 + fmaxf(s1[2], 0.f) * w1; acc[3] += fmaxf(s1[1], 0.f) * w0 + fmaxf(s1[3], 0.f) * w1;
                        acc[4] += fmaxf(s2[0], 0.f) * w0 + fmaxf(s2[2], 0.f) * w1; acc[5] += fmaxf(s2[1], 0.f) * w0 + fmaxf(s2[3], 0.f) * w1;
                        acc[6] += fmaxf(s3[0], 0.f) * w0 + fmaxf(s3[2], 0.f) * w1; acc[7] += fmaxf(s3[1], 0.f) * w0 + fmaxf(s3[3], 0.f) * w1;
                    }
#pragma unroll
                    for (u32 i = 0u; i < 8u; i++) {
                        acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 4u);
                        acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 8u);
                        acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 16u);
                    }
                    if (g == 0u) {
                        u32 p0 = j0 + (t << 1);
#pragma unroll
                        for (u32 c = 0u; c < 4u; c++) {
                            u32 k0 = p0 + (c << 3);
                            if (k0 < j_end) out_row[k0] = (k0 < my_to) ? acc[c * 2u] : neg;
                            if (k0 + 1u < j_end) out_row[k0 + 1u] = (k0 + 1u < my_to) ? acc[c * 2u + 1u] : neg;
                        }
                    }
                }
            }
            if (has_next) prim::block_sync();
            st = nst;
        }
        if (pass + 1u < passes) prim::block_sync();
    }
}

#define IDX_DEF(NAME, QT_NAME, FMT) \
    extern "C" __global__ IDX_BOUNDS void NAME(const float* q, const void* k, const float* weights, const u32* slots, const u32* positions, float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 dim, u32 masked, u32 tile_keys, const u32* page_table, u32 page_shift) { \
        __shared__ idx_smem sm; \
        idx_body<FMT>(q, k, weights, slots, positions, out, rows, out_keys, live_keys, context, heads, dim, masked, tile_keys, page_table, page_shift, sm); \
    } \
    extern "C" __global__ __launch_bounds__(256, IDX_QT_MIN_BLOCKS) void QT_NAME(const float* q, const void* k, const float* weights, const u32* slots, const u32* positions, float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 dim, u32 masked, u32 tile_keys, const u32* page_table, u32 page_shift) { \
        (void)heads; (void)dim; __shared__ idx_qt_smem sm; \
        idx_qtile_body<FMT>(q, k, weights, slots, positions, out, rows, out_keys, live_keys, context, masked, tile_keys, page_table, page_shift, sm); \
    }

IDX_DEF(tessera_indexer_fp8, tessera_indexer_qt_fp8, 0u)
IDX_DEF(tessera_indexer_bf16, tessera_indexer_qt_bf16, 1u)
IDX_DEF(tessera_indexer_f32, tessera_indexer_qt_f32, 2u)

#endif
