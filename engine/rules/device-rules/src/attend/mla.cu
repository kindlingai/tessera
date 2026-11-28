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
#define IDX_D128_BOUNDS __launch_bounds__(128, IDX_MIN_BLOCKS)
#else
#define IDX_BOUNDS __launch_bounds__(128)
#define IDX_D128_BOUNDS __launch_bounds__(128, 4)
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
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
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

struct mla_window {
    u32 row, h0, slot, from, to, p_begin, p_end;
};

template <u32 TILE_SHIFT, typename SM>
static __device__ __forceinline__ mla_window mla_window_for(
    const u32* ids, const u32* slots, const u32* positions, const float* sinks, u32 rows, u32 keys,
    u32 heads, u32 picks, u32 from_list, u32 before, u32 after, u32 splits, u32 has_sinks, SM& sm) {
    constexpr u32 MASK = (1u << TILE_SHIFT) - 1u;
    u32 tid = threadIdx.x, split = blockIdx.x, hgroups = heads >> 4u;
    u32 row = rows - 1u - blockIdx.y / hgroups;
    u32 h0 = (blockIdx.y % hgroups) << 4u, slot = slots[row], pos = positions[row];
    u32 from = (pos > before) ? (pos - before) : 0u;
    u32 to = (pos + after < keys) ? (pos + after + 1u) : keys;
    u32 start = (from_list == 0u) ? from : 0u;
    if (tid == 0u) {
        u32 stop = (from_list == 0u && to < picks) ? to : picks;
        if (from_list == 1u && picks > 0u && ids[u64(row) * picks + picks - 1u] >= keys) {
            u32 lo = 0u, hi = picks;
            for (u32 s = 0u; s < 16u; s++) {
                u32 mid = (lo + hi) >> 1u;
                if (mid < hi) { if (ids[u64(row) * picks + mid] < keys) lo = mid + 1u; else hi = mid; }
            }
            stop = lo;
        }
        sm.stop = stop;
    }
    if (tid < 16u) { attn_init_head_state(sm.m[tid], sm.l[tid], sinks, has_sinks, split, h0 + tid); }
    prim::block_sync();
    u32 stop = sm.stop, total_p = (stop > start) ? (stop - start) : 0u;
    u32 per = ((((total_p + MASK) >> TILE_SHIFT) + splits - 1u) / splits) << TILE_SHIFT;
    u32 p_begin = start + split * per, p_end = p_begin + per;
    if (p_begin > stop) p_begin = stop;
    if (p_end > stop) p_end = stop;
    return {row, h0, slot, from, to, p_begin, p_end};
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
    u32 wg = warp >> 2, wl = warp & 3u, split = blockIdx.x;
    if (blockIdx.y / (heads >> 4) >= rows || split >= splits) return;
    auto win = mla_window_for<5u>(ids, slots, positions, sinks, rows, keys, heads, picks,
                                  from_list, before, after, splits, has_sinks, sm);
    u32 row = win.row, h0 = win.h0, slot = win.slot;
    u32 from = win.from, to = win.to, p_begin = win.p_begin, p_end = win.p_end;
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
static __device__ __forceinline__ u32 mla_pack4_fp8(float4 v, float inv) {
    return u32(trellis::f2_e4m3x2(v.x * inv, v.y * inv)) |
           (u32(trellis::f2_e4m3x2(v.z * inv, v.w * inv)) << 16u);
}
static __device__ __forceinline__ u32 mla_max4_scale(float4 a, float4 b, float& inv) {
    float mx = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w)));
    mx = fmaxf(mx, fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
    mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1u, 4));
    mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2u, 4));
    return trellis::mx_scale(mx, inv) * 0x01010101u;
}
static __device__ __forceinline__ u32 mla_prmt(u32 a, u32 b, u32 sel) {
    u32 out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
    return out;
}

// SM120 FP8 MLA with double-buffered KV and 512V.
struct alignas(16) mla_fp8_smem {
    u8 nope[2][32][512], rope[2][32][64];
    float red[4][16][32];
    u16 p8[16][16];
    u32 p_sc[16];
    float m[16], l[16], corr[16];
    u32 stop, valid_mask;
};

static __device__ __forceinline__ void mla_stage_fp8(mla_fp8_smem& sm, u32 b, const u8* kv,
                                                      const u8* kv_host, const u32* pt, u32 slot,
                                                      u32 j, u32 ctx, u32 sh, u32 tid) {
    u32 r = tid >> 3, lane8 = tid & 7u;
    bool is_host = false;
    const u8* src = (j != 0xFFFFFFFFu)
        ? kv_row_ptr(kv, kv_host, pt, slot, j, ctx, sh, 576ull, is_host)
        : kv;
    if (is_host) {
#pragma unroll
        for (u32 i = 0u; i < 5u; i++) {
            u32 ch = lane8 + i * 8u;
            if (ch < 32u) *reinterpret_cast<uint4*>(&sm.nope[b][r][((ch ^ (r & 7u)) << 4u)]) =
                *reinterpret_cast<const uint4*>(src + ch * 16u);
            else if (ch < 36u) *reinterpret_cast<uint4*>(&sm.rope[b][r][(ch - 32u) << 4u]) =
                *reinterpret_cast<const uint4*>(src + ch * 16u);
        }
    } else {
        u32 bytes = (j != 0xFFFFFFFFu) ? 16u : 0u;
#pragma unroll
        for (u32 i = 0u; i < 5u; i++) {
            u32 ch = lane8 + i * 8u;
            if (ch < 32u) attn_cp_async16(&sm.nope[b][r][((ch ^ (r & 7u)) << 4u)], src + ch * 16u, bytes);
            else if (ch < 36u) attn_cp_async16(&sm.rope[b][r][(ch - 32u) << 4u], src + ch * 16u, bytes);
        }
    }
    attn_cp_commit();
}
static __device__ __forceinline__ const u8* mla_nope_at(const mla_fp8_smem& sm, u32 b,
                                                         u32 r, u32 col) {
    return &sm.nope[b][r][(((col >> 4u) ^ (r & 7u)) << 4u) + (col & 15u)];
}

static __device__ __forceinline__ void mla_fp8_v512_body(
    const float* q, const u8* kv, const u8* kv_host, const u32* ids, const u32* slots,
    const u32* positions, const float* sinks, float* part, float scale, u32 rows, u32 keys,
    u32 context, u32 heads, u32 picks, u32 from_list, u32 before, u32 after, u32 splits,
    const u32* page_table, u32 page_shift, u32 has_sinks, mla_fp8_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    u32 wg = warp >> 2u, wl = warp & 3u, split = blockIdx.x;
    if (blockIdx.y / (heads >> 4u) >= rows || split >= splits) return;
    auto win = mla_window_for<5u>(ids, slots, positions, sinks, rows, keys, heads, picks,
                                  from_list, before, after, splits, has_sinks, sm);
    u32 row = win.row, h0 = win.h0, slot = win.slot;
    u32 from = win.from, to = win.to, p_begin = win.p_begin, p_end = win.p_end;
    if (p_begin >= p_end) {
        attn_store_empty<512u, 516u, 256u>(part, sinks, has_sinks, row, heads, h0, 16u, split, splits, tid);
        return;
    }
    u32 qa[5][4], q_sa[5];
    const float* q0 = q + u64(row * heads + h0 + g) * 576ull;
    const float* q8 = q + u64(row * heads + h0 + g + 8u) * 576ull;
#pragma unroll
    for (u32 s = 0u; s < 5u; s++) {
        u32 k0 = (wl * 5u + s) * 32u + (t << 2u);
        float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), b0 = a0, a1 = a0, b1 = a0;
        if (k0 < 576u) {
            a0 = *reinterpret_cast<const float4*>(q0 + k0);
            b0 = *reinterpret_cast<const float4*>(q0 + k0 + 16u);
            a1 = *reinterpret_cast<const float4*>(q8 + k0);
            b1 = *reinterpret_cast<const float4*>(q8 + k0 + 16u);
        }
        float inv0, inv1;
        u32 sc0 = mla_max4_scale(a0, b0, inv0);
        u32 sc1 = mla_max4_scale(a1, b1, inv1);
        q_sa[s] = (lane & 1u) ? sc1 : sc0;
        qa[s][0] = mla_pack4_fp8(a0, inv0); qa[s][1] = mla_pack4_fp8(a1, inv1);
        qa[s][2] = mla_pack4_fp8(b0, inv0); qa[s][3] = mla_pack4_fp8(b1, inv1);
    }
    float o[8][4] = {};
    u32 tk = tid >> 3u, vrow = lane;
    u32 krow = wg * 16u + ((lane >> 4u) << 3u) + (lane & 7u);
    u32 kcol = ((lane >> 3u) & 1u) << 4u;
    u32 cur = mla_pick_id(ids, row, picks, from_list, from, to, p_begin + tk, p_end);
    mla_stage_fp8(sm, 0u, kv, kv_host, page_table, slot, cur, context, page_shift, tid);
    u32 b = 0u, d_base = warp << 6u;
    for (u32 p0 = p_begin; p0 < p_end; p0 += 32u) {
        if (tid < 32u) {
            u32 j = mla_pick_id(ids, row, picks, from_list, from, to, p0 + tid, p_end);
            u32 mask = __ballot_sync(0xFFFFFFFFu, j != 0xFFFFFFFFu);
            if (tid == 0u) sm.valid_mask = mask;
        }
        bool next = p0 + 32u < p_end;
        if (next) {
            u32 nj = mla_pick_id(ids, row, picks, from_list, from, to, p0 + 32u + tk, p_end);
            mla_stage_fp8(sm, b ^ 1u, kv, kv_host, page_table, slot, nj, context, page_shift, tid);
            attn_cp_wait1();
        } else attn_cp_wait0();
        prim::block_sync();
        u32 mask = sm.valid_mask;
        if (mask != 0u) {
            float s0[4] = {}, s1[4] = {};
#pragma unroll
            for (u32 s = 0u; s < 5u; s++) {
                u32 k0 = (wl * 5u + s) << 5u;
                if (k0 >= 576u) continue;
                u32 kb[4];
                const u8* kp = k0 < 512u ? mla_nope_at(sm, b, krow, k0 + kcol)
                                           : &sm.rope[b][krow][k0 - 512u + kcol];
                attn_ldm_x4(kb, kp);
                u32 b0[2] = {kb[0], kb[1]}, b1[2] = {kb[2], kb[3]};
                u32 sa = q_sa[s];
                trellis::mma_mx8(s0, qa[s], b0, sa, 0x7F7F7F7Fu);
                trellis::mma_mx8(s1, qa[s], b1, sa, 0x7F7F7F7Fu);
            }
            u32 cb = wg << 4u;
            *reinterpret_cast<float2*>(&sm.red[wl][g][cb + t * 2u]) = make_float2(s0[0], s0[1]);
            *reinterpret_cast<float2*>(&sm.red[wl][g + 8u][cb + t * 2u]) = make_float2(s0[2], s0[3]);
            *reinterpret_cast<float2*>(&sm.red[wl][g][cb + 8u + t * 2u]) = make_float2(s1[0], s1[1]);
            *reinterpret_cast<float2*>(&sm.red[wl][g + 8u][cb + 8u + t * 2u]) = make_float2(s1[2], s1[3]);
            prim::block_sync();
            u32 h = (warp << 1u) | (lane >> 4u), seg = lane & 15u, c = seg << 1u;
            float2 r0v = *reinterpret_cast<const float2*>(&sm.red[0][h][c]);
            float2 r1v = *reinterpret_cast<const float2*>(&sm.red[1][h][c]);
            float2 r2v = *reinterpret_cast<const float2*>(&sm.red[2][h][c]);
            float2 r3v = *reinterpret_cast<const float2*>(&sm.red[3][h][c]);
            float x0 = ((mask >> c) & 1u) ? (r0v.x + r1v.x + r2v.x + r3v.x) * scale : attn_neg_inf();
            float x1 = ((mask >> (c + 1u)) & 1u) ? (r0v.y + r1v.y + r2v.y + r3v.y) * scale : attn_neg_inf();
            float mx = fmaxf(x0, x1);
#pragma unroll
            for (u32 off = 8u; off > 0u; off >>= 1u) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off, 16));
            float old_m = sm.m[h], old_l = sm.l[h], nm = fmaxf(old_m, mx);
            float corr = old_l > 0.0f ? __expf(old_m - nm) : 0.0f;
            float p0v = ((mask >> c) & 1u) ? __expf(x0 - nm) : 0.0f;
            float p1v = ((mask >> (c + 1u)) & 1u) ? __expf(x1 - nm) : 0.0f;
            float psum = p0v + p1v, pmax = fmaxf(p0v, p1v);
#pragma unroll
            for (u32 off = 8u; off > 0u; off >>= 1u) {
                psum += __shfl_xor_sync(0xFFFFFFFFu, psum, off, 16);
                pmax = fmaxf(pmax, __shfl_xor_sync(0xFFFFFFFFu, pmax, off, 16));
            }
            float pinv;
            u32 pscale = trellis::mx_scale(pmax, pinv) * 0x01010101u;
            sm.p8[h][((seg & 3u) << 1u) | ((seg >> 2u) & 1u) | ((seg >> 3u) << 3u)] =
                trellis::f2_e4m3x2(p0v * pinv, p1v * pinv);
            if (seg == 0u) {
                sm.m[h] = nm; sm.l[h] = old_l * corr + psum; sm.corr[h] = corr;
                sm.p_sc[h] = pscale;
            }
            prim::block_sync();
            float c0 = sm.corr[g], c1 = sm.corr[g + 8u];
#pragma unroll
            for (u32 nt = 0u; nt < 8u; nt++) {
                o[nt][0] *= c0; o[nt][1] *= c0; o[nt][2] *= c1; o[nt][3] *= c1;
            }
            u32 pa[4] = {*reinterpret_cast<const u32*>(&sm.p8[g][t << 1u]),
                         *reinterpret_cast<const u32*>(&sm.p8[g + 8u][t << 1u]),
                         *reinterpret_cast<const u32*>(&sm.p8[g][8u + (t << 1u)]),
                         *reinterpret_cast<const u32*>(&sm.p8[g + 8u][8u + (t << 1u)])};
            u32 sa = sm.p_sc[g + ((lane & 1u) << 3u)];
#pragma unroll
            for (u32 nt = 0u; nt < 8u; nt += 2u) {
                u32 raw[4], even[2], odd[2];
                attn_ldm_x4_t(raw, mla_nope_at(sm, b, vrow, d_base + nt * 8u));
                even[0] = mla_prmt(raw[0], raw[1], 0x6420u);
                even[1] = mla_prmt(raw[2], raw[3], 0x6420u);
                odd[0] = mla_prmt(raw[0], raw[1], 0x7531u);
                odd[1] = mla_prmt(raw[2], raw[3], 0x7531u);
                trellis::mma_mx8(o[nt], pa, even, sa, 0x7F7F7F7Fu);
                trellis::mma_mx8(o[nt + 1u], pa, odd, sa, 0x7F7F7F7Fu);
            }
        }
        if (next) prim::block_sync();
        b ^= 1u;
    }
#pragma unroll
    for (u32 hi = 0u; hi < 2u; hi++) {
        u32 head = g + (hi << 3u);
        float inv = splits == 1u && sm.l[head] > 0.0f ? 1.0f / sm.l[head] : 1.0f;
        u64 base = splits == 1u ? u64(row * heads + h0 + head) * 512ull
                                 : (u64(row * heads + h0 + head) * splits + split) * 516ull;
#pragma unroll
        for (u32 nt = 0u; nt < 8u; nt++) {
            u32 col = d_base + (nt >> 1u) * 16u + (t << 2u) + (nt & 1u);
            float a = o[nt][hi * 2u] * inv, z = o[nt][hi * 2u + 1u] * inv;
            asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(part + base + col), "f"(a));
            asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(part + base + col + 2u), "f"(z));
        }
    }
    if (splits > 1u && tid < 16u) {
        u64 base = (u64(row * heads + h0 + tid) * splits + split) * 516ull;
        part[base + 512u] = sm.m[tid]; part[base + 513u] = sm.l[tid];
    }
}

extern "C" __global__ __launch_bounds__(256, MLA_MIN_BLOCKS) void tessera_mla_attend_fp8_v512(
    const float* q, const u8* kv, const u32* ids, const u32* slots, const u32* positions,
    const float* sinks, float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads,
    u32 picks, u32 from_list, u32 before, u32 after, u32 splits, const u32* page_table,
    u32 page_shift, u32 has_sinks, const u8* kv_host) {
    __shared__ mla_fp8_smem sm;
    mla_fp8_v512_body(q, kv, kv_host, ids, slots, positions, sinks, part, scale, rows, keys,
                      context, heads, picks, from_list, before, after, splits, page_table,
                      page_shift, has_sinks, sm);
}
#else
extern "C" __global__ __launch_bounds__(256, MLA_MIN_BLOCKS) void tessera_mla_attend_fp8_v512(
    const float* q, const u8* kv, const u32* ids, const u32* slots, const u32* positions,
    const float* sinks, float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads,
    u32 picks, u32 from_list, u32 before, u32 after, u32 splits, const u32* page_table,
    u32 page_shift, u32 has_sinks, const u8* kv_host) {
    __shared__ mla_smem sm;
    mla_attend_body<0u, 512u>(q, kv, kv_host, ids, slots, positions, sinks, part, scale, rows,
                              keys, context, heads, picks, from_list, before, after, splits,
                              page_table, page_shift, has_sinks, sm);
}
#endif

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

#endif
