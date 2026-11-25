#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_CUDA)

static __device__ __forceinline__ float attn_neg_inf() {
    return __int_as_float(0xff800000u);
}

static __device__ __forceinline__ void attn_fma4(float4& sum, const float* p, float w) {
    float4 v = *reinterpret_cast<const float4*>(p);
    sum.x = fmaf(v.x, w, sum.x);
    sum.y = fmaf(v.y, w, sum.y);
    sum.z = fmaf(v.z, w, sum.z);
    sum.w = fmaf(v.w, w, sum.w);
}

static __device__ __forceinline__ u32 attn_e4m3_f16x2(u16 q) {
    u32 h;
    asm("{.reg .b16 t; mov.b16 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(h) : "h"(q));
    return h;
}

template <u32 STEPS, u32 STRIDE>
static __device__ __forceinline__ void attn_qk_tile(const u32 (*q)[STRIDE], const u32 (*k)[STRIDE],
                                                    float red[4][16][16], u32 warp, u32 g, u32 t) {
    float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (u32 step = 0u; step < STEPS; step++) {
        u32 k0 = (warp * STEPS + step) * 8u;
        u32 a[4] = {q[g][k0 + t], q[g + 8u][k0 + t], q[g][k0 + t + 4u], q[g + 8u][k0 + t + 4u]};
        u32 b0[2] = {k[g][k0 + t], k[g][k0 + t + 4u]};
        u32 b1[2] = {k[g + 8u][k0 + t], k[g + 8u][k0 + t + 4u]};
        trellis::mma_m16n8k16(s0, a, b0);
        trellis::mma_m16n8k16(s1, a, b1);
    }
    red[warp][g][t * 2u] = s0[0];
    red[warp][g][t * 2u + 1u] = s0[1];
    red[warp][g + 8u][t * 2u] = s0[2];
    red[warp][g + 8u][t * 2u + 1u] = s0[3];
    red[warp][g][8u + t * 2u] = s1[0];
    red[warp][g][8u + t * 2u + 1u] = s1[1];
    red[warp][g + 8u][8u + t * 2u] = s1[2];
    red[warp][g + 8u][8u + t * 2u + 1u] = s1[3];
}

static __device__ __forceinline__ void attn_softmax_row(const float red[4][16][16], u32 valid_mask,
                                                        u32 tid, float scale, float m[16],
                                                        float l[16], float corr[16],
                                                        u32 p16[16][8]) {
    float sc[16];
    float tmax = attn_neg_inf();
#pragma unroll
    for (u32 c = 0u; c < 16u; c++) {
        if ((valid_mask >> c) & 1u) {
            float v = (red[0][tid][c] + red[1][tid][c] + red[2][tid][c] + red[3][tid][c]) * scale;
            sc[c] = v;
            if (v > tmax) tmax = v;
        } else {
            sc[c] = attn_neg_inf();
        }
    }
    float old_m = m[tid];
    float old_l = l[tid];
    float next_m = (old_m > tmax) ? old_m : tmax;
    float cval = (old_l > 0.0f) ? __expf(old_m - next_m) : 0.0f;
    float psum = 0.0f;
#pragma unroll
    for (u32 c = 0u; c < 16u; c += 2u) {
        float p0v = ((valid_mask >> c) & 1u) ? __expf(sc[c] - next_m) : 0.0f;
        float p1v = ((valid_mask >> (c + 1u)) & 1u) ? __expf(sc[c + 1u] - next_m) : 0.0f;
        psum += p0v + p1v;
        p16[tid][c >> 1] = trellis::f32x2_to_f16x2(p0v, p1v);
    }
    m[tid] = next_m;
    l[tid] = old_l * cval + psum;
    corr[tid] = cval;
}

template <u32 NT, u32 V_STRIDE16>
static __device__ __forceinline__ void attn_pv_tile(float o[NT][4], const u32 p16[16][8],
                                                    const float corr[16], const u16* v16,
                                                    u32 d_base, u32 g, u32 t) {
    u32 a_p[4] = {p16[g][t], p16[g + 8u][t], p16[g][t + 4u], p16[g + 8u][t + 4u]};
    float c0 = corr[g];
    float c1 = corr[g + 8u];
    u32 r0 = t << 1;
    u32 r1 = r0 + 8u;
#pragma unroll
    for (u32 nt = 0u; nt < NT; nt++) {
        o[nt][0] *= c0;
        o[nt][1] *= c0;
        o[nt][2] *= c1;
        o[nt][3] *= c1;
        u32 col = d_base + nt * 8u + g;
        u32 b_v[2];
        b_v[0] = u32(v16[r0 * V_STRIDE16 + col]) | (u32(v16[(r0 + 1u) * V_STRIDE16 + col]) << 16);
        b_v[1] = u32(v16[r1 * V_STRIDE16 + col]) | (u32(v16[(r1 + 1u) * V_STRIDE16 + col]) << 16);
        trellis::mma_m16n8k16(o[nt], a_p, b_v);
    }
}

template <u32 NT, u32 WIDTH, u32 STRIDE>
static __device__ __forceinline__ void attn_store_partials(
    float* part, const float o[NT][4], const float m[16], const float l[16], u32 row, u32 heads,
    u32 h0, u32 nh, u32 split, u32 splits, u32 d_base, u32 g, u32 t, u32 tid) {
    if (g < nh) {
        float* p_g = part + (u64(row * heads + h0 + g) * splits + split) * STRIDE;
#pragma unroll
        for (u32 nt = 0u; nt < NT; nt++) {
            *reinterpret_cast<float2*>(p_g + d_base + nt * 8u + (t << 1)) =
                make_float2(o[nt][0], o[nt][1]);
        }
    }
    if (g + 8u < nh) {
        float* p_g8 = part + (u64(row * heads + h0 + g + 8u) * splits + split) * STRIDE;
#pragma unroll
        for (u32 nt = 0u; nt < NT; nt++) {
            *reinterpret_cast<float2*>(p_g8 + d_base + nt * 8u + (t << 1)) =
                make_float2(o[nt][2], o[nt][3]);
        }
    }
    if (tid < nh) {
        u64 out_base = (u64(row * heads + h0 + tid) * splits + split) * STRIDE;
        part[out_base + WIDTH] = m[tid];
        part[out_base + WIDTH + 1u] = l[tid];
    }
}

struct alignas(16) mla_smem {
    u32 q[16][292];
    u32 kv[16][292];
    float red[4][16][16];
    u32 p16[16][8];
    float m[16];
    float l[16];
    float corr[16];
    u32 j[16];
    u32 stop;
    u32 valid_mask;
};

static __device__ __forceinline__ void mla_load_ids(const u32* ids, u32 row, u32 picks,
                                                    u32 from_list, u32 from, u32 to, u32 p0,
                                                    u32 p_end, u32 tid, mla_smem& sm) {
    if (tid < 16u) {
        u32 p = p0 + tid;
        u32 j = 0xFFFFFFFFu;
        if (p < p_end) {
            u32 cand = (from_list == 1u) ? ids[u64(row) * picks + p] : p;
            if (cand >= from && cand < to) j = cand;
        }
        sm.j[tid] = j;
        u32 mask = __ballot_sync(0xFFFFu, j != 0xFFFFFFFFu);
        if (tid == 0u) sm.valid_mask = mask;
    }
}

template <u32 KV_FMT>
static __device__ __forceinline__ void mla_attend_body(const float* q, const void* kv_ptr,
                                                       const u32* ids, const u32* slots,
                                                       const u32* positions, float* part,
                                                       float scale, u32 rows, u32 keys,
                                                       u32 context, u32 heads, u32 picks,
                                                       u32 from_list, u32 before, u32 after,
                                                       u32 splits, mla_smem& sm) {
    u32 tid = threadIdx.x;
    u32 warp = tid >> 5;
    u32 lane = tid & 31u;
    u32 g = lane >> 2;
    u32 t = lane & 3u;
    u32 hgroups = heads >> 4;
    u32 split = blockIdx.x;
    u32 row = blockIdx.y / hgroups;
    u32 hg = blockIdx.y % hgroups;
    if (row >= rows || split >= splits) return;
    u32 h0 = hg << 4;

    u32 slot = slots[row];
    u32 pos = positions[row];
    u32 from = (pos > before) ? (pos - before) : 0u;
    u32 to = (pos + after < keys) ? (pos + after + 1u) : keys;
    u32 start = (from_list == 0u) ? from : 0u;

    if (tid == 0u) {
        u32 stop = (from_list == 0u && to < picks) ? to : picks;
        if (from_list == 1u && picks > 0u) {
            u64 base = u64(row) * picks;
            if (ids[base + picks - 1u] >= keys) {
                u32 lo = 0u;
                u32 hi = picks;
                for (u32 step = 0u; step < 16u; step++) {
                    u32 mid = (lo + hi) >> 1;
                    if (mid < hi) {
                        if (ids[base + mid] < keys) lo = mid + 1u;
                        else hi = mid;
                    }
                }
                stop = lo;
            }
        }
        sm.stop = stop;
    }
    if (tid < 16u) {
        sm.m[tid] = attn_neg_inf();
        sm.l[tid] = 0.0f;
    }
    prim::block_sync();

    u32 stop = sm.stop;
    u32 total_p = (stop > start) ? (stop - start) : 0u;
    u32 tiles = (total_p + 15u) >> 4;
    u32 tiles_per = (tiles + splits - 1u) / splits;
    u32 per = tiles_per << 4;
    u32 p_begin = start + split * per;
    u32 p_end = p_begin + per;
    if (p_begin > stop) p_begin = stop;
    if (p_end > stop) p_end = stop;

    if (p_begin >= p_end) {
        if (tid < 16u) {
            u64 out_base = (u64(row * heads + h0 + tid) * splits + split) * 580ull;
            part[out_base + 576ull] = attn_neg_inf();
            part[out_base + 577ull] = 0.0f;
        }
        return;
    }

    // Stage Q[16, 576] once as f16x2 (no div/mod).
    u32 tk = tid >> 3;
    u32 lane8 = tid & 7u;
    const float* q_row = q + u64(row * heads + h0 + tk) * 576ull;
#pragma unroll
    for (u32 q4 = lane8; q4 < 144u; q4 += 8u) {
        float4 qv = *reinterpret_cast<const float4*>(q_row + u64(q4) * 4ull);
        u32 f0 = trellis::f32x2_to_f16x2(qv.x, qv.y);
        u32 f1 = trellis::f32x2_to_f16x2(qv.z, qv.w);
        *reinterpret_cast<uint2*>(&sm.q[tk][q4 * 2u]) = make_uint2(f0, f1);
    }
    mla_load_ids(ids, row, picks, from_list, from, to, p_begin, p_end, tid, sm);
    prim::block_sync();

    float o[18][4];
#pragma unroll
    for (u32 nt = 0u; nt < 18u; nt++) {
        o[nt][0] = 0.0f;
        o[nt][1] = 0.0f;
        o[nt][2] = 0.0f;
        o[nt][3] = 0.0f;
    }

    const u8* kv_u8 = static_cast<const u8*>(kv_ptr);
    const u16* kv_u16 = static_cast<const u16*>(kv_ptr);
    const float* kv_f32 = static_cast<const float*>(kv_ptr);
    const u16* kv16 = reinterpret_cast<const u16*>(sm.kv);
    u32 d_base = warp * 144u;

    for (u32 p0 = p_begin; p0 < p_end; p0 += 16u) {
        u32 vmask = sm.valid_mask;
        if (vmask != 0u) {
            u32 j = sm.j[tk];
            if constexpr (KV_FMT == 0u) {
                if (j != 0xFFFFFFFFu) {
                    const u8* row_u8 = kv_u8 + (u64(slot) * context + j) * 576ull;
                    for (u32 q16 = lane8; q16 < 36u; q16 += 8u) {
                        uint4 raw = *reinterpret_cast<const uint4*>(row_u8 + u64(q16) * 16ull);
                        u32 h0w = attn_e4m3_f16x2(static_cast<u16>(raw.x));
                        u32 h1w = attn_e4m3_f16x2(static_cast<u16>(raw.x >> 16));
                        u32 h2w = attn_e4m3_f16x2(static_cast<u16>(raw.y));
                        u32 h3w = attn_e4m3_f16x2(static_cast<u16>(raw.y >> 16));
                        u32 h4w = attn_e4m3_f16x2(static_cast<u16>(raw.z));
                        u32 h5w = attn_e4m3_f16x2(static_cast<u16>(raw.z >> 16));
                        u32 h6w = attn_e4m3_f16x2(static_cast<u16>(raw.w));
                        u32 h7w = attn_e4m3_f16x2(static_cast<u16>(raw.w >> 16));
                        *reinterpret_cast<uint4*>(&sm.kv[tk][q16 * 8u]) =
                            make_uint4(h0w, h1w, h2w, h3w);
                        *reinterpret_cast<uint4*>(&sm.kv[tk][q16 * 8u + 4u]) =
                            make_uint4(h4w, h5w, h6w, h7w);
                    }
                } else {
                    for (u32 q16 = lane8; q16 < 36u; q16 += 8u) {
                        *reinterpret_cast<uint4*>(&sm.kv[tk][q16 * 8u]) =
                            make_uint4(0u, 0u, 0u, 0u);
                        *reinterpret_cast<uint4*>(&sm.kv[tk][q16 * 8u + 4u]) =
                            make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            } else {
                u64 row_base = (u64(slot) * context + j) * 576ull;
                for (u32 q4 = lane8; q4 < 144u; q4 += 8u) {
                    u32 h0w = 0u;
                    u32 h1w = 0u;
                    if (j != 0xFFFFFFFFu) {
                        u64 base = row_base + u64(q4) * 4ull;
                        if constexpr (KV_FMT == 1u) {
                            uint2 raw = *reinterpret_cast<const uint2*>(kv_u16 + base);
                            h0w = trellis::f32x2_to_f16x2(__uint_as_float(raw.x << 16),
                                                          __uint_as_float(raw.x & 0xFFFF0000u));
                            h1w = trellis::f32x2_to_f16x2(__uint_as_float(raw.y << 16),
                                                          __uint_as_float(raw.y & 0xFFFF0000u));
                        } else {
                            float4 raw = *reinterpret_cast<const float4*>(kv_f32 + base);
                            h0w = trellis::f32x2_to_f16x2(raw.x, raw.y);
                            h1w = trellis::f32x2_to_f16x2(raw.z, raw.w);
                        }
                    }
                    *reinterpret_cast<uint2*>(&sm.kv[tk][q4 * 2u]) = make_uint2(h0w, h1w);
                }
            }
            prim::block_sync();

            attn_qk_tile<9u, 292u>(sm.q, sm.kv, sm.red, warp, g, t);
            prim::block_sync();

            if (tid < 16u) {
                attn_softmax_row(sm.red, vmask, tid, scale, sm.m, sm.l, sm.corr, sm.p16);
            }
            prim::block_sync();

            attn_pv_tile<18u, 584u>(o, sm.p16, sm.corr, kv16, d_base, g, t);
        }
        if (p0 + 16u < p_end) {
            mla_load_ids(ids, row, picks, from_list, from, to, p0 + 16u, p_end, tid, sm);
            prim::block_sync();
        }
    }

    attn_store_partials<18u, 576u, 580u>(part, o, sm.m, sm.l, row, heads, h0, 16u, split, splits,
                                         d_base, g, t, tid);
}

// Grouped-Q split-K MLA attention over FP8 KV.
extern "C" __global__ __launch_bounds__(128, 2) void tessera_mla_attend_fp8(
    const float* q, const u8* kv, const u32* ids, const u32* slots, const u32* positions,
    float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads, u32 picks, u32 from_list,
    u32 before, u32 after, u32 splits) {
    __shared__ mla_smem sm;
    mla_attend_body<0u>(q, kv, ids, slots, positions, part, scale, rows, keys, context, heads,
                        picks, from_list, before, after, splits, sm);
}

// Grouped-Q split-K MLA attention over BF16 KV.
extern "C" __global__ __launch_bounds__(128, 2) void tessera_mla_attend_bf16(
    const float* q, const u16* kv, const u32* ids, const u32* slots, const u32* positions,
    float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads, u32 picks, u32 from_list,
    u32 before, u32 after, u32 splits) {
    __shared__ mla_smem sm;
    mla_attend_body<1u>(q, kv, ids, slots, positions, part, scale, rows, keys, context, heads,
                        picks, from_list, before, after, splits, sm);
}

// Grouped-Q split-K MLA attention over F32 KV.
extern "C" __global__ __launch_bounds__(128, 2) void tessera_mla_attend_f32(
    const float* q, const float* kv, const u32* ids, const u32* slots, const u32* positions,
    float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads, u32 picks, u32 from_list,
    u32 before, u32 after, u32 splits) {
    __shared__ mla_smem sm;
    mla_attend_body<2u>(q, kv, ids, slots, positions, part, scale, rows, keys, context, heads,
                        picks, from_list, before, after, splits, sm);
}

// Combines split-K MLA partials for one row and head.
extern "C" __global__ __launch_bounds__(160, 4) void tessera_mla_reduce(
    const float* part, float* out, u32 lists, u32 splits) {
    u32 list = blockIdx.x;
    if (list >= lists) return;
    u32 tid = threadIdx.x;
    __shared__ float s_m[32];
    __shared__ float s_l[32];
    __shared__ float s_w[32];
    __shared__ float s_inv;
    const float* base = part + u64(list) * splits * 580ull;
    if (tid < splits) {
        s_m[tid] = base[u64(tid) * 580ull + 576ull];
        s_l[tid] = base[u64(tid) * 580ull + 577ull];
    }
    prim::block_sync();
    if (tid == 0u) {
        float best = attn_neg_inf();
        for (u32 s = 0u; s < splits; s++) {
            if (s_l[s] > 0.0f && s_m[s] > best) best = s_m[s];
        }
        float denom = 0.0f;
        for (u32 s = 0u; s < splits; s++) {
            float w = (s_l[s] > 0.0f) ? __expf(s_m[s] - best) : 0.0f;
            s_w[s] = w;
            denom += s_l[s] * w;
        }
        s_inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    }
    prim::block_sync();
    if (tid < 144u) {
        float inv = s_inv;
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        u32 d4 = tid << 2;
        for (u32 s = 0u; s < splits; s++) {
            float w = s_w[s];
            if (w > 0.0f) attn_fma4(sum, base + u64(s) * 580ull + d4, w);
        }
        *reinterpret_cast<float4*>(out + u64(list) * 576ull + d4) =
            make_float4(sum.x * inv, sum.y * inv, sum.z * inv, sum.w * inv);
    }
}

struct alignas(16) gqa_smem {
    u32 q[16][68];
    u32 k[16][68];
    u32 v[16][68];
    float red[4][16][16];
    u32 p16[16][8];
    float m[16];
    float l[16];
    float corr[16];
};

template <u32 KV_FMT>
static __device__ __forceinline__ void gqa_stage_vec(const void* ptr, u64 row_base, u32 lane8,
                                                     bool valid, u32* dst_row) {
    if constexpr (KV_FMT == 0u) {
        if (valid) {
            uint4 raw =
                *reinterpret_cast<const uint4*>(static_cast<const u8*>(ptr) + row_base + lane8 * 16u);
            u32 h0 = attn_e4m3_f16x2(static_cast<u16>(raw.x));
            u32 h1 = attn_e4m3_f16x2(static_cast<u16>(raw.x >> 16));
            u32 h2 = attn_e4m3_f16x2(static_cast<u16>(raw.y));
            u32 h3 = attn_e4m3_f16x2(static_cast<u16>(raw.y >> 16));
            u32 h4 = attn_e4m3_f16x2(static_cast<u16>(raw.z));
            u32 h5 = attn_e4m3_f16x2(static_cast<u16>(raw.z >> 16));
            u32 h6 = attn_e4m3_f16x2(static_cast<u16>(raw.w));
            u32 h7 = attn_e4m3_f16x2(static_cast<u16>(raw.w >> 16));
            *reinterpret_cast<uint4*>(dst_row + lane8 * 8u) = make_uint4(h0, h1, h2, h3);
            *reinterpret_cast<uint4*>(dst_row + lane8 * 8u + 4u) = make_uint4(h4, h5, h6, h7);
        } else {
            *reinterpret_cast<uint4*>(dst_row + lane8 * 8u) = make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(dst_row + lane8 * 8u + 4u) = make_uint4(0u, 0u, 0u, 0u);
        }
    } else if constexpr (KV_FMT == 1u) {
        const u16* p16 = static_cast<const u16*>(ptr) + row_base;
#pragma unroll
        for (u32 q4 = lane8; q4 < 32u; q4 += 8u) {
            u32 h0 = 0u;
            u32 h1 = 0u;
            if (valid) {
                uint2 raw = *reinterpret_cast<const uint2*>(p16 + u64(q4) * 4ull);
                h0 = trellis::f32x2_to_f16x2(__uint_as_float(raw.x << 16),
                                             __uint_as_float(raw.x & 0xFFFF0000u));
                h1 = trellis::f32x2_to_f16x2(__uint_as_float(raw.y << 16),
                                             __uint_as_float(raw.y & 0xFFFF0000u));
            }
            *reinterpret_cast<uint2*>(dst_row + q4 * 2u) = make_uint2(h0, h1);
        }
    } else {
        const float* pf = static_cast<const float*>(ptr) + row_base;
#pragma unroll
        for (u32 q4 = lane8; q4 < 32u; q4 += 8u) {
            u32 h0 = 0u;
            u32 h1 = 0u;
            if (valid) {
                float4 raw = *reinterpret_cast<const float4*>(pf + u64(q4) * 4ull);
                h0 = trellis::f32x2_to_f16x2(raw.x, raw.y);
                h1 = trellis::f32x2_to_f16x2(raw.z, raw.w);
            }
            *reinterpret_cast<uint2*>(dst_row + q4 * 2u) = make_uint2(h0, h1);
        }
    }
}

template <u32 KV_FMT>
static __device__ __forceinline__ void gqa_attend_body(const float* q, const void* k_ptr,
                                                       const void* v_ptr, const u32* slots,
                                                       const u32* positions, float* part,
                                                       float scale, u32 rows, u32 keys,
                                                       u32 context, u32 heads, u32 kv_heads,
                                                       u32 before, u32 after, u32 splits,
                                                       gqa_smem& sm) {
    u32 tid = threadIdx.x;
    u32 warp = tid >> 5;
    u32 lane = tid & 31u;
    u32 g = lane >> 2;
    u32 t = lane & 3u;
    u32 gqa = heads / kv_heads;
    u32 nh = (gqa < 16u) ? gqa : 16u;
    u32 per_kv = (gqa + 15u) >> 4;
    u32 kg_groups = kv_heads * per_kv;
    u32 split = blockIdx.x;
    u32 row = blockIdx.y / kg_groups;
    u32 kg = blockIdx.y % kg_groups;
    if (row >= rows || split >= splits) return;
    u32 kv_head = kg / per_kv;
    u32 h0 = kv_head * gqa + ((kg % per_kv) << 4);

    u32 slot = slots[row];
    u32 pos = positions[row];
    u32 from = (pos > before) ? (pos - before) : 0u;
    u32 to = (pos + after < keys) ? (pos + after + 1u) : keys;
    u32 total_p = (to > from) ? (to - from) : 0u;
    u32 tiles = (total_p + 15u) >> 4;
    u32 per = ((tiles + splits - 1u) / splits) << 4;
    u32 p_begin = from + split * per;
    u32 p_end = p_begin + per;
    if (p_begin > to) p_begin = to;
    if (p_end > to) p_end = to;

    if (p_begin >= p_end) {
        if (tid < nh) {
            u64 out_base = (u64(row * heads + h0 + tid) * splits + split) * 132ull;
            part[out_base + 128ull] = attn_neg_inf();
            part[out_base + 129ull] = 0.0f;
        }
        return;
    }

    u32 tk = tid >> 3;
    u32 lane8 = tid & 7u;
    if (tk < nh) {
        const float* q_row = q + u64(row * heads + h0 + tk) * 128ull;
#pragma unroll
        for (u32 q4 = lane8; q4 < 32u; q4 += 8u) {
            float4 qv = *reinterpret_cast<const float4*>(q_row + u64(q4) * 4ull);
            *reinterpret_cast<uint2*>(&sm.q[tk][q4 * 2u]) =
                make_uint2(trellis::f32x2_to_f16x2(qv.x, qv.y),
                           trellis::f32x2_to_f16x2(qv.z, qv.w));
        }
    } else {
#pragma unroll
        for (u32 q4 = lane8; q4 < 32u; q4 += 8u) {
            *reinterpret_cast<uint2*>(&sm.q[tk][q4 * 2u]) = make_uint2(0u, 0u);
        }
    }
    if (tid < 16u) {
        sm.m[tid] = attn_neg_inf();
        sm.l[tid] = 0.0f;
        sm.corr[tid] = 0.0f;
#pragma unroll
        for (u32 c = 0u; c < 8u; c++) sm.p16[tid][c] = 0u;
    }
    prim::block_sync();

    float o[4][4];
#pragma unroll
    for (u32 nt = 0u; nt < 4u; nt++) {
        o[nt][0] = 0.0f;
        o[nt][1] = 0.0f;
        o[nt][2] = 0.0f;
        o[nt][3] = 0.0f;
    }

    const u16* v16 = reinterpret_cast<const u16*>(sm.v);
    u32 d_base = warp << 5;

    for (u32 j0 = p_begin; j0 < p_end; j0 += 16u) {
        u32 j = j0 + tk;
        bool valid = j < p_end;
        u64 kv_base = ((u64(slot) * context + j) * kv_heads + kv_head) * 128ull;
        gqa_stage_vec<KV_FMT>(k_ptr, kv_base, lane8, valid, sm.k[tk]);
        gqa_stage_vec<KV_FMT>(v_ptr, kv_base, lane8, valid, sm.v[tk]);
        prim::block_sync();

        attn_qk_tile<2u, 68u>(sm.q, sm.k, sm.red, warp, g, t);
        prim::block_sync();

        u32 nvalid = (p_end - j0 < 16u) ? (p_end - j0) : 16u;
        u32 vmask = (1u << nvalid) - 1u;
        if (tid < nh) {
            attn_softmax_row(sm.red, vmask, tid, scale, sm.m, sm.l, sm.corr, sm.p16);
        }
        prim::block_sync();

        attn_pv_tile<4u, 136u>(o, sm.p16, sm.corr, v16, d_base, g, t);
        if (j0 + 16u < p_end) prim::block_sync();
    }

    attn_store_partials<4u, 128u, 132u>(part, o, sm.m, sm.l, row, heads, h0, nh, split, splits,
                                        d_base, g, t, tid);
}

#define GQA_DEF(NAME, T, FMT)                                                                 \
    extern "C" __global__ __launch_bounds__(128, 4) void NAME(                                \
        const float* q, const T* k, const T* v, const u32* slots, const u32* positions,       \
        float* part, float scale, u32 rows, u32 keys, u32 context, u32 heads, u32 kv_heads,   \
        u32 before, u32 after, u32 splits) {                                                  \
        __shared__ gqa_smem sm;                                                               \
        gqa_attend_body<FMT>(q, k, v, slots, positions, part, scale, rows, keys, context,     \
                             heads, kv_heads, before, after, splits, sm);                     \
    }

GQA_DEF(tessera_gqa_attend_fp8, u8, 0u)
GQA_DEF(tessera_gqa_attend_bf16, u16, 1u)
GQA_DEF(tessera_gqa_attend_f32, float, 2u)

// Warp-level reduction of split-K GQA partials for head_dim 128.
extern "C" __global__ __launch_bounds__(128, 4) void tessera_gqa_reduce(
    const float* part, float* out, u32 lists, u32 splits) {
    u32 list = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (list >= lists) return;
    u32 lane = threadIdx.x & 31u;
    const float* base = part + u64(list) * splits * 132ull;
    float my_m = (lane < splits) ? base[u64(lane) * 132ull + 128ull] : attn_neg_inf();
    float my_l = (lane < splits) ? base[u64(lane) * 132ull + 129ull] : 0.0f;
    float best = (my_l > 0.0f) ? my_m : attn_neg_inf();
#pragma unroll
    for (u32 off = 16u; off > 0u; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, best, off);
        if (other > best) best = other;
    }
    float my_w = (lane < splits && my_l > 0.0f) ? __expf(my_m - best) : 0.0f;
    float denom = my_l * my_w;
#pragma unroll
    for (u32 off = 16u; off > 0u; off >>= 1) {
        denom += __shfl_xor_sync(0xFFFFFFFFu, denom, off);
    }
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    u32 d4 = lane << 2;
    for (u32 s = 0u; s < splits; s++) {
        float w = __shfl_sync(0xFFFFFFFFu, my_w, s);
        if (w > 0.0f) attn_fma4(sum, base + u64(s) * 132ull + d4, w);
    }
    *reinterpret_cast<float4*>(out + u64(list) * 128ull + d4) =
        make_float4(sum.x * inv, sum.y * inv, sum.z * inv, sum.w * inv);
}

constexpr u32 IDX_SM_WORDS = 4608u;

struct alignas(16) idx_smem {
    u32 q[IDX_SM_WORDS];
    u32 k[IDX_SM_WORDS];
    float w[128];
};

static __device__ __forceinline__ u32 idx_f32x2_bf16x2(float lo, float hi) {
    u32 out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
    return out;
}

static __device__ __forceinline__ u32 idx_e4m3x2_bf16x2(u16 q) {
    u32 h = attn_e4m3_f16x2(q);
    return idx_f32x2_bf16x2(trellis::f16_to_f32(h & 0xFFFFu), trellis::f16_to_f32(h >> 16u));
}

static __device__ __forceinline__ void idx_mma_bf16(float d[4], const u32 a[4], const u32 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <u32 KV_FMT>
static __device__ __forceinline__ void idx_load_k16(const void* k_ptr, u32* sk, u32 slot,
                                                    u32 context, u32 dim, u32 stride, u32 j0,
                                                    u32 to, u32 lane) {
    u32 q_words = dim >> 2;
    u32 total_k4 = q_words << 4;
    const u8* k_u8 = static_cast<const u8*>(k_ptr);
    const u16* k_u16 = static_cast<const u16*>(k_ptr);
    const float* k_f32 = static_cast<const float*>(k_ptr);
    for (u32 idx = lane; idx < total_k4; idx += 32u) {
        u32 kr = idx / q_words;
        u32 q4 = idx % q_words;
        u32 j = j0 + kr;
        uint2 pack = make_uint2(0u, 0u);
        if (j < to) {
            u64 base = (u64(slot) * context + j) * dim + (u64(q4) << 2);
            if constexpr (KV_FMT == 0u) {
                u32 raw = *reinterpret_cast<const u32*>(k_u8 + base);
                pack = make_uint2(idx_e4m3x2_bf16x2(static_cast<u16>(raw)),
                                  idx_e4m3x2_bf16x2(static_cast<u16>(raw >> 16)));
            } else if constexpr (KV_FMT == 1u) {
                pack = *reinterpret_cast<const uint2*>(k_u16 + base);
            } else {
                float4 v = *reinterpret_cast<const float4*>(k_f32 + base);
                pack = make_uint2(idx_f32x2_bf16x2(v.x, v.y), idx_f32x2_bf16x2(v.z, v.w));
            }
        }
        *reinterpret_cast<uint2*>(sk + kr * stride + (q4 << 1)) = pack;
    }
}

template <u32 KV_FMT>
static __device__ __forceinline__ void idx_body(const float* q, const void* k_ptr,
                                                const float* weights, const u32* slots,
                                                const u32* positions, float* out, u32 rows,
                                                u32 out_keys, u32 live_keys, u32 context,
                                                u32 heads, u32 dim, u32 masked, u32 tile_keys,
                                                idx_smem& sm) {
    u32 row = blockIdx.y;
    if (row >= rows) return;
    u32 j_begin = blockIdx.x * tile_keys;
    u32 j_end = j_begin + tile_keys;
    if (j_end > out_keys) j_end = out_keys;
    if (j_begin >= j_end) return;

    u32 tid = threadIdx.x;
    u32 slot = slots[row];
    u32 to = live_keys;
    if (masked == 1u) {
        u32 pos = positions[row];
        if (pos < to) to = pos + 1u;
    }
    float* out_row = out + u64(row) * out_keys;
    float neg = attn_neg_inf();

    // Early exit for fully masked causal tiles.
    if (j_begin >= to) {
        for (u32 j = j_begin + tid; j < j_end; j += 128u) {
            out_row[j] = neg;
        }
        return;
    }

    if (tid < heads) {
        sm.w[tid] = weights[u64(row) * heads + tid];
    }
    u32 stride = (dim >> 1) + 2u;
    u32 q_words = dim >> 2;
    u32 total_q4 = heads * q_words;
    const float* q_base = q + u64(row) * heads * dim;
    for (u32 idx = tid; idx < total_q4; idx += 128u) {
        u32 h = idx / q_words;
        u32 q4 = idx % q_words;
        float4 v = *reinterpret_cast<const float4*>(q_base + u64(h) * dim + (u64(q4) << 2));
        uint2 pack = make_uint2(idx_f32x2_bf16x2(v.x, v.y), idx_f32x2_bf16x2(v.z, v.w));
        *reinterpret_cast<uint2*>(sm.q + h * stride + (q4 << 1)) = pack;
    }
    prim::block_sync();

    u32 warp = tid >> 5;
    u32 lane = tid & 31u;
    u32 g = lane >> 2;
    u32 t = lane & 3u;
    u32* sk = sm.k + warp * 16u * stride;
    u32 d_steps = dim >> 4;

    for (u32 j0 = j_begin + (warp << 4); j0 < j_end; j0 += 64u) {
        if (j0 >= to) {
            if (lane < 16u && j0 + lane < j_end) out_row[j0 + lane] = neg;
            continue;
        }
        idx_load_k16<KV_FMT>(k_ptr, sk, slot, context, dim, stride, j0, to, lane);
        __syncwarp();

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (u32 h0 = 0u; h0 < heads; h0 += 16u) {
            float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (u32 ds = 0u; ds < d_steps; ds++) {
                u32 k0 = ds << 3;
                u32 a[4] = {
                    sm.q[(h0 + g) * stride + k0 + t],
                    sm.q[(h0 + g + 8u) * stride + k0 + t],
                    sm.q[(h0 + g) * stride + k0 + t + 4u],
                    sm.q[(h0 + g + 8u) * stride + k0 + t + 4u],
                };
                u32 b0[2] = {sk[g * stride + k0 + t], sk[g * stride + k0 + t + 4u]};
                u32 b1[2] = {sk[(g + 8u) * stride + k0 + t], sk[(g + 8u) * stride + k0 + t + 4u]};
                idx_mma_bf16(s0, a, b0);
                idx_mma_bf16(s1, a, b1);
            }
            float w0 = sm.w[h0 + g];
            float w1 = sm.w[h0 + g + 8u];
            acc[0] += fmaxf(s0[0], 0.0f) * w0 + fmaxf(s0[2], 0.0f) * w1;
            acc[1] += fmaxf(s0[1], 0.0f) * w0 + fmaxf(s0[3], 0.0f) * w1;
            acc[2] += fmaxf(s1[0], 0.0f) * w0 + fmaxf(s1[2], 0.0f) * w1;
            acc[3] += fmaxf(s1[1], 0.0f) * w0 + fmaxf(s1[3], 0.0f) * w1;
        }
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
            acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 4u);
            acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 8u);
            acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 16u);
        }
        if (g == 0u) {
            u32 k0 = j0 + (t << 1);
            u32 k2 = j0 + 8u + (t << 1);
            if (k0 < j_end) out_row[k0] = (k0 < to) ? acc[0] : neg;
            if (k0 + 1u < j_end) out_row[k0 + 1u] = (k0 + 1u < to) ? acc[1] : neg;
            if (k2 < j_end) out_row[k2] = (k2 < to) ? acc[2] : neg;
            if (k2 + 1u < j_end) out_row[k2 + 1u] = (k2 + 1u < to) ? acc[3] : neg;
        }
    }
}

extern "C" __global__ __launch_bounds__(128) void tessera_indexer_fp8(
    const float* q, const void* k, const float* weights, const u32* slots, const u32* positions,
    float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 dim, u32 masked,
    u32 tile_keys) {
    __shared__ idx_smem sm;
    idx_body<0u>(q, k, weights, slots, positions, out, rows, out_keys, live_keys, context, heads,
                 dim, masked, tile_keys, sm);
}

extern "C" __global__ __launch_bounds__(128) void tessera_indexer_bf16(
    const float* q, const void* k, const float* weights, const u32* slots, const u32* positions,
    float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 dim, u32 masked,
    u32 tile_keys) {
    __shared__ idx_smem sm;
    idx_body<1u>(q, k, weights, slots, positions, out, rows, out_keys, live_keys, context, heads,
                 dim, masked, tile_keys, sm);
}

extern "C" __global__ __launch_bounds__(128) void tessera_indexer_f32(
    const float* q, const void* k, const float* weights, const u32* slots, const u32* positions,
    float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 dim, u32 masked,
    u32 tile_keys) {
    __shared__ idx_smem sm;
    idx_body<2u>(q, k, weights, slots, positions, out, rows, out_keys, live_keys, context, heads,
                 dim, masked, tile_keys, sm);
}

#endif
