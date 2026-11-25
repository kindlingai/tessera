#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_CUDA)

static __device__ __forceinline__ float mla_neg_inf() {
    return __int_as_float(0xff800000u);
}

static __device__ __forceinline__ u32 mla_e4m3_f16x2(u16 q) {
    u32 h;
    asm("{.reg .b16 t; mov.b16 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(h) : "h"(q));
    return h;
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
    u32 has_valid;
};

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
        sm.m[tid] = mla_neg_inf();
        sm.l[tid] = 0.0f;
    }
    prim::block_sync();

    u32 stop = sm.stop;
    u32 total_p = (stop > start) ? (stop - start) : 0u;
    u32 per = (total_p + splits - 1u) / splits;
    u32 p_begin = start + split * per;
    u32 p_end = p_begin + per;
    if (p_begin > stop) p_begin = stop;
    if (p_end > stop) p_end = stop;

    if (p_begin >= p_end) {
        if (tid < 16u) {
            u64 out_base = (u64(row * heads + h0 + tid) * splits + split) * 580ull;
            part[out_base + 576ull] = mla_neg_inf();
            part[out_base + 577ull] = 0.0f;
        }
        return;
    }

    // Stage Q[16, 576] once as f16x2.
    for (u32 qi = tid; qi < 16u * 144u; qi += 128u) {
        u32 h = qi / 144u;
        u32 q4 = qi % 144u;
        u64 q_at = u64(row * heads + h0 + h) * 576ull + u64(q4) * 4ull;
        float4 qv = *reinterpret_cast<const float4*>(q + q_at);
        u32 f0 = trellis::f32x2_to_f16x2(qv.x, qv.y);
        u32 f1 = trellis::f32x2_to_f16x2(qv.z, qv.w);
        *reinterpret_cast<uint2*>(&sm.q[h][q4 * 2u]) = make_uint2(f0, f1);
    }
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
    u32 r0 = t << 1;
    u32 r1 = r0 + 8u;

    for (u32 p0 = p_begin; p0 < p_end; p0 += 16u) {
        if (tid == 0u) sm.has_valid = 0u;
        prim::block_sync();
        if (tid < 16u) {
            u32 p = p0 + tid;
            u32 j = 0xFFFFFFFFu;
            if (p < p_end) {
                u32 cand = (from_list == 1u) ? ids[u64(row) * picks + p] : p;
                if (cand >= from && cand < to) {
                    j = cand;
                    sm.has_valid = 1u;
                }
            }
            sm.j[tid] = j;
        }
        prim::block_sync();
        if (sm.has_valid == 0u) continue;

        for (u32 qi = tid; qi < 16u * 144u; qi += 128u) {
            u32 tk = qi / 144u;
            u32 q4 = qi % 144u;
            u32 j = sm.j[tk];
            u32 h0w = 0u;
            u32 h1w = 0u;
            if (j != 0xFFFFFFFFu) {
                u64 base = (u64(slot) * context + j) * 576ull + u64(q4) * 4ull;
                if constexpr (KV_FMT == 0u) {
                    u32 raw = *reinterpret_cast<const u32*>(kv_u8 + base);
                    h0w = mla_e4m3_f16x2(static_cast<u16>(raw));
                    h1w = mla_e4m3_f16x2(static_cast<u16>(raw >> 16));
                } else if constexpr (KV_FMT == 1u) {
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
        prim::block_sync();

        float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (u32 step = 0u; step < 9u; step++) {
            u32 k0 = (warp * 9u + step) * 8u;
            u32 a[4];
            a[0] = sm.q[g][k0 + t];
            a[1] = sm.q[g + 8u][k0 + t];
            a[2] = sm.q[g][k0 + t + 4u];
            a[3] = sm.q[g + 8u][k0 + t + 4u];
            u32 b0[2] = {sm.kv[g][k0 + t], sm.kv[g][k0 + t + 4u]};
            u32 b1[2] = {sm.kv[g + 8u][k0 + t], sm.kv[g + 8u][k0 + t + 4u]};
            trellis::mma_m16n8k16(s0, a, b0);
            trellis::mma_m16n8k16(s1, a, b1);
        }
        sm.red[warp][g][t * 2u] = s0[0];
        sm.red[warp][g][t * 2u + 1u] = s0[1];
        sm.red[warp][g + 8u][t * 2u] = s0[2];
        sm.red[warp][g + 8u][t * 2u + 1u] = s0[3];
        sm.red[warp][g][8u + t * 2u] = s1[0];
        sm.red[warp][g][8u + t * 2u + 1u] = s1[1];
        sm.red[warp][g + 8u][8u + t * 2u] = s1[2];
        sm.red[warp][g + 8u][8u + t * 2u + 1u] = s1[3];
        prim::block_sync();

        if (tid < 16u) {
            float sc[16];
            float tmax = mla_neg_inf();
#pragma unroll
            for (u32 c = 0u; c < 16u; c++) {
                if (sm.j[c] != 0xFFFFFFFFu) {
                    float v = (sm.red[0][tid][c] + sm.red[1][tid][c] + sm.red[2][tid][c] +
                               sm.red[3][tid][c]) *
                              scale;
                    sc[c] = v;
                    if (v > tmax) tmax = v;
                } else {
                    sc[c] = mla_neg_inf();
                }
            }
            float old_m = sm.m[tid];
            float old_l = sm.l[tid];
            float next_m = (old_m > tmax) ? old_m : tmax;
            float corr = (old_l > 0.0f) ? expf(old_m - next_m) : 0.0f;
            float psum = 0.0f;
#pragma unroll
            for (u32 c = 0u; c < 16u; c += 2u) {
                float p0v = (sm.j[c] != 0xFFFFFFFFu) ? expf(sc[c] - next_m) : 0.0f;
                float p1v = (sm.j[c + 1u] != 0xFFFFFFFFu) ? expf(sc[c + 1u] - next_m) : 0.0f;
                psum += p0v + p1v;
                sm.p16[tid][c >> 1] = trellis::f32x2_to_f16x2(p0v, p1v);
            }
            sm.m[tid] = next_m;
            sm.l[tid] = old_l * corr + psum;
            sm.corr[tid] = corr;
        }
        prim::block_sync();

        u32 a_p[4] = {sm.p16[g][t], sm.p16[g + 8u][t], sm.p16[g][t + 4u], sm.p16[g + 8u][t + 4u]};
        float c0 = sm.corr[g];
        float c1 = sm.corr[g + 8u];
#pragma unroll
        for (u32 nt = 0u; nt < 18u; nt++) {
            o[nt][0] *= c0;
            o[nt][1] *= c0;
            o[nt][2] *= c1;
            o[nt][3] *= c1;
            u32 col = d_base + nt * 8u + g;
            u32 b_v[2];
            b_v[0] = u32(kv16[r0 * 584u + col]) | (u32(kv16[(r0 + 1u) * 584u + col]) << 16);
            b_v[1] = u32(kv16[r1 * 584u + col]) | (u32(kv16[(r1 + 1u) * 584u + col]) << 16);
            trellis::mma_m16n8k16(o[nt], a_p, b_v);
        }
    }

    float* p_g = part + (u64(row * heads + h0 + g) * splits + split) * 580ull;
    float* p_g8 = part + (u64(row * heads + h0 + g + 8u) * splits + split) * 580ull;
#pragma unroll
    for (u32 nt = 0u; nt < 18u; nt++) {
        u32 col2 = d_base + nt * 8u + (t << 1);
        *reinterpret_cast<float2*>(p_g + col2) = make_float2(o[nt][0], o[nt][1]);
        *reinterpret_cast<float2*>(p_g8 + col2) = make_float2(o[nt][2], o[nt][3]);
    }
    if (tid < 16u) {
        u64 out_base = (u64(row * heads + h0 + tid) * splits + split) * 580ull;
        part[out_base + 576ull] = sm.m[tid];
        part[out_base + 577ull] = sm.l[tid];
    }
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
        float best = mla_neg_inf();
        for (u32 s = 0u; s < splits; s++) {
            if (s_l[s] > 0.0f && s_m[s] > best) best = s_m[s];
        }
        float denom = 0.0f;
        for (u32 s = 0u; s < splits; s++) {
            float w = (s_l[s] > 0.0f) ? expf(s_m[s] - best) : 0.0f;
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
            if (w > 0.0f) {
                float4 v = *reinterpret_cast<const float4*>(base + u64(s) * 580ull + d4);
                sum.x = fmaf(v.x, w, sum.x);
                sum.y = fmaf(v.y, w, sum.y);
                sum.z = fmaf(v.z, w, sum.z);
                sum.w = fmaf(v.w, w, sum.w);
            }
        }
        *reinterpret_cast<float4*>(out + u64(list) * 576ull + d4) =
            make_float4(sum.x * inv, sum.y * inv, sum.z * inv, sum.w * inv);
    }
}

#endif
