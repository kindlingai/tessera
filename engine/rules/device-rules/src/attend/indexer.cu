// Indexer shares mla.cu helpers in one CUDA unit.
#if defined(TESSERA_CUDA)

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

static __device__ __forceinline__ void idx_relu_accum(float acc[2], const float s[4], float w0, float w1) {
    acc[0] += fmaxf(s[0], 0.f) * w0 + fmaxf(s[2], 0.f) * w1;
    acc[1] += fmaxf(s[1], 0.f) * w0 + fmaxf(s[3], 0.f) * w1;
}
static __device__ __forceinline__ void idx_qt_reduce_store(
    float acc[8], float* out, u64 out_off, u32 j0, u32 j_end, u32 my_to, u32 g, u32 t) {
    bool hi1 = (g & 1u) != 0u;
    float p0 = hi1 ? acc[4] : acc[0], p1 = hi1 ? acc[5] : acc[1];
    float p2 = hi1 ? acc[6] : acc[2], p3 = hi1 ? acc[7] : acc[3];
    p0 += __shfl_xor_sync(0xFFFFFFFFu, hi1 ? acc[0] : acc[4], 4u);
    p1 += __shfl_xor_sync(0xFFFFFFFFu, hi1 ? acc[1] : acc[5], 4u);
    p2 += __shfl_xor_sync(0xFFFFFFFFu, hi1 ? acc[2] : acc[6], 4u);
    p3 += __shfl_xor_sync(0xFFFFFFFFu, hi1 ? acc[3] : acc[7], 4u);
    bool hi2 = (g & 2u) != 0u;
    float q0 = hi2 ? p2 : p0, q1 = hi2 ? p3 : p1;
    q0 += __shfl_xor_sync(0xFFFFFFFFu, hi2 ? p0 : p2, 8u);
    q1 += __shfl_xor_sync(0xFFFFFFFFu, hi2 ? p1 : p3, 8u);
    q0 += __shfl_xor_sync(0xFFFFFFFFu, q0, 16u);
    q1 += __shfl_xor_sync(0xFFFFFFFFu, q1, 16u);
    if (g < 4u) {
        u32 c = ((g & 1u) << 1u) | (g >> 1u);
        u32 k0 = j0 + (c << 3u) + (t << 1u);
        float neg = attn_neg_inf();
        if (k0 + 1u < j_end && ((out_off + k0) & 1ull) == 0ull) {
            *reinterpret_cast<float2*>(out + out_off + k0) =
                make_float2((k0 < my_to) ? q0 : neg, (k0 + 1u < my_to) ? q1 : neg);
        } else {
            if (k0 < j_end) out[out_off + k0] = (k0 < my_to) ? q0 : neg;
            if (k0 + 1u < j_end) out[out_off + k0 + 1u] = (k0 + 1u < my_to) ? q1 : neg;
        }
    }
}

struct idx_qt_ctx { u32 r0, row, nr, j_begin, j_end, max_to, same_slot, my_to; u64 out_off; };
static __device__ __forceinline__ bool idx_qt_setup(
    const u32* slots, const u32* positions, float* out, u32 rows, u32 out_keys, u32 live_keys,
    u32 masked, u32 tile_keys, u32 tid, u32 warp, u32 lane,
    u32& sm_max_to, u32& sm_same_slot, idx_qt_ctx& c) {
    c.r0 = blockIdx.y << 3u; c.row = c.r0 + warp;
    c.j_begin = blockIdx.x * tile_keys; c.j_end = min(c.j_begin + tile_keys, out_keys);
    if (c.r0 >= rows || c.j_begin >= c.j_end) return false;
    c.nr = min(rows - c.r0, 8u);
    if (tid < 8u) {
        u32 r = c.r0 + tid, ok = tid < c.nr;
        u32 to = ok ? ((masked == 1u && positions[r] < live_keys) ? (positions[r] + 1u) : live_keys) : 0u;
        u32 same = ok ? (slots[r] == slots[c.r0] ? 1u : 0u) : 1u;
        for (u32 o = 1u; o < 8u; o <<= 1u) { to = max(to, __shfl_xor_sync(0xFFu, to, o)); same &= __shfl_xor_sync(0xFFu, same, o); }
        if (tid == 0u) { sm_max_to = to; sm_same_slot = same; }
    }
    prim::block_sync();
    c.max_to = sm_max_to; c.same_slot = sm_same_slot;
    c.my_to = (c.row < rows) ? ((masked == 1u && positions[c.row] < live_keys) ? (positions[c.row] + 1u) : live_keys) : 0u;
    c.out_off = u64(c.row) * out_keys;
    if (c.j_begin >= c.max_to) {
        if (c.row < rows) for (u32 j = c.j_begin + lane; j < c.j_end; j += 32u) out[c.out_off + j] = attn_neg_inf();
        return false;
    }
    return true;
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
    idx_qt_ctx c;
    if (!idx_qt_setup(slots, positions, out, rows, out_keys, live_keys, masked, tile_keys, tid, warp, lane, sm.max_to, sm.same_slot, c)) return;
    u32 r0 = c.r0, row = c.row, nr = c.nr, j_begin = c.j_begin, j_end = c.j_end, max_to = c.max_to, same_slot = c.same_slot, my_to = c.my_to;
    u64 out_off = c.out_off;
    float neg = attn_neg_inf();
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
                    if (j0 + lane < j_end) out[out_off + j0 + lane] = neg;
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
                        idx_relu_accum(acc + 0u, s0, w0, w1); idx_relu_accum(acc + 2u, s1, w0, w1);
                        idx_relu_accum(acc + 4u, s2, w0, w1); idx_relu_accum(acc + 6u, s3, w0, w1);
                    }
                    idx_qt_reduce_store(acc, out, out_off, j0, j_end, my_to, g, t);
                }
            }
            if (has_next) prim::block_sync();
            st = nst;
        }
        if (pass + 1u < passes) prim::block_sync();
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
struct alignas(16) idx_qt_fp8_smem {
    u8 k[2][32][128];
    u32 max_to, same_slot;
};
struct alignas(16) idx_fp8_smem {
    u8 q[64][128];
    u8 k[4][16][128];
    u32 q_sc[64][4];
    float w[64];
};
static __device__ __forceinline__ void idx_qt_stage_fp8(
    const void* kp, u8 dst[32][128], u32 slot, u32 ctx, u32 j0, u32 lim, const u32* pt, u32 sh, u32 tid) {
    u32 kr = tid >> 3u, c = tid & 7u, j = j0 + kr, bytes = (j < lim) ? 16u : 0u;
    const u8* base = static_cast<const u8*>(kp);
    const u8* src = (j < lim) ? (base + kv_row_at(pt, slot, j, ctx, sh) * 128ull + u64(c) * 16ull) : base;
    attn_cp_async16(&dst[kr][(c ^ (kr & 7u)) << 4u], src, bytes);
    attn_cp_commit();
}
static __device__ __forceinline__ void idx_qt_fp8_score(
    const u8 tile[32][128], const u32 qa[2][4][4], const u32 q_sa[2], const float w_reg[2][2],
    u32 krow0, u32 krow1, u32 l_ch, u32 l_row7, float acc[8]) {
    float s0[4][4] = {}, s1[4][4] = {};
    u32 kb0[4], kb1[4], sw0 = (l_ch ^ l_row7) << 4u;
    attn_ldm_x4(kb0, &tile[krow0][sw0]);
    attn_ldm_x4(kb1, &tile[krow1][sw0]);
#pragma unroll
    for (u32 ds = 0u; ds < 4u; ds++) {
        u32 sa0 = q_sa[0] >> (ds << 3u), sa1 = q_sa[1] >> (ds << 3u);
        u32 sw_n = ((((ds + 1u) << 1u) | l_ch) ^ l_row7) << 4u;
        trellis::mma_mx8(s0[0], qa[0][ds], kb0, sa0, 0x7F7F7F7Fu); trellis::mma_mx8(s1[0], qa[1][ds], kb0, sa1, 0x7F7F7F7Fu);
        trellis::mma_mx8(s0[1], qa[0][ds], kb0 + 2u, sa0, 0x7F7F7F7Fu); trellis::mma_mx8(s1[1], qa[1][ds], kb0 + 2u, sa1, 0x7F7F7F7Fu);
        if (ds + 1u < 4u) attn_ldm_x4(kb0, &tile[krow0][sw_n]);
        trellis::mma_mx8(s0[2], qa[0][ds], kb1, sa0, 0x7F7F7F7Fu); trellis::mma_mx8(s1[2], qa[1][ds], kb1, sa1, 0x7F7F7F7Fu);
        trellis::mma_mx8(s0[3], qa[0][ds], kb1 + 2u, sa0, 0x7F7F7F7Fu); trellis::mma_mx8(s1[3], qa[1][ds], kb1 + 2u, sa1, 0x7F7F7F7Fu);
        if (ds + 1u < 4u) attn_ldm_x4(kb1, &tile[krow1][sw_n]);
    }
    float w00 = w_reg[0][0], w01 = w_reg[0][1], w10 = w_reg[1][0], w11 = w_reg[1][1];
#pragma unroll
    for (u32 q = 0u; q < 4u; q++) { idx_relu_accum(acc + (q << 1u), s0[q], w00, w01); idx_relu_accum(acc + (q << 1u), s1[q], w10, w11); }
}
static __device__ __forceinline__ void idx_qtile_fp8_body(
    const float* q, const void* k_ptr, const float* weights, const u32* slots, const u32* positions,
    float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 masked, u32 tile_keys,
    const u32* page_table, u32 page_shift, idx_qt_fp8_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    idx_qt_ctx cx;
    if (!idx_qt_setup(slots, positions, out, rows, out_keys, live_keys, masked, tile_keys, tid, warp, lane, sm.max_to, sm.same_slot, cx)) return;
    u32 qa[2][4][4], q_sa[2];
    float w_reg[2][2];
    if (cx.row < rows) {
        const float* qb = q + u64(cx.row) * 4096ull;
        const float* wb = weights + u64(cx.row) * 32ull;
#pragma unroll
        for (u32 hi = 0u; hi < 2u; hi++) {
            u32 h0 = hi * 16u, sa_pack = 0u;
            w_reg[hi][0] = wb[h0 + g]; w_reg[hi][1] = wb[h0 + g + 8u];
            const float* q0 = qb + u64(h0 + g) * 128ull;
            const float* q8 = qb + u64(h0 + g + 8u) * 128ull;
#pragma unroll
            for (u32 ds = 0u; ds < 4u; ds++) {
                u32 k0 = (ds << 5u) + (t << 2u);
                float inv0, inv1;
                float4 a0 = *reinterpret_cast<const float4*>(q0 + k0);
                float4 b0 = *reinterpret_cast<const float4*>(q0 + k0 + 16u);
                float4 a1 = *reinterpret_cast<const float4*>(q8 + k0);
                float4 b1 = *reinterpret_cast<const float4*>(q8 + k0 + 16u);
                u32 sc0 = mla_max4_scale(a0, b0, inv0), sc1 = mla_max4_scale(a1, b1, inv1);
                sa_pack |= (((lane & 1u) ? sc1 : sc0) & 0xFFu) << (ds << 3u);
                qa[hi][ds][0] = mla_pack4_fp8(a0, inv0); qa[hi][ds][1] = mla_pack4_fp8(a1, inv1);
                qa[hi][ds][2] = mla_pack4_fp8(b0, inv0); qa[hi][ds][3] = mla_pack4_fp8(b1, inv1);
            }
            q_sa[hi] = sa_pack;
        }
    }
    u32 krow0 = ((lane >> 4u) << 3u) + (lane & 7u), krow1 = 16u + krow0, l_ch = (lane >> 3u) & 1u, l_row7 = lane & 7u;
    u32 passes = cx.same_slot ? 1u : cx.nr;
    for (u32 pass = 0u; pass < passes; pass++) {
        u32 cur_slot = slots[cx.r0 + pass];
        bool active = cx.same_slot ? (cx.row < rows) : (warp == pass);
        u32 lim = cx.same_slot ? cx.max_to : ((masked == 1u && positions[cx.r0 + pass] < live_keys) ? (positions[cx.r0 + pass] + 1u) : live_keys);
        u32 j_lim = min(lim, cx.j_end), st = 0u, j0 = cx.j_begin;
        if (cx.j_begin < j_lim) idx_qt_stage_fp8(k_ptr, sm.k[0], cur_slot, context, cx.j_begin, lim, page_table, page_shift, tid);
        for (; j0 < j_lim; j0 += 32u) {
            u32 nst = st ^ 1u;
            bool has_next = (j0 + 32u < j_lim);
            if (has_next) { idx_qt_stage_fp8(k_ptr, sm.k[nst], cur_slot, context, j0 + 32u, lim, page_table, page_shift, tid); attn_cp_wait1(); }
            else attn_cp_wait0();
            prim::block_sync();
            if (active) {
                if (j0 >= cx.my_to) { if (j0 + lane < cx.j_end) out[cx.out_off + j0 + lane] = attn_neg_inf(); }
                else {
                    float acc[8] = {};
                    idx_qt_fp8_score(sm.k[st], qa, q_sa, w_reg, krow0, krow1, l_ch, l_row7, acc);
                    idx_qt_reduce_store(acc, out, cx.out_off, j0, cx.j_end, cx.my_to, g, t);
                }
            }
            if (has_next) prim::block_sync();
            st = nst;
        }
        if (active && j0 < cx.j_end) for (u32 j = j0 + lane; j < cx.j_end; j += 32u) out[cx.out_off + j] = attn_neg_inf();
        if (pass + 1u < passes) prim::block_sync();
    }
}
static __device__ __forceinline__ void idx_fp8_d128_body(
    const float* q, const void* k_ptr, const float* weights, const u32* slots, const u32* positions,
    float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 masked, u32 tile_keys,
    const u32* page_table, u32 page_shift, idx_fp8_smem& sm) {
    u32 row = blockIdx.y, j_begin = blockIdx.x * tile_keys, j_end = min(j_begin + tile_keys, out_keys);
    if (row >= rows || j_begin >= j_end) return;
    u32 tid = threadIdx.x, slot = slots[row], to = (masked == 1u && positions[row] < live_keys) ? (positions[row] + 1u) : live_keys;
    float* out_row = out + u64(row) * out_keys;
    float neg = attn_neg_inf();
    if (j_begin >= to) { for (u32 j = j_begin + tid; j < j_end; j += 128u) out_row[j] = neg; return; }
    if (tid < heads) sm.w[tid] = weights[u64(row) * heads + tid];
    u32 grp4 = tid >> 2u, t = tid & 3u, total_blocks = heads << 2u;
    const float* q_base = q + u64(row) * heads * 128ull;
    for (u32 blk = grp4; blk < total_blocks; blk += 32u) {
        u32 h = blk >> 2u, ds = blk & 3u;
        const float* qp = q_base + u64(h) * 128ull + u64(ds << 5u) + u64(t << 2u);
        float inv;
        float4 a0 = *reinterpret_cast<const float4*>(qp);
        float4 b0 = *reinterpret_cast<const float4*>(qp + 16u);
        u32 sc = mla_max4_scale(a0, b0, inv), ch0 = (ds << 1u) ^ (h & 7u), ch1 = ((ds << 1u) | 1u) ^ (h & 7u);
        *reinterpret_cast<u32*>(&sm.q[h][(ch0 << 4u) + (t << 2u)]) = mla_pack4_fp8(a0, inv);
        *reinterpret_cast<u32*>(&sm.q[h][(ch1 << 4u) + (t << 2u)]) = mla_pack4_fp8(b0, inv);
        if (t == 0u) sm.q_sc[h][ds] = sc;
    }
    prim::block_sync();
    u32 warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, q_r_off = lane & 15u, q_ch_bit = lane >> 4u;
    u32 k_r = ((lane >> 4u) << 3u) + (lane & 7u), k_ch_bit = (lane >> 3u) & 1u, l_row7 = lane & 7u;
    const u8* kp = static_cast<const u8*>(k_ptr);
    for (u32 j0 = j_begin + (warp << 4u); j0 < j_end; j0 += 64u) {
        if (j0 >= to) { if (lane < 16u && j0 + lane < j_end) out_row[j0 + lane] = neg; continue; }
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
            u32 idx = lane + (i << 5u), kr = idx >> 3u, c = idx & 7u, j = j0 + kr;
            uint4 pk = (j < to) ? *reinterpret_cast<const uint4*>(kp + kv_row_at(page_table, slot, j, context, page_shift) * 128ull + u64(c) * 16ull) : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(&sm.k[warp][kr][(c ^ (kr & 7u)) << 4u]) = pk;
        }
        __syncwarp();
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (u32 h0 = 0u; h0 < heads; h0 += 16u) {
            float s0[4] = {}, s1[4] = {};
            u32 q_r = h0 + q_r_off, qr7 = q_r & 7u;
#pragma unroll
            for (u32 ds = 0u; ds < 4u; ds++) {
                u32 qa[4], kb[4], q_sw = (((ds << 1u) | q_ch_bit) ^ qr7) << 4u, k_sw = (((ds << 1u) | k_ch_bit) ^ l_row7) << 4u;
                attn_ldm_x4(qa, &sm.q[q_r][q_sw]); attn_ldm_x4(kb, &sm.k[warp][k_r][k_sw]);
                u32 b0[2] = {kb[0], kb[1]}, b1[2] = {kb[2], kb[3]}, sa = sm.q_sc[h0 + g + ((lane & 1u) << 3u)][ds];
                trellis::mma_mx8(s0, qa, b0, sa, 0x7F7F7F7Fu); trellis::mma_mx8(s1, qa, b1, sa, 0x7F7F7F7Fu);
            }
            float w0 = sm.w[h0 + g], w1 = sm.w[h0 + g + 8u];
            acc[0] += fmaxf(s0[0], 0.f) * w0 + fmaxf(s0[2], 0.f) * w1; acc[1] += fmaxf(s0[1], 0.f) * w0 + fmaxf(s0[3], 0.f) * w1;
            acc[2] += fmaxf(s1[0], 0.f) * w0 + fmaxf(s1[2], 0.f) * w1; acc[3] += fmaxf(s1[1], 0.f) * w0 + fmaxf(s1[3], 0.f) * w1;
        }
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
            acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 4u);
            acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 8u);
            acc[i] += __shfl_xor_sync(0xFFFFFFFFu, acc[i], 16u);
        }
        if (g == 0u) {
            u32 k0 = j0 + (t << 1u), k2 = j0 + 8u + (t << 1u);
            if (k0 < j_end) out_row[k0] = (k0 < to) ? acc[0] : neg;
            if (k0 + 1u < j_end) out_row[k0 + 1u] = (k0 + 1u < to) ? acc[1] : neg;
            if (k2 < j_end) out_row[k2] = (k2 < to) ? acc[2] : neg;
            if (k2 + 1u < j_end) out_row[k2 + 1u] = (k2 + 1u < to) ? acc[3] : neg;
        }
    }
}
#endif

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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
extern "C" __global__ IDX_D128_BOUNDS void tessera_indexer_mx8(const float* q, const void* k, const float* weights, const u32* slots, const u32* positions, float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 dim, u32 masked, u32 tile_keys, const u32* page_table, u32 page_shift) {
    (void)dim;
    __shared__ idx_fp8_smem sm;
    idx_fp8_d128_body(q, k, weights, slots, positions, out, rows, out_keys, live_keys, context, heads, masked, tile_keys, page_table, page_shift, sm);
}
extern "C" __global__ __launch_bounds__(256, IDX_QT_MIN_BLOCKS) void tessera_indexer_qt_mx8(const float* q, const void* k, const float* weights, const u32* slots, const u32* positions, float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 dim, u32 masked, u32 tile_keys, const u32* page_table, u32 page_shift) {
    (void)heads; (void)dim;
    __shared__ idx_qt_fp8_smem sm;
    idx_qtile_fp8_body(q, k, weights, slots, positions, out, rows, out_keys, live_keys, context, masked, tile_keys, page_table, page_shift, sm);
}
#else
extern "C" __global__ IDX_D128_BOUNDS void tessera_indexer_mx8(const float* q, const void* k, const float* weights, const u32* slots, const u32* positions, float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 dim, u32 masked, u32 tile_keys, const u32* page_table, u32 page_shift) {
    __shared__ idx_smem sm;
    idx_body<0u>(q, k, weights, slots, positions, out, rows, out_keys, live_keys, context, heads, dim, masked, tile_keys, page_table, page_shift, sm);
}
extern "C" __global__ __launch_bounds__(256, IDX_QT_MIN_BLOCKS) void tessera_indexer_qt_mx8(const float* q, const void* k, const float* weights, const u32* slots, const u32* positions, float* out, u32 rows, u32 out_keys, u32 live_keys, u32 context, u32 heads, u32 dim, u32 masked, u32 tile_keys, const u32* page_table, u32 page_shift) {
    (void)heads; (void)dim;
    __shared__ idx_qt_smem sm;
    idx_qtile_body<0u>(q, k, weights, slots, positions, out, rows, out_keys, live_keys, context, masked, tile_keys, page_table, page_shift, sm);
}
#endif
IDX_DEF(tessera_indexer_bf16, tessera_indexer_qt_bf16, 1u)
IDX_DEF(tessera_indexer_f32, tessera_indexer_qt_f32, 2u)

#endif
