#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

constant uint AM_THREADS = 256u;
constant uint AM_WARPS = 8u;
constant uint AM_BQ = 16u;
constant uint AM_BK = 32u;
constant float AM_LOG2E = 1.4426950408889634f;
constant float AM_LN2 = 0.6931471805599453f;

enum AmFmt : uint {
    AM_F32 = 0u,
    AM_BF16 = 1u,
    AM_F16 = 2u,
    AM_FP8 = 3u,
};

static inline uint am_kept_row(
    uint slot,
    uint pos,
    uint kept_mode,
    uint kept_param,
    uint page_shift,
    uint page_mask,
    device const uint *page_table) {
    return kept_mode == 0u
        ? slot * kept_param + pos
        : (page_table[slot * kept_param + (pos >> page_shift)] << page_shift) | (pos & page_mask);
}

// Decodes two E4M3 bytes to half2.
static inline half2 am_fp8_pair(uchar2 raw) {
    uint bits = uint(raw.x) | (uint(raw.y) << 16u);
    uint shifted = ((bits & 0x00800080u) << 8u) | ((bits & 0x007F007Fu) << 7u);
    return as_type<half2>(shifted) * half2(256.0h);
}

template <uint FMT>
static inline half2 am_load2_half(device const uchar *base, ulong off) {
    if constexpr (FMT == AM_F32) {
        return half2(*(device const float2 *)(base + (off << 2u)));
    } else if constexpr (FMT == AM_BF16) {
        ushort2 bits = *(device const ushort2 *)(base + (off << 1u));
        return half2(as_type<float2>(uint2(bits) << 16u));
    } else if constexpr (FMT == AM_F16) {
        return *(device const half2 *)(base + (off << 1u));
    } else {
        return am_fp8_pair(*(device const uchar2 *)(base + off));
    }
}

template <uint FMT>
static inline half4 am_load4_half(device const uchar *base, ulong off) {
    if constexpr (FMT == AM_FP8) {
        uchar4 raw = *(device const uchar4 *)(base + off);
        return half4(am_fp8_pair(raw.xy), am_fp8_pair(raw.zw));
    } else if constexpr (FMT == AM_F16) {
        return *(device const half4 *)(base + (off << 1u));
    } else {
        return half4(am_load2_half<FMT>(base, off), am_load2_half<FMT>(base, off + 2u));
    }
}

// Reduces warp partials in threadgroup memory and writes output.
template <uint D>
static inline void am_dec_finish(
    device float *out,
    uint row,
    uint qh,
    uint heads,
    uint splits,
    uint sp,
    uint tid,
    threadgroup float sm_o[AM_WARPS][D],
    threadgroup float sm_m[AM_WARPS],
    threadgroup float sm_l[AM_WARPS]) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < D) {
        float best = -INFINITY;
        for (uint w = 0u; w < AM_WARPS; w++) {
            if (sm_l[w] > 0.0f) best = max(best, sm_m[w]);
        }
        float denom = 0.0f, sum = 0.0f;
        for (uint w = 0u; w < AM_WARPS; w++) {
            if (sm_l[w] > 0.0f) {
                float sc = exp2(sm_m[w] - best);
                denom += sm_l[w] * sc;
                sum += sm_o[w][tid] * sc;
            }
        }
        if (splits == 1u) {
            float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
            out[(ulong(row) * heads + qh) * ulong(D) + tid] = sum * inv;
        } else {
            ulong pbase = ((ulong(row) * heads + qh) * splits + sp) * ulong(D + 2u);
            if (tid == 0u) {
                out[pbase] = (best == -INFINITY) ? -INFINITY : (best * AM_LN2);
                out[pbase + 1u] = denom;
            }
            out[pbase + 2u + tid] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Single-pass GQA=6 decode for D=256 with Q in threadgroup memory.
template <uint FMT>
static inline void am_dec_g6(
    device const float *q,
    device const uchar *k_buf,
    device const uchar *v_buf,
    device const uint *page_table,
    device float *out,
    float scale,
    uint row,
    uint kvh,
    uint sp,
    uint heads,
    uint kv_heads,
    uint splits,
    uint slot,
    uint k_start,
    uint k_end,
    uint kept_mode,
    uint kept_param,
    uint page_shift,
    uint page_mask,
    uint tid,
    threadgroup half sm_q[6][256],
    threadgroup float sm_o[AM_WARPS][256],
    threadgroup float sm_m[AM_WARPS],
    threadgroup float sm_l[AM_WARPS]) {
    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    float qscale = scale * AM_LOG2E;
    #pragma unroll
    for (uint g = 0u; g < 6u; g++) {
        uint qh = kvh * 6u + g;
        sm_q[g][tid] = half(q[(ulong(row) * heads + qh) * 256ull + tid] * qscale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_g[6] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_g[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float4 a0[6], a1[6];
    #pragma unroll
    for (uint g = 0u; g < 6u; g++) {
        a0[g] = float4(0.0f);
        a1[g] = float4(0.0f);
    }

    uint d0 = lane * 4u;
    uint d1 = 128u + lane * 4u;
    for (uint t = k_start + simd; t < k_end; t += AM_WARPS) {
        uint kv_row = am_kept_row(slot, t, kept_mode, kept_param, page_shift, page_mask, page_table);
        ulong kv_base = (ulong(kv_row) * kv_heads + kvh) * 256ull;
        half4 kv0 = am_load4_half<FMT>(k_buf, kv_base + d0);
        half4 kv1 = am_load4_half<FMT>(k_buf, kv_base + d1);
        float corr_g[6], wt_g[6];
        #pragma unroll
        for (uint g = 0u; g < 6u; g++) {
            half4 q0 = *(threadgroup const half4 *)(&sm_q[g][d0]);
            half4 q1 = *(threadgroup const half4 *)(&sm_q[g][d1]);
            float d = float(dot(q0, kv0) + dot(q1, kv1));
            float score = simd_sum(d);
            float next = max(m_g[g], score);
            float corr = (l_g[g] > 0.0f) ? exp2(m_g[g] - next) : 0.0f;
            float wt = exp2(score - next);
            m_g[g] = next;
            l_g[g] = l_g[g] * corr + wt;
            corr_g[g] = corr;
            wt_g[g] = wt;
        }
        float4 vv0 = float4(am_load4_half<FMT>(v_buf, kv_base + d0));
        float4 vv1 = float4(am_load4_half<FMT>(v_buf, kv_base + d1));
        #pragma unroll
        for (uint g = 0u; g < 6u; g++) {
            a0[g] = fma(vv0, float4(wt_g[g]), a0[g] * corr_g[g]);
            a1[g] = fma(vv1, float4(wt_g[g]), a1[g] * corr_g[g]);
        }
    }

    #pragma unroll
    for (uint g = 0u; g < 6u; g++) {
        uint qh = kvh * 6u + g;
        if (lane == 0u) {
            sm_m[simd] = m_g[g];
            sm_l[simd] = l_g[g];
        }
        *(threadgroup float4 *)(&sm_o[simd][d0]) = a0[g];
        *(threadgroup float4 *)(&sm_o[simd][d1]) = a1[g];
        am_dec_finish<256u>(out, row, qh, heads, splits, sp, tid, sm_o, sm_m, sm_l);
    }
}

struct AmDecCtx {
    device const float *q;
    device const uchar *k_buf;
    device const uchar *v_buf;
    device const uint *page_table;
    device float *out;
    float qscale;
    uint row;
    uint kvh;
    uint sp;
    uint heads;
    uint kv_heads;
    uint gqa;
    uint splits;
    uint slot;
    uint k_start;
    uint k_end;
    uint kept_mode;
    uint kept_param;
    uint page_shift;
    uint page_mask;
};

// Chunked GQA split-K decode for G heads.
template <uint D, uint FMT, uint G, typename HV, typename FV, uint N>
static inline void am_dec_core(
    thread const AmDecCtx &c,
    uint g0,
    uint tid,
    threadgroup float sm_o[AM_WARPS][D],
    threadgroup float sm_m[AM_WARPS],
    threadgroup float sm_l[AM_WARPS]) {
    constexpr uint CHUNK = D / N;
    constexpr uint VW = CHUNK / 32u;
    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    float m_g[G], l_g[G];
    HV qv[G][N];
    FV av[G][N];
    #pragma unroll
    for (uint g = 0u; g < G; g++) {
        uint qh = c.kvh * c.gqa + g0 + g;
        ulong qb = (ulong(c.row) * c.heads + qh) * ulong(D);
        #pragma unroll
        for (uint u = 0u; u < N; u++) {
            qv[g][u] = HV(*(device const FV *)(c.q + qb + u * CHUNK + lane * VW) * c.qscale);
            av[g][u] = FV(0.0f);
        }
        m_g[g] = -INFINITY;
        l_g[g] = 0.0f;
    }
    for (uint t = c.k_start + simd; t < c.k_end; t += AM_WARPS) {
        uint kv_row = am_kept_row(c.slot, t, c.kept_mode, c.kept_param, c.page_shift, c.page_mask, c.page_table);
        ulong kv_base = (ulong(kv_row) * c.kv_heads + c.kvh) * ulong(D);
        HV kv[N];
        #pragma unroll
        for (uint u = 0u; u < N; u++) {
            if constexpr (VW == 4u) {
                kv[u] = am_load4_half<FMT>(c.k_buf, kv_base + u * CHUNK + lane * VW);
            } else {
                kv[u] = am_load2_half<FMT>(c.k_buf, kv_base + lane * VW);
            }
        }
        float corr_g[G], wt_g[G];
        #pragma unroll
        for (uint g = 0u; g < G; g++) {
            float d = 0.0f;
            #pragma unroll
            for (uint u = 0u; u < N; u++) {
                d += float(dot(qv[g][u], kv[u]));
            }
            float score = simd_sum(d);
            float next = max(m_g[g], score);
            float corr = (l_g[g] > 0.0f) ? exp2(m_g[g] - next) : 0.0f;
            float wt = exp2(score - next);
            m_g[g] = next;
            l_g[g] = l_g[g] * corr + wt;
            corr_g[g] = corr;
            wt_g[g] = wt;
        }
        FV vv[N];
        #pragma unroll
        for (uint u = 0u; u < N; u++) {
            if constexpr (VW == 4u) {
                vv[u] = FV(am_load4_half<FMT>(c.v_buf, kv_base + u * CHUNK + lane * VW));
            } else {
                vv[u] = FV(am_load2_half<FMT>(c.v_buf, kv_base + lane * VW));
            }
        }
        #pragma unroll
        for (uint g = 0u; g < G; g++) {
            #pragma unroll
            for (uint u = 0u; u < N; u++) {
                av[g][u] = fma(vv[u], FV(wt_g[g]), av[g][u] * corr_g[g]);
            }
        }
    }
    #pragma unroll
    for (uint g = 0u; g < G; g++) {
        uint qh = c.kvh * c.gqa + g0 + g;
        if (lane == 0u) {
            sm_m[simd] = m_g[g];
            sm_l[simd] = l_g[g];
        }
        #pragma unroll
        for (uint u = 0u; u < N; u++) {
            *(threadgroup FV *)(&sm_o[simd][u * CHUNK + lane * VW]) = av[g][u];
        }
        am_dec_finish<D>(c.out, c.row, qh, c.heads, c.splits, c.sp, tid, sm_o, sm_m, sm_l);
    }
}

template <uint D, uint FMT, uint G>
static inline void am_dec_group(
    thread const AmDecCtx &c,
    uint g0,
    uint tid,
    threadgroup float sm_o[AM_WARPS][D],
    threadgroup float sm_m[AM_WARPS],
    threadgroup float sm_l[AM_WARPS]) {
    if constexpr (D == 64u) {
        am_dec_core<64u, FMT, G, half2, float2, 1u>(c, g0, tid, sm_o, sm_m, sm_l);
    } else if constexpr (D == 128u) {
        am_dec_core<128u, FMT, G, half4, float4, 1u>(c, g0, tid, sm_o, sm_m, sm_l);
    } else {
        am_dec_core<256u, FMT, G, half4, float4, 2u>(c, g0, tid, sm_o, sm_m, sm_l);
    }
}

template <uint D, uint FMT>
static inline void am_dec_body(
    device const float *q,
    device const uchar *k_buf,
    device const uchar *v_buf,
    device const uint *slots,
    device const uint *positions,
    device const uint *page_table,
    device float *out,
    float scale,
    uint rows,
    uint keys,
    uint heads,
    uint kv_heads,
    uint splits,
    uint kept_mode,
    uint kept_param,
    uint page_shift,
    uint page_mask,
    uint tg_x,
    uint tid,
    threadgroup half sm_q[6][D],
    threadgroup float sm_o[AM_WARPS][D],
    threadgroup float sm_m[AM_WARPS],
    threadgroup float sm_l[AM_WARPS]) {
    uint sp = tg_x % splits;
    uint rk = tg_x / splits;
    uint kvh = rk % kv_heads;
    uint row = rk / kv_heads;
    if (row >= rows) return;

    uint gqa = heads / kv_heads;
    uint slot = slots[row];
    uint pos = positions[row];
    uint nkeys = min(pos + 1u, keys);

    uint chunk = (nkeys + splits - 1u) / splits;
    uint k_start = min(sp * chunk, nkeys);
    uint k_end = min(k_start + chunk, nkeys);

    if constexpr (D == 256u) {
        if (gqa == 6u) {
            am_dec_g6<FMT>(
                q, k_buf, v_buf, page_table, out, scale, row, kvh, sp, heads, kv_heads,
                splits, slot, k_start, k_end, kept_mode, kept_param, page_shift, page_mask, tid,
                sm_q, sm_o, sm_m, sm_l);
            return;
        }
    }
    AmDecCtx ctx = {
        q, k_buf, v_buf, page_table, out, scale * AM_LOG2E,
        row, kvh, sp, heads, kv_heads, gqa, splits, slot,
        k_start, k_end, kept_mode, kept_param, page_shift, page_mask
    };
    (void)sm_q;
    if ((gqa % 3u) == 0u) {
        for (uint g0 = 0u; g0 < gqa; g0 += 3u) {
            am_dec_group<D, FMT, 3u>(ctx, g0, tid, sm_o, sm_m, sm_l);
        }
    } else if ((gqa & 3u) == 0u) {
        for (uint g0 = 0u; g0 < gqa; g0 += 4u) {
            am_dec_group<D, FMT, 4u>(ctx, g0, tid, sm_o, sm_m, sm_l);
        }
    } else if ((gqa & 1u) == 0u) {
        for (uint g0 = 0u; g0 < gqa; g0 += 2u) {
            am_dec_group<D, FMT, 2u>(ctx, g0, tid, sm_o, sm_m, sm_l);
        }
    } else {
        for (uint g0 = 0u; g0 < gqa; g0 += 1u) {
            am_dec_group<D, FMT, 1u>(ctx, g0, tid, sm_o, sm_m, sm_l);
        }
    }
}

// Combines split-K partials into final output.
kernel void am_dec_reduce(
    device const float *part [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &lists [[buffer(2)]],
    constant uint &splits [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint list = tg.x;
    if (list >= lists || tid >= width) return;
    ulong stride = ulong(width + 2u);
    ulong base = ulong(list) * splits * stride;

    float best = -INFINITY;
    for (uint s = 0u; s < splits; s++) {
        float l = part[base + ulong(s) * stride + 1u];
        if (l > 0.0f) {
            best = max(best, part[base + ulong(s) * stride]);
        }
    }
    float denom = 0.0f;
    float sum = 0.0f;
    for (uint s = 0u; s < splits; s++) {
        float l = part[base + ulong(s) * stride + 1u];
        if (l > 0.0f) {
            float m = part[base + ulong(s) * stride];
            float sc = exp(m - best);
            denom += l * sc;
            sum += part[base + ulong(s) * stride + 2u + tid] * sc;
        }
    }
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    out[ulong(list) * width + tid] = sum * inv;
}

// Native Metal FlashAttention-2 prefill for causal ragged attention.
template <uint D, uint FMT>
static inline void am_pf_body(
    device const float *q,
    device const uchar *k_buf,
    device const uchar *v_buf,
    device const uint *slots,
    device const uint *positions,
    device const uint *page_table,
    device float *out,
    float scale,
    uint rows,
    uint keys,
    uint heads,
    uint kv_heads,
    uint kept_mode,
    uint kept_param,
    uint page_shift,
    uint page_mask,
    uint tg_x,
    uint tid,
    threadgroup half *qs,
    threadgroup half *kvs,
    threadgroup float *ss,
    threadgroup half *ps,
    threadgroup float *alpha_s,
    threadgroup float *m_s,
    threadgroup float *l_s,
    threadgroup uint *to_s) {
    constexpr uint STRIDE = D + 8u;
    constexpr uint DTILES = D / 32u;
    constexpr uint Q_ELEMS = AM_BQ * D;

    uint head = tg_x % heads;
    uint qb = tg_x / heads;
    uint q0 = qb * AM_BQ;
    if (q0 >= rows) return;

    uint kvh = head / (heads / kv_heads);
    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    uint qg = simd >> 2u;   // 0..1 (8 rows each)
    uint dg = simd & 3u;    // 0..3 (8 keys in S, D/4 cols in O)
    uint tile_row = ((lane >> 2u) & 4u) + ((lane >> 1u) & 3u);
    uint tile_col = (((lane >> 2u) & 2u) << 1u) + ((lane & 1u) << 1u);
    float qscale = scale * AM_LOG2E;

    for (uint idx = tid * 4u; idx < Q_ELEMS; idx += AM_THREADS * 4u) {
        uint r = idx / D;
        uint d = idx % D;
        half4 qh = half4(0.0h);
        if (q0 + r < rows) {
            ulong q_off = (ulong(q0 + r) * heads + head) * D + d;
            qh = half4(*(device const float4 *)(q + q_off) * qscale);
        }
        *(threadgroup half4 *)(qs + r * STRIDE + d) = qh;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pass = 0u; pass < AM_BQ; pass++) {
        uint lead = q0 + pass;
        bool live = (lead < rows);
        uint slot = live ? slots[lead] : 0u;
        bool fresh = live;
        if (live) {
            for (uint j = 0u; j < pass; j++) {
                if (slots[q0 + j] == slot) {
                    fresh = false;
                    break;
                }
            }
        }
        if (!fresh) continue;

        if (tid < AM_BQ) {
            uint r = q0 + tid;
            uint lim = (r < rows && slots[r] == slot) ? min(positions[r] + 1u, keys) : 0u;
            to_s[tid] = lim;
            m_s[tid] = -INFINITY;
            l_s[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint last = 0u;
        uint min_live = 0xFFFFFFFFu;
        for (uint r = 0u; r < AM_BQ; r++) {
            uint lim = to_s[r];
            last = max(last, lim);
            if (lim > 0u) min_live = min(min_live, lim);
        }
        if (min_live == 0xFFFFFFFFu) min_live = 0u;

        simdgroup_float8x8 o_acc[DTILES];
        #pragma unroll
        for (uint j = 0u; j < DTILES; j++) {
            o_acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }

        uint kr = tid >> 3u;
        uint kt = tid & 7u;
        constexpr uint PER_T = D / 8u;

        for (uint k0 = 0u; k0 < last; k0 += AM_BK) {
            uint k_pos = k0 + kr;
            uint kv_row = (k_pos < last)
                              ? am_kept_row(slot, k_pos, kept_mode, kept_param, page_shift, page_mask, page_table)
                              : 0u;
            ulong kv_base = (ulong(kv_row) * kv_heads + kvh) * D + kt * PER_T;

            if (k_pos < last) {
                #pragma unroll
                for (uint d = 0u; d < PER_T; d += 4u) {
                    *(threadgroup half4 *)(kvs + kr * STRIDE + kt * PER_T + d) =
                        am_load4_half<FMT>(k_buf, kv_base + d);
                }
            } else {
                #pragma unroll
                for (uint d = 0u; d < PER_T; d += 4u) {
                    *(threadgroup half4 *)(kvs + kr * STRIDE + kt * PER_T + d) = half4(0.0h);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 s_acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0u; d < D; d += 8u) {
                simdgroup_half8x8 qa, kb;
                simdgroup_load(qa, qs + (qg * 8u) * STRIDE + d, STRIDE);
                simdgroup_load(kb, kvs + (dg * 8u) * STRIDE + d, STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s_acc, qa, kb, s_acc);
            }
            simdgroup_store(s_acc, ss + (qg * 8u) * 36u + dg * 8u, 36u);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool full_interior = (k0 + AM_BK <= min_live);
            #pragma unroll
            for (uint ri = 0u; ri < 2u; ri++) {
                uint r = simd * 2u + ri;
                uint lim = to_s[r];
                if (full_interior && lim > 0u) {
                    float val = ss[r * 36u + lane];
                    float rmax = simd_max(val);
                    float m_old = m_s[r];
                    float m_new = max(m_old, rmax);
                    float alpha = (m_old == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                    float p = exp2(val - m_new);
                    float psum = simd_sum(p);
                    ps[r * 36u + lane] = half(p);
                    if (lane == 0u) {
                        m_s[r] = m_new;
                        l_s[r] = l_s[r] * alpha + psum;
                        alpha_s[r] = alpha;
                    }
                } else {
                    bool seen = (k0 + lane) < lim;
                    float val = seen ? ss[r * 36u + lane] : -INFINITY;
                    float rmax = simd_max(val);
                    float m_old = m_s[r];
                    float m_new = max(m_old, rmax);
                    float safe = (m_new == -INFINITY) ? 0.0f : m_new;
                    float alpha = (m_old == -INFINITY) ? 0.0f : exp2(m_old - safe);
                    float p = seen ? exp2(val - safe) : 0.0f;
                    float psum = simd_sum(p);
                    ps[r * 36u + lane] = half(p);
                    if (lane == 0u) {
                        m_s[r] = m_new;
                        l_s[r] = l_s[r] * alpha + psum;
                        alpha_s[r] = alpha;
                    }
                }
            }
            if (k_pos < last) {
                #pragma unroll
                for (uint d = 0u; d < PER_T; d += 4u) {
                    *(threadgroup half4 *)(kvs + kr * STRIDE + kt * PER_T + d) =
                        am_load4_half<FMT>(v_buf, kv_base + d);
                }
            } else {
                #pragma unroll
                for (uint d = 0u; d < PER_T; d += 4u) {
                    *(threadgroup half4 *)(kvs + kr * STRIDE + kt * PER_T + d) = half4(0.0h);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float a = alpha_s[qg * 8u + tile_row];
            #pragma unroll
            for (uint j = 0u; j < DTILES; j++) {
                o_acc[j].thread_elements()[0] *= a;
                o_acc[j].thread_elements()[1] *= a;
            }
            uint col0 = dg * (D / 4u);
            for (uint k8 = 0u; k8 < AM_BK; k8 += 8u) {
                simdgroup_half8x8 pa;
                simdgroup_load(pa, ps + (qg * 8u) * 36u + k8, 36u);
                #pragma unroll
                for (uint j = 0u; j < DTILES; j++) {
                    simdgroup_half8x8 vb;
                    simdgroup_load(vb, kvs + k8 * STRIDE + col0 + j * 8u, STRIDE);
                    simdgroup_multiply_accumulate(o_acc[j], pa, vb, o_acc[j]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint r = qg * 8u + tile_row;
        if (to_s[r] > 0u) {
            float inv = (l_s[r] > 0.0f) ? (1.0f / l_s[r]) : 0.0f;
            device float *row_out = out + (ulong(q0 + r) * heads + head) * D;
            uint cbase = dg * (D / 4u) + tile_col;
            #pragma unroll
            for (uint j = 0u; j < DTILES; j++) {
                row_out[cbase + j * 8u] = o_acc[j].thread_elements()[0] * inv;
                row_out[cbase + j * 8u + 1u] = o_acc[j].thread_elements()[1] * inv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define AM_DEC_ARGS                                                                          \
    device const float *q [[buffer(0)]],                                                     \
    device const uchar *k_buf [[buffer(1)]],                                                 \
    device const uchar *v_buf [[buffer(2)]],                                                 \
    device const uint *slots [[buffer(3)]],                                                  \
    device const uint *positions [[buffer(4)]],                                              \
    device const uint *page_table [[buffer(5)]],                                             \
    device float *out [[buffer(6)]],                                                         \
    constant float &scale [[buffer(7)]],                                                     \
    constant uint &rows [[buffer(8)]],                                                       \
    constant uint &keys [[buffer(9)]],                                                       \
    constant uint &heads [[buffer(10)]],                                                     \
    constant uint &kv_heads [[buffer(11)]],                                                  \
    constant uint &splits [[buffer(12)]],                                                    \
    constant uint &kept_mode [[buffer(13)]],                                                 \
    constant uint &kept_param [[buffer(14)]],                                                \
    constant uint &page_shift [[buffer(15)]],                                                \
    constant uint &page_mask [[buffer(16)]],                                                 \
    uint3 tg [[threadgroup_position_in_grid]],                                               \
    uint tid [[thread_index_in_threadgroup]]

#define AM_DEC_ENTRY(NAME, D, FMT)                                                           \
    kernel void NAME(AM_DEC_ARGS) {                                                          \
        threadgroup half sm_q[6][D];                                                         \
        threadgroup float sm_o[AM_WARPS][D];                                                 \
        threadgroup float sm_m[AM_WARPS];                                                    \
        threadgroup float sm_l[AM_WARPS];                                                    \
        am_dec_body<D, FMT>(                                                                 \
            q, k_buf, v_buf, slots, positions, page_table, out, scale, rows, keys, heads,    \
            kv_heads, splits, kept_mode, kept_param, page_shift, page_mask, tg.x, tid, sm_q, \
            sm_o, sm_m, sm_l);                                                               \
    }

#define AM_DEC_DIM(D) \
    AM_DEC_ENTRY(am_dec_##D##_f32, D##u, AM_F32) \
    AM_DEC_ENTRY(am_dec_##D##_bf16, D##u, AM_BF16) \
    AM_DEC_ENTRY(am_dec_##D##_f16, D##u, AM_F16) \
    AM_DEC_ENTRY(am_dec_##D##_fp8, D##u, AM_FP8)

AM_DEC_DIM(64)
AM_DEC_DIM(128)
AM_DEC_DIM(256)

#define AM_PF_ENTRY(NAME, D, FMT)                                                            \
    kernel void NAME(AM_DEC_ARGS) {                                                          \
        (void)splits;                                                                        \
        threadgroup half qs[AM_BQ * (D + 8u)];                                               \
        threadgroup half kvs[AM_BK * (D + 8u)];                                              \
        threadgroup float ss[AM_BQ * 36u];                                                   \
        threadgroup half ps[AM_BQ * 36u];                                                    \
        threadgroup float alpha_s[AM_BQ];                                                    \
        threadgroup float m_s[AM_BQ];                                                        \
        threadgroup float l_s[AM_BQ];                                                        \
        threadgroup uint to_s[AM_BQ];                                                        \
        am_pf_body<D, FMT>(                                                                  \
            q, k_buf, v_buf, slots, positions, page_table, out, scale, rows, keys, heads,    \
            kv_heads, kept_mode, kept_param, page_shift, page_mask, tg.x, tid, qs, kvs, ss,  \
            ps, alpha_s, m_s, l_s, to_s);                                                    \
    }

#define AM_PF_DIM(D) \
    AM_PF_ENTRY(am_pf_##D##_f32, D##u, AM_F32) \
    AM_PF_ENTRY(am_pf_##D##_bf16, D##u, AM_BF16) \
    AM_PF_ENTRY(am_pf_##D##_f16, D##u, AM_F16) \
    AM_PF_ENTRY(am_pf_##D##_fp8, D##u, AM_FP8)

AM_PF_DIM(64)
AM_PF_DIM(128)
AM_PF_DIM(256)
