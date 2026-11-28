#include <metal_stdlib>

using namespace metal;

constant uint DM_THREADS = 128u;
constant uint DM_DEC_THREADS = 256u;
constant uint DM_CHUNK = 8u;
constant uint CM_MAX_TAPS = 16u;
constant uint CM_BR = 16u;

template <typename V>
static inline V dm_group_sum(V x) {
    x += simd_shuffle_xor(x, ushort(4u));
    x += simd_shuffle_xor(x, ushort(8u));
    x += simd_shuffle_xor(x, ushort(16u));
    return x;
}

template <typename V, uint I_PER>
static inline V dm_step(
    thread V S[I_PER],
    threadgroup const float *k_row,
    threadgroup const float *q_row,
    V v_in,
    float fade,
    float beta,
    float scale) {
    float4 kv[I_PER / 4u];
    V p0 = V(0.0f), p1 = V(0.0f);
    #pragma unroll
    for (uint g = 0u; g < I_PER / 4u; g++) {
        float4 k4 = *(threadgroup const float4 *)(k_row + g * 4u);
        kv[g] = k4;
        S[g * 4u + 0u] *= fade;
        S[g * 4u + 1u] *= fade;
        S[g * 4u + 2u] *= fade;
        S[g * 4u + 3u] *= fade;
        p0 = fma(S[g * 4u + 0u], V(k4.x), fma(S[g * 4u + 1u], V(k4.y), p0));
        p1 = fma(S[g * 4u + 2u], V(k4.z), fma(S[g * 4u + 3u], V(k4.w), p1));
    }
    V chg = (v_in - dm_group_sum(p0 + p1)) * beta;

    V r0 = V(0.0f), r1 = V(0.0f);
    #pragma unroll
    for (uint g = 0u; g < I_PER / 4u; g++) {
        float4 k4 = kv[g];
        float4 q4 = *(threadgroup const float4 *)(q_row + g * 4u);
        S[g * 4u + 0u] = fma(V(k4.x), chg, S[g * 4u + 0u]);
        S[g * 4u + 1u] = fma(V(k4.y), chg, S[g * 4u + 1u]);
        S[g * 4u + 2u] = fma(V(k4.z), chg, S[g * 4u + 2u]);
        S[g * 4u + 3u] = fma(V(k4.w), chg, S[g * 4u + 3u]);
        r0 = fma(S[g * 4u + 0u], V(q4.x), fma(S[g * 4u + 1u], V(q4.y), r0));
        r1 = fma(S[g * 4u + 2u], V(q4.z), fma(S[g * 4u + 3u], V(q4.w), r1));
    }
    return dm_group_sum(r0 + r1) * scale;
}

template <uint DK>
struct DmSmem {
    uint first_sg[4];
    uint r[DM_CHUNK];
    uint pos[DM_CHUNK];
    float fade[DM_CHUNK];
    float rate[DM_CHUNK];
    float k[DM_CHUNK][DK];
    float q[DM_CHUNK][DK];
    uint cnt;
    uint next;
};

// Prefill and general ragged Gated DeltaNet recurrence.
template <uint DK>
static inline void dm_delta_body(
    device const float *q,
    device const float *k,
    device const float *v,
    device const float *decay,
    device const float *rate,
    device const uint *slots,
    device const uint *positions,
    device const float *state,
    device float *out,
    device float *keep,
    float scale,
    uint seqs,
    uint heads,
    uint kv_heads,
    uint width,
    uint rows,
    uint write_keep,
    uint tg_x,
    uint tid,
    threadgroup DmSmem<DK> &sm) {
    constexpr uint I_PER = DK / 8u;

    uint j_tiles = (width + 31u) >> 5u;
    uint j_tile = tg_x % j_tiles;
    uint sh = tg_x / j_tiles;
    uint s = sh / heads;
    uint h = sh % heads;
    if (s >= seqs || rows == 0u) return;

    uint sg = tid >> 5u;
    uint lane = tid & 31u;

    // Skip slots with no active rows.
    uint my_first = rows;
    for (uint r = tid; r < rows; r += DM_THREADS) {
        if (slots[r] == s) {
            my_first = r;
            break;
        }
    }
    uint sg_first = simd_min(my_first);
    if (lane == 0u) sm.first_sg[sg] = sg_first;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cursor = min(min(sm.first_sg[0], sm.first_sg[1]), min(sm.first_sg[2], sm.first_sg[3]));
    if (cursor >= rows) return;

    uint kh = h / (heads / kv_heads);
    uint j_sub = lane & 3u;
    uint i_part = lane >> 2u;
    uint i_base = i_part * I_PER;
    uint j_base = j_tile * 32u + sg * 8u;
    bool active_j = (j_base < width);
    uint j_col = j_base + j_sub * 2u;
    ulong state_base = ulong(sh) * ulong(DK) * ulong(width);

    float2 S[I_PER];
    bool loaded = false;
    uint prev = 0xFFFFFFFFu;
    float2 last2 = float2(0.0f);

    while (cursor < rows) {
        if (tid == 0u) {
            uint cnt = 0u;
            uint r = cursor;
            while (r < rows && cnt < DM_CHUNK) {
                if (slots[r] == s) {
                    sm.r[cnt] = r;
                    sm.pos[cnt] = positions[r];
                    cnt++;
                }
                r++;
            }
            sm.cnt = cnt;
            sm.next = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint cnt = sm.cnt;
        cursor = sm.next;
        if (cnt == 0u) break;

        if (tid < cnt) {
            ulong rh = ulong(sm.r[tid]) * ulong(heads) + ulong(h);
            sm.fade[tid] = exp(decay[rh]);
            sm.rate[tid] = rate[rh];
        }

        constexpr uint DK4 = DK / 4u;
        if (lane < DK4) {
            for (uint c = sg; c < cnt; c += 4u) {
                ulong kq_off = ulong(sm.r[c] * kv_heads + kh) * ulong(DK) + ulong(lane * 4u);
                *(threadgroup float4 *)(&sm.k[c][lane * 4u]) = *(device const float4 *)(k + kq_off);
                *(threadgroup float4 *)(&sm.q[c][lane * 4u]) = *(device const float4 *)(q + kq_off);
            }
        }

        if (!loaded) {
            loaded = true;
            if (active_j) {
                if (sm.pos[0] == 0u) {
                    #pragma unroll
                    for (uint u = 0u; u < I_PER; u++) S[u] = float2(0.0f);
                } else {
                    #pragma unroll
                    for (uint u = 0u; u < I_PER; u++) {
                        S[u] = *(device const float2 *)(state + state_base + ulong(i_base + u) * ulong(width) + ulong(j_col));
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_j) {
            for (uint c = 0u; c < cnt; c++) {
                uint r = sm.r[c];
                uint pos = sm.pos[c];
                if ((c > 0u || prev != 0xFFFFFFFFu) && pos == 0u && prev != 0u) {
                    #pragma unroll
                    for (uint u = 0u; u < I_PER; u++) S[u] = float2(0.0f);
                }
                ulong vo = ulong(r * heads + h) * ulong(width) + ulong(j_col);
                if (pos == prev) {
                    if (i_part == 0u) *(device float2 *)(out + vo) = last2;
                } else {
                    float2 v2 = *(device const float2 *)(v + vo);
                    last2 = dm_step<float2, I_PER>(
                        S, &sm.k[c][i_base], &sm.q[c][i_base], v2, sm.fade[c], sm.rate[c], scale);
                    prev = pos;
                    if (i_part == 0u) *(device float2 *)(out + vo) = last2;
                }
            }
        }
        if (cursor < rows) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (write_keep != 0u && loaded && active_j) {
        #pragma unroll
        for (uint u = 0u; u < I_PER; u++) {
            *(device float2 *)(keep + state_base + ulong(i_base + u) * ulong(width) + ulong(j_col)) = S[u];
        }
    }
}

// Coalesced float4 decode kernel for 128-wide keys.
kernel void dm_dec_128(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device const float *decay [[buffer(3)]],
    device const float *rate [[buffer(4)]],
    device const uint *slots [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    device const float *state [[buffer(7)]],
    device float *out [[buffer(8)]],
    device float *keep [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    constant uint &seqs [[buffer(11)]],
    constant uint &heads [[buffer(12)]],
    constant uint &kv_heads [[buffer(13)]],
    constant uint &width [[buffer(14)]],
    constant uint &rows [[buffer(15)]],
    constant uint &write_keep [[buffer(16)]],
    uint tg_x [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint sm_r[16];
    threadgroup uint sm_pos[16];
    threadgroup float sm_fade[16];
    threadgroup float sm_rate[16];
    threadgroup float sm_k[16][128];
    threadgroup float sm_q[16][128];
    threadgroup uint sm_cnt;

    uint j_tiles = (width + 127u) >> 7u;
    uint j_tile = tg_x % j_tiles;
    uint sh = tg_x / j_tiles;
    uint s = sh / heads;
    uint h = sh % heads;
    if (s >= seqs || rows == 0u) return;

    if (tid == 0u) {
        uint cnt = 0u;
        for (uint r = 0u; r < rows && cnt < 16u; r++) {
            if (slots[r] == s) {
                sm_r[cnt] = r;
                sm_pos[cnt] = positions[r];
                cnt++;
            }
        }
        sm_cnt = cnt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cnt = sm_cnt;
    if (cnt == 0u) return;

    uint sg = tid >> 5u;
    uint lane = tid & 31u;
    uint kh = h / (heads / kv_heads);

    if (tid < cnt) {
        ulong rh = ulong(sm_r[tid]) * ulong(heads) + ulong(h);
        sm_fade[tid] = exp(decay[rh]);
        sm_rate[tid] = rate[rh];
    }
    for (uint c = sg; c < cnt; c += 8u) {
        ulong kq_off = ulong(sm_r[c] * kv_heads + kh) * 128ull + ulong(lane * 4u);
        *(threadgroup float4 *)(&sm_k[c][lane * 4u]) = *(device const float4 *)(k + kq_off);
        *(threadgroup float4 *)(&sm_q[c][lane * 4u]) = *(device const float4 *)(q + kq_off);
    }

    uint j_sub = lane & 3u;
    uint i_part = lane >> 2u;
    uint i_base = i_part * 16u;
    uint j_col = j_tile * 128u + sg * 16u + j_sub * 4u;
    bool active_j = (j_col < width);
    ulong state_base = ulong(sh) * 128ull * ulong(width);

    float4 S[16];
    if (active_j) {
        if (sm_pos[0] == 0u) {
            #pragma unroll
            for (uint u = 0u; u < 16u; u++) S[u] = float4(0.0f);
        } else {
            #pragma unroll
            for (uint u = 0u; u < 16u; u++) {
                S[u] = *(device const float4 *)(state + state_base + ulong(i_base + u) * ulong(width) + ulong(j_col));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!active_j) return;

    uint prev = 0xFFFFFFFFu;
    float4 last4 = float4(0.0f);
    for (uint c = 0u; c < cnt; c++) {
        uint r = sm_r[c];
        uint pos = sm_pos[c];
        if (c > 0u && pos == 0u && prev != 0u) {
            #pragma unroll
            for (uint u = 0u; u < 16u; u++) S[u] = float4(0.0f);
        }
        ulong vo = ulong(r * heads + h) * ulong(width) + ulong(j_col);
        if (pos == prev) {
            if (i_part == 0u) *(device float4 *)(out + vo) = last4;
        } else {
            float4 v4 = *(device const float4 *)(v + vo);
            last4 = dm_step<float4, 16u>(
                S, &sm_k[c][i_base], &sm_q[c][i_base], v4, sm_fade[c], sm_rate[c], scale);
            prev = pos;
            if (i_part == 0u) *(device float4 *)(out + vo) = last4;
        }
    }

    if (write_keep != 0u) {
        #pragma unroll
        for (uint u = 0u; u < 16u; u++) {
            *(device float4 *)(keep + state_base + ulong(i_base + u) * ulong(width) + ulong(j_col)) = S[u];
        }
    }
}

#define DM_DELTA_ENTRY(NAME, DK)                                                   \
kernel void NAME(                                                                  \
    device const float *q [[buffer(0)]],                                           \
    device const float *k [[buffer(1)]],                                           \
    device const float *v [[buffer(2)]],                                           \
    device const float *decay [[buffer(3)]],                                       \
    device const float *rate [[buffer(4)]],                                        \
    device const uint *slots [[buffer(5)]],                                        \
    device const uint *positions [[buffer(6)]],                                    \
    device const float *state [[buffer(7)]],                                       \
    device float *out [[buffer(8)]],                                               \
    device float *keep [[buffer(9)]],                                              \
    constant float &scale [[buffer(10)]],                                          \
    constant uint &seqs [[buffer(11)]],                                            \
    constant uint &heads [[buffer(12)]],                                           \
    constant uint &kv_heads [[buffer(13)]],                                        \
    constant uint &width [[buffer(14)]],                                           \
    constant uint &rows [[buffer(15)]],                                            \
    constant uint &write_keep [[buffer(16)]],                                      \
    uint tg_x [[threadgroup_position_in_grid]],                                    \
    uint tid [[thread_index_in_threadgroup]]) {                                    \
    threadgroup DmSmem<DK> sm;                                                     \
    dm_delta_body<DK>(                                                             \
        q, k, v, decay, rate, slots, positions, state, out, keep,                  \
        scale, seqs, heads, kv_heads, width, rows, write_keep, tg_x, tid, sm);     \
}

DM_DELTA_ENTRY(dm_delta_64, 64u)
DM_DELTA_ENTRY(dm_delta_128, 128u)

// Native Metal causal conv1d for f32 and bf16 weights.
static inline float cm_load_w(device const uchar *w_raw, ulong idx, uint w_bf16) {
    if (w_bf16) {
        ushort bits = ((device const ushort *)w_raw)[idx];
        return as_type<float>(uint(bits) << 16u);
    }
    return ((device const float *)w_raw)[idx];
}

template <uint MT>
static inline void cm_load_taps(
    device const uchar *w_raw,
    uint c0,
    uint taps,
    uint w_bf16,
    thread float4 tw[MT]) {
    if constexpr (MT == 4u) {
        if (taps == 4u) {
            float4 r0, r1, r2, r3;
            if (w_bf16) {
                const device ushort4 *wb = (const device ushort4 *)w_raw;
                ushort4 u0 = wb[c0 + 0u], u1 = wb[c0 + 1u], u2 = wb[c0 + 2u], u3 = wb[c0 + 3u];
                r0 = as_type<float4>(uint4(u0) << 16u);
                r1 = as_type<float4>(uint4(u1) << 16u);
                r2 = as_type<float4>(uint4(u2) << 16u);
                r3 = as_type<float4>(uint4(u3) << 16u);
            } else {
                const device float4 *wf = (const device float4 *)w_raw;
                r0 = wf[c0 + 0u];
                r1 = wf[c0 + 1u];
                r2 = wf[c0 + 2u];
                r3 = wf[c0 + 3u];
            }
            tw[0] = float4(r0.x, r1.x, r2.x, r3.x);
            tw[1] = float4(r0.y, r1.y, r2.y, r3.y);
            tw[2] = float4(r0.z, r1.z, r2.z, r3.z);
            tw[3] = float4(r0.w, r1.w, r2.w, r3.w);
            return;
        }
    }
    for (uint j = 0u; j < taps; j++) {
        tw[j] = float4(
            cm_load_w(w_raw, ulong(c0 + 0u) * ulong(taps) + j, w_bf16),
            cm_load_w(w_raw, ulong(c0 + 1u) * ulong(taps) + j, w_bf16),
            cm_load_w(w_raw, ulong(c0 + 2u) * ulong(taps) + j, w_bf16),
            cm_load_w(w_raw, ulong(c0 + 3u) * ulong(taps) + j, w_bf16));
    }
}

template <uint MT>
static inline float4 cm_step(
    float4 cur,
    thread float4 hist[MT],
    thread const float4 tw[MT],
    uint taps,
    uint dilation,
    uint hist_len) {
    if constexpr (MT == 4u) {
        if (taps == 4u && dilation == 1u) {
            float4 sum = fma(hist[0], tw[0], fma(hist[1], tw[1], fma(hist[2], tw[2], cur * tw[3])));
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = cur;
            return sum;
        }
    }
    float4 sum = cur * tw[taps - 1u];
    for (uint j = 0u; j < taps - 1u; j++) {
        sum = fma(hist[j * dilation], tw[j], sum);
    }
    for (uint k = 0u; k < hist_len - 1u; k++) hist[k] = hist[k + 1u];
    hist[hist_len - 1u] = cur;
    return sum;
}

template <uint MT>
static inline void cm_build_hist(
    device const float *x,
    device const uint *slots,
    device const uint *positions,
    device const float *state,
    uint s,
    uint r_start,
    uint c0,
    uint width,
    uint hist_len,
    bool include_cur,
    thread float4 hist[MT],
    thread uint &eff_r) {
    uint rec_r[MT + 1u];
    uint rec_pos[MT + 1u];
    uint want = include_cur ? hist_len : (hist_len + 1u);
    uint cnt = 0u;
    bool hit_zero = false;

    for (int p = int(r_start); p >= 0; p--) {
        if (slots[p] != s) continue;
        uint pp = positions[p];
        if (cnt > 0u && pp == rec_pos[cnt - 1u]) {
            rec_r[cnt - 1u] = uint(p);
        } else {
            if (cnt > 0u && rec_pos[cnt - 1u] == 0u) {
                hit_zero = true;
                break;
            }
            if (cnt == want) break;
            rec_r[cnt] = uint(p);
            rec_pos[cnt] = pp;
            cnt++;
        }
    }
    if (cnt > 0u && rec_pos[cnt - 1u] == 0u) {
        hit_zero = true;
    }

    uint skip = include_cur ? 0u : 1u;
    eff_r = (cnt > 0u) ? rec_r[0] : r_start;
    uint have_hist = (cnt > skip) ? (cnt - skip) : 0u;
    if (have_hist > hist_len) have_hist = hist_len;

    for (uint k = 0u; k < have_hist; k++) {
        uint rr = rec_r[skip + k];
        hist[hist_len - 1u - k] = *(device const float4 *)(x + ulong(rr) * ulong(width) + ulong(c0));
    }
    uint need = hist_len - have_hist;
    if (need > 0u) {
        if (hit_zero) {
            for (uint k = 0u; k < need; k++) hist[k] = float4(0.0f);
        } else {
            ulong sbase = ulong(s) * ulong(hist_len) * ulong(width) + ulong(c0);
            for (uint k = 0u; k < need; k++) {
                hist[k] = *(device const float4 *)(state + sbase + ulong(k + have_hist) * ulong(width));
            }
        }
    }
}

template <uint MT>
static inline void cm_conv_body(
    device const float *x,
    device const uchar *w_raw,
    device const uint *slots,
    device const uint *positions,
    device const float *state,
    device float *out,
    device float *keep,
    uint seqs,
    uint width,
    uint taps,
    uint dilation,
    uint rows,
    uint mode,
    uint w_bf16,
    uint write_keep,
    uint2 tg,
    uint tid,
    uint tg_threads,
    threadgroup int tg_last_sg[8]) {
    uint hist_len = (taps - 1u) * dilation;
    uint c0 = (tg.x * tg_threads + tid) * 4u;
    if (rows == 0u) return;

    if (mode == 2u) {
        uint s = tg.y;
        if (s >= seqs || c0 >= width) return;
        bool any = false;
        for (uint r = 0u; r < rows; r++) {
            if (slots[r] == s) {
                any = true;
                break;
            }
        }
        if (!any) return;

        float4 tw[MT];
        cm_load_taps<MT>(w_raw, c0, taps, w_bf16, tw);

        ulong sbase = ulong(s) * ulong(hist_len) * ulong(width) + ulong(c0);
        float4 hist[MT];
        bool loaded = false;
        uint prev = 0xFFFFFFFFu;
        float4 last4 = float4(0.0f);

        for (uint r = 0u; r < rows; r++) {
            if (slots[r] != s) continue;
            uint pos = positions[r];
            if (!loaded) {
                loaded = true;
                if (pos == 0u) {
                    for (uint k = 0u; k < hist_len; k++) hist[k] = float4(0.0f);
                } else {
                    for (uint k = 0u; k < hist_len; k++) {
                        hist[k] = *(device const float4 *)(state + sbase + ulong(k) * ulong(width));
                    }
                }
            } else if (pos == 0u && prev != 0u) {
                for (uint k = 0u; k < hist_len; k++) hist[k] = float4(0.0f);
            }
            ulong at = ulong(r) * ulong(width) + ulong(c0);
            if (pos != prev) {
                float4 cur = *(device const float4 *)(x + at);
                last4 = cm_step<MT>(cur, hist, tw, taps, dilation, hist_len);
                prev = pos;
            }
            *(device float4 *)(out + at) = last4;
        }
        if (write_keep != 0u && loaded) {
            for (uint k = 0u; k < hist_len; k++) {
                *(device float4 *)(keep + sbase + ulong(k) * ulong(width)) = hist[k];
            }
        }
        return;
    }

    if (mode == 0u) {
        if (c0 >= width) return;
        uint r0 = tg.y * CM_BR;
        if (r0 >= rows) return;
        uint r1 = min(rows, r0 + CM_BR);

        float4 tw[MT];
        cm_load_taps<MT>(w_raw, c0, taps, w_bf16, tw);

        uint cur_s = 0xFFFFFFFFu;
        uint prev = 0xFFFFFFFFu;
        float4 hist[MT];
        float4 last4 = float4(0.0f);

        for (uint r = r0; r < r1; r++) {
            uint s = slots[r];
            uint pos = positions[r];
            ulong at = ulong(r) * ulong(width) + ulong(c0);
            if (s != cur_s) {
                uint eff_r = r;
                cm_build_hist<MT>(x, slots, positions, state, s, r, c0, width, hist_len, false, hist, eff_r);
                float4 cur = *(device const float4 *)(x + ulong(eff_r) * ulong(width) + ulong(c0));
                last4 = cm_step<MT>(cur, hist, tw, taps, dilation, hist_len);
                prev = pos;
                cur_s = s;
            } else {
                if (pos == 0u && prev != 0u) {
                    for (uint k = 0u; k < hist_len; k++) hist[k] = float4(0.0f);
                }
                if (pos != prev) {
                    float4 cur = *(device const float4 *)(x + at);
                    last4 = cm_step<MT>(cur, hist, tw, taps, dilation, hist_len);
                    prev = pos;
                }
            }
            *(device float4 *)(out + at) = last4;
        }
        return;
    }

    // Mode 1 writes each slot's final kept history.
    if (write_keep == 0u) return;
    uint s = tg.y;
    if (s >= seqs) return;
    int my_last = -1;
    for (int r = int(rows) - 1 - int(tid); r >= 0; r -= int(tg_threads)) {
        if (slots[r] == s) {
            my_last = r;
            break;
        }
    }
    int sg_last = simd_max(my_last);
    uint n_sg = tg_threads >> 5u;
    if ((tid & 31u) == 0u) tg_last_sg[tid >> 5u] = sg_last;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int r_last = -1;
    for (uint i = 0u; i < n_sg; i++) r_last = max(r_last, tg_last_sg[i]);
    if (r_last < 0 || c0 >= width) return;

    float4 hist[MT];
    uint eff_r = uint(r_last);
    cm_build_hist<MT>(x, slots, positions, state, s, uint(r_last), c0, width, hist_len, true, hist, eff_r);
    ulong sbase = ulong(s) * ulong(hist_len) * ulong(width) + ulong(c0);
    for (uint k = 0u; k < hist_len; k++) {
        *(device float4 *)(keep + sbase + ulong(k) * ulong(width)) = hist[k];
    }
}

#define CM_CONV_ENTRY(NAME, MT)                                                    \
kernel void NAME(                                                                  \
    device const float *x [[buffer(0)]],                                           \
    device const uchar *w_raw [[buffer(1)]],                                       \
    device const uint *slots [[buffer(2)]],                                        \
    device const uint *positions [[buffer(3)]],                                    \
    device const float *state [[buffer(4)]],                                       \
    device float *out [[buffer(5)]],                                               \
    device float *keep [[buffer(6)]],                                              \
    constant uint &seqs [[buffer(7)]],                                             \
    constant uint &width [[buffer(8)]],                                            \
    constant uint &taps [[buffer(9)]],                                             \
    constant uint &dilation [[buffer(10)]],                                        \
    constant uint &rows [[buffer(11)]],                                            \
    constant uint &mode [[buffer(12)]],                                            \
    constant uint &w_bf16 [[buffer(13)]],                                          \
    constant uint &write_keep [[buffer(14)]],                                      \
    uint2 tg [[threadgroup_position_in_grid]],                                     \
    uint tid [[thread_index_in_threadgroup]],                                      \
    uint2 tg_threads [[threads_per_threadgroup]]) {                                \
    threadgroup int tg_last_sg[8];                                                 \
    cm_conv_body<MT>(                                                              \
        x, w_raw, slots, positions, state, out, keep, seqs, width, taps,           \
        dilation, rows, mode, w_bf16, write_keep, tg, tid, tg_threads.x,           \
        tg_last_sg);                                                               \
}

CM_CONV_ENTRY(cm_conv_t4, 4u)
CM_CONV_ENTRY(cm_conv_rows, CM_MAX_TAPS)
