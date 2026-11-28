#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

constant uint FS_BQ = 32u;
constant uint FS_BK = 32u;
constant uint FS_D = 128u;
constant uint FS_WIDE = 136u;
constant uint FS_NEAR = 40u;
constant uint FS_NONE = 0xFFFFFFFFu;

static inline float fs_floor() {
    return as_type<float>(0xff800000u);
}

// Rounds four floats to bf16 bits, ties to even.
static inline ushort4 fs_bf16(float4 v) {
    uint4 b = as_type<uint4>(v);
    return ushort4((b + uint4(0x7FFFu) + ((b >> uint4(16u)) & uint4(1u))) >> uint4(16u));
}

static inline void fs_put4(threadgroup ushort *at, ushort4 v) {
    at[0] = v.x;
    at[1] = v.y;
    at[2] = v.z;
    at[3] = v.w;
}

// Stages `live` rows of `src` as bf16, `step` apart.
static inline void fs_stage(
    threadgroup ushort *bits,
    device const float *src,
    ulong row0,
    uint step,
    uint live,
    uint tid) {
    for (uint i = tid; i < FS_BK * (FS_D / 4u); i += 128u) {
        uint t = i >> 5u;
        uint d = (i & 31u) << 2u;
        ushort4 v = ushort4(0u);
        if (t < live) {
            v = fs_bf16(*(device const float4 *)(src + (row0 + ulong(t) * step) * FS_D + d));
        }
        fs_put4(bits + t * FS_WIDE + d, v);
    }
}

// Ragged rows in a window, on bf16 matrix units.
kernel void flash_sunk_metal(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *values [[buffer(2)]],
    device const uint *slots [[buffer(3)]],
    device const uint *positions [[buffer(4)]],
    device const float *sinks [[buffer(5)]],
    device const float *rel_bias [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant uint &queries [[buffer(9)]],
    constant uint &keys [[buffer(10)]],
    constant uint &heads [[buffer(11)]],
    constant uint &kv_heads [[buffer(12)]],
    constant uint &before [[buffer(13)]],
    constant uint &after [[buffer(14)]],
    constant uint &sink_mode [[buffer(15)]],
    constant uint &rel_win [[buffer(16)]],
    constant uint &kept [[buffer(17)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat q_s[FS_BQ][FS_WIDE];
    threadgroup bfloat kv_s[FS_BK][FS_WIDE];
    threadgroup bfloat w_s[FS_BQ][FS_NEAR];
    threadgroup float p_s[FS_BQ][FS_NEAR];
    threadgroup float alpha_s[FS_BQ];
    threadgroup float l_s[FS_BQ];
    threadgroup uint from_s[FS_BQ];
    threadgroup uint to_s[FS_BQ];
    threadgroup uint pos_s[FS_BQ];
    threadgroup ushort *q_bits = (threadgroup ushort *)&q_s[0][0];
    threadgroup ushort *kv_bits = (threadgroup ushort *)&kv_s[0][0];
    threadgroup ushort *w_bits = (threadgroup ushort *)&w_s[0][0];

    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    uint head = tg.x % heads;
    uint q0 = (tg.x / heads) * FS_BQ;
    uint kvh = head / (heads / kv_heads);
    // The two tile elements this thread owns.
    uint er = ((lane >> 2u) & 4u) + ((lane >> 1u) & 3u);
    uint ec = ((lane >> 2u) & 2u) * 2u + (lane & 1u) * 2u;

    fs_stage(q_bits, q, ulong(q0) * heads + head, heads, min(FS_BQ, queries - q0), tid);

    float o[4][4][2];
    float m[8];
    float l[8];
    for (uint pass = 0u; pass < FS_BQ; pass++) {
        // The first row of a slot serves that slot.
        uint lead = q0 + pass;
        uint slot = FS_NONE;
        if (lead < queries) {
            slot = slots[lead];
            for (uint j = 0u; j < pass; j++) {
                if (slots[q0 + j] == slot) slot = FS_NONE;
            }
        }
        if (slot == FS_NONE) continue;

        if (tid < FS_BQ) {
            uint r = q0 + tid;
            bool mine = r < queries && slots[r] == slot;
            uint p = mine ? positions[r] : 0u;
            uint from = p > before ? p - before : 0u;
            uint to = (p < keys && keys - p - 1u > after) ? p + after + 1u : keys;
            pos_s[tid] = p;
            from_s[tid] = mine ? from : 0u;
            to_s[tid] = mine ? to : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint lo = keys;
        uint hi = 0u;
        for (uint r = 0u; r < FS_BQ; r++) {
            if (to_s[r] > from_s[r]) {
                lo = min(lo, from_s[r]);
                hi = max(hi, to_s[r]);
            }
        }
        for (uint i = 0u; i < 8u; i++) {
            m[i] = fs_floor();
            l[i] = 0.0f;
        }
        for (uint i = 0u; i < 4u; i++) {
            for (uint j = 0u; j < 4u; j++) {
                o[i][j][0] = 0.0f;
                o[i][j][1] = 0.0f;
            }
        }

        for (uint k0 = (lo / FS_BK) * FS_BK; k0 < hi; k0 += FS_BK) {
            // A ring of whole blocks keeps each block unbroken.
            ulong row0 = (ulong(slot) * kept + (k0 % kept)) * kv_heads + kvh;
            uint live = min(FS_BK, hi - k0);
            fs_stage(kv_bits, k, row0, kv_heads, live, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each SIMD group scores one 16x16 quadrant.
            uint sr = (simd & 1u) * 16u;
            uint sc = (simd >> 1u) * 16u;
            simdgroup_float8x8 s00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 s01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 s10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 s11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0u; d < FS_D; d += 8u) {
                // Key rows hold the width, so they load transposed.
                simdgroup_bfloat8x8 ta0, ta1, tb0, tb1;
                simdgroup_load(ta0, &q_s[sr][d], FS_WIDE);
                simdgroup_load(ta1, &q_s[sr + 8u][d], FS_WIDE);
                simdgroup_load(tb0, &kv_s[sc][d], FS_WIDE, ulong2(0ul), true);
                simdgroup_load(tb1, &kv_s[sc + 8u][d], FS_WIDE, ulong2(0ul), true);
                simdgroup_multiply_accumulate(s00, ta0, tb0, s00);
                simdgroup_multiply_accumulate(s01, ta0, tb1, s01);
                simdgroup_multiply_accumulate(s10, ta1, tb0, s10);
                simdgroup_multiply_accumulate(s11, ta1, tb1, s11);
            }
            simdgroup_store(s00, &p_s[sr][sc], ulong(FS_NEAR));
            simdgroup_store(s01, &p_s[sr][sc + 8u], ulong(FS_NEAR));
            simdgroup_store(s10, &p_s[sr + 8u][sc], ulong(FS_NEAR));
            simdgroup_store(s11, &p_s[sr + 8u][sc + 8u], ulong(FS_NEAR));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Values replace the keys while scores soften.
            fs_stage(kv_bits, values, row0, kv_heads, live, tid);
            uint t = k0 + lane;
            for (uint i = 0u; i < 8u; i++) {
                uint row = simd * 8u + i;
                uint p = pos_s[row];
                bool seen = t >= from_s[row] && t < to_s[row];
                float s = seen ? p_s[row][lane] * scale : fs_floor();
                if (seen && rel_win > 0u && t <= p && p - t < rel_win) {
                    ulong list = ulong(q0 + row) * heads + head;
                    s += rel_bias[list * rel_win + (p - t)];
                }
                float top = s;
                top = max(top, simd_shuffle_xor(top, 16));
                top = max(top, simd_shuffle_xor(top, 8));
                top = max(top, simd_shuffle_xor(top, 4));
                top = max(top, simd_shuffle_xor(top, 2));
                top = max(top, simd_shuffle_xor(top, 1));
                float m_new = max(m[i], top);
                float safe = (m_new == fs_floor()) ? 0.0f : m_new;
                float alpha = exp(m[i] - safe);
                float weight = exp(s - safe);
                float sum = weight;
                sum += simd_shuffle_xor(sum, 16);
                sum += simd_shuffle_xor(sum, 8);
                sum += simd_shuffle_xor(sum, 4);
                sum += simd_shuffle_xor(sum, 2);
                sum += simd_shuffle_xor(sum, 1);
                w_bits[row * FS_NEAR + lane] = fs_bf16(float4(weight, 0.0f, 0.0f, 0.0f)).x;
                l[i] = l[i] * alpha + sum;
                m[i] = m_new;
                if (lane == 0u) alpha_s[row] = alpha;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each SIMD group folds 32 output columns.
            uint oc = simd * 32u;
            simdgroup_float8x8 acc[4][4];
            for (uint i = 0u; i < 4u; i++) {
                for (uint j = 0u; j < 4u; j++) {
                    acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            for (uint k8 = 0u; k8 < FS_BK; k8 += 8u) {
                simdgroup_bfloat8x8 tv[4];
                for (uint j = 0u; j < 4u; j++) {
                    simdgroup_load(tv[j], &kv_s[k8][oc + j * 8u], FS_WIDE);
                }
                for (uint i = 0u; i < 4u; i++) {
                    simdgroup_bfloat8x8 tw;
                    simdgroup_load(tw, &w_s[i * 8u][k8], FS_NEAR);
                    for (uint j = 0u; j < 4u; j++) {
                        simdgroup_multiply_accumulate(acc[i][j], tw, tv[j], acc[i][j]);
                    }
                }
            }
            for (uint i = 0u; i < 4u; i++) {
                float alpha = alpha_s[i * 8u + er];
                for (uint j = 0u; j < 4u; j++) {
                    o[i][j][0] = o[i][j][0] * alpha + acc[i][j].thread_elements()[0];
                    o[i][j][1] = o[i][j][1] * alpha + acc[i][j].thread_elements()[1];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Folds any sink without overflowing or flushing subnormals.
        for (uint i = 0u; i < 8u; i++) {
            float f = 1.0f;
            int k = 0;
            uint r = q0 + simd * 8u + i;
            if (sink_mode > 0u && r < queries) {
                float sink = sinks[sink_mode == 1u ? head : r * heads + head];
                if (sink > fs_floor()) {
                    if (sink > m[i]) {
                        float d = m[i] - sink;
                        if (d < -174.0f) {
                            f = 0.0f;
                            l[i] = 1.0f;
                        } else {
                            k = int(ceil(d * 1.4426950408889634f));
                            f = exp(d - float(k) * 0.6931471805599453f);
                            int k1 = k >> 1;
                            l[i] = ldexp(ldexp(l[i] * f, k1), k - k1) + 1.0f;
                        }
                    } else {
                        l[i] += exp(sink - m[i]);
                    }
                }
            }
            if (lane == 0u) {
                alpha_s[simd * 8u + i] = f;
                pos_s[simd * 8u + i] = as_type<uint>(k);
                l_s[simd * 8u + i] = l[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0u; i < 4u; i++) {
            uint r = i * 8u + er;
            if (to_s[r] > from_s[r]) {
                float f = alpha_s[r];
                int k = as_type<int>(pos_s[r]);
                int k1 = k >> 1;
                int k2 = k - k1;
                ulong at = (ulong(q0 + r) * heads + head) * FS_D + simd * 32u + ec;
                for (uint j = 0u; j < 4u; j++) {
                    float v0 = (o[i][j][0] * f) / l_s[r];
                    float v1 = (o[i][j][1] * f) / l_s[r];
                    if (k < 0) {
                        v0 = ldexp(ldexp(v0, k1), k2);
                        v1 = ldexp(ldexp(v1, k1), k2);
                    }
                    out[at + j * 8u] = v0;
                    out[at + j * 8u + 1u] = v1;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
