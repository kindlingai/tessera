constant uint IDX_WARPS = 4u;
constant uint IDX_THREADS = 128u;
constant uint IDX_MAX_TILE = 256u;

static inline float idx_load1_plain(device const uchar *base, ulong elem_off, uint fmt) {
    if (fmt == KV_F32) {
        return *(device const float *)(base + (elem_off << 2u));
    }
    if (fmt == KV_BF16) {
        ushort raw = *(device const ushort *)(base + (elem_off << 1u));
        return as_type<float>(uint(raw) << 16u);
    }
    return float(*(device const half *)(base + (elem_off << 1u)));
}

template <uint HQ, uint BK, uint STRIDE, typename T>
struct IdxSmem {
    T q_s[HQ * STRIDE];
    T k_s[IDX_WARPS * BK * STRIDE];
    float w_s[HQ];
    float acc_s[IDX_MAX_TILE];
};

// Simdgroup-matrix Lightning Indexer scoring kernel.
template <uint K_FMT, uint HQ, uint BK, uint STRIDE, typename T, typename T4, typename Mat8>
static inline void idx_mma_body(
    device const uchar *q_buf,
    device const uchar *k_buf,
    device const uchar *w_buf,
    device const uint *slots,
    device const uint *positions,
    device const uint *page_table,
    device uchar *out_buf,
    uint rows,
    uint out_keys,
    uint live_keys,
    uint heads,
    uint dim,
    uint tile_keys,
    uint kept_param,
    uint page_shift,
    uint page_mask,
    uint queries,
    uint past,
    uint back,
    uint ahead,
    uint flags,
    uint3 tg,
    uint tid,
    threadgroup IdxSmem<HQ, BK, STRIDE, T> &sm) {
    uint kept_mode = flags & 3u;
    uint ragged = (flags >> 2u) & 1u;
    uint masked = (flags >> 3u) & 1u;
    uint q_fmt = (flags >> 4u) & 3u;
    uint w_fmt = (flags >> 6u) & 3u;
    uint out_fmt = (flags >> 8u) & 3u;

    uint row = tg.y;
    uint j_begin = tg.x * tile_keys;
    uint j_end = min(j_begin + tile_keys, out_keys);
    if (row >= rows || j_begin >= j_end) return;

    uint slot = 0u;
    uint from = 0u;
    uint to = out_keys;
    if (ragged == 1u) {
        slot = slots[row];
        to = live_keys;
        if (masked == 1u) {
            uint pos = positions[row];
            if (pos < to) to = pos + 1u;
        }
    } else {
        slot = row / queries;
        if (masked == 1u) {
            uint pos = past + (row % queries);
            if (pos > back) from = pos - back;
            if (out_keys > pos && out_keys - pos - 1u > ahead) to = pos + ahead + 1u;
        }
    }

    ulong out_row = ulong(row) * out_keys;
    if (j_begin >= to || j_end <= from) {
        for (uint j = j_begin + tid; j < j_end; j += IDX_THREADS) {
            kv_store1(out_buf, out_row + j, -INFINITY, out_fmt);
        }
        return;
    }

    bool multi = (heads > HQ);
    if (multi) {
        uint span = j_end - j_begin;
        for (uint idx = tid; idx < span; idx += IDX_THREADS) {
            sm.acc_s[idx] = 0.0f;
        }
    }

    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    uint tile_row = ((lane >> 2u) & 4u) + ((lane >> 1u) & 3u);
    uint tile_col = (((lane >> 2u) & 2u) << 1u) + ((lane & 1u) << 1u);
    uint q4_dim = dim >> 2u;
    threadgroup T *k_mine = sm.k_s + simd * (BK * STRIDE);

    for (uint hc = 0u; hc < heads; hc += HQ) {
        uint nh = min(HQ, heads - hc);
        uint nh_pad = (nh + 7u) & ~7u;
        if (tid < nh) {
            sm.w_s[tid] = idx_load1_plain(w_buf, ulong(row) * heads + hc + tid, w_fmt);
        } else if (tid < HQ) {
            sm.w_s[tid] = 0.0f;
        }
        uint tot_q = nh_pad * q4_dim;
        for (uint idx = tid; idx < tot_q; idx += IDX_THREADS) {
            uint h = idx / q4_dim;
            uint q4 = idx - h * q4_dim;
            T4 qv = T4(0);
            if (h < nh) {
                ulong qoff = (ulong(row) * heads + hc + h) * dim + ulong(q4) * 4ull;
                qv = T4(kv_load4_plain(q_buf, qoff, q_fmt));
            }
            *(threadgroup T4 *)(sm.q_s + h * STRIDE + q4 * 4u) = qv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j0 = j_begin + simd * BK; j0 < j_end; j0 += IDX_WARPS * BK) {
            if (j0 >= to || j0 + BK <= from) {
                if (!multi && lane < BK && j0 + lane < j_end) {
                    kv_store1(out_buf, out_row + j0 + lane, -INFINITY, out_fmt);
                }
                continue;
            }
            uint tot_k = BK * q4_dim;
            for (uint idx = lane; idx < tot_k; idx += 32u) {
                uint kr = idx / q4_dim;
                uint q4 = idx - kr * q4_dim;
                uint j = j0 + kr;
                T4 kv = T4(0);
                if (j >= from && j < to && j < j_end) {
                    uint krow = (ragged == 1u)
                        ? kv_kept_row(slot, j, kept_mode, kept_param, page_shift, page_mask, page_table)
                        : (slot * out_keys + j);
                    ulong koff = ulong(krow) * dim + ulong(q4) * 4ull;
                    if constexpr (sizeof(T) == 2u) {
                        kv = kv_load4_h4<K_FMT>(k_buf, koff);
                    } else {
                        kv = kv_load4_f32<K_FMT>(k_buf, koff);
                    }
                }
                *(threadgroup T4 *)(k_mine + kr * STRIDE + q4 * 4u) = kv;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            for (uint h0 = 0u; h0 < nh_pad; h0 += 8u) {
                simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint d = 0u; d < dim; d += 8u) {
                    Mat8 qa, kb0;
                    simdgroup_load(qa, sm.q_s + h0 * STRIDE + d, STRIDE);
                    simdgroup_load(kb0, k_mine + d, STRIDE, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s0, qa, kb0, s0);
                    if constexpr (BK == 16u) {
                        Mat8 kb1;
                        simdgroup_load(kb1, k_mine + 8u * STRIDE + d, STRIDE, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(s1, qa, kb1, s1);
                    }
                }
                uint h = h0 + tile_row;
                float wh = (h < nh) ? sm.w_s[h] : 0.0f;
                acc0 += max(s0.thread_elements()[0], 0.0f) * wh;
                acc1 += max(s0.thread_elements()[1], 0.0f) * wh;
                if constexpr (BK == 16u) {
                    acc2 += max(s1.thread_elements()[0], 0.0f) * wh;
                    acc3 += max(s1.thread_elements()[1], 0.0f) * wh;
                }
            }

            // Sum across the 8 tile_row lanes sharing each tile_col.
            acc0 += simd_shuffle_xor(acc0, 2u);
            acc0 += simd_shuffle_xor(acc0, 4u);
            acc0 += simd_shuffle_xor(acc0, 16u);
            acc1 += simd_shuffle_xor(acc1, 2u);
            acc1 += simd_shuffle_xor(acc1, 4u);
            acc1 += simd_shuffle_xor(acc1, 16u);
            if constexpr (BK == 16u) {
                acc2 += simd_shuffle_xor(acc2, 2u);
                acc2 += simd_shuffle_xor(acc2, 4u);
                acc2 += simd_shuffle_xor(acc2, 16u);
                acc3 += simd_shuffle_xor(acc3, 2u);
                acc3 += simd_shuffle_xor(acc3, 4u);
                acc3 += simd_shuffle_xor(acc3, 16u);
            }

            if (tile_row == 0u) {
                uint k0 = j0 + tile_col;
                uint k1 = k0 + 1u;
                if (multi) {
                    if (k0 < j_end) sm.acc_s[k0 - j_begin] += acc0;
                    if (k1 < j_end) sm.acc_s[k1 - j_begin] += acc1;
                    if constexpr (BK == 16u) {
                        uint k2 = j0 + 8u + tile_col;
                        uint k3 = k2 + 1u;
                        if (k2 < j_end) sm.acc_s[k2 - j_begin] += acc2;
                        if (k3 < j_end) sm.acc_s[k3 - j_begin] += acc3;
                    }
                } else {
                    if (k0 < j_end) {
                        float v = (k0 >= from && k0 < to) ? acc0 : -INFINITY;
                        kv_store1(out_buf, out_row + k0, v, out_fmt);
                    }
                    if (k1 < j_end) {
                        float v = (k1 >= from && k1 < to) ? acc1 : -INFINITY;
                        kv_store1(out_buf, out_row + k1, v, out_fmt);
                    }
                    if constexpr (BK == 16u) {
                        uint k2 = j0 + 8u + tile_col;
                        uint k3 = k2 + 1u;
                        if (k2 < j_end) {
                            float v = (k2 >= from && k2 < to) ? acc2 : -INFINITY;
                            kv_store1(out_buf, out_row + k2, v, out_fmt);
                        }
                        if (k3 < j_end) {
                            float v = (k3 >= from && k3 < to) ? acc3 : -INFINITY;
                            kv_store1(out_buf, out_row + k3, v, out_fmt);
                        }
                    }
                }
            }
        }
        if (multi) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (multi) {
        uint span = j_end - j_begin;
        for (uint idx = tid; idx < span; idx += IDX_THREADS) {
            uint j = j_begin + idx;
            float v = (j >= from && j < to) ? sm.acc_s[idx] : -INFINITY;
            kv_store1(out_buf, out_row + j, v, out_fmt);
        }
    }
}

#define IDX_ARGS                                                                             \
    device const uchar *q_buf [[buffer(0)]],                                                 \
    device const uchar *k_buf [[buffer(1)]],                                                 \
    device const uchar *w_buf [[buffer(2)]],                                                 \
    device const uint *slots [[buffer(3)]],                                                  \
    device const uint *positions [[buffer(4)]],                                              \
    device const uint *page_table [[buffer(5)]],                                             \
    device uchar *out_buf [[buffer(6)]],                                                     \
    constant uint &rows [[buffer(7)]],                                                       \
    constant uint &out_keys [[buffer(8)]],                                                   \
    constant uint &live_keys [[buffer(9)]],                                                  \
    constant uint &heads [[buffer(10)]],                                                     \
    constant uint &dim [[buffer(11)]],                                                       \
    constant uint &tile_keys [[buffer(12)]],                                                 \
    constant uint &kept_param [[buffer(13)]],                                                \
    constant uint &page_shift [[buffer(14)]],                                                \
    constant uint &page_mask [[buffer(15)]],                                                 \
    constant uint &queries [[buffer(16)]],                                                   \
    constant uint &past [[buffer(17)]],                                                      \
    constant uint &back [[buffer(18)]],                                                      \
    constant uint &ahead [[buffer(19)]],                                                     \
    constant uint &flags [[buffer(20)]],                                                     \
    uint3 tg [[threadgroup_position_in_grid]],                                               \
    uint tid [[thread_index_in_threadgroup]]

#define IDX_HALF_ENTRY(NAME, FMT, HQ, BK, STRIDE)                                            \
    kernel void NAME(IDX_ARGS) {                                                             \
        threadgroup IdxSmem<HQ, BK, STRIDE, half> sm;                                        \
        idx_mma_body<FMT, HQ, BK, STRIDE, half, half4, simdgroup_half8x8>(                   \
            q_buf, k_buf, w_buf, slots, positions, page_table, out_buf, rows, out_keys,      \
            live_keys, heads, dim, tile_keys, kept_param, page_shift, page_mask, queries,    \
            past, back, ahead, flags, tg, tid, sm);                                          \
    }

#define IDX_F32_ENTRY(NAME, FMT, HQ, BK, STRIDE)                                             \
    kernel void NAME(IDX_ARGS) {                                                             \
        threadgroup IdxSmem<HQ, BK, STRIDE, float> sm;                                       \
        idx_mma_body<FMT, HQ, BK, STRIDE, float, float4, simdgroup_float8x8>(                \
            q_buf, k_buf, w_buf, slots, positions, page_table, out_buf, rows, out_keys,      \
            live_keys, heads, dim, tile_keys, kept_param, page_shift, page_mask, queries,    \
            past, back, ahead, flags, tg, tid, sm);                                          \
    }

IDX_F32_ENTRY(idx_128_f32, KV_F32, 16u, 8u, 136u)
IDX_HALF_ENTRY(idx_128_bf16, KV_BF16, 32u, 16u, 136u)
IDX_HALF_ENTRY(idx_128_f16, KV_F16, 32u, 16u, 136u)
IDX_HALF_ENTRY(idx_128_fp8, KV_FP8, 32u, 16u, 136u)

IDX_F32_ENTRY(idx_256_f32, KV_F32, 8u, 8u, 264u)
IDX_HALF_ENTRY(idx_256_bf16, KV_BF16, 16u, 8u, 264u)
IDX_HALF_ENTRY(idx_256_f16, KV_F16, 16u, 8u, 264u)
IDX_HALF_ENTRY(idx_256_fp8, KV_FP8, 16u, 8u, 264u)
