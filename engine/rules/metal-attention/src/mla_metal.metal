constant uint MLA_MMA_THREADS = 512u;
constant uint MLA_GEN_THREADS = 256u;
constant float MLA_LOG2E = 1.4426950408889634f;
constant uint MLA_D = 576u;
constant uint MLA_STRIDE = 584u;

static inline uint mla_pick_id(
    device const uint *ids,
    uint row,
    uint picks,
    uint from_list,
    uint from,
    uint to,
    uint p,
    uint p_end) {
    if (p >= p_end) return 0xFFFFFFFFu;
    uint c = (from_list == 1u) ? ids[ulong(row) * picks + p] : p;
    return (c >= from && c < to) ? c : 0xFFFFFFFFu;
}

static inline void mla_store2(device uchar *base, ulong elem_off, float2 v, uint out_fmt) {
    if (out_fmt == KV_F32) {
        *(device float2 *)(base + (elem_off << 2u)) = v;
    } else if (out_fmt == KV_BF16) {
        *(device ushort2 *)(base + (elem_off << 1u)) = ushort2(kv_f32_to_bf16(v.x), kv_f32_to_bf16(v.y));
    } else {
        *(device half2 *)(base + (elem_off << 1u)) = half2(v);
    }
}

static inline void mla_store4(device uchar *base, ulong elem_off, float4 v, uint out_fmt) {
    if (out_fmt == KV_F32) {
        *(device float4 *)(base + (elem_off << 2u)) = v;
    } else if (out_fmt == KV_BF16) {
        *(device ushort4 *)(base + (elem_off << 1u)) = ushort4(
            kv_f32_to_bf16(v.x), kv_f32_to_bf16(v.y), kv_f32_to_bf16(v.z), kv_f32_to_bf16(v.w));
    } else {
        *(device half4 *)(base + (elem_off << 1u)) = half4(v);
    }
}

static inline void mla_init_state(
    thread float &m,
    thread float &l,
    device const float *sinks,
    uint has_sinks,
    uint split,
    uint h) {
    if (has_sinks != 0u && split == 0u) {
        float sv = sinks[h];
        m = (sv > -INFINITY) ? (sv * MLA_LOG2E) : -INFINITY;
        l = (sv > -INFINITY) ? 1.0f : 0.0f;
    } else {
        m = -INFINITY;
        l = 0.0f;
    }
}

static inline uint mla_find_stop(
    device const uint *ids,
    uint row,
    uint keys,
    uint picks,
    uint from_list,
    uint to) {
    uint stop = (from_list == 0u && to < picks) ? to : picks;
    if (from_list == 1u && picks > 0u && ids[ulong(row) * picks + picks - 1u] >= keys) {
        uint lo = 0u;
        uint hi = picks;
        while (lo < hi) {
            uint mid = lo + ((hi - lo) >> 1u);
            if (ids[ulong(row) * picks + mid] < keys) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        stop = lo;
    }
    return stop;
}

template <uint NTH>
static inline void mla_store_empty(
    device uchar *out_buf,
    device const float *sinks,
    uint has_sinks,
    uint out_fmt,
    uint row,
    uint heads,
    uint h0,
    uint nh,
    uint out_width,
    uint split,
    uint splits,
    uint tid) {
    if (splits == 1u) {
        uint quads = out_width >> 2u;
        for (uint idx = tid; idx < nh * quads; idx += NTH) {
            uint h = idx / quads;
            uint q4 = idx - h * quads;
            ulong off = (ulong(row) * heads + h0 + h) * out_width + ulong(q4) * 4ull;
            mla_store4(out_buf, off, float4(0.0f), out_fmt);
        }
    } else {
        device float *part = (device float *)out_buf;
        ulong stride = ulong(out_width + 2u);
        if (tid < nh) {
            float m, l;
            mla_init_state(m, l, sinks, has_sinks, split, h0 + tid);
            ulong ob = ((ulong(row) * heads + h0 + tid) * splits + split) * stride;
            part[ob] = m;
            part[ob + 1u] = l;
        }
        if (has_sinks != 0u && split == 0u) {
            uint quads = out_width >> 2u;
            for (uint idx = tid; idx < nh * quads; idx += NTH) {
                uint h = idx / quads;
                uint q4 = idx - h * quads;
                ulong ob = ((ulong(row) * heads + h0 + h) * splits + split) * stride + 2ull;
                *(device float2 *)(part + ob + ulong(q4) * 4ull) = float2(0.0f);
                *(device float2 *)(part + ob + ulong(q4) * 4ull + 2ull) = float2(0.0f);
            }
        }
    }
}

template <uint BK, typename T>
struct MlaMmaSmem {
    T kv_s[BK * MLA_STRIDE];
    float s_red[8][16][BK];
    T p_s[16u * BK];
    float m_s[16];
    float l_s[16];
    float corr_s[16];
    uint valid_s[BK];
    uint stop_s;
};

template <uint MAX_W>
struct MlaGenSmem {
    float kv_s[4u * MAX_W];
    uint valid_s[4];
    uint stop_s;
};

struct MlaCtx {
    device const uchar *q_buf;
    device const uchar *kv_buf;
    device const uint *ids;
    device const uint *slots;
    device const uint *positions;
    device const float *sinks;
    device const uint *page_table;
    device uchar *out_buf;
    float scale;
    uint rows, keys, heads, width, out_width, picks, before, after, splits;
    uint kept_param, page_shift, page_mask, flags;
};

struct MlaRange {
    uint row, h0, nh, slot, from, to, p_begin, p_end;
    uint kept_mode, from_list, has_sinks, q_fmt, out_fmt;
    bool active;
};

static inline MlaRange mla_setup_mma(
    thread const MlaCtx &c,
    uint bk,
    uint3 tg,
    uint tid,
    threadgroup float *m_s,
    threadgroup float *l_s,
    threadgroup uint &stop_s) {
    MlaRange r;
    r.kept_mode = c.flags & 3u;
    r.from_list = (c.flags >> 2u) & 1u;
    r.has_sinks = (c.flags >> 3u) & 1u;
    r.q_fmt = (c.flags >> 4u) & 3u;
    r.out_fmt = (c.flags >> 6u) & 3u;
    uint hgroups = (c.heads + 15u) >> 4u;
    uint split = tg.x;
    r.row = tg.y / hgroups;
    uint hg = tg.y - r.row * hgroups;
    r.active = (r.row < c.rows && split < c.splits);
    if (!r.active) return r;
    r.h0 = hg << 4u;
    r.nh = min(16u, c.heads - r.h0);
    r.slot = c.slots[r.row];
    uint pos = c.positions[r.row];
    r.from = (pos > c.before) ? (pos - c.before) : 0u;
    r.to = (pos + c.after < c.keys) ? (pos + c.after + 1u) : c.keys;
    uint start = (r.from_list == 0u) ? r.from : 0u;
    if (tid == 0u) {
        stop_s = mla_find_stop(c.ids, r.row, c.keys, c.picks, r.from_list, r.to);
    }
    if (tid < 16u) {
        if (tid < r.nh) {
            float m, l;
            mla_init_state(m, l, c.sinks, r.has_sinks, split, r.h0 + tid);
            m_s[tid] = m;
            l_s[tid] = l;
        } else {
            m_s[tid] = -INFINITY;
            l_s[tid] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint stop = stop_s;
    uint total_p = (stop > start) ? (stop - start) : 0u;
    uint tiles = (total_p + bk - 1u) / bk;
    uint per_tiles = (tiles + c.splits - 1u) / c.splits;
    r.p_begin = min(start + split * per_tiles * bk, stop);
    r.p_end = min(r.p_begin + per_tiles * bk, stop);
    return r;
}

template <uint NT, uint OUT_W>
static inline void mla_store_mma(
    thread const MlaCtx &c,
    thread const simdgroup_float8x8 (&o_acc)[NT],
    threadgroup const float *m_s,
    threadgroup const float *l_s,
    thread const MlaRange &r,
    uint split,
    uint qg,
    uint col_base,
    uint tile_row,
    uint tile_col,
    uint tid) {
    uint h = qg * 8u + tile_row;
    if (c.splits == 1u) {
        if (h < r.nh) {
            float inv = (l_s[h] > 0.0f) ? (1.0f / l_s[h]) : 0.0f;
            ulong ob = (ulong(r.row) * c.heads + r.h0 + h) * OUT_W + col_base + tile_col;
            for (uint j = 0u; j < NT; j++) {
                float2 v = float2(
                    o_acc[j].thread_elements()[0] * inv, o_acc[j].thread_elements()[1] * inv);
                mla_store2(c.out_buf, ob + j * 8u, v, r.out_fmt);
            }
        }
    } else {
        device float *part = (device float *)c.out_buf;
        ulong stride = ulong(OUT_W + 2u);
        if (h < r.nh) {
            ulong ob = ((ulong(r.row) * c.heads + r.h0 + h) * c.splits + split) * stride + 2ull
                + col_base + tile_col;
            for (uint j = 0u; j < NT; j++) {
                *(device float2 *)(part + ob + j * 8u) =
                    float2(o_acc[j].thread_elements()[0], o_acc[j].thread_elements()[1]);
            }
        }
        if (tid < r.nh) {
            ulong ob = ((ulong(r.row) * c.heads + r.h0 + tid) * c.splits + split) * stride;
            part[ob] = m_s[tid];
            part[ob + 1u] = l_s[tid];
        }
    }
}

template <uint BK, typename PT>
static inline void mla_softmax_tile(
    threadgroup const float (&s_red)[8][16][BK],
    threadgroup const uint *valid_s,
    threadgroup float *m_s,
    threadgroup float *l_s,
    threadgroup float *corr_s,
    threadgroup PT *p_s,
    uint tid) {
    if (tid >= 16u * BK) return;
    uint sh = (BK == 16u) ? (tid >> 4u) : (tid >> 3u);
    uint sk = tid & (BK - 1u);
    bool ok = (valid_s[sk] != 0u);
    float score = -INFINITY;
    if (ok) {
        score = s_red[0][sh][sk] + s_red[1][sh][sk] + s_red[2][sh][sk] + s_red[3][sh][sk]
            + s_red[4][sh][sk] + s_red[5][sh][sk] + s_red[6][sh][sk] + s_red[7][sh][sk];
    }
    float tmax = score;
    tmax = max(tmax, simd_shuffle_xor(tmax, 1u));
    tmax = max(tmax, simd_shuffle_xor(tmax, 2u));
    tmax = max(tmax, simd_shuffle_xor(tmax, 4u));
    if constexpr (BK == 16u) {
        tmax = max(tmax, simd_shuffle_xor(tmax, 8u));
    }
    float old_m = m_s[sh];
    float old_l = l_s[sh];
    float next_m = max(old_m, tmax);
    float safe = (next_m == -INFINITY) ? 0.0f : next_m;
    float corr = (old_l > 0.0f) ? exp2(old_m - safe) : 0.0f;
    float pv = ok ? exp2(score - safe) : 0.0f;
    float psum = pv;
    psum += simd_shuffle_xor(psum, 1u);
    psum += simd_shuffle_xor(psum, 2u);
    psum += simd_shuffle_xor(psum, 4u);
    if constexpr (BK == 16u) {
        psum += simd_shuffle_xor(psum, 8u);
    }
    p_s[sh * BK + sk] = PT(pv);
    if (sk == 0u) {
        m_s[sh] = next_m;
        l_s[sh] = old_l * corr + psum;
        corr_s[sh] = corr;
    }
}

// Grouped-Q MMA MLA kernel.
template <uint KV_FMT, uint OUT_W, uint BK, typename T, typename T4, typename Mat8>
static inline void mla_mma_body(
    thread const MlaCtx &c,
    uint3 tg,
    uint tid,
    threadgroup MlaMmaSmem<BK, T> &sm) {
    constexpr uint NT = OUT_W >> 6u;

    MlaRange r = mla_setup_mma(c, BK, tg, tid, sm.m_s, sm.l_s, sm.stop_s);
    if (!r.active) return;
    if (r.p_begin >= r.p_end) {
        mla_store_empty<MLA_MMA_THREADS>(
            c.out_buf, c.sinks, r.has_sinks, r.out_fmt, r.row, c.heads, r.h0, r.nh, OUT_W, tg.x,
            c.splits, tid);
        return;
    }

    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    uint qg = simd >> 3u;
    uint dg = simd & 7u;
    uint tile_row = ((lane >> 2u) & 4u) + ((lane >> 1u) & 3u);
    uint tile_col = (((lane >> 2u) & 2u) << 1u) + ((lane & 1u) << 1u);
    float qscale = c.scale * MLA_LOG2E;

    Mat8 q_mat[9];
    if constexpr (BK == 16u) {
        if (simd < r.nh) {
            ulong qb = (ulong(r.row) * c.heads + r.h0 + simd) * MLA_D;
            for (uint q4 = lane; q4 < 144u; q4 += 32u) {
                *(threadgroup T4 *)(sm.kv_s + simd * MLA_STRIDE + q4 * 4u) =
                    T4(kv_load4_plain(c.q_buf, qb + q4 * 4u, r.q_fmt) * qscale);
            }
        } else {
            for (uint q4 = lane; q4 < 144u; q4 += 32u) {
                *(threadgroup T4 *)(sm.kv_s + simd * MLA_STRIDE + q4 * 4u) = T4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0u; s < 9u; s++) {
            uint d = dg * 72u + s * 8u;
            simdgroup_load(q_mat[s], sm.kv_s + (qg * 8u) * MLA_STRIDE + d, MLA_STRIDE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        uint kr = tid >> 6u;
        uint kt = tid & 63u;
        for (uint half_g = 0u; half_g < 2u; half_g++) {
            uint hr = half_g * 8u + kr;
            if (hr < r.nh) {
                ulong qb = (ulong(r.row) * c.heads + r.h0 + hr) * MLA_D;
                for (uint q4 = kt; q4 < 144u; q4 += 64u) {
                    *(threadgroup T4 *)(sm.kv_s + kr * MLA_STRIDE + q4 * 4u) =
                        T4(kv_load4_plain(c.q_buf, qb + q4 * 4u, r.q_fmt) * qscale);
                }
            } else {
                for (uint q4 = kt; q4 < 144u; q4 += 64u) {
                    *(threadgroup T4 *)(sm.kv_s + kr * MLA_STRIDE + q4 * 4u) = T4(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (qg == half_g) {
                for (uint s = 0u; s < 9u; s++) {
                    uint d = dg * 72u + s * 8u;
                    simdgroup_load(q_mat[s], sm.kv_s + d, MLA_STRIDE);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    simdgroup_float8x8 o_acc[NT];
    for (uint j = 0u; j < NT; j++) {
        o_acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint kr = (BK == 16u) ? simd : (tid >> 6u);
    uint kt = (BK == 16u) ? lane : (tid & 63u);
    uint kstep = (BK == 16u) ? 32u : 64u;
    uint col_base = dg * (NT * 8u);
    for (uint p0 = r.p_begin; p0 < r.p_end; p0 += BK) {
        uint j_id =
            mla_pick_id(c.ids, r.row, c.picks, r.from_list, r.from, r.to, p0 + kr, r.p_end);
        if (kt == 0u) {
            sm.valid_s[kr] = (j_id != 0xFFFFFFFFu) ? 1u : 0u;
        }
        if (j_id != 0xFFFFFFFFu) {
            uint kv_row = kv_kept_row(
                r.slot, j_id, r.kept_mode, c.kept_param, c.page_shift, c.page_mask, c.page_table);
            ulong kv_base = ulong(kv_row) * MLA_D;
            for (uint q4 = kt; q4 < 144u; q4 += kstep) {
                if constexpr (BK == 16u) {
                    *(threadgroup T4 *)(sm.kv_s + kr * MLA_STRIDE + q4 * 4u) =
                        kv_load4_h4<KV_FMT>(c.kv_buf, kv_base + q4 * 4u);
                } else {
                    *(threadgroup T4 *)(sm.kv_s + kr * MLA_STRIDE + q4 * 4u) =
                        kv_load4_f32<KV_FMT>(c.kv_buf, kv_base + q4 * 4u);
                }
            }
        } else {
            for (uint q4 = kt; q4 < 144u; q4 += kstep) {
                *(threadgroup T4 *)(sm.kv_s + kr * MLA_STRIDE + q4 * 4u) = T4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint s = 0u; s < 9u; s++) {
            uint d = dg * 72u + s * 8u;
            Mat8 kb0;
            simdgroup_load(kb0, sm.kv_s + d, MLA_STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s0, q_mat[s], kb0, s0);
            if constexpr (BK == 16u) {
                Mat8 kb1;
                simdgroup_load(kb1, sm.kv_s + 8u * MLA_STRIDE + d, MLA_STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s1, q_mat[s], kb1, s1);
            }
        }
        simdgroup_store(s0, &sm.s_red[dg][qg * 8u][0u], BK);
        if constexpr (BK == 16u) {
            simdgroup_store(s1, &sm.s_red[dg][qg * 8u][8u], BK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mla_softmax_tile<BK, T>(sm.s_red, sm.valid_s, sm.m_s, sm.l_s, sm.corr_s, sm.p_s, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float row_corr = sm.corr_s[qg * 8u + tile_row];
        for (uint j = 0u; j < NT; j++) {
            o_acc[j].thread_elements()[0] *= row_corr;
            o_acc[j].thread_elements()[1] *= row_corr;
        }
        Mat8 p0m, p1m;
        simdgroup_load(p0m, sm.p_s + (qg * 8u) * BK, BK);
        if constexpr (BK == 16u) {
            simdgroup_load(p1m, sm.p_s + (qg * 8u) * BK + 8u, BK);
        }
        for (uint j = 0u; j < NT; j++) {
            uint col = col_base + j * 8u;
            Mat8 v0;
            simdgroup_load(v0, sm.kv_s + col, MLA_STRIDE);
            simdgroup_multiply_accumulate(o_acc[j], p0m, v0, o_acc[j]);
            if constexpr (BK == 16u) {
                Mat8 v1;
                simdgroup_load(v1, sm.kv_s + 8u * MLA_STRIDE + col, MLA_STRIDE);
                simdgroup_multiply_accumulate(o_acc[j], p1m, v1, o_acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    mla_store_mma<NT, OUT_W>(
        c, o_acc, sm.m_s, sm.l_s, r, tg.x, qg, col_base, tile_row, tile_col, tid);
}

// General grouped-Q FP32 MLA up to MAX_W.
template <uint KV_FMT, uint MAX_W>
static inline void mla_gen_body(
    device const uchar *q_buf,
    device const uchar *kv_buf,
    device const uint *ids,
    device const uint *slots,
    device const uint *positions,
    device const float *sinks,
    device const uint *page_table,
    device uchar *out_buf,
    float scale,
    uint rows,
    uint keys,
    uint heads,
    uint width,
    uint out_width,
    uint picks,
    uint before,
    uint after,
    uint splits,
    uint kept_param,
    uint page_shift,
    uint page_mask,
    uint flags,
    uint3 tg,
    uint tid,
    threadgroup MlaGenSmem<MAX_W> &sm) {
    constexpr uint BK = 4u;
    constexpr uint MAX_QUADS = MAX_W / 128u;

    uint kept_mode = flags & 3u;
    uint from_list = (flags >> 2u) & 1u;
    uint has_sinks = (flags >> 3u) & 1u;
    uint q_fmt = (flags >> 4u) & 3u;
    uint out_fmt = (flags >> 6u) & 3u;

    uint hgroups = (heads + 7u) >> 3u;
    uint split = tg.x;
    uint row = tg.y / hgroups;
    uint hg = tg.y - row * hgroups;
    if (row >= rows || split >= splits) return;
    uint h0 = hg << 3u;
    uint nh = min(8u, heads - h0);

    uint slot = slots[row];
    uint pos = positions[row];
    uint from = (pos > before) ? (pos - before) : 0u;
    uint to = (pos + after < keys) ? (pos + after + 1u) : keys;
    uint start = (from_list == 0u) ? from : 0u;

    threadgroup float *kv_s = sm.kv_s;
    threadgroup uint *valid_s = sm.valid_s;

    if (tid == 0u) {
        sm.stop_s = mla_find_stop(ids, row, keys, picks, from_list, to);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint stop = sm.stop_s;
    uint total_p = (stop > start) ? (stop - start) : 0u;
    uint tiles = (total_p + BK - 1u) / BK;
    uint per_tiles = (tiles + splits - 1u) / splits;
    uint p_begin = min(start + split * per_tiles * BK, stop);
    uint p_end = min(p_begin + per_tiles * BK, stop);
    if (p_begin >= p_end) {
        mla_store_empty<MLA_GEN_THREADS>(
            out_buf, sinks, has_sinks, out_fmt, row, heads, h0, nh, out_width, split, splits, tid);
        return;
    }

    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    uint kr = tid >> 6u;
    uint kt = tid & 63u;
    uint quads = width >> 2u;
    float qscale = scale * MLA_LOG2E;

    float4 qr[MAX_QUADS];
    float4 acc[MAX_QUADS];
    float m_val = -INFINITY;
    float l_val = 0.0f;
    if (simd < nh) {
        mla_init_state(m_val, l_val, sinks, has_sinks, split, h0 + simd);
        ulong qb = (ulong(row) * heads + h0 + simd) * width;
        #pragma unroll
        for (uint s = 0u; s < MAX_QUADS; s++) {
            uint d = s * 128u + lane * 4u;
            qr[s] = (d < width) ? (kv_load4_plain(q_buf, qb + d, q_fmt) * qscale) : float4(0.0f);
            acc[s] = float4(0.0f);
        }
    }

    for (uint p0 = p_begin; p0 < p_end; p0 += BK) {
        uint j_id = mla_pick_id(ids, row, picks, from_list, from, to, p0 + kr, p_end);
        if (kt == 0u) {
            valid_s[kr] = (j_id != 0xFFFFFFFFu) ? 1u : 0u;
        }
        if (j_id != 0xFFFFFFFFu) {
            uint kv_row =
                kv_kept_row(slot, j_id, kept_mode, kept_param, page_shift, page_mask, page_table);
            ulong kv_base = ulong(kv_row) * width;
            for (uint q4 = kt; q4 < quads; q4 += 64u) {
                *(threadgroup float4 *)(kv_s + kr * MAX_W + q4 * 4u) =
                    kv_load4_f32<KV_FMT>(kv_buf, kv_base + q4 * 4u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd < nh) {
            #pragma unroll
            for (uint k = 0u; k < BK; k++) {
                if (valid_s[k] == 0u) continue;
                threadgroup const float *krow = kv_s + k * MAX_W;
                float dsum = 0.0f;
                #pragma unroll
                for (uint s = 0u; s < MAX_QUADS; s++) {
                    uint d = s * 128u + lane * 4u;
                    if (d < width) {
                        dsum += dot(qr[s], *(threadgroup const float4 *)(krow + d));
                    }
                }
                float score = simd_sum(dsum);
                float next_m = max(m_val, score);
                float corr = (l_val > 0.0f) ? exp2(m_val - next_m) : 0.0f;
                float wt = exp2(score - next_m);
                m_val = next_m;
                l_val = l_val * corr + wt;
                #pragma unroll
                for (uint s = 0u; s < MAX_QUADS; s++) {
                    uint d = s * 128u + lane * 4u;
                    if (d < out_width) {
                        float4 vv = *(threadgroup const float4 *)(krow + d);
                        acc[s] = fma(vv, float4(wt), acc[s] * corr);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd >= nh) return;
    if (splits == 1u) {
        float inv = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        ulong ob = (ulong(row) * heads + h0 + simd) * out_width;
        #pragma unroll
        for (uint s = 0u; s < MAX_QUADS; s++) {
            uint d = s * 128u + lane * 4u;
            if (d < out_width) {
                mla_store4(out_buf, ob + d, acc[s] * inv, out_fmt);
            }
        }
    } else {
        device float *part = (device float *)out_buf;
        ulong stride = ulong(out_width + 2u);
        ulong ob = ((ulong(row) * heads + h0 + simd) * splits + split) * stride;
        if (lane == 0u) {
            part[ob] = m_val;
            part[ob + 1u] = l_val;
        }
        #pragma unroll
        for (uint s = 0u; s < MAX_QUADS; s++) {
            uint d = s * 128u + lane * 4u;
            if (d < out_width) {
                *(device float2 *)(part + ob + 2ull + d) = float2(acc[s].x, acc[s].y);
                *(device float2 *)(part + ob + 2ull + d + 2u) = float2(acc[s].z, acc[s].w);
            }
        }
    }
}

// Combines split-K MLA partials into final outputs.
kernel void mla_reduce(
    device const float *part [[buffer(0)]],
    device uchar *out_buf [[buffer(1)]],
    constant uint &lists [[buffer(2)]],
    constant uint &splits [[buffer(3)]],
    constant uint &out_width [[buffer(4)]],
    constant uint &out_fmt [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint list = tg.x;
    if (list >= lists) return;
    ulong stride = ulong(out_width + 2u);
    ulong base = ulong(list) * splits * stride;

    float best = -INFINITY;
    for (uint s = 0u; s < splits; s++) {
        float l = part[base + ulong(s) * stride + 1u];
        if (l > 0.0f) {
            best = max(best, part[base + ulong(s) * stride]);
        }
    }
    float denom = 0.0f;
    for (uint s = 0u; s < splits; s++) {
        float l = part[base + ulong(s) * stride + 1u];
        if (l > 0.0f) {
            denom += l * exp2(part[base + ulong(s) * stride] - best);
        }
    }
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    ulong ob = ulong(list) * out_width;
    for (uint d = tid; d < out_width; d += MLA_GEN_THREADS) {
        float sum = 0.0f;
        for (uint s = 0u; s < splits; s++) {
            float l = part[base + ulong(s) * stride + 1u];
            if (l > 0.0f) {
                float sc = exp2(part[base + ulong(s) * stride] - best);
                sum += part[base + ulong(s) * stride + 2u + d] * sc;
            }
        }
        kv_store1(out_buf, ob + d, sum * inv, out_fmt);
    }
}

#define MLA_ARGS                                                                             \
    device const uchar *q_buf [[buffer(0)]],                                                 \
    device const uchar *kv_buf [[buffer(1)]],                                                \
    device const uint *ids [[buffer(2)]],                                                    \
    device const uint *slots [[buffer(3)]],                                                  \
    device const uint *positions [[buffer(4)]],                                              \
    device const float *sinks [[buffer(5)]],                                                 \
    device const uint *page_table [[buffer(6)]],                                             \
    device uchar *out_buf [[buffer(7)]],                                                     \
    constant float &scale [[buffer(8)]],                                                     \
    constant uint &rows [[buffer(9)]],                                                       \
    constant uint &keys [[buffer(10)]],                                                      \
    constant uint &heads [[buffer(11)]],                                                     \
    constant uint &width [[buffer(12)]],                                                     \
    constant uint &out_width [[buffer(13)]],                                                 \
    constant uint &picks [[buffer(14)]],                                                     \
    constant uint &before [[buffer(15)]],                                                    \
    constant uint &after [[buffer(16)]],                                                     \
    constant uint &splits [[buffer(17)]],                                                    \
    constant uint &kept_param [[buffer(18)]],                                                \
    constant uint &page_shift [[buffer(19)]],                                                \
    constant uint &page_mask [[buffer(20)]],                                                 \
    constant uint &flags [[buffer(21)]],                                                     \
    uint3 tg [[threadgroup_position_in_grid]],                                               \
    uint tid [[thread_index_in_threadgroup]]

#define MLA_BUILD_CTX                                                                        \
    MlaCtx c = {                                                                             \
        q_buf, kv_buf, ids, slots, positions, sinks, page_table, out_buf, scale, rows, keys, \
        heads, width, out_width, picks, before, after, splits, kept_param, page_shift,       \
        page_mask, flags                                                                     \
    }

#define MLA_MMA_HALF_ENTRY(NAME, FMT, OUT_W)                                                 \
    kernel void NAME(MLA_ARGS) {                                                             \
        MLA_BUILD_CTX;                                                                       \
        threadgroup MlaMmaSmem<16u, half> sm;                                                \
        mla_mma_body<FMT, OUT_W, 16u, half, half4, simdgroup_half8x8>(c, tg, tid, sm);       \
    }

#define MLA_MMA_F32_ENTRY(NAME, FMT, OUT_W)                                                  \
    kernel void NAME(MLA_ARGS) {                                                             \
        MLA_BUILD_CTX;                                                                       \
        threadgroup MlaMmaSmem<8u, float> sm;                                                \
        mla_mma_body<FMT, OUT_W, 8u, float, float4, simdgroup_float8x8>(c, tg, tid, sm);     \
    }

#define MLA_GEN_ENTRY(NAME, FMT, MAX_W)                                                      \
    kernel void NAME(MLA_ARGS) {                                                             \
        threadgroup MlaGenSmem<MAX_W> sm;                                                    \
        mla_gen_body<FMT, MAX_W>(                                                            \
            q_buf, kv_buf, ids, slots, positions, sinks, page_table, out_buf, scale, rows,   \
            keys, heads, width, out_width, picks, before, after, splits, kept_param,         \
            page_shift, page_mask, flags, tg, tid, sm);                                      \
    }

MLA_MMA_F32_ENTRY(mla_mma_512_f32, KV_F32, 512u)
MLA_MMA_HALF_ENTRY(mla_mma_512_bf16, KV_BF16, 512u)
MLA_MMA_HALF_ENTRY(mla_mma_512_f16, KV_F16, 512u)
MLA_MMA_HALF_ENTRY(mla_mma_512_fp8, KV_FP8, 512u)

MLA_MMA_F32_ENTRY(mla_mma_576_f32, KV_F32, 576u)
MLA_MMA_HALF_ENTRY(mla_mma_576_bf16, KV_BF16, 576u)
MLA_MMA_HALF_ENTRY(mla_mma_576_f16, KV_F16, 576u)
MLA_MMA_HALF_ENTRY(mla_mma_576_fp8, KV_FP8, 576u)

MLA_GEN_ENTRY(mla_gen_256_f32, KV_F32, 256u)
MLA_GEN_ENTRY(mla_gen_256_bf16, KV_BF16, 256u)
MLA_GEN_ENTRY(mla_gen_256_f16, KV_F16, 256u)
MLA_GEN_ENTRY(mla_gen_256_fp8, KV_FP8, 256u)

MLA_GEN_ENTRY(mla_gen_1024_f32, KV_F32, 1024u)
MLA_GEN_ENTRY(mla_gen_1024_bf16, KV_BF16, 1024u)
MLA_GEN_ENTRY(mla_gen_1024_f16, KV_F16, 1024u)
MLA_GEN_ENTRY(mla_gen_1024_fp8, KV_FP8, 1024u)
