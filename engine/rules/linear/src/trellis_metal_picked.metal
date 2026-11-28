// Weight blocks of one picked matrix for output `ob`.
template <uint K>
static inline TmWalk tm_picked_walk(
    device const uchar *w, device const uchar *mat, uint n, uint k, uint ob) {
    uint side = *(device const uint *)(w + 8u);
    bool input_side = (side & TM_SIDE_MASK) == TM_INPUT_SIDE;
    ulong local = input_side ? (ulong(ob) * 64u) : (ulong(ob) * (k >> 1u));
    ulong stride = (input_side ? ulong(n >> 1u) : 64ull) * ulong(32u * K);
    return {k >> 7u, 0u, stride, mat + local * (32u * K)};
}

// Matrix `e` of a set, and its code width.
static inline device const uchar *tm_picked_mat(
    device const uchar *w, uint n, uint k, uint bits, uint e, thread uint &ek) {
    if (bits == 0u) {
        device const uchar *meta = w + TM_MIX_TABLE + ulong(e) * TM_MIX_ROW;
        ek = *(device const uint *)(meta + 8u);
        return w + *(device const ulong *)meta;
    }
    ek = bits;
    uint side = *(device const uint *)(w + 8u);
    if (side & TM_PAGED) return w + *(device const ulong *)(w + TM_TABLE + ulong(e) * 8u);
    return w + TM_HEAD + ulong(e) * ulong(k >> 4u) * ulong(n >> 4u) * ulong(32u * bits);
}

// One output row and block on its picked matrix.
template <uint K, uint BOOK, bool TURNED>
static inline void tm_picked_body(
    device const float *x,
    device const uchar *w,
    device const uchar *mat,
    device const float *su,
    device const float *sv,
    device const float *extra,
    device float *out,
    uint n,
    uint k,
    uint repeat,
    uint post,
    uint e,
    uint ob,
    uint r,
    uint tid,
    threadgroup float xs[4][128]) {
    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    uint cc = lane >> 2u;
    uint rr = lane & 3u;
    TmWalk walk = tm_picked_walk<K>(w, mat, n, k, ob);
    TmWordMap wm = tm_word_map<K, BOOK>(lane);
    device const float *xr = x + ulong(r / repeat) * k;

    float acc0 = 0.0f, acc1 = 0.0f;
    tm_dec1_run<K, BOOK>(xr, su + ulong(e) * k, TURNED, walk, wm, simd, lane, xs, acc0, acc1);

    acc0 += simd_shuffle_xor(acc0, ushort(1u));
    acc0 += simd_shuffle_xor(acc0, ushort(2u));
    acc1 += simd_shuffle_xor(acc1, ushort(1u));
    acc1 += simd_shuffle_xor(acc1, ushort(2u));
    uint lcol = (simd << 4u) + cc;
    if constexpr (!TURNED) {
        if (rr == 0u) {
            device float *row = out + ulong(r) * n + (ob << 7u);
            row[lcol] = acc0;
            row[lcol + 8u] = acc1;
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rr == 0u) {
            xs[0][lcol] = acc0;
            xs[0][lcol + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            uint base = (ob << 7u) + (lane << 2u);
            float4 sc = *(device const float4 *)(sv + ulong(e) * n + base);
            float4 y = tm_turn_out(tm_get4(&xs[0][lane << 2u]), sc, lane);
            if (post == 1u) {
                y *= *(device const float4 *)(extra + base);
            } else if (post == 2u) {
                float4 ex = *(device const float4 *)(extra + ulong(r) * n + base);
                y = (y / (1.0f + exp(-y))) * ex;
            }
            *(device float4 *)(out + ulong(r) * n + base) = y;
        }
    }
}

#define TM_PICKED_ARGS                                                                       \
    device const float *x [[buffer(0)]],                                                     \
    device const uchar *w [[buffer(1)]],                                                     \
    device const uint *ids [[buffer(2)]],                                                    \
    device const float *su [[buffer(3)]],                                                    \
    device const float *sv [[buffer(4)]],                                                    \
    device const float *extra [[buffer(5)]],                                                 \
    device float *out [[buffer(6)]],                                                         \
    constant uint &rows [[buffer(7)]],                                                       \
    constant uint &n [[buffer(8)]],                                                          \
    constant uint &k [[buffer(9)]],                                                          \
    constant uint &count [[buffer(10)]],                                                     \
    constant uint &repeat [[buffer(11)]],                                                    \
    constant uint &first [[buffer(12)]],                                                     \
    constant uint &bits [[buffer(13)]],                                                      \
    constant uint &book [[buffer(14)]],                                                      \
    constant uint &turned [[buffer(15)]],                                                    \
    constant uint &post [[buffer(16)]],                                                      \
    uint3 tg [[threadgroup_position_in_grid]],                                               \
    uint tid [[thread_index_in_threadgroup]]

// Finds the group's matrix. Rows off this rank store zeros.
#define TM_PICKED_START                                                                      \
    threadgroup float xs[4][128];                                                            \
    uint ob = tg.x;                                                                          \
    uint r = tg.y;                                                                           \
    if (ob >= (n >> 7u) || r >= rows) return;                                                \
    uint raw_id = ids[r];                                                                    \
    if (raw_id < first || raw_id - first >= count) {                                         \
        if (tid < 128u) out[ulong(r) * n + (ob << 7u) + tid] = 0.0f;                         \
        return;                                                                              \
    }                                                                                        \
    uint e = raw_id - first;                                                                 \
    uint ek;                                                                                 \
    device const uchar *mat = tm_picked_mat(w, n, k, bits, e, ek);

#define TM_PICKED_K(BOOK, TURNED, POST)                                                      \
    if (ek == 2u)                                                                            \
        tm_picked_body<2u, BOOK, TURNED>(x, w, mat, su, sv, extra, out, n, k, repeat, POST, e, ob, r, tid, xs); \
    else if (ek == 3u)                                                                       \
        tm_picked_body<3u, BOOK, TURNED>(x, w, mat, su, sv, extra, out, n, k, repeat, POST, e, ob, r, tid, xs); \
    else                                                                                     \
        tm_picked_body<4u, BOOK, TURNED>(x, w, mat, su, sv, extra, out, n, k, repeat, POST, e, ob, r, tid, xs);

#define TM_PICKED_BOOK(BOOK)                                                                 \
    if (turned) {                                                                            \
        TM_PICKED_K(BOOK, true, post)                                                        \
    } else {                                                                                 \
        TM_PICKED_K(BOOK, false, 0u)                                                         \
    }

// Rotated PMCGX experts, each at its own width.
kernel void tm_pickmix_pmcgx_t(TM_PICKED_ARGS) {
    (void)book; (void)turned;
    TM_PICKED_START
    TM_PICKED_K(TM_BOOK_PMCGX, true, post)
}

// Picked experts of any width, book and rotation.
kernel void tm_picked_gen(TM_PICKED_ARGS) {
    TM_PICKED_START
    if (book == TM_BOOK_PMCGX) {
        TM_PICKED_BOOK(TM_BOOK_PMCGX)
    } else if (book == TM_BOOK_MUL1) {
        TM_PICKED_BOOK(TM_BOOK_MUL1)
    } else {
        TM_PICKED_BOOK(TM_BOOK_MCG)
    }
}

// Groups routed rows by expert into fixed-size work tiles.
kernel void tm_pick_group(
    device const uint *ids [[buffer(0)]],
    device uint *meta [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &first [[buffer(4)]],
    constant uint &bm [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint tg_counts[TM_MAX_GROUP_EXPERTS + 1u];
    threadgroup atomic_uint tg_cursor[TM_MAX_GROUP_EXPERTS + 1u];
    uint buckets = count + 1u;
    for (uint e = tid; e < buckets; e += 256u) {
        atomic_store_explicit(&tg_counts[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = tid; r < rows; r += 256u) {
        uint raw_id = ids[r];
        uint b = (raw_id >= first && raw_id - first < count) ? (raw_id - first) : count;
        atomic_fetch_add_explicit(&tg_counts[b], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint max_tiles = (rows / bm) + buckets;
    device uint *tiles = meta + 1u;
    device uint *order = meta + 1u + 3u * max_tiles;
    if (tid == 0u) {
        uint sum = 0u;
        uint nt = 0u;
        for (uint b = 0u; b < buckets; b++) {
            uint c = atomic_load_explicit(&tg_counts[b], memory_order_relaxed);
            atomic_store_explicit(&tg_cursor[b], sum, memory_order_relaxed);
            for (uint off = 0u; off < c; off += bm) {
                tiles[3u * nt] = b;
                tiles[3u * nt + 1u] = sum + off;
                tiles[3u * nt + 2u] = min(bm, c - off);
                nt++;
            }
            sum += c;
        }
        meta[0] = nt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = tid; r < rows; r += 256u) {
        uint raw_id = ids[r];
        uint b = (raw_id >= first && raw_id - first < count) ? (raw_id - first) : count;
        uint pos = atomic_fetch_add_explicit(&tg_cursor[b], 1u, memory_order_relaxed);
        order[pos] = r;
    }
}

// Stores two decoded quads of one 16-wide strip.
static inline void tm_ws_put(threadgroup half *d, half4 wa, half4 wb) {
    d[0] = wa.x;
    d[32] = wa.y;
    d[256] = wa.z;
    d[288] = wa.w;
    d[8] = wb.x;
    d[40] = wb.y;
    d[264] = wb.z;
    d[296] = wb.w;
}

// Writes accumulators out through `xs` reused as floats.
template <uint TM>
static inline void tm_pickmix_pf_out(
    thread simdgroup_float8x8 acc[TM][4],
    threadgroup half xs[TM * 8u][128],
    threadgroup const uint *row_ids,
    device const float *sv,
    device const float *extra,
    device float *out,
    uint n,
    uint post,
    uint e,
    uint ob,
    uint nr,
    uint simd,
    uint lane) {
    constexpr uint half_tm = TM / 2u;
    constexpr uint half_rows = half_tm * 8u;
    uint base = (ob << 7u) + (lane << 2u);
    float4 sv_sc = *(device const float4 *)(sv + ulong(e) * n + base);
    float4 ex1 = (post == 1u) ? *(device const float4 *)(extra + base) : float4(1.0f);
    threadgroup float *out_s = (threadgroup float *)&xs[0][0];
    for (uint p = 0u; p < 2u; p++) {
        uint r_off = p * half_rows;
        if (r_off >= nr) break;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0u; i < half_tm; i++) {
            threadgroup float *dst = out_s + (i * 8u) * 128u + simd * 32u;
            for (uint j = 0u; j < 4u; j++) {
                simdgroup_store(acc[p * half_tm + i][j], dst + j * 8u, 128u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r_end = min(half_rows, nr - r_off);
        for (uint r = simd; r < r_end; r += 4u) {
            uint rr_idx = row_ids[r_off + r];
            float4 y = tm_turn_out(tm_get4(out_s + r * 128u + (lane << 2u)), sv_sc, lane);
            if (post == 1u) y *= ex1;
            else if (post == 2u) {
                float4 ex = *(device const float4 *)(extra + ulong(rr_idx) * n + base);
                y = (y / (1.0f + exp(-y))) * ex;
            }
            *(device float4 *)(out + ulong(rr_idx) * n + base) = y;
        }
    }
}

// One rotated PMCGX tile. `GUARD` skips row groups past `nr`.
template <uint K, uint TM, bool GUARD>
static inline void tm_pickmix_pf_body(
    device const float *x, device const uchar *w, device const uchar *mat,
    device const float *su, device const float *sv,
    device const float *extra, device float *out,
    uint n, uint k, uint repeat, uint post, uint e, uint ob,
    uint nr, uint tid,
    threadgroup half xs[TM * 8u][128],
    threadgroup half ws[4][16 * 32],
    threadgroup const uint *row_ids) {
    constexpr uint tw = 8u * K;
    uint simd = tid >> 5u, lane = tid & 31u, cc = lane >> 2u, rr = lane & 3u;
    TmWalk walk = tm_picked_walk<K>(w, mat, n, k, ob);
    TmWordMap wm = tm_word_map<K, TM_BOOK_PMCGX>(lane);
    threadgroup half *wown = ws[simd];
    uint wr0 = (rr << 1u) * 32u + cc;
    uint sh = (K == 4u) ? 20u : wm.sh;
    uint rsh = 32u - sh, rsh3 = 26u - sh;
    uint i0 = wm.i0, i1 = wm.i1;
    device const float *sue = su + ulong(e) * k;
    simdgroup_float8x8 acc[TM][4];
    for (uint tm = 0u; tm < TM; tm++) {
        for (uint j = 0u; j < 4u; j++) {
            acc[tm][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    device const uchar *gb = walk.at;
    for (uint b = 0u; b < walk.blocks; b++) {
        if (b > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        tm_pf_stage<TM, true, half, 128u>(
            x, sue, k, b, 0u, nr, repeat, true, simd, 4u, lane, xs, row_ids);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const uint *tiles = (device const uint *)gb;
        for (uint u = 0u; u < 8u; u++) {
            uint at = (u * 8u + simd * 2u) * tw;
            half4 wa, wb;
            tm_decode_words<K, TM_BOOK_PMCGX>(
                tiles[at + i0], tiles[at + i1], sh, rsh, rsh3, wa, wb);
            tm_ws_put(wown + wr0, wa, wb);
            at += tw;
            tm_decode_words<K, TM_BOOK_PMCGX>(
                tiles[at + i0], tiles[at + i1], sh, rsh, rsh3, wa, wb);
            tm_ws_put(wown + wr0 + 16u, wa, wb);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k8 = 0u; k8 < 16u; k8 += 8u) {
                simdgroup_half8x8 tb[4];
                for (uint j = 0u; j < 4u; j++) {
                    simdgroup_load(tb[j], wown + k8 * 32u + j * 8u, 32u);
                }
                for (uint tm = 0u; tm < TM; tm++) {
                    if (GUARD && tm * 8u >= nr) continue;
                    simdgroup_half8x8 ta;
                    simdgroup_load(ta, &xs[tm * 8u][u * 16u + k8], 128u);
                    for (uint j = 0u; j < 4u; j++) {
                        simdgroup_multiply_accumulate(acc[tm][j], ta, tb[j], acc[tm][j]);
                    }
                }
            }
        }
        gb += walk.stride;
    }
    tm_pickmix_pf_out<TM>(acc, xs, row_ids, sv, extra, out, n, post, e, ob, nr, simd, lane);
}

// One general work tile using 8x8 f32 MMA.
template <uint K, uint BOOK, uint TM, bool TURNED>
static inline void tm_pick_pf_body(
    device const float *x, device const uchar *w, device const uchar *mat,
    device const float *su, device const float *sv,
    device const float *extra, device float *out,
    uint n, uint k, uint repeat, uint post, uint e, uint ob,
    uint nr, uint tid,
    threadgroup float xs[TM * 8u][TM_XS_STRIDE],
    threadgroup float ws[8][16 * TM_WS_STRIDE],
    threadgroup const uint *row_ids) {
    uint simd = tid >> 5u, lane = tid & 31u, cc = lane >> 2u, rr = lane & 3u;
    TmWalk walk = tm_picked_walk<K>(w, mat, n, k, ob);
    TmWordMap wm = tm_word_map<K, BOOK>(lane);
    threadgroup float *wown = ws[simd];
    uint wr0 = (rr << 1u) * TM_WS_STRIDE + cc;
    uint base = (ob << 7u) + (lane << 2u);
    float4 sv_sc = TURNED ? *(device const float4 *)(sv + ulong(e) * n + base) : float4(1.0f);
    float4 ex1 = (TURNED && post == 1u) ? *(device const float4 *)(extra + base) : float4(1.0f);

    tm_pf_tile<K, BOOK, TM, true>(
        x, su + ulong(e) * k, walk, wm, k, 0u, nr, repeat, TURNED, simd, lane, wr0, wown, xs, row_ids);
    for (uint r = simd; r < nr; r += 8u) {
        uint rr_idx = row_ids[r];
        float4 y = tm_get4(&xs[r][lane << 2u]);
        if constexpr (TURNED) {
            y = tm_turn_out(y, sv_sc, lane);
            if (post == 1u) y *= ex1;
            else if (post == 2u) {
                float4 ex = *(device const float4 *)(extra + ulong(rr_idx) * n + base);
                y = (y / (1.0f + exp(-y))) * ex;
            }
        }
        *(device float4 *)(out + ulong(rr_idx) * n + base) = y;
    }
}

// Loads one work tile from `ids` and zeros off-rank rows.
#define TM_PICK_TILE_START(TM)                                                               \
    threadgroup uint row_ids[TM * 8u];                                                       \
    (void)first;                                                                             \
    uint ob = tg.x;                                                                          \
    uint t = tg.y;                                                                           \
    uint total = ids[0];                                                                     \
    if (ob >= (n >> 7u) || t >= total) return;                                               \
    uint bm = TM * 8u;                                                                       \
    uint b = ids[1u + 3u * t];                                                               \
    uint start = ids[1u + 3u * t + 1u];                                                      \
    uint nr = ids[1u + 3u * t + 2u];                                                         \
    uint max_tiles = (rows / bm) + count + 1u;                                               \
    device const uint *order = ids + 1u + 3u * max_tiles;                                    \
    if (b == count) {                                                                        \
        for (uint i = 0u; i < nr; i++) {                                                     \
            uint r = order[start + i];                                                       \
            if (tid < 128u) out[ulong(r) * n + (ob << 7u) + tid] = 0.0f;                     \
        }                                                                                    \
        return;                                                                              \
    }                                                                                        \
    if (tid < nr) row_ids[tid] = order[start + tid];                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                         \
    uint e = b;                                                                              \
    uint ek;                                                                                 \
    device const uchar *mat = tm_picked_mat(w, n, k, bits, e, ek);

#define TM_PICK_PF_K(BOOK, TM, TURNED, POST)                                                 \
    if (ek == 2u)                                                                            \
        tm_pick_pf_body<2u, BOOK, TM, TURNED>(                                               \
            x, w, mat, su, sv, extra, out, n, k, repeat, POST, e, ob, nr, tid, xs, ws, row_ids); \
    else if (ek == 3u)                                                                       \
        tm_pick_pf_body<3u, BOOK, TM, TURNED>(                                               \
            x, w, mat, su, sv, extra, out, n, k, repeat, POST, e, ob, nr, tid, xs, ws, row_ids); \
    else                                                                                     \
        tm_pick_pf_body<4u, BOOK, TM, TURNED>(                                               \
            x, w, mat, su, sv, extra, out, n, k, repeat, POST, e, ob, nr, tid, xs, ws, row_ids);

#define TM_PICK_PF_BOOK(BOOK, TM)                                                            \
    if (turned) {                                                                            \
        TM_PICK_PF_K(BOOK, TM, true, post)                                                   \
    } else {                                                                                 \
        TM_PICK_PF_K(BOOK, TM, false, 0u)                                                    \
    }

#define TM_PICKMIX_PF_K(K, TM, GUARD)                                                        \
    tm_pickmix_pf_body<K, TM, GUARD>(                                                        \
        x, w, mat, su, sv, extra, out, n, k, repeat, post, e, ob, nr, tid, xs, ws, row_ids);

// Four SIMD groups per work tile. Short tiles take `GUARD`.
#define TM_PICKMIX_PF_KERNEL(NAME, TM, GUARD)                                                \
    kernel void NAME(TM_PICKED_ARGS) {                                                       \
        (void)book; (void)turned;                                                            \
        threadgroup half xs[TM * 8u][128];                                                   \
        threadgroup half ws[4][16 * 32];                                                     \
        TM_PICK_TILE_START(TM)                                                               \
        if (ek == 2u) {                                                                      \
            TM_PICKMIX_PF_K(2u, TM, GUARD)                                                   \
        } else if (ek == 3u) {                                                               \
            TM_PICKMIX_PF_K(3u, TM, GUARD)                                                   \
        } else {                                                                             \
            TM_PICKMIX_PF_K(4u, TM, GUARD)                                                   \
        }                                                                                    \
    }

#define TM_PICK_PF_GEN_KERNEL(NAME, TM)                                                      \
    kernel void NAME(TM_PICKED_ARGS) {                                                       \
        threadgroup float xs[TM * 8u][TM_XS_STRIDE];                                         \
        threadgroup float ws[8][16 * TM_WS_STRIDE];                                          \
        TM_PICK_TILE_START(TM)                                                               \
        if (book == TM_BOOK_PMCGX) {                                                         \
            TM_PICK_PF_BOOK(TM_BOOK_PMCGX, TM)                                               \
        } else if (book == TM_BOOK_MUL1) {                                                   \
            TM_PICK_PF_BOOK(TM_BOOK_MUL1, TM)                                                \
        } else {                                                                             \
            TM_PICK_PF_BOOK(TM_BOOK_MCG, TM)                                                 \
        }                                                                                    \
    }

TM_PICKMIX_PF_KERNEL(tm_pickmix_pf16_pmcgx_t, 2u, true)
TM_PICKMIX_PF_KERNEL(tm_pickmix_pf32_pmcgx_t, 4u, false)
TM_PICK_PF_GEN_KERNEL(tm_pick_pf16_gen, 2u)
TM_PICK_PF_GEN_KERNEL(tm_pick_pf32_gen, 4u)
