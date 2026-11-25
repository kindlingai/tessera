#include <tessera/1/core>

using namespace tessera;

#if defined(TESSERA_CUDA)

constexpr u32 CALIB_MAX_EXPERTS = 1024u;
constexpr u32 CALIB_TILE = 128u;
constexpr u32 CALIB_BK = 32u;
constexpr u32 CALIB_SYRK_STRIDE = CALIB_TILE + 8u;

static __device__ __forceinline__ u32 calib_bf16x2(float lo, float hi) {
    u32 out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
    return out;
}

static __device__ __forceinline__ float calib_silu(float v) {
    return v / (1.0f + expf(-v));
}

static __device__ __forceinline__ void calib_mma(float d[4], const u32 a[4], const u32 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ void calib_decode_triu_raw(u32 tidx, u32 blocks, u32& br, u32& bc) {
    u64 b2 = 2ull * blocks + 1ull;
    u64 disc = b2 * b2 - 8ull * tidx;
    u64 root = static_cast<u64>(sqrt(static_cast<double>(disc)));
    while ((root + 1ull) * (root + 1ull) <= disc) root++;
    while (root * root > disc) root--;
    br = static_cast<u32>((b2 - root) >> 1);
    u64 start = (static_cast<u64>(br) * (2ull * blocks + 1ull - br)) >> 1;
    while (start > tidx) {
        br--;
        start = (static_cast<u64>(br) * (2ull * blocks + 1ull - br)) >> 1;
    }
    bc = br + static_cast<u32>(tidx - start);
}

static __device__ __forceinline__ u32 calib_decode_triu(u32 tidx, u32 blocks, u32& br, u32& bc) {
    if ((blocks & 7u) == 0u) {
        u32 g_cnt = blocks >> 3;
        u32 rem = tidx;
        u32 sr = 0u;
        while (sr + 1u < g_cnt) {
            u32 row_tiles = 36u + 64u * (g_cnt - 1u - sr);
            if (rem < row_tiles) break;
            rem -= row_tiles;
            sr++;
        }
        if (rem < 36u) {
            u32 lr, lc;
            calib_decode_triu_raw(rem, 8u, lr, lc);
            br = (sr << 3) + lr;
            bc = (sr << 3) + lc;
        } else {
            rem -= 36u;
            u32 sc = sr + 1u + (rem >> 6);
            u32 loc = rem & 63u;
            br = (sr << 3) + (loc >> 3);
            bc = (sc << 3) + (loc & 7u);
        }
        return static_cast<u32>((u64(br) * (2ull * blocks + 1ull - br)) >> 1) + (bc - br);
    }
    calib_decode_triu_raw(tidx, blocks, br, bc);
    return tidx;
}

// Pass 1: per-block expert histogram with no global atomics.
extern "C" __global__ __launch_bounds__(256, 2) void calib_sort_hist(
    const u32* ids, u32* block_hist, u32 total_picks, u32 experts, u32 first, u32 num_blocks) {
    __shared__ u32 sm[CALIB_MAX_EXPERTS];
    u32 tid = threadIdx.x;
    for (u32 e = tid; e < experts; e += blockDim.x) sm[e] = 0u;
    prim::block_sync();
    u32 b = blockIdx.x;
    if (b >= num_blocks) return;
    u32 lo = static_cast<u32>((u64(total_picks) * b) / num_blocks);
    u32 hi = static_cast<u32>((u64(total_picks) * (b + 1u)) / num_blocks);
    for (u32 i = lo + tid; i < hi; i += blockDim.x) {
        u32 raw = ids[i];
        if (raw >= first) {
            u32 e = raw - first;
            if (e < experts) atomicAdd(&sm[e], 1u);
        }
    }
    prim::block_sync();
    u32* dst = block_hist + u64(b) * experts;
    for (u32 e = tid; e < experts; e += blockDim.x) dst[e] = sm[e];
}

// Pass 2: deterministic prefix sums, counts, and M-tile table.
extern "C" __global__ __launch_bounds__(256, 1) void calib_sort_prefix(
    const u32* block_hist, u32* block_off, u32* offsets, long long* total_counts, u32* tile_meta,
    u32 experts, u32 num_blocks, u32 tile_m) {
    __shared__ u32 ec[CALIB_MAX_EXPERTS];
    u32 tid = threadIdx.x;
    for (u32 e = tid; e < experts; e += blockDim.x) {
        u32 sum = 0u;
        for (u32 b = 0u; b < num_blocks; b++) {
            u64 at = u64(b) * experts + e;
            block_off[at] = sum;
            sum += block_hist[at];
        }
        ec[e] = sum;
        if (total_counts != nullptr) total_counts[e] += static_cast<long long>(sum);
    }
    prim::block_sync();
    if (tid == 0u) {
        u32 cur = 0u;
        u32 tiles = 0u;
        for (u32 e = 0u; e < experts; e++) {
            offsets[e] = cur;
            u32 n = ec[e];
            cur += n;
            if (tile_meta != nullptr && n > 0u && tile_m > 0u) {
                for (u32 m0 = 0u; m0 < n; m0 += tile_m) {
                    tile_meta[1u + 2u * tiles] = e;
                    tile_meta[2u + 2u * tiles] = m0;
                    tiles++;
                }
            }
        }
        offsets[experts] = cur;
        if (tile_meta != nullptr) tile_meta[0] = tiles;
    }
}

// Pass 3: stable per-block scatter into sorted pick permutation.
extern "C" __global__ __launch_bounds__(32, 4) void calib_sort_scatter(
    const u32* ids, const u32* block_off, const u32* offsets, u32* perm, u32 total_picks,
    u32 experts, u32 first, u32 num_blocks) {
    __shared__ u32 cur[CALIB_MAX_EXPERTS];
    u32 b = blockIdx.x;
    if (b >= num_blocks) return;
    u32 tid = threadIdx.x;
    const u32* bo = block_off + u64(b) * experts;
    for (u32 e = tid; e < experts; e += blockDim.x) cur[e] = offsets[e] + bo[e];
    prim::block_sync();
    if (tid == 0u) {
        u32 lo = static_cast<u32>((u64(total_picks) * b) / num_blocks);
        u32 hi = static_cast<u32>((u64(total_picks) * (b + 1u)) / num_blocks);
        for (u32 i = lo; i < hi; i++) {
            u32 raw = ids[i];
            if (raw >= first) {
                u32 e = raw - first;
                if (e < experts) perm[cur[e]++] = i;
            }
        }
    }
}

static __device__ __forceinline__ void calib_ldm_x4_trans(u32 out[4], const void* smem_ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(addr));
}

static __device__ __forceinline__ void calib_cp_async16(void* dst, const void* src) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(src) : "memory");
}

static __device__ __forceinline__ void calib_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void calib_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

struct alignas(16) syrk_smem {
    u16 a[2][CALIB_BK][CALIB_SYRK_STRIDE];
    u16 b[2][CALIB_BK][CALIB_SYRK_STRIDE];
    u32 rows[2][CALIB_BK];
};

static __device__ __forceinline__ void syrk_fill_rows(
    const u32* perm, u32 start, u32 n_tok, u32 t0, u32 repeat, u32 shift, u32 dst_rows[CALIB_BK]) {
    u32 tid = threadIdx.x;
    if (tid < CALIB_BK) {
        u32 t = t0 + tid;
        u32 r = 0xFFFFFFFFu;
        if (t < n_tok) {
            if (perm == nullptr) {
                r = t;
            } else {
                u32 p = perm[start + t];
                r = shift < 32u ? (p >> shift) : (p / repeat);
            }
        }
        dst_rows[tid] = r;
    }
}

template <bool X_F32>
static __device__ __forceinline__ void calib_load8_bf16(
    const void* x_ptr, u32 row, u32 stride, u32 col_off, u16* dp) {
    if (row != 0xFFFFFFFFu) {
        u64 off = u64(row) * stride + col_off;
        if constexpr (X_F32) {
            const float* fp = static_cast<const float*>(x_ptr) + off;
            float4 f0 = *reinterpret_cast<const float4*>(fp);
            float4 f1 = *reinterpret_cast<const float4*>(fp + 4u);
            *reinterpret_cast<uint4*>(dp) =
                make_uint4(calib_bf16x2(f0.x, f0.y), calib_bf16x2(f0.z, f0.w),
                           calib_bf16x2(f1.x, f1.y), calib_bf16x2(f1.z, f1.w));
        } else {
            calib_cp_async16(dp, static_cast<const u16*>(x_ptr) + off);
        }
    } else {
        *reinterpret_cast<uint4*>(dp) = make_uint4(0u, 0u, 0u, 0u);
    }
}

template <bool X_F32>
static __device__ __forceinline__ void syrk_load_panel(
    const void* x_ptr, const u32 src_rows[CALIB_BK], u32 col0, u32 dim,
    u16 dst[CALIB_BK][CALIB_SYRK_STRIDE]) {
    u32 tid = threadIdx.x;
    u32 row_sub = tid >> 4;
    u32 col_u4 = (tid & 15u) << 3;
#pragma unroll
    for (u32 p = 0u; p < 2u; p++) {
        u32 lt = row_sub + (p << 4);
        calib_load8_bf16<X_F32>(x_ptr, src_rows[lt], dim, col0 + col_u4, &dst[lt][col_u4]);
    }
}

template <bool X_F32>
static __device__ __forceinline__ void syrk_body(
    const void* x_ptr, const u32* perm, const u32* offsets, float* acc_buf, u32 total_rows, u32 dim,
    u32 repeat, u32 full_out, syrk_smem& sm) {
    u32 blocks = dim >> 7;
    u32 total_tiles = (blocks * (blocks + 1u)) >> 1;
    u32 tidx = blockIdx.x;
    u32 e = blockIdx.y;
    if (tidx >= total_tiles) return;
    u32 start = offsets != nullptr ? offsets[e] : 0u;
    u32 n_tok = offsets != nullptr ? (offsets[e + 1u] - start) : total_rows;
    if (n_tok == 0u) return;
    u32 br, bc;
    u32 canon_tidx = calib_decode_triu(tidx, blocks, br, bc);
    u32 row0 = br << 7;
    u32 col0 = bc << 7;
    bool diag = (br == bc);
    u32 shift = (repeat > 0u && (repeat & (repeat - 1u)) == 0u) ? static_cast<u32>(__ffs(repeat) - 1)
                                                                : 32u;

    float acc[2][8][4];
#pragma unroll
    for (u32 mi = 0u; mi < 2u; mi++) {
#pragma unroll
        for (u32 ni = 0u; ni < 8u; ni++) {
            acc[mi][ni][0] = 0.0f;
            acc[mi][ni][1] = 0.0f;
            acc[mi][ni][2] = 0.0f;
            acc[mi][ni][3] = 0.0f;
        }
    }

    u32 warp = threadIdx.x >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 wm = warp >> 1;
    u32 wn = warp & 1u;
    u32 g = lane >> 2;
    u32 t4 = lane & 3u;
    u32 a_ld_r = ((lane >> 4) << 3) + (lane & 7u);
    u32 a_ld_c = ((lane >> 3) & 1u) << 3;
    u32 b_ld_r = lane & 15u;
    u32 b_ld_c = (lane >> 4) << 3;

    u32 stage = 0u;
    syrk_fill_rows(perm, start, n_tok, 0u, repeat, shift, sm.rows[0]);
    prim::block_sync();
    syrk_load_panel<X_F32>(x_ptr, sm.rows[0], row0, dim, sm.a[0]);
    if (!diag) syrk_load_panel<X_F32>(x_ptr, sm.rows[0], col0, dim, sm.b[0]);
    if constexpr (!X_F32) {
        calib_cp_commit();
    }
    if (CALIB_BK < n_tok) {
        syrk_fill_rows(perm, start, n_tok, CALIB_BK, repeat, shift, sm.rows[1]);
    }
    if constexpr (!X_F32) {
        calib_cp_wait<0>();
    }
    prim::block_sync();

    for (u32 t0 = 0u; t0 < n_tok; t0 += CALIB_BK) {
        u32 next_stage = stage ^ 1u;
        bool has_next = (t0 + CALIB_BK < n_tok);
        if (has_next) {
            syrk_load_panel<X_F32>(x_ptr, sm.rows[next_stage], row0, dim, sm.a[next_stage]);
            if (!diag) {
                syrk_load_panel<X_F32>(x_ptr, sm.rows[next_stage], col0, dim, sm.b[next_stage]);
            }
            if constexpr (!X_F32) {
                calib_cp_commit();
            }
        }
        const u16 (*sa)[CALIB_SYRK_STRIDE] = sm.a[stage];
        const u16 (*sb)[CALIB_SYRK_STRIDE] = diag ? sm.a[stage] : sm.b[stage];
#pragma unroll
        for (u32 ks = 0u; ks < 2u; ks++) {
            u32 kb = ks << 4;
            u32 a_frag[2][4];
#pragma unroll
            for (u32 mi = 0u; mi < 2u; mi++) {
                u32 d0 = (wm << 5) + (mi << 4) + a_ld_c;
                calib_ldm_x4_trans(a_frag[mi], &sa[kb + a_ld_r][d0]);
            }
#pragma unroll
            for (u32 ni = 0u; ni < 8u; ni += 2u) {
                u32 c0 = (wn << 6) + (ni << 3) + b_ld_c;
                u32 bf[4];
                calib_ldm_x4_trans(bf, &sb[kb + b_ld_r][c0]);
                u32 b0[2] = {bf[0], bf[1]};
                u32 b1[2] = {bf[2], bf[3]};
                calib_mma(acc[0][ni], a_frag[0], b0);
                calib_mma(acc[1][ni], a_frag[1], b0);
                calib_mma(acc[0][ni + 1u], a_frag[0], b1);
                calib_mma(acc[1][ni + 1u], a_frag[1], b1);
            }
        }
        if (t0 + 2u * CALIB_BK < n_tok) {
            syrk_fill_rows(perm, start, n_tok, t0 + 2u * CALIB_BK, repeat, shift, sm.rows[stage]);
        }
        if (has_next) {
            if constexpr (!X_F32) {
                calib_cp_wait<0>();
            }
            prim::block_sync();
        }
        stage = next_stage;
    }

    if (full_out == 0u) {
        float* tile_dst = acc_buf + (u64(e) * total_tiles + canon_tidx) * (CALIB_TILE * CALIB_TILE);
#pragma unroll
        for (u32 mi = 0u; mi < 2u; mi++) {
            u32 r = (wm << 5) + (mi << 4) + g;
#pragma unroll
            for (u32 ni = 0u; ni < 8u; ni++) {
                u32 c = (wn << 6) + (ni << 3) + (t4 << 1);
                u64 o0 = u64(r) * CALIB_TILE + c;
                u64 o1 = u64(r + 8u) * CALIB_TILE + c;
                float2* p0 = reinterpret_cast<float2*>(tile_dst + o0);
                float2* p1 = reinterpret_cast<float2*>(tile_dst + o1);
                float2 d0 = *p0;
                float2 d1 = *p1;
                d0.x += acc[mi][ni][0];
                d0.y += acc[mi][ni][1];
                d1.x += acc[mi][ni][2];
                d1.y += acc[mi][ni][3];
                *p0 = d0;
                *p1 = d1;
            }
        }
    } else {
        float* mat = acc_buf + u64(e) * dim * dim;
#pragma unroll
        for (u32 mi = 0u; mi < 2u; mi++) {
            u32 lr = (wm << 5) + (mi << 4) + g;
#pragma unroll
            for (u32 ni = 0u; ni < 8u; ni++) {
                u32 lc = (wn << 6) + (ni << 3) + (t4 << 1);
                u32 gr0 = row0 + lr;
                u32 gr1 = gr0 + 8u;
                u32 gc0 = col0 + lc;
                u32 gc1 = gc0 + 1u;
                mat[u64(gr0) * dim + gc0] += acc[mi][ni][0];
                mat[u64(gr0) * dim + gc1] += acc[mi][ni][1];
                mat[u64(gr1) * dim + gc0] += acc[mi][ni][2];
                mat[u64(gr1) * dim + gc1] += acc[mi][ni][3];
                if (!diag) {
                    mat[u64(gc0) * dim + gr0] += acc[mi][ni][0];
                    mat[u64(gc1) * dim + gr0] += acc[mi][ni][1];
                    mat[u64(gc0) * dim + gr1] += acc[mi][ni][2];
                    mat[u64(gc1) * dim + gr1] += acc[mi][ni][3];
                }
            }
        }
    }
}

// Grouped upper-triangle SYRK over BF16 inputs.
extern "C" __global__ __launch_bounds__(256, 2) void calib_syrk_bf16(
    const u16* x, const u32* perm, const u32* offsets, float* acc_buf, u32 total_rows, u32 dim,
    u32 repeat, u32 full_out) {
    __shared__ syrk_smem sm;
    syrk_body<false>(x, perm, offsets, acc_buf, total_rows, dim, repeat, full_out, sm);
}

// Grouped upper-triangle SYRK over F32 inputs rounded to BF16.
extern "C" __global__ __launch_bounds__(256, 2) void calib_syrk_f32(
    const float* x, const u32* perm, const u32* offsets, float* acc_buf, u32 total_rows, u32 dim,
    u32 repeat, u32 full_out) {
    __shared__ syrk_smem sm;
    syrk_body<true>(x, perm, offsets, acc_buf, total_rows, dim, repeat, full_out, sm);
}

static __device__ __forceinline__ void calib_ldm_x4(u32 out[4], const void* smem_ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(addr));
}

constexpr u32 CALIB_GEMM_STRIDE = CALIB_BK + 8u;

struct alignas(16) gemm_smem {
    u16 a[2][CALIB_TILE][CALIB_GEMM_STRIDE];
    u16 b[2][CALIB_TILE][CALIB_GEMM_STRIDE];
    u32 rows[CALIB_TILE];
};

static __device__ __forceinline__ void pbf16_load_x(
    const u16* x_ptr, const u32 src_rows[CALIB_TILE], u32 k0, u32 k,
    u16 dst[CALIB_TILE][CALIB_GEMM_STRIDE]) {
    u32 r_base = threadIdx.x >> 2;
    u32 k_sub = (threadIdx.x & 3u) << 3;
#pragma unroll
    for (u32 p = 0u; p < 2u; p++) {
        u32 lr = r_base + (p << 6);
        calib_load8_bf16<false>(x_ptr, src_rows[lr], k, k0 + k_sub, &dst[lr][k_sub]);
    }
}

static __device__ __forceinline__ void pbf16_load_w(
    const u16* w0, const u16* w1, u32 k0, u32 k, bool fused,
    u16 dst[CALIB_TILE][CALIB_GEMM_STRIDE]) {
    u32 r_base = threadIdx.x >> 2;
    u32 k_sub = (threadIdx.x & 3u) << 3;
    calib_cp_async16(&dst[r_base][k_sub], w0 + u64(r_base) * k + k0 + k_sub);
    const u16* p1 = fused ? (w1 + u64(r_base) * k) : (w0 + u64(r_base + 64u) * k);
    calib_cp_async16(&dst[r_base + 64u][k_sub], p1 + k0 + k_sub);
}

template <bool OUT_F32>
static __device__ __forceinline__ void pbf16_store2(
    void* out_ptr, u64 off, float v0, float v1) {
    if constexpr (OUT_F32) {
        *reinterpret_cast<float2*>(static_cast<float*>(out_ptr) + off) = make_float2(v0, v1);
    } else {
        *reinterpret_cast<u32*>(static_cast<u16*>(out_ptr) + off) = calib_bf16x2(v0, v1);
    }
}

template <bool OUT_F32, bool FUSED>
static __device__ __forceinline__ void pbf16_grouped_body(
    const u16* x_ptr, const u16* w0, const u16* w1, const u32* offsets, const u32* perm,
    const u32* tile_meta, void* out_ptr, u32 n, u32 k, u32 stride, u32 w_off, u32 repeat,
    gemm_smem& sm) {
    u32 total_m = tile_meta[0];
    if (blockIdx.y >= total_m) return;
    u32 n_tiles = gridDim.x;
    u32 linear = blockIdx.y * n_tiles + blockIdx.x;
    u32 gm = 8u;
    u32 mg = linear / (gm * n_tiles);
    u32 rem = linear - mg * (gm * n_tiles);
    u32 gm_size = total_m - mg * gm;
    if (gm_size > gm) gm_size = gm;
    u32 n_tile = rem / gm_size;
    u32 m_tile = mg * gm + (rem - n_tile * gm_size);
    u32 e = tile_meta[1u + 2u * m_tile];
    u32 m0 = tile_meta[2u + 2u * m_tile];
    u32 start = offsets[e];
    u32 n_tok = offsets[e + 1u] - start;
    u32 n0 = FUSED ? (n_tile << 6) : (n_tile << 7);
    u64 w_shift = (u64(e) * stride + w_off + n0) * k;
    const u16* wb0 = w0 + w_shift;
    const u16* wb1 = FUSED ? (w1 + w_shift) : nullptr;
    u32 shift = (repeat > 0u && (repeat & (repeat - 1u)) == 0u) ? static_cast<u32>(__ffs(repeat) - 1)
                                                                : 32u;

    if (threadIdx.x < CALIB_TILE) {
        u32 t = m0 + threadIdx.x;
        u32 r = 0xFFFFFFFFu;
        if (t < n_tok) {
            u32 p = perm[start + t];
            r = shift < 32u ? (p >> shift) : (p / repeat);
        }
        sm.rows[threadIdx.x] = r;
    }
    prim::block_sync();

    float acc[2][8][4];
#pragma unroll
    for (u32 mi = 0u; mi < 2u; mi++) {
#pragma unroll
        for (u32 ni = 0u; ni < 8u; ni++) {
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.0f;
        }
    }

    u32 warp = threadIdx.x >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 wm = warp >> 1;
    u32 wn = warp & 1u;
    u32 g = lane >> 2;
    u32 t4 = lane & 3u;
    u32 a_ld_r = lane & 15u;
    u32 a_ld_c = (lane >> 4) << 3;
    u32 b_ld_r = ((lane >> 4) << 3) + (lane & 7u);
    u32 b_ld_c = ((lane >> 3) & 1u) << 3;

    u32 stage = 0u;
    pbf16_load_x(x_ptr, sm.rows, 0u, k, sm.a[0]);
    pbf16_load_w(wb0, wb1, 0u, k, FUSED, sm.b[0]);
    calib_cp_commit();
    calib_cp_wait<0>();
    prim::block_sync();

    for (u32 k0 = 0u; k0 < k; k0 += CALIB_BK) {
        u32 next_stage = stage ^ 1u;
        bool has_next = (k0 + CALIB_BK < k);
        if (has_next) {
            pbf16_load_x(x_ptr, sm.rows, k0 + CALIB_BK, k, sm.a[next_stage]);
            pbf16_load_w(wb0, wb1, k0 + CALIB_BK, k, FUSED, sm.b[next_stage]);
            calib_cp_commit();
        }
#pragma unroll
        for (u32 ks = 0u; ks < 2u; ks++) {
            u32 kb = ks << 4;
            u32 a_frag[2][4];
#pragma unroll
            for (u32 mi = 0u; mi < 2u; mi++) {
                u32 r0 = (wm << 5) + (mi << 4) + a_ld_r;
                calib_ldm_x4(a_frag[mi], &sm.a[stage][r0][kb + a_ld_c]);
            }
#pragma unroll
            for (u32 ni = 0u; ni < 8u; ni += 2u) {
                u32 c0 = FUSED ? (((ni >> 2) << 6) + (wn << 5) + ((ni & 3u) << 3))
                               : ((wn << 6) + (ni << 3));
                u32 bf[4];
                calib_ldm_x4(bf, &sm.b[stage][c0 + b_ld_r][kb + b_ld_c]);
                u32 b0[2] = {bf[0], bf[1]};
                u32 b1[2] = {bf[2], bf[3]};
                calib_mma(acc[0][ni], a_frag[0], b0);
                calib_mma(acc[1][ni], a_frag[1], b0);
                calib_mma(acc[0][ni + 1u], a_frag[0], b1);
                calib_mma(acc[1][ni + 1u], a_frag[1], b1);
            }
        }
        if (has_next) {
            calib_cp_wait<0>();
            prim::block_sync();
        }
        stage = next_stage;
    }

#pragma unroll
    for (u32 mi = 0u; mi < 2u; mi++) {
        u32 t_r0 = m0 + (wm << 5) + (mi << 4) + g;
        u32 t_r1 = t_r0 + 8u;
        bool ok0 = t_r0 < n_tok;
        bool ok1 = t_r1 < n_tok;
        u32 pick0 = ok0 ? perm[start + t_r0] : 0u;
        u32 pick1 = ok1 ? perm[start + t_r1] : 0u;
        if constexpr (FUSED) {
#pragma unroll
            for (u32 ni = 0u; ni < 4u; ni++) {
                u32 col = n0 + (wn << 5) + (ni << 3) + (t4 << 1);
                if (ok0) {
                    float v0 = calib_silu(acc[mi][ni][0]) * acc[mi][ni + 4u][0];
                    float v1 = calib_silu(acc[mi][ni][1]) * acc[mi][ni + 4u][1];
                    pbf16_store2<OUT_F32>(out_ptr, u64(pick0) * n + col, v0, v1);
                }
                if (ok1) {
                    float v2 = calib_silu(acc[mi][ni][2]) * acc[mi][ni + 4u][2];
                    float v3 = calib_silu(acc[mi][ni][3]) * acc[mi][ni + 4u][3];
                    pbf16_store2<OUT_F32>(out_ptr, u64(pick1) * n + col, v2, v3);
                }
            }
        } else {
#pragma unroll
            for (u32 ni = 0u; ni < 8u; ni++) {
                u32 col = n0 + (wn << 6) + (ni << 3) + (t4 << 1);
                if (ok0) {
                    pbf16_store2<OUT_F32>(out_ptr, u64(pick0) * n + col, acc[mi][ni][0],
                                          acc[mi][ni][1]);
                }
                if (ok1) {
                    pbf16_store2<OUT_F32>(out_ptr, u64(pick1) * n + col, acc[mi][ni][2],
                                          acc[mi][ni][3]);
                }
            }
        }
    }
}

#define PBF16_GEMM_DEF(NAME, OF)                                                                   \
    extern "C" __global__ __launch_bounds__(256, 2) void NAME(                                     \
        const u16* x, const u16* w, const u32* offsets, const u32* perm, const u32* tile_meta,     \
        void* out, u32 n, u32 k, u32 stride, u32 w_off, u32 repeat) {                              \
        __shared__ gemm_smem sm;                                                                   \
        pbf16_grouped_body<OF, false>(x, w, nullptr, offsets, perm, tile_meta, out, n, k, stride,  \
                                      w_off, repeat, sm);                                          \
    }

PBF16_GEMM_DEF(pbf16_gemm_bf16_bf16, false)
PBF16_GEMM_DEF(pbf16_gemm_bf16_f32, true)

#define PBF16_GU_DEF(NAME, OF)                                                                     \
    extern "C" __global__ __launch_bounds__(256, 2) void NAME(                                     \
        const u16* x, const u16* wg, const u16* wu, const u32* offsets, const u32* perm,           \
        const u32* tile_meta, void* out, u32 n, u32 k, u32 stride, u32 repeat) {                   \
        __shared__ gemm_smem sm;                                                                   \
        pbf16_grouped_body<OF, true>(x, wg, wu, offsets, perm, tile_meta, out, n, k, stride, 0u,   \
                                     repeat, sm);                                                  \
    }

PBF16_GU_DEF(pbf16_gu_bf16_bf16, false)
PBF16_GU_DEF(pbf16_gu_bf16_f32, true)

#endif
