#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_CUDA)

static __device__ __forceinline__ float i4_b1(const u8* p, u32 col, bool bf16) {
    return bf16 ? __uint_as_float(u32(reinterpret_cast<const u16*>(p)[col]) << 16u)
                : reinterpret_cast<const float*>(p)[col];
}

static __device__ __forceinline__ float i4_epilogue(
    float v, const trellis::o_t* other, const u8* bias_p, u64 off, u32 col, u32 mode) {
    u32 base = mode & 15u;
    bool bf16 = (mode & 16u) != 0u;
    if (base == 1u) return v + trellis::load_o(other, off);
    if (base == 3u) return (v / (1.0f + expf(-v))) * trellis::load_o(other, off);
    if (base == 4u) return v * trellis::load_o(other, off);
    if (base == 5u) return v / (1.0f + expf(-v));
    if (base == 8u) return v + i4_b1(bias_p, col, bf16) + trellis::load_o(other, off);
    if (base == 2u || base == 6u || base == 7u) {
        float x = v + (bf16 ? i4_b1(reinterpret_cast<const u8*>(other), col, true)
                            : trellis::load_o(other, col));
        if (base == 6u) {
            return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        }
        if (base == 7u) return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
        return x;
    }
    return v;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void i4_cp16(void* smem, const void* gmem, bool pred) {
    u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem));
    if (pred) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem));
    else *reinterpret_cast<uint4*>(smem) = make_uint4(0u, 0u, 0u, 0u);
}

static __device__ __forceinline__ void i4_ldm4(u32 (&r)[4], const void* smem_ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

static __device__ __forceinline__ u32 i4_dec(u32 b) {
    u32 p = b * 0x1001u, v, out;
    asm("lop3.b32 %0, %1, 0x000F000F, 0x64006400, 0xEA;" : "=r"(v) : "r"(p));
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(v), "r"(0x64086408u));
    return out;
}

static __device__ __forceinline__ u32 i4_dec_s(u32 b, u32 sc_h) {
    u32 v = i4_dec(b), out;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(v), "r"(sc_h));
    return out;
}

union alignas(16) i4_r16_smem {
    u32 sw[2][16][36];
    float red[4][16][17];
};

static __device__ __forceinline__ void i4_hilo_pair(const trellis::x_t* p, u32& hi, u32& lo) {
    float2 v = trellis::load_x2(p);
    hi = trellis::f32x2_to_f16x2(v.x, v.y);
    lo = trellis::f32x2_to_f16x2(
        v.x - trellis::f16_to_f32(hi & 0xFFFFu),
        v.y - trellis::f16_to_f32(hi >> 16u));
}

template <bool SPLIT>
static __device__ __forceinline__ void fg_int4_r16(
    const trellis::x_t* x, const u8* w, const trellis::o_t* other, const u8* bias_p, trellis::y_t* out,
    u32 m, u32 n, u32 k, u32 mode, u32 ks, i4_r16_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    u32 r0 = blockIdx.y << 4u, nb = n >> 4u;
    u32 ob = SPLIT ? (blockIdx.x % nb) : blockIdx.x, p = SPLIT ? (blockIdx.x / nb) : 0u;
    u32 n0 = ob << 4u, chunks = k >> 8u, ch_per = SPLIT ? (chunks / ks) : chunks;
    u32 ch0 = SPLIT ? (p * ch_per) : 0u, ch1 = ch0 + ch_per;
    u64 row_bytes = u64(k >> 5u) * 18ull;
    u32 r_l = tid >> 3u, c_l = tid & 7u;
    auto load_w = [&](u32 st, u32 ch) {
        u64 off = u64(ch) * 144ull;
        i4_cp16(&sm.sw[st][r_l][c_l << 2u], w + u64(n0 + r_l) * row_bytes + off + (c_l << 4u), true);
        if (tid < 16u) i4_cp16(&sm.sw[st][tid][32u], w + u64(n0 + tid) * row_bytes + off + 128u, true);
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    load_w(0u, ch0);
    for (u32 ch = ch0; ch < ch1; ch++) {
        u32 st = (ch - ch0) & 1u;
        bool has_nxt = (ch + 1u < ch1);
        if (has_nxt) {
            load_w(st ^ 1u, ch + 1u);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        prim::block_sync();
        const u8* sb0 = reinterpret_cast<const u8*>(sm.sw[st][g]);
        const u8* sb1 = reinterpret_cast<const u8*>(sm.sw[st][g + 8u]);
#pragma unroll
        for (u32 bl = 0u; bl < 2u; bl++) {
            u32 blk = (warp << 1u) + bl, b_off = blk * 18u;
            float sc0 = trellis::f16_to_f32(*reinterpret_cast<const u16*>(sb0 + b_off));
            float sc1 = trellis::f16_to_f32(*reinterpret_cast<const u16*>(sb1 + b_off));
            float tmp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
            for (u32 ks_i = 0u; ks_i < 2u; ks_i++) {
                u32 q_off = b_off + 2u + (ks_i << 3u) + t;
                u32 a[4] = {i4_dec(sb0[q_off]), i4_dec(sb1[q_off]),
                            i4_dec(sb0[q_off + 4u]), i4_dec(sb1[q_off + 4u])};
                u32 k_col = (ch << 8u) + (blk << 5u) + (ks_i << 4u) + (t << 1u);
#pragma unroll
                for (u32 ti = 0u; ti < 2u; ti++) {
                    u32 tok = r0 + (ti << 3u) + g, bh[2] = {0u, 0u}, blv[2] = {0u, 0u};
                    if (tok < m) {
                        const trellis::x_t* xp = x + u64(tok) * k + k_col;
                        i4_hilo_pair(xp, bh[0], blv[0]);
                        i4_hilo_pair(xp + 8u, bh[1], blv[1]);
                    }
                    trellis::mma_m16n8k16(tmp[ti], a, bh);
                    trellis::mma_m16n8k16(tmp[ti], a, blv);
                }
            }
#pragma unroll
            for (u32 ti = 0u; ti < 2u; ti++) {
                acc[ti][0] = fmaf(sc0, tmp[ti][0], acc[ti][0]);
                acc[ti][1] = fmaf(sc0, tmp[ti][1], acc[ti][1]);
                acc[ti][2] = fmaf(sc1, tmp[ti][2], acc[ti][2]);
                acc[ti][3] = fmaf(sc1, tmp[ti][3], acc[ti][3]);
            }
        }
        if (has_nxt) prim::block_sync();
    }
    prim::block_sync();
#pragma unroll
    for (u32 ti = 0u; ti < 2u; ti++) {
        u32 tr = (ti << 3u) + (t << 1u);
        sm.red[warp][tr][g] = acc[ti][0]; sm.red[warp][tr + 1u][g] = acc[ti][1];
        sm.red[warp][tr][g + 8u] = acc[ti][2]; sm.red[warp][tr + 1u][g + 8u] = acc[ti][3];
    }
    prim::block_sync();
    u32 col = tid & 15u, tr0 = tid >> 4u;
    for (u32 tr = tr0; tr < 16u; tr += 8u) {
        if (r0 + tr >= m) continue;
        float v = sm.red[0][tr][col] + sm.red[1][tr][col] +
                  sm.red[2][tr][col] + sm.red[3][tr][col];
        u64 off = u64(SPLIT ? (p * m + r0 + tr) : (r0 + tr)) * n + n0 + col;
        trellis::store_y(out, off, SPLIT ? v : i4_epilogue(v, other, bias_p, off, n0 + col, mode));
    }
}

union alignas(16) i4_r64_smem {
    struct {
        u32 sw[128][36];
        u32 sx[2][64][32];
    } k;
    float out[32][132];
};

static __device__ __forceinline__ void i4_load_x(
    u32 (*sx_st)[32], const trellis::x_t* x, u32 r0, u32 k_sub, u32 m, u32 k, u32 r0_t, u32 col16) {
#pragma unroll
    for (u32 dr = 0u; dr < 64u; dr += 32u) {
        u32 r = r0_t + dr;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (r0 + r < m) {
            const trellis::x_t* xp = x + u64(r0 + r) * k + k_sub + (col16 << 3u);
            float4 x0 = trellis::load_x4(xp);
            float4 x1 = trellis::load_x4(xp + 4u);
            v = make_uint4(trellis::f32x2_to_f16x2(x0.x, x0.y),
                           trellis::f32x2_to_f16x2(x0.z, x0.w),
                           trellis::f32x2_to_f16x2(x1.x, x1.y),
                           trellis::f32x2_to_f16x2(x1.z, x1.w));
        }
        *reinterpret_cast<uint4*>(&sx_st[r][(col16 ^ (r & 7u)) << 2u]) = v;
    }
}

template <bool SPLIT>
static __device__ __forceinline__ void fg_int4_r64(
    const trellis::x_t* x, const u8* w, const trellis::o_t* other, const u8* bias_p, trellis::y_t* out,
    u32 m, u32 n, u32 k, u32 mode, u32 ks, i4_r64_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    u32 g = lane >> 2u, t = lane & 3u, wn = warp & 3u, wm = warp >> 2u;
    u32 nb = (n + 127u) >> 7u, mb = (m + 63u) >> 6u;
    u32 ob = SPLIT ? (blockIdx.x % nb) : blockIdx.x, p = SPLIT ? (blockIdx.x / nb) : 0u;
    u32 lin = blockIdx.y * nb + ob, grp_sz = 8u * mb;
    u32 gid = lin / grp_sz, rem = lin - gid * grp_sz;
    u32 first_n = gid * 8u, width = min(8u, nb - first_n);
    u32 by = rem / width, bx = first_n + (rem - by * width);
    u32 n0 = bx << 7u, r0 = by << 6u;

    u32 chunks = k >> 8u, ch_per = SPLIT ? (chunks / ks) : chunks;
    u32 ch0 = SPLIT ? (p * ch_per) : 0u, ch1 = ch0 + ch_per;
    u64 row_bytes = u64(k >> 5u) * 18ull;
    u32 r0_t = tid >> 3u, col16 = tid & 7u;
    u32 n_base = wn << 5u, m_base = wm << 5u;
    u32 lb_row = ((lane >> 4u) << 3u) | (lane & 7u), lb_half = (lane >> 3u) & 1u;

    float acc[2][4][4];
#pragma unroll
    for (u32 ni = 0u; ni < 2u; ni++)
#pragma unroll
        for (u32 mi = 0u; mi < 4u; mi++)
            acc[ni][mi][0] = acc[ni][mi][1] = acc[ni][mi][2] = acc[ni][mi][3] = 0.0f;

    const u8* sb0[2] = {reinterpret_cast<const u8*>(sm.k.sw[n_base + g]),
                        reinterpret_cast<const u8*>(sm.k.sw[n_base + 16u + g])};
    const u8* sb1[2] = {reinterpret_cast<const u8*>(sm.k.sw[n_base + g + 8u]),
                        reinterpret_cast<const u8*>(sm.k.sw[n_base + 24u + g])};

    for (u32 ch = ch0; ch < ch1; ch++) {
        u64 ch_off = u64(ch) * 144ull;
#pragma unroll
        for (u32 dr = 0u; dr < 128u; dr += 32u) {
            u32 nr = r0_t + dr;
            i4_cp16(&sm.k.sw[nr][col16 << 2u],
                    w + u64(n0 + nr) * row_bytes + ch_off + (col16 << 4u), n0 + nr < n);
        }
        if (tid < 128u) {
            i4_cp16(&sm.k.sw[tid][32u],
                    w + u64(n0 + tid) * row_bytes + ch_off + 128u, n0 + tid < n);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        i4_load_x(sm.k.sx[0], x, r0, ch << 8u, m, k, r0_t, col16);
        asm volatile("cp.async.wait_group 0;\n" ::);
        prim::block_sync();
#pragma unroll
        for (u32 sub = 0u; sub < 4u; sub++) {
            u32 st = sub & 1u;
#pragma unroll
            for (u32 bl = 0u; bl < 2u; bl++) {
                u32 b_off = ((sub << 1u) + bl) * 18u;
                u32 sc0_h[2], sc1_h[2];
#pragma unroll
                for (u32 ni = 0u; ni < 2u; ni++) {
                    u32 s0 = *reinterpret_cast<const u16*>(sb0[ni] + b_off);
                    u32 s1 = *reinterpret_cast<const u16*>(sb1[ni] + b_off);
                    sc0_h[ni] = s0 | (s0 << 16u);
                    sc1_h[ni] = s1 | (s1 << 16u);
                }
#pragma unroll
                for (u32 ks_i = 0u; ks_i < 2u; ks_i++) {
                    u32 k16 = (bl << 1u) + ks_i, fb[4][2];
                    {
                        u32 mr0 = m_base + lb_row, tmp[4];
                        i4_ldm4(tmp, &sm.k.sx[st][mr0][(((k16 << 1u) | lb_half) ^ (mr0 & 7u)) << 2u]);
                        fb[0][0] = tmp[0]; fb[0][1] = tmp[1]; fb[1][0] = tmp[2]; fb[1][1] = tmp[3];
                    }
                    {
                        u32 mr2 = m_base + 16u + lb_row, tmp[4];
                        i4_ldm4(tmp, &sm.k.sx[st][mr2][(((k16 << 1u) | lb_half) ^ (mr2 & 7u)) << 2u]);
                        fb[2][0] = tmp[0]; fb[2][1] = tmp[1]; fb[3][0] = tmp[2]; fb[3][1] = tmp[3];
                    }
                    u32 q_off = b_off + 2u + (ks_i << 3u) + t;
#pragma unroll
                    for (u32 ni = 0u; ni < 2u; ni++) {
                        u32 a[4] = {i4_dec_s(sb0[ni][q_off], sc0_h[ni]),
                                    i4_dec_s(sb1[ni][q_off], sc1_h[ni]),
                                    i4_dec_s(sb0[ni][q_off + 4u], sc0_h[ni]),
                                    i4_dec_s(sb1[ni][q_off + 4u], sc1_h[ni])};
#pragma unroll
                        for (u32 mi = 0u; mi < 4u; mi++) trellis::mma_m16n8k16(acc[ni][mi], a, fb[mi]);
                    }
                }
            }
            if (sub + 1u < 4u) {
                i4_load_x(sm.k.sx[st ^ 1u], x, r0, (ch << 8u) + ((sub + 1u) << 6u), m, k, r0_t, col16);
            }
            prim::block_sync();
        }
    }
    u32 sr = tid >> 5u, sc4 = (tid & 31u) << 2u, col = n0 + sc4;
    for (u32 q = 0u; q < 2u; q++) {
        if (wm == q) {
            for (u32 ni = 0u; ni < 2u; ni++) {
                u32 c0 = n_base + (ni << 4u) + g;
                for (u32 mi = 0u; mi < 4u; mi++) {
                    u32 lr = (mi << 3u) + (t << 1u);
                    sm.out[lr][c0] = acc[ni][mi][0]; sm.out[lr][c0 + 8u] = acc[ni][mi][2];
                    sm.out[lr + 1u][c0] = acc[ni][mi][1]; sm.out[lr + 1u][c0 + 8u] = acc[ni][mi][3];
                }
            }
        }
        prim::block_sync();
        for (u32 dr = 0u; dr < 32u; dr += 8u) {
            u32 row = r0 + (q << 5u) + sr + dr;
            if (row >= m || col >= n) continue;
            float4 v = *reinterpret_cast<const float4*>(&sm.out[sr + dr][sc4]);
            u64 off = u64(SPLIT ? (p * m + row) : row) * n + col;
            if (!SPLIT && mode != 0u) {
                v.x = i4_epilogue(v.x, other, bias_p, off, col, mode);
                v.y = i4_epilogue(v.y, other, bias_p, off + 1u, col + 1u, mode);
                v.z = i4_epilogue(v.z, other, bias_p, off + 2u, col + 2u, mode);
                v.w = i4_epilogue(v.w, other, bias_p, off + 3u, col + 3u, mode);
            }
            trellis::store_y4(out + off, v);
        }
        if (q == 0u) prim::block_sync();
    }
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define I4_RUN(SM, FN, SPLIT, KS) \
    __shared__ SM sm; \
    FN<SPLIT>(x, w, other, bias_p, out, m, n, k, mode, KS, sm)
#else
#define I4_RUN(SM, FN, SPLIT, KS) \
    (void)x; (void)w; (void)other; (void)bias_p; (void)out; (void)m; (void)n; (void)k; (void)mode
#endif

#define I4_PAIR(TAG, TH, BLK, SM, FN) \
    extern "C" __global__ __launch_bounds__(TH, BLK) void tessera_mm_int4_##TAG( \
        const trellis::x_t* x, const u8* w, const trellis::o_t* other, const u8* bias_p, trellis::y_t* out, u32 m, u32 n, u32 k, u32 mode) { \
        I4_RUN(SM, FN, false, 1u); \
    } \
    extern "C" __global__ __launch_bounds__(TH, BLK) void tessera_mm_int4_##TAG##_sk( \
        const trellis::x_t* x, const u8* w, const trellis::o_t* other, const u8* bias_p, trellis::y_t* out, u32 m, u32 n, u32 k, u32 mode, u32 ks) { \
        (void)ks; \
        I4_RUN(SM, FN, true, ks); \
    }

// Streaming and staged-X INT4_G32 Tensor-Core matmul entries.
I4_PAIR(r16, 128, 4, i4_r16_smem, fg_int4_r16)
I4_PAIR(r64, 256, 2, i4_r64_smem, fg_int4_r64)

#undef I4_PAIR
#undef I4_RUN

#endif
