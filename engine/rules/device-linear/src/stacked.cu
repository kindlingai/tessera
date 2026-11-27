#include <tessera/1/core>

using namespace tessera;

#if defined(TESSERA_CUDA)

static __device__ __forceinline__ uint4 st_ld_nc4(const void* p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}

static __device__ __forceinline__ u32 st_bf16x2(float lo, float hi) {
    u32 out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
    return out;
}

static __device__ __forceinline__ void st_mma_bf16(float d[4], const u32 a[4], const u32 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ void st_stacked_x_bf16(
    const float* xb, u32 r0, u32 rows, u32 x_rs, u32 k0, u32 r, u32 xc, u32 xw, u32 (*sxh)[18],
    u32 (*sxl)[18]) {
    if (r0 + r < rows) {
        float4 xv = *reinterpret_cast<const float4*>(xb + u64(r0 + r) * x_rs + k0 + xc);
        u32 h0 = st_bf16x2(xv.x, xv.y), h1 = st_bf16x2(xv.z, xv.w);
        u32 l0 = st_bf16x2(xv.x - __uint_as_float(h0 << 16u),
                           xv.y - __uint_as_float(h0 & 0xFFFF0000u));
        u32 l1 = st_bf16x2(xv.z - __uint_as_float(h1 << 16u),
                           xv.w - __uint_as_float(h1 & 0xFFFF0000u));
        *reinterpret_cast<uint2*>(&sxh[r][xw]) = make_uint2(h0, h1);
        *reinterpret_cast<uint2*>(&sxl[r][xw]) = make_uint2(l0, l1);
    } else {
        *reinterpret_cast<uint2*>(&sxh[r][xw]) = make_uint2(0u, 0u);
        *reinterpret_cast<uint2*>(&sxl[r][xw]) = make_uint2(0u, 0u);
    }
}

template <u32 SW_STRIDE>
static __device__ __forceinline__ void st_stacked_w_t(u16 (*sw)[SW_STRIDE], u32 wc, u32 wk,
                                                      uint4 cur) {
    sw[wc + 0u][wk] = static_cast<u16>(cur.x);
    sw[wc + 1u][wk] = static_cast<u16>(cur.x >> 16u);
    sw[wc + 2u][wk] = static_cast<u16>(cur.y);
    sw[wc + 3u][wk] = static_cast<u16>(cur.y >> 16u);
    sw[wc + 4u][wk] = static_cast<u16>(cur.z);
    sw[wc + 5u][wk] = static_cast<u16>(cur.z >> 16u);
    sw[wc + 6u][wk] = static_cast<u16>(cur.w);
    sw[wc + 7u][wk] = static_cast<u16>(cur.w >> 16u);
}

static __device__ __forceinline__ void st_stacked_a(const u32 (*sxh)[18], const u32 (*sxl)[18],
                                                    u32 row, u32 off, u32 ah[4], u32 al[4]) {
    ah[0] = sxh[row][off];
    ah[1] = sxh[row + 8u][off];
    ah[2] = sxh[row][off + 4u];
    ah[3] = sxh[row + 8u][off + 4u];
    al[0] = sxl[row][off];
    al[1] = sxl[row + 8u][off];
    al[2] = sxl[row][off + 4u];
    al[3] = sxl[row + 8u][off + 4u];
}

struct alignas(16) st_stacked_smem {
    u32 sxh[16][18];
    u32 sxl[16][18];
    u16 sw[32][34];
};

static __device__ __forceinline__ void st_rope_tail(
    const float* rot, const u32* pos, float* ob, u32 batch, u32 batches, u32 r0, u32 tile_rows,
    u32 rows, u32 o_rs, u32 rope_w, u32 rot_stride, u32 rot_off, u32 rope_out_off, u32 turned,
    u32 adjacent, float theta, u32 tid, u32 nthreads) {
    u32 nr = min(rows - r0, tile_rows);
    u32 pairs = rope_w >> 1u;
    u32 total = nr * pairs;
    for (u32 i = tid; i < total; i += nthreads) {
        u32 r = i / pairs;
        u32 pair = i - r * pairs;
        u32 row = r0 + r;
        float angle = (pair < turned)
            ? float(pos[row]) * powf(theta, -float(2u * pair) / float(rope_w))
            : 0.0f;
        float sn, c;
        sincosf(angle, &sn, &c);
        u32 a = adjacent ? (2u * pair) : pair;
        u32 b = adjacent ? (a + 1u) : (pairs + pair);
        const float* src = rot + (u64(row) * batches + batch) * rot_stride + rot_off;
        float* dst = ob + u64(row) * o_rs + rope_out_off;
        float x0 = src[a], x1 = src[b];
        dst[a] = x0 * c - x1 * sn;
        dst[b] = x0 * sn + x1 * c;
    }
}

// Tensor-Core BF16 transposed stacked matmul, 16x32 per CTA.
extern "C" __global__ __launch_bounds__(128, 6) void tessera_stacked_wide(
    const float* x, const u16* w, const float* rot, const u32* pos, float* out, u32 batches,
    u32 rows, u32 n, u32 k, u32 block, u32 first, u32 k_step, u32 x_bs, u32 x_rs, u32 o_bs,
    u32 o_rs, u32 mat_off, u32 rope_w, u32 rot_stride, u32 rot_off, u32 rope_out_off, u32 turned,
    u32 adjacent, float theta) {
    __shared__ st_stacked_smem sm;
    u32 tile = blockIdx.x, rtiles = (rows + 15u) >> 4u;
    u32 batch = blockIdx.y / rtiles, r0 = (blockIdx.y % rtiles) * 16u;
    u32 tid = threadIdx.x, wt = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    u32 xr = tid >> 3u, xc = (tid & 7u) << 2u, xw = (tid & 7u) << 1u;
    u32 wk = tid >> 2u, wc = (tid & 3u) << 3u;
    const float* xb = x + u64(batch) * x_bs;
    const u16* wb = w + u64(batch) * block + first + (tile << 5u) + wc;
    float* ob = out + u64(batch) * o_bs;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint4 cur = st_ld_nc4(wb + u64(wk) * k_step);
    for (u32 k0 = 0u; k0 < k; k0 += 32u) {
        st_stacked_x_bf16(xb, r0, rows, x_rs, k0, xr, xc, xw, sm.sxh, sm.sxl);
        st_stacked_w_t(sm.sw, wc, wk, cur);
        prim::block_sync();
        if (k0 + 32u < k) {
            cur = st_ld_nc4(wb + u64(k0 + 32u + wk) * k_step);
        }
#pragma unroll
        for (u32 ks = 0u; ks < 2u; ks++) {
            u32 off = (ks << 3u) + t;
            u32 ah[4], al[4];
            st_stacked_a(sm.sxh, sm.sxl, g, off, ah, al);
            u32 wc0 = (wt << 3u) + g;
            u32 wk0 = (ks << 4u) + (t << 1u);
            u32 b[2] = {*reinterpret_cast<const u32*>(&sm.sw[wc0][wk0]),
                        *reinterpret_cast<const u32*>(&sm.sw[wc0][wk0 + 8u])};
            st_mma_bf16(acc, ah, b);
            st_mma_bf16(acc, al, b);
        }
        prim::block_sync();
    }
    u32 col = (tile << 5u) + (wt << 3u) + (t << 1u);
    if (r0 + g < rows) {
        *reinterpret_cast<float2*>(ob + u64(r0 + g) * o_rs + mat_off + col) =
            make_float2(acc[0], acc[1]);
    }
    if (r0 + g + 8u < rows) {
        *reinterpret_cast<float2*>(ob + u64(r0 + g + 8u) * o_rs + mat_off + col) =
            make_float2(acc[2], acc[3]);
    }
    if (tile == 0u && rope_w > 0u) {
        st_rope_tail(rot, pos, ob, batch, batches, r0, 16u, rows, o_rs, rope_w, rot_stride,
                     rot_off, rope_out_off, turned, adjacent, theta, tid, 128u);
    }
}

// Widened 64x64 BF16 Tensor-Core stacked matmul (non-transposed or transposed).
template <bool TRANSPOSED>
static __device__ __forceinline__ void st_stacked_64(
    const float* x, const u16* w, float* out, u32 batches, u32 rows, u32 n, u32 k, u32 block,
    u32 first, u32 step, u32 x_bs, u32 x_rs, u32 o_bs, u32 o_rs, u32 mat_off, const float* rot,
    const u32* pos, u32 rope_w, u32 rot_stride, u32 rot_off, u32 rope_out_off, u32 turned,
    u32 adjacent, float theta) {
    __shared__ alignas(16) u32 sxh[64][18];
    __shared__ alignas(16) u32 sxl[64][18];
    __shared__ alignas(16) u16 sw_u16[64][40];
    u32 tile = blockIdx.x, rtiles = (rows + 63u) >> 6u;
    u32 batch = blockIdx.y / rtiles, r0 = (blockIdx.y % rtiles) << 6u, n0 = tile << 6u;
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    u32 wm = warp >> 1u, wn = warp & 1u, m_base = wm << 4u, n_base = wn << 5u;
    u32 xr0 = tid >> 3u, xc = (tid & 7u) << 2u, xw = (tid & 7u) << 1u;
    const float* xb = x + u64(batch) * x_bs;
    const u16* wb = w + u64(batch) * block + first;
    float* ob = out + u64(batch) * o_bs;
    u32 wr_r = tid >> 2u, wr_c = (tid & 3u) << 3u;
    u32 wk = tid >> 3u, wc = (tid & 7u) << 3u;
    uint4 cur = TRANSPOSED ? st_ld_nc4(wb + u64(wk) * step + n0 + wc)
                           : st_ld_nc4(wb + u64(n0 + wr_r) * step + wr_c);
    float acc[4][4] = {};
    for (u32 k0 = 0u; k0 < k; k0 += 32u) {
#pragma unroll
        for (u32 dr = 0u; dr < 64u; dr += 32u) {
            st_stacked_x_bf16(xb, r0, rows, x_rs, k0, xr0 + dr, xc, xw, sxh, sxl);
        }
        if constexpr (TRANSPOSED) {
            st_stacked_w_t(sw_u16, wc, wk, cur);
        } else {
            *reinterpret_cast<uint4*>(&sw_u16[wr_r][wr_c]) = cur;
        }
        prim::block_sync();
        if (k0 + 32u < k) {
            cur = TRANSPOSED ? st_ld_nc4(wb + u64(k0 + 32u + wk) * step + n0 + wc)
                             : st_ld_nc4(wb + u64(n0 + wr_r) * step + k0 + 32u + wr_c);
        }
#pragma unroll
        for (u32 ks = 0u; ks < 2u; ks++) {
            u32 off = (ks << 3u) + t;
            u32 ah[4], al[4];
            st_stacked_a(sxh, sxl, m_base + g, off, ah, al);
            u32 wk0 = (ks << 4u) + (t << 1u);
#pragma unroll
            for (u32 ni = 0u; ni < 4u; ni++) {
                u32 nc = n_base + (ni << 3u) + g;
                u32 b[2] = {*reinterpret_cast<const u32*>(&sw_u16[nc][wk0]),
                            *reinterpret_cast<const u32*>(&sw_u16[nc][wk0 + 8u])};
                st_mma_bf16(acc[ni], ah, b);
                st_mma_bf16(acc[ni], al, b);
            }
        }
        prim::block_sync();
    }
#pragma unroll
    for (u32 ni = 0u; ni < 4u; ni++) {
        u32 col = n0 + n_base + (ni << 3u) + (t << 1u);
        if (r0 + m_base + g < rows) {
            *reinterpret_cast<float2*>(ob + u64(r0 + m_base + g) * o_rs + mat_off + col) =
                make_float2(acc[ni][0], acc[ni][1]);
        }
        if (r0 + m_base + g + 8u < rows) {
            *reinterpret_cast<float2*>(ob + u64(r0 + m_base + g + 8u) * o_rs + mat_off + col) =
                make_float2(acc[ni][2], acc[ni][3]);
        }
    }
    if constexpr (TRANSPOSED) {
        if (tile == 0u && rope_w > 0u) {
            st_rope_tail(rot, pos, ob, batch, batches, r0, 64u, rows, o_rs, rope_w, rot_stride,
                         rot_off, rope_out_off, turned, adjacent, theta, tid, 256u);
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 3) void tessera_stacked_lanes_64(
    const float* x, const u16* w, float* out, u32 rows, u32 n, u32 k, u32 block, u32 first,
    u32 col_step, u32 x_bs, u32 x_rs, u32 o_bs, u32 o_rs) {
    st_stacked_64<false>(x, w, out, 1u, rows, n, k, block, first, col_step, x_bs, x_rs, o_bs, o_rs,
                         0u, nullptr, nullptr, 0u, 0u, 0u, 0u, 0u, 0u, 0.0f);
}

extern "C" __global__ __launch_bounds__(256, 3) void tessera_stacked_wide_64(
    const float* x, const u16* w, const float* rot, const u32* pos, float* out, u32 batches,
    u32 rows, u32 n, u32 k, u32 block, u32 first, u32 k_step, u32 x_bs, u32 x_rs, u32 o_bs,
    u32 o_rs, u32 mat_off, u32 rope_w, u32 rot_stride, u32 rot_off, u32 rope_out_off, u32 turned,
    u32 adjacent, float theta) {
    st_stacked_64<true>(x, w, out, batches, rows, n, k, block, first, k_step, x_bs, x_rs, o_bs,
                        o_rs, mat_off, rot, pos, rope_w, rot_stride, rot_off, rope_out_off, turned,
                        adjacent, theta);
}

#endif
