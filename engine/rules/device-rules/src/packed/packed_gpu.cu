// Packed varlen FlashAttention-2 kernels for CUDA and ROCm.

#include <tessera/1/core>

using namespace tessera;

#if defined(TESSERA_CUDA) || defined(TESSERA_HIP)

namespace {

constexpr float FA2_NEG_INF = -1e30f;
constexpr float FA2_MASK_CUT = -1e20f;
constexpr float FA2_LOG2E = 1.4426950408889634f;

TESSERA_FN float fmax_fast(float a, float b) {
    return a > b ? a : b;
}

TESSERA_FN float clamp_f16(float x) {
    float hi = x < 65504.0f ? x : 65504.0f;
    return hi > -65504.0f ? hi : -65504.0f;
}

TESSERA_FN u32 umin_u32(u32 a, u32 b) {
    return a < b ? a : b;
}

TESSERA_FN u32 umax_u32(u32 a, u32 b) {
    return a > b ? a : b;
}

#if defined(TESSERA_CUDA)

TESSERA_FN float exp2_fast(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

TESSERA_FN u32 shfl_xor_u32(u32 v, int mask) {
    return __shfl_xor_sync(0xffffffffu, v, mask);
}

TESSERA_FN u32 pack_f16x2(float lo, float hi) {
    u32 out;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(clamp_f16(lo)), "f"(clamp_f16(hi)));
    return out;
}

TESSERA_FN u32 load_f16x2(const float* p) {
    float2 v = *reinterpret_cast<const float2*>(p);
    return pack_f16x2(v.x, v.y);
}

TESSERA_FN void mma_f16(float d[4], const u32 a[4], const u32 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

TESSERA_FN void ldm_x4(u32& r0, u32& r1, u32& r2, u32& r3, const void* ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(ptr));
    asm("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

TESSERA_FN void ldm_x4_t(u32& r0, u32& r1, u32& r2, u32& r3, const void* ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(ptr));
    asm("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

#else

typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef u32 u32x4 __attribute__((ext_vector_type(4)));
typedef u32 u32x8 __attribute__((ext_vector_type(8)));

TESSERA_FN void lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0)\ns_barrier" ::: "memory");
}

TESSERA_FN float exp2_fast(float x) {
    float r;
    asm("v_exp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

TESSERA_FN u32 shfl_xor_u32(u32 v, int mask) {
    return __shfl_xor(v, mask);
}

TESSERA_FN float clamp_fast(float x) {
    return fminf(fmaxf(x, -65504.0f), 65504.0f);
}

TESSERA_FN u32 pack_f16_raw(float a, float b) {
    auto h = __builtin_amdgcn_cvt_pkrtz(a, b);
    return __builtin_bit_cast(u32, h);
}

TESSERA_FN u32 pack_clamped(float a, float b) {
    return pack_f16_raw(clamp_fast(a), clamp_fast(b));
}

template <u32 BK>
TESSERA_FN u32 vt_sw(u32 d) {
    return ((d ^ (d >> 2u)) & ((BK == 64u) ? 7u : 3u)) << 3u;
}

template <u32 HPAD, u32 BK>
struct rocm_kv_buf {
    _Float16 s_k[BK][HPAD + 8u];
    _Float16 s_vt[HPAD][BK];
};

template <u32 HPAD, u32 WARPS, u32 BK, bool DBUF>
struct rocm_smem {
    union {
        rocm_kv_buf<HPAD, BK> kv[DBUF ? 2u : 1u];
        struct {
            u32 s_from[WARPS * 16u];
            u32 s_to[WARPS * 16u];
            u32 s_wmin[WARPS];
            u32 s_wmax[WARPS];
        } bnd;
    };
};

#endif

struct TileRange {
    u32 kv_start;
    u32 kv_end;
    u32 w_min_from;
    u32 w_max_from;
    u32 w_min_to;
    u32 w_max_to;
};

// Loads per-row bounds and computes warp and block KV spans.
template <u32 WARPS, u32 QPW>
TESSERA_FN void load_bounds(const u32* bounds, u32 s_from[WARPS * QPW * 16u],
                            u32 s_to[WARPS * QPW * 16u], u32 s_wmin[WARPS], u32 s_wmax[WARPS],
                            u32 q0, u32 m, u32 causal, u32 tid, u32 warp, u32 lane,
                            TileRange tr[QPW], u32& kv_start, u32& kv_end) {
    constexpr u32 BQ = WARPS * QPW * 16u;
    constexpr u32 NT = WARPS * 32u;
    for (u32 i = tid; i < BQ; i += NT) {
        u32 r = q0 + i;
        u32 f = 0u;
        u32 t = 0u;
        if (r < m) {
            f = bounds[u64(r) * 2u];
            u32 end = bounds[u64(r) * 2u + 1u];
            t = (causal != 0u && r + 1u < end) ? (r + 1u) : end;
            if (t > m) t = m;
            if (f > t) f = t;
        }
        s_from[i] = f;
        s_to[i] = t;
    }
    prim::block_sync();

    u32 w_lo = m;
    u32 w_hi = 0u;
#pragma unroll
    for (u32 qw = 0u; qw < QPW; qw++) {
        u32 qi = (warp * QPW + qw) * 16u + (lane & 15u);
        u32 rf = s_from[qi];
        u32 rt = s_to[qi];
        bool has = rf < rt;
        bool in_m = (q0 + qi) < m;
        u32 mn_f = has ? rf : m;
        u32 mx_t = has ? rt : 0u;
        u32 mx_f = in_m ? rf : 0u;
        u32 mn_t = in_m ? rt : 0xFFFFFFFFu;
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            mn_f = umin_u32(mn_f, shfl_xor_u32(mn_f, off));
            mx_t = umax_u32(mx_t, shfl_xor_u32(mx_t, off));
            mx_f = umax_u32(mx_f, shfl_xor_u32(mx_f, off));
            mn_t = umin_u32(mn_t, shfl_xor_u32(mn_t, off));
        }
        tr[qw] = TileRange{0u, 0u, mn_f, mx_f, mn_t, mx_t};
        w_lo = umin_u32(w_lo, mn_f);
        w_hi = umax_u32(w_hi, mx_t);
    }

    if (lane == 0u) {
        s_wmin[warp] = w_lo;
        s_wmax[warp] = w_hi;
    }
    prim::block_sync();

    kv_start = s_wmin[0];
    kv_end = s_wmax[0];
#pragma unroll
    for (u32 w = 1u; w < WARPS; w++) {
        kv_start = umin_u32(kv_start, s_wmin[w]);
        kv_end = umax_u32(kv_end, s_wmax[w]);
    }
}

struct Fa2Tile {
    u32 hq, q0, hkv;
    float sc_l2;
};

template <u32 BQ>
TESSERA_FN Fa2Tile fa2_tile_info(u32 m, u32 heads, u32 kv_heads, float scale, u32 causal) {
    u32 num_qt = (m + BQ - 1u) / BQ;
    u32 hq = blockIdx.x;
    u32 q_raw = blockIdx.y;
    if (causal == 0u && heads == kv_heads) {
        u32 lin = blockIdx.y * heads + blockIdx.x;
        hq = lin / num_qt;
        q_raw = lin - hq * num_qt;
    }
    u32 q_tile = (causal != 0u) ? (num_qt - 1u - q_raw) : q_raw;
    return Fa2Tile{hq, q_tile * BQ, hq / (heads / kv_heads), scale * FA2_LOG2E};
}

#if defined(TESSERA_CUDA)

template <u32 HPAD, u32 WARPS, u32 QPW>
TESSERA_FN void fa2_cuda_run(const float* q, const float* k, const float* v, const u32* bounds,
                             float* o, float scale, u32 m, u32 heads, u32 kv_heads, u32 hd,
                             u32 causal) {
    constexpr u32 BQ = WARPS * QPW * 16u;
    constexpr u32 NT = WARPS * 32u;
    constexpr u32 STRIDE = HPAD + 8u;
    constexpr u32 KS = HPAD / 16u;
    constexpr u32 DT = HPAD / 8u;
    constexpr u32 VECS = HPAD / 4u;

    __shared__ __align__(16) u16 s_k[64][STRIDE];
    __shared__ __align__(16) u16 s_v[64][STRIDE];
    __shared__ u32 s_from[BQ];
    __shared__ u32 s_to[BQ];
    __shared__ u32 s_wmin[WARPS];
    __shared__ u32 s_wmax[WARPS];

    u32 tid = threadIdx.x;
    u32 warp = tid >> 5u;
    u32 lane = tid & 31u;
    u32 r = lane >> 2u;
    u32 c = (lane & 3u) << 1u;

    Fa2Tile tile = fa2_tile_info<BQ>(m, heads, kv_heads, scale, causal);
    u32 hq = tile.hq, q0 = tile.q0, hkv = tile.hkv;
    float sc_l2 = tile.sc_l2;

    TileRange tr[QPW];
    u32 kv_start = 0u;
    u32 kv_end = 0u;
    load_bounds<WARPS, QPW>(bounds, s_from, s_to, s_wmin, s_wmax, q0, m, causal, tid, warp, lane,
                            tr, kv_start, kv_end);

    u32 q_frag[QPW][KS][4];
#pragma unroll
    for (u32 qw = 0u; qw < QPW; qw++) {
        u32 q_row0 = q0 + (warp * QPW + qw) * 16u + r;
        u32 q_row1 = q_row0 + 8u;
        const float* qp0 = (q_row0 < m) ? (q + (u64(q_row0) * heads + hq) * hd) : nullptr;
        const float* qp1 = (q_row1 < m) ? (q + (u64(q_row1) * heads + hq) * hd) : nullptr;
#pragma unroll
        for (u32 ks = 0u; ks < KS; ks++) {
            u32 col0 = ks * 16u + c;
            u32 col1 = col0 + 8u;
            q_frag[qw][ks][0] = (qp0 != nullptr && col0 < hd) ? load_f16x2(qp0 + col0) : 0u;
            q_frag[qw][ks][1] = (qp1 != nullptr && col0 < hd) ? load_f16x2(qp1 + col0) : 0u;
            q_frag[qw][ks][2] = (qp0 != nullptr && col1 < hd) ? load_f16x2(qp0 + col1) : 0u;
            q_frag[qw][ks][3] = (qp1 != nullptr && col1 < hd) ? load_f16x2(qp1 + col1) : 0u;
        }
    }

    float o_acc[QPW][DT][4];
    float m_max[QPW][2];
    float l_sum[QPW][2];
#pragma unroll
    for (u32 qw = 0u; qw < QPW; qw++) {
#pragma unroll
        for (u32 dt = 0u; dt < DT; dt++) {
            o_acc[qw][dt][0] = 0.0f;
            o_acc[qw][dt][1] = 0.0f;
            o_acc[qw][dt][2] = 0.0f;
            o_acc[qw][dt][3] = 0.0f;
        }
        m_max[qw][0] = FA2_NEG_INF;
        m_max[qw][1] = FA2_NEG_INF;
        l_sum[qw][0] = 0.0f;
        l_sum[qw][1] = 0.0f;
    }

    u32 krow_l = (lane & 7u) + ((lane >> 4u) << 3u);
    u32 kcol_off = ((lane >> 3u) & 1u) << 3u;
    u32 vrow_off = lane & 15u;
    u32 vcol_off = (lane >> 4u) << 3u;

    for (u32 k0 = kv_start; k0 < kv_end; k0 += 64u) {
        u32 kt = umin_u32(64u, kv_end - k0);
        for (u32 idx = tid; idx < 64u * VECS; idx += NT) {
            u32 kr = idx / VECS;
            u32 kc = (idx % VECS) * 4u;
            if (kr < kt && kc < hd) {
                u64 off = (u64(k0 + kr) * kv_heads + hkv) * hd + kc;
                float4 kv = *reinterpret_cast<const float4*>(k + off);
                float4 vv = *reinterpret_cast<const float4*>(v + off);
                *reinterpret_cast<uint2*>(&s_k[kr][kc]) =
                    make_uint2(pack_f16x2(kv.x, kv.y), pack_f16x2(kv.z, kv.w));
                *reinterpret_cast<uint2*>(&s_v[kr][kc]) =
                    make_uint2(pack_f16x2(vv.x, vv.y), pack_f16x2(vv.z, vv.w));
            } else {
                *reinterpret_cast<uint2*>(&s_k[kr][kc]) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(&s_v[kr][kc]) = make_uint2(0u, 0u);
            }
        }
        prim::block_sync();

#pragma unroll
        for (u32 qw = 0u; qw < QPW; qw++) {
            bool warp_on = tr[qw].w_min_from < tr[qw].w_max_to && (k0 + 64u) > tr[qw].w_min_from &&
                           k0 < tr[qw].w_max_to;
            if (!warp_on) continue;

            float sc[8][4];
#pragma unroll
            for (u32 nt = 0u; nt < 8u; nt++) {
                sc[nt][0] = 0.0f;
                sc[nt][1] = 0.0f;
                sc[nt][2] = 0.0f;
                sc[nt][3] = 0.0f;
            }
#pragma unroll
            for (u32 ks = 0u; ks < KS; ks++) {
                u32 kcol = ks * 16u + kcol_off;
#pragma unroll
                for (u32 np = 0u; np < 4u; np++) {
                    u32 b0, b1, b2, b3;
                    ldm_x4(b0, b1, b2, b3, &s_k[np * 16u + krow_l][kcol]);
                    u32 ba[2] = {b0, b1};
                    u32 bb[2] = {b2, b3};
                    mma_f16(sc[np * 2u], q_frag[qw][ks], ba);
                    mma_f16(sc[np * 2u + 1u], q_frag[qw][ks], bb);
                }
            }

            bool all_in = (k0 >= tr[qw].w_max_from) && (k0 + 64u <= tr[qw].w_min_to);
            u32 row_base = (warp * QPW + qw) * 16u + r;
            u32 my_from0 = all_in ? 0u : s_from[row_base];
            u32 my_to0 = all_in ? 0xFFFFFFFFu : s_to[row_base];
            u32 my_from1 = all_in ? 0u : s_from[row_base + 8u];
            u32 my_to1 = all_in ? 0xFFFFFFFFu : s_to[row_base + 8u];
            float rmax0 = FA2_NEG_INF;
            float rmax1 = FA2_NEG_INF;
#pragma unroll
            for (u32 nt = 0u; nt < 8u; nt++) {
                u32 kp0 = k0 + nt * 8u + c;
                u32 kp1 = kp0 + 1u;
                float s0 = sc[nt][0] * sc_l2;
                float s1 = sc[nt][1] * sc_l2;
                float s2 = sc[nt][2] * sc_l2;
                float s3 = sc[nt][3] * sc_l2;
                if (!all_in) {
                    if (kp0 < my_from0 || kp0 >= my_to0) s0 = FA2_NEG_INF;
                    if (kp1 < my_from0 || kp1 >= my_to0) s1 = FA2_NEG_INF;
                    if (kp0 < my_from1 || kp0 >= my_to1) s2 = FA2_NEG_INF;
                    if (kp1 < my_from1 || kp1 >= my_to1) s3 = FA2_NEG_INF;
                }
                sc[nt][0] = s0;
                sc[nt][1] = s1;
                sc[nt][2] = s2;
                sc[nt][3] = s3;
                rmax0 = fmax_fast(rmax0, fmax_fast(s0, s1));
                rmax1 = fmax_fast(rmax1, fmax_fast(s2, s3));
            }
            rmax0 = fmax_fast(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
            rmax0 = fmax_fast(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
            rmax1 = fmax_fast(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
            rmax1 = fmax_fast(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));

            float m_new0 = fmax_fast(m_max[qw][0], rmax0);
            float m_new1 = fmax_fast(m_max[qw][1], rmax1);
            float alpha0 = (m_max[qw][0] > FA2_MASK_CUT) ? exp2_fast(m_max[qw][0] - m_new0) : 0.0f;
            float alpha1 = (m_max[qw][1] > FA2_MASK_CUT) ? exp2_fast(m_max[qw][1] - m_new1) : 0.0f;

#pragma unroll
            for (u32 dt = 0u; dt < DT; dt++) {
                o_acc[qw][dt][0] *= alpha0;
                o_acc[qw][dt][1] *= alpha0;
                o_acc[qw][dt][2] *= alpha1;
                o_acc[qw][dt][3] *= alpha1;
            }
            l_sum[qw][0] *= alpha0;
            l_sum[qw][1] *= alpha1;

            float rsum0 = 0.0f;
            float rsum1 = 0.0f;
            u32 pf[4][4];
#pragma unroll
            for (u32 ks = 0u; ks < 4u; ks++) {
                u32 n0 = ks * 2u;
                u32 n1 = n0 + 1u;
                float p00 = (sc[n0][0] > FA2_MASK_CUT) ? exp2_fast(sc[n0][0] - m_new0) : 0.0f;
                float p01 = (sc[n0][1] > FA2_MASK_CUT) ? exp2_fast(sc[n0][1] - m_new0) : 0.0f;
                float p02 = (sc[n0][2] > FA2_MASK_CUT) ? exp2_fast(sc[n0][2] - m_new1) : 0.0f;
                float p03 = (sc[n0][3] > FA2_MASK_CUT) ? exp2_fast(sc[n0][3] - m_new1) : 0.0f;
                float p10 = (sc[n1][0] > FA2_MASK_CUT) ? exp2_fast(sc[n1][0] - m_new0) : 0.0f;
                float p11 = (sc[n1][1] > FA2_MASK_CUT) ? exp2_fast(sc[n1][1] - m_new0) : 0.0f;
                float p12 = (sc[n1][2] > FA2_MASK_CUT) ? exp2_fast(sc[n1][2] - m_new1) : 0.0f;
                float p13 = (sc[n1][3] > FA2_MASK_CUT) ? exp2_fast(sc[n1][3] - m_new1) : 0.0f;
                rsum0 += (p00 + p01) + (p10 + p11);
                rsum1 += (p02 + p03) + (p12 + p13);
                pf[ks][0] = pack_f16x2(p00, p01);
                pf[ks][1] = pack_f16x2(p02, p03);
                pf[ks][2] = pack_f16x2(p10, p11);
                pf[ks][3] = pack_f16x2(p12, p13);
            }
            rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 1);
            rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 2);
            rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 1);
            rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 2);
            l_sum[qw][0] += rsum0;
            l_sum[qw][1] += rsum1;
            m_max[qw][0] = m_new0;
            m_max[qw][1] = m_new1;

#pragma unroll
            for (u32 ks = 0u; ks < 4u; ks++) {
                u32 vrow = ks * 16u + vrow_off;
#pragma unroll
                for (u32 dp = 0u; dp < DT / 2u; dp++) {
                    u32 b0, b1, b2, b3;
                    ldm_x4_t(b0, b1, b2, b3, &s_v[vrow][dp * 16u + vcol_off]);
                    u32 ba[2] = {b0, b1};
                    u32 bb[2] = {b2, b3};
                    mma_f16(o_acc[qw][dp * 2u], pf[ks], ba);
                    mma_f16(o_acc[qw][dp * 2u + 1u], pf[ks], bb);
                }
            }
        }
        prim::block_sync();
    }

#pragma unroll
    for (u32 qw = 0u; qw < QPW; qw++) {
        u32 q_row0 = q0 + (warp * QPW + qw) * 16u + r;
        u32 q_row1 = q_row0 + 8u;
        float inv0 = (l_sum[qw][0] > 0.0f) ? (1.0f / l_sum[qw][0]) : 0.0f;
        float inv1 = (l_sum[qw][1] > 0.0f) ? (1.0f / l_sum[qw][1]) : 0.0f;
        float* op0 = (q_row0 < m) ? (o + (u64(q_row0) * heads + hq) * hd) : nullptr;
        float* op1 = (q_row1 < m) ? (o + (u64(q_row1) * heads + hq) * hd) : nullptr;
#pragma unroll
        for (u32 dt = 0u; dt < DT; dt++) {
            u32 dc = dt * 8u + c;
            if (dc < hd) {
                if (op0 != nullptr) {
                    *reinterpret_cast<float2*>(op0 + dc) =
                        make_float2(o_acc[qw][dt][0] * inv0, o_acc[qw][dt][1] * inv0);
                }
                if (op1 != nullptr) {
                    *reinterpret_cast<float2*>(op1 + dc) =
                        make_float2(o_acc[qw][dt][2] * inv1, o_acc[qw][dt][3] * inv1);
                }
            }
        }
    }
}

#else

// Prefetches one K and V tile into registers.
template <u32 HPAD, u32 WARPS, u32 BK, u32 LP>
TESSERA_FN void rocm_fetch_kv(const float* k, const float* v, u32 k0, u32 kv_end, u32 kv_heads,
                              u32 hkv, u32 hd, u32 tid, float4 rk[LP], float4 rv[LP]) {
    constexpr u32 NT = WARPS * 32u;
    constexpr u32 VECS = HPAD / 4u;
    constexpr u32 TOT = BK * VECS;
    u32 kt = (k0 < kv_end) ? umin_u32(BK, kv_end - k0) : 0u;
#pragma unroll
    for (u32 p = 0u; p < LP; p++) {
        u32 idx = tid + p * NT;
        if (idx < TOT) {
            u32 kr = idx / VECS;
            u32 kc = (idx % VECS) * 4u;
            if (kr < kt && kc < hd) {
                u64 off = (u64(k0 + kr) * kv_heads + hkv) * hd + kc;
                rk[p] = *reinterpret_cast<const float4*>(k + off);
                rv[p] = *reinterpret_cast<const float4*>(v + off);
            } else {
                rk[p] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                rv[p] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }
}

// Converts and writes one K and transposed V tile.
template <u32 HPAD, u32 WARPS, u32 BK, u32 LP>
TESSERA_FN void rocm_store_kv(rocm_kv_buf<HPAD, BK>& dst, u32 tid, const float4 rk[LP],
                              const float4 rv[LP]) {
    constexpr u32 NT = WARPS * 32u;
    constexpr u32 VECS = HPAD / 4u;
    constexpr u32 TOT = BK * VECS;
#pragma unroll
    for (u32 p = 0u; p < LP; p++) {
        u32 idx = tid + p * NT;
        if (idx < TOT) {
            u32 kr = idx / VECS;
            u32 kc = (idx % VECS) * 4u;
            u32 k01 = pack_clamped(rk[p].x, rk[p].y);
            u32 k23 = pack_clamped(rk[p].z, rk[p].w);
            *reinterpret_cast<uint2*>(&dst.s_k[kr][kc]) = make_uint2(k01, k23);
            f16x2 v01 = __builtin_bit_cast(f16x2, pack_clamped(rv[p].x, rv[p].y));
            f16x2 v23 = __builtin_bit_cast(f16x2, pack_clamped(rv[p].z, rv[p].w));
            dst.s_vt[kc][kr ^ vt_sw<BK>(kc)] = v01[0];
            dst.s_vt[kc + 1u][kr ^ vt_sw<BK>(kc + 1u)] = v01[1];
            dst.s_vt[kc + 2u][kr ^ vt_sw<BK>(kc + 2u)] = v23[0];
            dst.s_vt[kc + 3u][kr ^ vt_sw<BK>(kc + 3u)] = v23[1];
        }
    }
}

template <u32 HPAD, u32 BK>
TESSERA_FN f16x16 rocm_load_vf(const rocm_kv_buf<HPAD, BK>& src, u32 dt_idx, u32 nt_idx,
                               u32 col16) {
    u32 d = dt_idx * 16u + col16;
    u32 sw = vt_sw<BK>(d);
    u32x4 lo = *reinterpret_cast<const u32x4*>(&src.s_vt[d][(nt_idx * 16u) ^ sw]);
    u32x4 hi = *reinterpret_cast<const u32x4*>(&src.s_vt[d][(nt_idx * 16u + 8u) ^ sw]);
    u32x8 vw = {lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
    return __builtin_bit_cast(f16x16, vw);
}

// Runs dual-transpose QK, register-only softmax, and PV WMMA.
template <u32 HPAD, u32 BK>
TESSERA_FN void rocm_tile_step(const f16x16 q_frag[HPAD / 16u], const rocm_kv_buf<HPAD, BK>& src,
                               v8f o_acc[HPAD / 16u], float& m_max, float& l_sum,
                               const TileRange& tr, u32 k0, u32 my_from, u32 my_to, u32 col16,
                               u32 half, u32 sel0, u32 sel1) {
    constexpr u32 KS = HPAD / 16u;
    constexpr u32 DT = HPAD / 16u;
    constexpr u32 NT_K = BK / 16u;
    bool warp_on = tr.w_min_from < tr.w_max_to && (k0 + BK) > tr.w_min_from && k0 < tr.w_max_to;
    if (!warp_on) return;

    v8f sc[NT_K];
#pragma unroll
    for (u32 nt = 0u; nt < NT_K; nt++) sc[nt] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

    f16x16 kf0 = *reinterpret_cast<const f16x16*>(&src.s_k[col16][0]);
#pragma unroll
    for (u32 ks = 0u; ks < KS; ks++) {
#pragma unroll
        for (u32 nt = 0u; nt < NT_K; nt++) {
            u32 next_nt = nt + 1u;
            u32 next_ks = ks;
            if (next_nt == NT_K) {
                next_nt = 0u;
                next_ks = ks + 1u;
            }
            f16x16 kf_next;
            if (next_ks < KS) {
                kf_next = *reinterpret_cast<const f16x16*>(
                    &src.s_k[next_nt * 16u + col16][next_ks * 16u]);
            }
            sc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf0, q_frag[ks], sc[nt]);
            kf0 = kf_next;
        }
    }

    bool all_in = (k0 >= tr.w_max_from) && (k0 + BK <= tr.w_min_to);
    if (__builtin_expect(!all_in, 0)) {
#pragma unroll
        for (u32 nt = 0u; nt < NT_K; nt++) {
#pragma unroll
            for (u32 e = 0u; e < 8u; e++) {
                u32 kp = k0 + nt * 16u + 2u * e + half;
                if (kp < my_from || kp >= my_to) sc[nt][e] = FA2_NEG_INF;
            }
        }
    }

    float rm = FA2_NEG_INF;
#pragma unroll
    for (u32 nt = 0u; nt < NT_K; nt++) {
        float a0 = fmax_fast(fmax_fast(sc[nt][0], sc[nt][1]), fmax_fast(sc[nt][2], sc[nt][3]));
        float a1 = fmax_fast(fmax_fast(sc[nt][4], sc[nt][5]), fmax_fast(sc[nt][6], sc[nt][7]));
        rm = fmax_fast(rm, fmax_fast(a0, a1));
    }
    rm = fmax_fast(rm, __shfl_xor(rm, 16));
    float m_new = fmax_fast(m_max, rm);
    float m_safe = (m_new > FA2_MASK_CUT) ? m_new : 0.0f;
    float alpha = exp2_fast(m_max - m_safe);
#pragma unroll
    for (u32 dt = 0u; dt < DT; dt++) {
#pragma unroll
        for (u32 e = 0u; e < 8u; e++) o_acc[dt][e] *= alpha;
    }

    float rsum = 0.0f;
#pragma unroll
    for (u32 nt = 0u; nt < NT_K; nt++) {
        float p0 = exp2_fast(sc[nt][0] - m_safe);
        float p1 = exp2_fast(sc[nt][1] - m_safe);
        float p2 = exp2_fast(sc[nt][2] - m_safe);
        float p3 = exp2_fast(sc[nt][3] - m_safe);
        float p4 = exp2_fast(sc[nt][4] - m_safe);
        float p5 = exp2_fast(sc[nt][5] - m_safe);
        float p6 = exp2_fast(sc[nt][6] - m_safe);
        float p7 = exp2_fast(sc[nt][7] - m_safe);
        rsum += ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
        u32 u0 = pack_f16_raw(p0, p1);
        u32 u1 = pack_f16_raw(p2, p3);
        u32 u2 = pack_f16_raw(p4, p5);
        u32 u3 = pack_f16_raw(p6, p7);
        u32 q0p = __shfl_xor(u0, 16);
        u32 q1p = __shfl_xor(u1, 16);
        u32 q2p = __shfl_xor(u2, 16);
        u32 q3p = __shfl_xor(u3, 16);
        u32x8 pw = {
            __builtin_amdgcn_perm(q0p, u0, sel0), __builtin_amdgcn_perm(q0p, u0, sel1),
            __builtin_amdgcn_perm(q1p, u1, sel0), __builtin_amdgcn_perm(q1p, u1, sel1),
            __builtin_amdgcn_perm(q2p, u2, sel0), __builtin_amdgcn_perm(q2p, u2, sel1),
            __builtin_amdgcn_perm(q3p, u3, sel0), __builtin_amdgcn_perm(q3p, u3, sel1),
        };
        f16x16 pf = __builtin_bit_cast(f16x16, pw);
        f16x16 vf0 = rocm_load_vf<HPAD, BK>(src, 0u, nt, col16);
#pragma unroll
        for (u32 dt = 0u; dt < DT; dt++) {
            f16x16 vf_next;
            if (dt + 1u < DT) vf_next = rocm_load_vf<HPAD, BK>(src, dt + 1u, nt, col16);
            o_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf0, pf, o_acc[dt]);
            vf0 = vf_next;
        }
    }
    rsum += __shfl_xor(rsum, 16);
    l_sum = l_sum * alpha + rsum;
    m_max = m_new;
}

template <u32 HPAD, u32 WARPS, u32 BK, bool DBUF>
TESSERA_FN void fa2_rocm_run(const float* q, const float* k, const float* v, const u32* bounds,
                             float* o, float scale, u32 m, u32 heads, u32 kv_heads, u32 hd,
                             u32 causal) {
    constexpr u32 BQ = WARPS * 16u;
    constexpr u32 NT = WARPS * 32u;
    constexpr u32 KS = HPAD / 16u;
    constexpr u32 DT = HPAD / 16u;
    constexpr u32 LP = (BK * (HPAD / 4u) + NT - 1u) / NT;

    __shared__ alignas(16) rocm_smem<HPAD, WARPS, BK, DBUF> sm;

    u32 tid = threadIdx.x;
    u32 warp = tid >> 5u;
    u32 lane = tid & 31u;
    u32 col16 = lane & 15u;
    u32 half = lane >> 4u;

    Fa2Tile tile = fa2_tile_info<BQ>(m, heads, kv_heads, scale, causal);
    u32 hq = tile.hq, q0 = tile.q0, hkv = tile.hkv;
    float sc_l2 = tile.sc_l2;

    TileRange tr[1];
    u32 kv_start = 0u;
    u32 kv_end = 0u;
    load_bounds<WARPS, 1u>(bounds, sm.bnd.s_from, sm.bnd.s_to, sm.bnd.s_wmin, sm.bnd.s_wmax, q0, m,
                           causal, tid, warp, lane, tr, kv_start, kv_end);

    u32 q_local = warp * 16u + col16;
    u32 my_from = sm.bnd.s_from[q_local];
    u32 my_to = sm.bnd.s_to[q_local];
    lds_barrier();

    u32 qa_row = q0 + q_local;
    const float* qp = (qa_row < m) ? (q + (u64(qa_row) * heads + hq) * hd) : nullptr;

    f16x16 q_frag[KS];
#pragma unroll
    for (u32 ks = 0u; ks < KS; ks++) {
        u32x8 qw_bits = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
            u32 col = ks * 16u + i * 4u;
            if (qp != nullptr && col < hd) {
                float4 fv = *reinterpret_cast<const float4*>(qp + col);
                qw_bits[i * 2u] = pack_clamped(fv.x * sc_l2, fv.y * sc_l2);
                qw_bits[i * 2u + 1u] = pack_clamped(fv.z * sc_l2, fv.w * sc_l2);
            }
        }
        q_frag[ks] = __builtin_bit_cast(f16x16, qw_bits);
    }

    v8f o_acc[DT];
#pragma unroll
    for (u32 dt = 0u; dt < DT; dt++) o_acc[dt] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    float m_max = FA2_NEG_INF;
    float l_sum = 0.0f;
    u32 sel0 = (half == 0u) ? 0x05040100u : 0x01000504u;
    u32 sel1 = (half == 0u) ? 0x07060302u : 0x03020706u;

    if (kv_start < kv_end) {
        float4 rk[LP];
        float4 rv[LP];
        rocm_fetch_kv<HPAD, WARPS, BK, LP>(k, v, kv_start, kv_end, kv_heads, hkv, hd, tid, rk, rv);
        if constexpr (DBUF) {
            rocm_store_kv<HPAD, WARPS, BK, LP>(sm.kv[0], tid, rk, rv);
            lds_barrier();
            u32 buf = 0u;
            for (u32 k0 = kv_start; k0 < kv_end; k0 += BK) {
                u32 next_k0 = k0 + BK;
                if (next_k0 < kv_end) {
                    rocm_fetch_kv<HPAD, WARPS, BK, LP>(k, v, next_k0, kv_end, kv_heads, hkv, hd,
                                                       tid, rk, rv);
                }
                rocm_tile_step<HPAD, BK>(q_frag, sm.kv[buf], o_acc, m_max, l_sum, tr[0], k0,
                                         my_from, my_to, col16, half, sel0, sel1);
                if (next_k0 < kv_end) {
                    rocm_store_kv<HPAD, WARPS, BK, LP>(sm.kv[buf ^ 1u], tid, rk, rv);
                    lds_barrier();
                    buf ^= 1u;
                }
            }
        } else {
            for (u32 k0 = kv_start; k0 < kv_end; k0 += BK) {
                rocm_store_kv<HPAD, WARPS, BK, LP>(sm.kv[0], tid, rk, rv);
                lds_barrier();
                u32 next_k0 = k0 + BK;
                if (next_k0 < kv_end) {
                    rocm_fetch_kv<HPAD, WARPS, BK, LP>(k, v, next_k0, kv_end, kv_heads, hkv, hd,
                                                       tid, rk, rv);
                }
                rocm_tile_step<HPAD, BK>(q_frag, sm.kv[0], o_acc, m_max, l_sum, tr[0], k0, my_from,
                                         my_to, col16, half, sel0, sel1);
                if (next_k0 < kv_end) lds_barrier();
            }
        }
    }

    u32 row = q0 + warp * 16u + col16;
    if (row < m) {
        float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        float* op = o + (u64(row) * heads + hq) * hd;
#pragma unroll
        for (u32 dt = 0u; dt < DT; dt++) {
#pragma unroll
            for (u32 e = 0u; e < 8u; e++) {
                u32 dc = dt * 16u + 2u * e + half;
                if (dc < hd) op[dc] = o_acc[dt][e] * inv;
            }
        }
    }
}

#endif

}  // namespace

#if defined(TESSERA_CUDA)
#define FA2_PACKED_DEF(NAME, HPAD, CW, CQ, RW, RBK, RDB)                                     \
    extern "C" __global__ __launch_bounds__(CW * 32, CW == 4u ? 2 : 1) void NAME(            \
        const float* q, const float* k, const float* v, const u32* bounds, float* o,         \
        float scale, u32 m, u32 heads, u32 kv_heads, u32 hd, u32 causal) {                   \
        fa2_cuda_run<HPAD, CW, CQ>(q, k, v, bounds, o, scale, m, heads, kv_heads, hd,        \
                                   causal);                                                  \
    }
#else
#define FA2_PACKED_DEF(NAME, HPAD, CW, CQ, RW, RBK, RDB)                                     \
    extern "C" __global__ __launch_bounds__(RW * 32) void NAME(                              \
        const float* q, const float* k, const float* v, const u32* bounds, float* o,         \
        float scale, u32 m, u32 heads, u32 kv_heads, u32 hd, u32 causal) {                   \
        fa2_rocm_run<HPAD, RW, RBK, RDB>(q, k, v, bounds, o, scale, m, heads, kv_heads, hd,  \
                                         causal);                                            \
    }
#endif

FA2_PACKED_DEF(fa2_packed_d16, 16u, 4u, 1u, 4u, 64u, false)
FA2_PACKED_DEF(fa2_packed_d64, 64u, 8u, 2u, 16u, 64u, false)
FA2_PACKED_DEF(fa2_packed_d80, 80u, 8u, 2u, 16u, 32u, true)
FA2_PACKED_DEF(fa2_packed_d128, 128u, 8u, 1u, 8u, 32u, true)

#undef FA2_PACKED_DEF

#endif
