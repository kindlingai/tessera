// Native Metal decode and MMA kernels for stacked matmuls.

static inline float st_bf16_to_f32(ushort u) {
    return as_type<float>(uint(u) << 16u);
}

static inline ushort st_f32_to_bf16(float f) {
    uint u = as_type<uint>(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return ushort((u >> 16u) | 0x0040u);
    }
    uint bias = 0x7FFFu + ((u >> 16u) & 1u);
    return ushort((u + bias) >> 16u);
}

static inline float st_load1_dyn(const device uchar* p, ulong idx, uint fmt) {
    if (fmt == 0u) {
        return reinterpret_cast<const device float*>(p)[idx];
    }
    if (fmt == 1u) {
        return st_bf16_to_f32(reinterpret_cast<const device ushort*>(p)[idx]);
    }
    return float(reinterpret_cast<const device half*>(p)[idx]);
}

static inline float4 st_load4_dyn(const device uchar* p, ulong idx, uint fmt) {
    if (fmt == 0u) {
        const device float* f = reinterpret_cast<const device float*>(p) + idx;
        return float4(*reinterpret_cast<const device packed_float4*>(f));
    }
    if (fmt == 1u) {
        const device ushort* u = reinterpret_cast<const device ushort*>(p) + idx;
        ushort4 v = ushort4(*reinterpret_cast<const device packed_ushort4*>(u));
        return float4(
            st_bf16_to_f32(v.x),
            st_bf16_to_f32(v.y),
            st_bf16_to_f32(v.z),
            st_bf16_to_f32(v.w));
    }
    const device half* h = reinterpret_cast<const device half*>(p) + idx;
    return float4(half4(*reinterpret_cast<const device packed_half4*>(h)));
}

static inline void st_store1_dyn(device uchar* p, ulong idx, float v, uint fmt) {
    if (fmt == 0u) {
        reinterpret_cast<device float*>(p)[idx] = v;
    } else if (fmt == 1u) {
        reinterpret_cast<device ushort*>(p)[idx] = st_f32_to_bf16(v);
    } else {
        reinterpret_cast<device half*>(p)[idx] = half(v);
    }
}

static inline void st_store2_dyn(device uchar* p, ulong idx, float2 v, uint fmt) {
    if (fmt == 0u) {
        device float* f = reinterpret_cast<device float*>(p) + idx;
        *reinterpret_cast<device packed_float2*>(f) = packed_float2(v);
    } else if (fmt == 1u) {
        device ushort* u = reinterpret_cast<device ushort*>(p) + idx;
        *reinterpret_cast<device packed_ushort2*>(u) =
            packed_ushort2(st_f32_to_bf16(v.x), st_f32_to_bf16(v.y));
    } else {
        device half* h = reinterpret_cast<device half*>(p) + idx;
        *reinterpret_cast<device packed_half2*>(h) = packed_half2(half2(v));
    }
}

static inline void st_store4_dyn(device uchar* p, ulong idx, float4 v, uint fmt) {
    if (fmt == 0u) {
        device float* f = reinterpret_cast<device float*>(p) + idx;
        *reinterpret_cast<device packed_float4*>(f) = packed_float4(v);
    } else if (fmt == 1u) {
        device ushort* u = reinterpret_cast<device ushort*>(p) + idx;
        *reinterpret_cast<device packed_ushort4*>(u) = packed_ushort4(
            st_f32_to_bf16(v.x),
            st_f32_to_bf16(v.y),
            st_f32_to_bf16(v.z),
            st_f32_to_bf16(v.w));
    } else {
        device half* h = reinterpret_cast<device half*>(p) + idx;
        *reinterpret_cast<device packed_half4*>(h) = packed_half4(half4(v));
    }
}

template <uint FMT>
static inline float st_w_load1(const device uchar* p, ulong idx) {
    if constexpr (FMT == 0u) {
        return reinterpret_cast<const device float*>(p)[idx];
    } else if constexpr (FMT == 1u) {
        return st_bf16_to_f32(reinterpret_cast<const device ushort*>(p)[idx]);
    } else if constexpr (FMT == 2u) {
        return float(reinterpret_cast<const device half*>(p)[idx]);
    } else {
        return FP8_LUT[p[idx]];
    }
}

template <uint FMT>
static inline float2 st_w_load2(const device uchar* p, ulong idx) {
    if constexpr (FMT == 0u) {
        const device float* f = reinterpret_cast<const device float*>(p) + idx;
        return float2(*reinterpret_cast<const device packed_float2*>(f));
    } else if constexpr (FMT == 1u) {
        const device ushort* u = reinterpret_cast<const device ushort*>(p) + idx;
        ushort2 v = ushort2(*reinterpret_cast<const device packed_ushort2*>(u));
        return float2(st_bf16_to_f32(v.x), st_bf16_to_f32(v.y));
    } else if constexpr (FMT == 2u) {
        const device half* h = reinterpret_cast<const device half*>(p) + idx;
        return float2(half2(*reinterpret_cast<const device packed_half2*>(h)));
    } else {
        uchar2 v = uchar2(*reinterpret_cast<const device packed_uchar2*>(p + idx));
        return float2(FP8_LUT[v.x], FP8_LUT[v.y]);
    }
}

template <uint FMT>
static inline float4 st_w_load4(const device uchar* p, ulong idx) {
    if constexpr (FMT == 0u) {
        const device float* f = reinterpret_cast<const device float*>(p) + idx;
        return float4(*reinterpret_cast<const device packed_float4*>(f));
    } else if constexpr (FMT == 1u) {
        const device ushort* u = reinterpret_cast<const device ushort*>(p) + idx;
        ushort4 v = ushort4(*reinterpret_cast<const device packed_ushort4*>(u));
        return float4(
            st_bf16_to_f32(v.x),
            st_bf16_to_f32(v.y),
            st_bf16_to_f32(v.z),
            st_bf16_to_f32(v.w));
    } else if constexpr (FMT == 2u) {
        const device half* h = reinterpret_cast<const device half*>(p) + idx;
        return float4(half4(*reinterpret_cast<const device packed_half4*>(h)));
    } else {
        uchar4 v = uchar4(*reinterpret_cast<const device packed_uchar4*>(p + idx));
        return float4(
            FP8_LUT[v.x],
            FP8_LUT[v.y],
            FP8_LUT[v.z],
            FP8_LUT[v.w]);
    }
}

// Decode non-transposed stacked weights with 2 columns per warp.
template <uint W_FMT>
static inline void st_dec_nt_body(
    const device uchar* x,
    const device uchar* w,
    device uchar* out,
    uint batches,
    uint rows,
    uint n,
    uint k,
    uint block,
    uint first,
    uint col_step,
    uint flags,
    uint2 tgid,
    uint tid) {
    uint batch = tgid.y;
    if (batch >= batches) {
        return;
    }
    uint wid = tid >> 5u;
    uint lane = tid & 31u;
    uint c0 = tgid.x * 8u + wid * 2u;
    uint c1 = c0 + 1u;
    if (c0 >= n) {
        return;
    }
    bool has1 = c1 < n;
    uint x_fmt = flags & 3u;
    uint o_fmt = (flags >> 2u) & 3u;
    ulong w_base = ulong(batch) * ulong(block) + ulong(first);
    ulong wb0 = w_base + ulong(c0) * ulong(col_step);
    ulong wb1 = w_base + ulong(c1) * ulong(col_step);
    ulong x_base = ulong(batch) * ulong(rows) * ulong(k);
    ulong o_base = ulong(batch) * ulong(rows) * ulong(n);
    uint k4 = k & ~3u;

    if (rows == 1u) {
        float a0 = 0.0f;
        float a1 = 0.0f;
        if (x_fmt == 0u && has1) {
            const device float* xf = reinterpret_cast<const device float*>(x) + x_base;
            for (uint i = lane * 4u; i < k4; i += 128u) {
                float4 xv = float4(*reinterpret_cast<const device packed_float4*>(xf + i));
                a0 += dot(xv, st_w_load4<W_FMT>(w, wb0 + i));
                a1 += dot(xv, st_w_load4<W_FMT>(w, wb1 + i));
            }
        } else {
            for (uint i = lane * 4u; i < k4; i += 128u) {
                float4 xv = st_load4_dyn(x, x_base + i, x_fmt);
                a0 += dot(xv, st_w_load4<W_FMT>(w, wb0 + i));
                if (has1) {
                    a1 += dot(xv, st_w_load4<W_FMT>(w, wb1 + i));
                }
            }
        }
        if (lane == 0u) {
            for (uint i = k4; i < k; ++i) {
                float xv = st_load1_dyn(x, x_base + i, x_fmt);
                a0 += xv * st_w_load1<W_FMT>(w, wb0 + i);
                if (has1) {
                    a1 += xv * st_w_load1<W_FMT>(w, wb1 + i);
                }
            }
        }
        a0 = simd_sum(a0);
        if (has1) a1 = simd_sum(a1);
        if (lane == 0u) {
            st_store1_dyn(out, o_base + c0, a0, o_fmt);
            if (has1) st_store1_dyn(out, o_base + c1, a1, o_fmt);
        }
        return;
    }

    float a00 = 0.0f, a01 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f;
    float a20 = 0.0f, a21 = 0.0f;
    float a30 = 0.0f, a31 = 0.0f;
    bool r2 = rows > 2u;
    bool r3 = rows > 3u;

    if (x_fmt == 0u && has1 && rows == 4u) {
        const device float* xf = reinterpret_cast<const device float*>(x) + x_base;
        for (uint i = lane * 4u; i < k4; i += 128u) {
            float4 w0 = st_w_load4<W_FMT>(w, wb0 + i);
            float4 w1 = st_w_load4<W_FMT>(w, wb1 + i);
            float4 x0 = float4(*reinterpret_cast<const device packed_float4*>(xf + i));
            float4 x1 = float4(*reinterpret_cast<const device packed_float4*>(xf + ulong(k) + i));
            float4 x2 = float4(*reinterpret_cast<const device packed_float4*>(xf + ulong(2u * k) + i));
            float4 x3 = float4(*reinterpret_cast<const device packed_float4*>(xf + ulong(3u * k) + i));
            a00 += dot(x0, w0);
            a01 += dot(x0, w1);
            a10 += dot(x1, w0);
            a11 += dot(x1, w1);
            a20 += dot(x2, w0);
            a21 += dot(x2, w1);
            a30 += dot(x3, w0);
            a31 += dot(x3, w1);
        }
    } else {
        for (uint i = lane * 4u; i < k4; i += 128u) {
            float4 w0 = st_w_load4<W_FMT>(w, wb0 + i);
            float4 w1 = has1 ? st_w_load4<W_FMT>(w, wb1 + i) : float4(0.0f);
            float4 x0 = st_load4_dyn(x, x_base + i, x_fmt);
            float4 x1 = st_load4_dyn(x, x_base + ulong(k) + i, x_fmt);
            a00 += dot(x0, w0);
            if (has1) a01 += dot(x0, w1);
            a10 += dot(x1, w0);
            if (has1) a11 += dot(x1, w1);
            if (r2) {
                float4 x2 = st_load4_dyn(x, x_base + ulong(2u * k) + i, x_fmt);
                a20 += dot(x2, w0);
                if (has1) a21 += dot(x2, w1);
            }
            if (r3) {
                float4 x3 = st_load4_dyn(x, x_base + ulong(3u * k) + i, x_fmt);
                a30 += dot(x3, w0);
                if (has1) a31 += dot(x3, w1);
            }
        }
    }
    if (lane == 0u && k4 < k) {
        float t0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float t1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = k4; i < k; ++i) {
            float w0 = st_w_load1<W_FMT>(w, wb0 + i);
            float w1 = has1 ? st_w_load1<W_FMT>(w, wb1 + i) : 0.0f;
            for (uint r = 0u; r < rows; ++r) {
                float xv = st_load1_dyn(x, x_base + ulong(r) * ulong(k) + i, x_fmt);
                t0[r] += xv * w0;
                t1[r] += xv * w1;
            }
        }
        a00 += t0[0];
        a01 += t1[0];
        a10 += t0[1];
        a11 += t1[1];
        a20 += t0[2];
        a21 += t1[2];
        a30 += t0[3];
        a31 += t1[3];
    }

    a00 = simd_sum(a00);
    a10 = simd_sum(a10);
    if (has1) {
        a01 = simd_sum(a01);
        a11 = simd_sum(a11);
    }
    if (lane == 0u) {
        st_store1_dyn(out, o_base + c0, a00, o_fmt);
        st_store1_dyn(out, o_base + ulong(n) + c0, a10, o_fmt);
        if (has1) {
            st_store1_dyn(out, o_base + c1, a01, o_fmt);
            st_store1_dyn(out, o_base + ulong(n) + c1, a11, o_fmt);
        }
    }
    if (r2) {
        a20 = simd_sum(a20);
        if (has1) a21 = simd_sum(a21);
        if (lane == 0u) {
            st_store1_dyn(out, o_base + ulong(2u * n) + c0, a20, o_fmt);
            if (has1) st_store1_dyn(out, o_base + ulong(2u * n) + c1, a21, o_fmt);
        }
    }
    if (r3) {
        a30 = simd_sum(a30);
        if (has1) a31 = simd_sum(a31);
        if (lane == 0u) {
            st_store1_dyn(out, o_base + ulong(3u * n) + c0, a30, o_fmt);
            if (has1) st_store1_dyn(out, o_base + ulong(3u * n) + c1, a31, o_fmt);
        }
    }
}

struct StDecTSmem {
    float x[4][128];
};

// Decode transposed stacked weights with 256 columns per group.
template <uint W_FMT>
static inline void st_dec_t_body(
    threadgroup StDecTSmem& sm,
    const device uchar* x,
    const device uchar* w,
    device uchar* out,
    uint batches,
    uint rows,
    uint n,
    uint k,
    uint block,
    uint first,
    uint k_step,
    uint flags,
    uint2 tgid,
    uint tid) {
    uint batch = tgid.y;
    if (batch >= batches) {
        return;
    }
    uint col = tgid.x * 256u + tid * 2u;
    bool full2 = col + 2u <= n;
    uint x_fmt = flags & 3u;
    uint o_fmt = (flags >> 2u) & 3u;
    ulong w_base = ulong(batch) * ulong(block) + ulong(first);
    ulong x_base = ulong(batch) * ulong(rows) * ulong(k);
    ulong o_base = ulong(batch) * ulong(rows) * ulong(n);
    bool r1 = rows > 1u;
    bool r2 = rows > 2u;
    bool r3 = rows > 3u;

    float2 a0 = float2(0.0f);
    float2 a1 = float2(0.0f);
    float2 a2 = float2(0.0f);
    float2 a3 = float2(0.0f);

    for (uint k0 = 0u; k0 < k; k0 += 128u) {
        uint k_len = min(128u, k - k0);
        if (tid < k_len) {
            sm.x[0][tid] = st_load1_dyn(x, x_base + k0 + tid, x_fmt);
            if (r1) sm.x[1][tid] = st_load1_dyn(x, x_base + ulong(k) + k0 + tid, x_fmt);
            if (r2) sm.x[2][tid] = st_load1_dyn(x, x_base + ulong(2u * k) + k0 + tid, x_fmt);
            if (r3) sm.x[3][tid] = st_load1_dyn(x, x_base + ulong(3u * k) + k0 + tid, x_fmt);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (full2) {
            ulong wrow = w_base + ulong(k0) * ulong(k_step) + col;
            ulong ks = ulong(k_step);
            uint ki = 0u;
            if (rows == 1u) {
                for (; ki + 4u <= k_len; ki += 4u) {
                    float2 w0 = st_w_load2<W_FMT>(w, wrow);
                    float2 w1 = st_w_load2<W_FMT>(w, wrow + ks);
                    float2 w2 = st_w_load2<W_FMT>(w, wrow + 2u * ks);
                    float2 w3 = st_w_load2<W_FMT>(w, wrow + 3u * ks);
                    a0 = fma(float2(sm.x[0][ki]), w0, a0);
                    a0 = fma(float2(sm.x[0][ki + 1u]), w1, a0);
                    a0 = fma(float2(sm.x[0][ki + 2u]), w2, a0);
                    a0 = fma(float2(sm.x[0][ki + 3u]), w3, a0);
                    wrow += 4u * ks;
                }
                for (; ki < k_len; ++ki) {
                    a0 = fma(float2(sm.x[0][ki]), st_w_load2<W_FMT>(w, wrow), a0);
                    wrow += ks;
                }
            } else if (rows == 4u) {
                for (; ki + 2u <= k_len; ki += 2u) {
                    float2 w0 = st_w_load2<W_FMT>(w, wrow);
                    float2 w1 = st_w_load2<W_FMT>(w, wrow + ks);
                    a0 = fma(float2(sm.x[0][ki]), w0, a0);
                    a1 = fma(float2(sm.x[1][ki]), w0, a1);
                    a2 = fma(float2(sm.x[2][ki]), w0, a2);
                    a3 = fma(float2(sm.x[3][ki]), w0, a3);
                    a0 = fma(float2(sm.x[0][ki + 1u]), w1, a0);
                    a1 = fma(float2(sm.x[1][ki + 1u]), w1, a1);
                    a2 = fma(float2(sm.x[2][ki + 1u]), w1, a2);
                    a3 = fma(float2(sm.x[3][ki + 1u]), w1, a3);
                    wrow += 2u * ks;
                }
                for (; ki < k_len; ++ki) {
                    float2 w0 = st_w_load2<W_FMT>(w, wrow);
                    a0 = fma(float2(sm.x[0][ki]), w0, a0);
                    a1 = fma(float2(sm.x[1][ki]), w0, a1);
                    a2 = fma(float2(sm.x[2][ki]), w0, a2);
                    a3 = fma(float2(sm.x[3][ki]), w0, a3);
                    wrow += ks;
                }
            } else {
                for (; ki < k_len; ++ki) {
                    float2 w0 = st_w_load2<W_FMT>(w, wrow);
                    a0 = fma(float2(sm.x[0][ki]), w0, a0);
                    if (r1) a1 = fma(float2(sm.x[1][ki]), w0, a1);
                    if (r2) a2 = fma(float2(sm.x[2][ki]), w0, a2);
                    wrow += ks;
                }
            }
        } else if (col < n) {
            ulong wrow = w_base + ulong(k0) * ulong(k_step) + col;
            ulong ks = ulong(k_step);
            for (uint ki = 0u; ki < k_len; ++ki) {
                float w0 = st_w_load1<W_FMT>(w, wrow);
                a0.x = fma(sm.x[0][ki], w0, a0.x);
                if (r1) a1.x = fma(sm.x[1][ki], w0, a1.x);
                if (r2) a2.x = fma(sm.x[2][ki], w0, a2.x);
                if (r3) a3.x = fma(sm.x[3][ki], w0, a3.x);
                wrow += ks;
            }
        }
        if (k0 + 128u < k) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (full2) {
        st_store2_dyn(out, o_base + col, a0, o_fmt);
        if (r1) st_store2_dyn(out, o_base + ulong(n) + col, a1, o_fmt);
        if (r2) st_store2_dyn(out, o_base + ulong(2u * n) + col, a2, o_fmt);
        if (r3) st_store2_dyn(out, o_base + ulong(3u * n) + col, a3, o_fmt);
    } else if (col < n) {
        st_store1_dyn(out, o_base + col, a0.x, o_fmt);
        if (r1) st_store1_dyn(out, o_base + ulong(n) + col, a1.x, o_fmt);
        if (r2) st_store1_dyn(out, o_base + ulong(2u * n) + col, a2.x, o_fmt);
        if (r3) st_store1_dyn(out, o_base + ulong(3u * n) + col, a3.x, o_fmt);
    }
}

template <bool SWAPPED, uint BM, uint BN>
union StMmaSmem {
    struct {
        float x[BM][36];
        float w[SWAPPED ? 32 : BN][SWAPPED ? (BN + 4) : 36];
    } tile;
    float out[BM][BN + 4];
};

// Prefill simdgroup_float8x8 MMA kernel for stacked block weights.
template <bool SWAPPED, uint W_FMT, uint BM, uint BN, uint WM, uint WN>
static inline void st_mma_body(
    threadgroup StMmaSmem<SWAPPED, BM, BN>& sm,
    const device uchar* x,
    const device uchar* w,
    device uchar* out,
    uint batches,
    uint rows,
    uint n,
    uint k,
    uint block,
    uint first,
    uint step,
    uint flags,
    uint2 tgid,
    uint tid) {
    constexpr uint THREADS = WM * WN * 32u;
    constexpr uint RM = BM / (WM * 8u);
    constexpr uint RN = BN / (WN * 8u);
    constexpr uint W_STRIDE = SWAPPED ? (BN + 4u) : 36u;

    uint r_tiles = (rows + BM - 1u) / BM;
    uint batch = tgid.y / r_tiles;
    if (batch >= batches) {
        return;
    }
    uint r0 = (tgid.y - batch * r_tiles) * BM;
    uint n0 = tgid.x * BN;
    uint wid = tid >> 5u;
    uint wm = wid / WN;
    uint wn = wid % WN;
    uint m_base = wm * (RM * 8u);
    uint n_base = wn * (RN * 8u);

    uint x_fmt = flags & 3u;
    uint o_fmt = (flags >> 2u) & 3u;
    ulong w_base = ulong(batch) * ulong(block) + ulong(first);
    ulong x_base = ulong(batch) * ulong(rows) * ulong(k);
    ulong o_base = ulong(batch) * ulong(rows) * ulong(n);

    simdgroup_float8x8 acc[RM][RN];
    for (uint mi = 0u; mi < RM; ++mi) {
        for (uint ni = 0u; ni < RN; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint xr = tid >> 3u;
    uint xc = (tid & 7u) << 2u;

    for (uint k0 = 0u; k0 < k; k0 += 32u) {
        float4 xv = float4(0.0f);
        if (r0 + xr < rows) {
            ulong xoff = x_base + ulong(r0 + xr) * ulong(k) + k0 + xc;
            if (k0 + xc + 4u <= k) {
                xv = st_load4_dyn(x, xoff, x_fmt);
            } else {
                xv = float4(
                    k0 + xc < k ? st_load1_dyn(x, xoff, x_fmt) : 0.0f,
                    k0 + xc + 1u < k ? st_load1_dyn(x, xoff + 1u, x_fmt) : 0.0f,
                    k0 + xc + 2u < k ? st_load1_dyn(x, xoff + 2u, x_fmt) : 0.0f,
                    k0 + xc + 3u < k ? st_load1_dyn(x, xoff + 3u, x_fmt) : 0.0f);
            }
        }
        *reinterpret_cast<threadgroup float4*>(&sm.tile.x[xr][xc]) = xv;

        if constexpr (!SWAPPED) {
            for (uint pass = 0u; pass < 2u; ++pass) {
                uint idx = tid + pass * THREADS;
                uint wr = idx >> 3u;
                uint wc = (idx & 7u) << 2u;
                float4 wv = float4(0.0f);
                if (n0 + wr < n) {
                    ulong woff = w_base + ulong(n0 + wr) * ulong(step) + k0 + wc;
                    if (k0 + wc + 4u <= k) {
                        wv = st_w_load4<W_FMT>(w, woff);
                    } else {
                        wv = float4(
                            k0 + wc < k ? st_w_load1<W_FMT>(w, woff) : 0.0f,
                            k0 + wc + 1u < k ? st_w_load1<W_FMT>(w, woff + 1u) : 0.0f,
                            k0 + wc + 2u < k ? st_w_load1<W_FMT>(w, woff + 2u) : 0.0f,
                            k0 + wc + 3u < k ? st_w_load1<W_FMT>(w, woff + 3u) : 0.0f);
                    }
                }
                *reinterpret_cast<threadgroup float4*>(&sm.tile.w[wr][wc]) = wv;
            }
        } else {
            constexpr uint COLS4 = BN >> 2u;
            for (uint pass = 0u; pass < 2u; ++pass) {
                uint idx = tid + pass * THREADS;
                uint wk = idx / COLS4;
                uint wc = (idx % COLS4) << 2u;
                float4 wv = float4(0.0f);
                if (k0 + wk < k) {
                    ulong woff = w_base + ulong(k0 + wk) * ulong(step) + n0 + wc;
                    if (n0 + wc + 4u <= n) {
                        wv = st_w_load4<W_FMT>(w, woff);
                    } else {
                        wv = float4(
                            n0 + wc < n ? st_w_load1<W_FMT>(w, woff) : 0.0f,
                            n0 + wc + 1u < n ? st_w_load1<W_FMT>(w, woff + 1u) : 0.0f,
                            n0 + wc + 2u < n ? st_w_load1<W_FMT>(w, woff + 2u) : 0.0f,
                            n0 + wc + 3u < n ? st_w_load1<W_FMT>(w, woff + 3u) : 0.0f);
                    }
                }
                *reinterpret_cast<threadgroup float4*>(&sm.tile.w[wk][wc]) = wv;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ks = 0u; ks < 32u; ks += 8u) {
            simdgroup_float8x8 ma[RM];
            for (uint mi = 0u; mi < RM; ++mi) {
                simdgroup_load(ma[mi], &sm.tile.x[m_base + mi * 8u][ks], 36u, ulong2(0), false);
            }
            for (uint ni = 0u; ni < RN; ++ni) {
                simdgroup_float8x8 mb;
                if constexpr (!SWAPPED) {
                    simdgroup_load(
                        mb, &sm.tile.w[n_base + ni * 8u][ks], W_STRIDE, ulong2(0), true);
                } else {
                    simdgroup_load(
                        mb, &sm.tile.w[ks][n_base + ni * 8u], W_STRIDE, ulong2(0), false);
                }
                for (uint mi = 0u; mi < RM; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][ni], ma[mi], mb, acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint mi = 0u; mi < RM; ++mi) {
        for (uint ni = 0u; ni < RN; ++ni) {
            simdgroup_store(
                acc[mi][ni], &sm.out[m_base + mi * 8u][n_base + ni * 8u], BN + 4u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint COLS4 = BN >> 2u;
    constexpr uint VECS = BM * COLS4;
    for (uint i = tid; i < VECS; i += THREADS) {
        uint lr = i / COLS4;
        uint lc = (i % COLS4) << 2u;
        if (r0 + lr < rows && n0 + lc < n) {
            float4 v = *reinterpret_cast<const threadgroup float4*>(&sm.out[lr][lc]);
            ulong dst = o_base + ulong(r0 + lr) * ulong(n) + n0 + lc;
            if (n0 + lc + 4u <= n) {
                st_store4_dyn(out, dst, v, o_fmt);
            } else {
                st_store1_dyn(out, dst, v.x, o_fmt);
                if (n0 + lc + 1u < n) st_store1_dyn(out, dst + 1u, v.y, o_fmt);
                if (n0 + lc + 2u < n) st_store1_dyn(out, dst + 2u, v.z, o_fmt);
                if (n0 + lc + 3u < n) st_store1_dyn(out, dst + 3u, v.w, o_fmt);
            }
        }
    }
}

#define DEFINE_ST_DEC_NT(NAME, FMT)                                                 \
    kernel void NAME(                                                               \
        const device uchar* x [[buffer(0)]],                                        \
        const device uchar* w [[buffer(1)]],                                        \
        device uchar* out [[buffer(2)]],                                            \
        constant uint& batches [[buffer(3)]],                                       \
        constant uint& rows [[buffer(4)]],                                          \
        constant uint& n [[buffer(5)]],                                             \
        constant uint& k [[buffer(6)]],                                             \
        constant uint& block [[buffer(7)]],                                         \
        constant uint& first [[buffer(8)]],                                         \
        constant uint& step [[buffer(9)]],                                          \
        constant uint& flags [[buffer(10)]],                                        \
        uint2 tgid [[threadgroup_position_in_grid]],                                \
        uint tid [[thread_index_in_threadgroup]]) {                                 \
        st_dec_nt_body<FMT>(                                                        \
            x, w, out, batches, rows, n, k, block, first, step, flags, tgid, tid);  \
    }

#define DEFINE_ST_DEC_T(NAME, FMT)                                                  \
    kernel void NAME(                                                               \
        const device uchar* x [[buffer(0)]],                                        \
        const device uchar* w [[buffer(1)]],                                        \
        device uchar* out [[buffer(2)]],                                            \
        constant uint& batches [[buffer(3)]],                                       \
        constant uint& rows [[buffer(4)]],                                          \
        constant uint& n [[buffer(5)]],                                             \
        constant uint& k [[buffer(6)]],                                             \
        constant uint& block [[buffer(7)]],                                         \
        constant uint& first [[buffer(8)]],                                         \
        constant uint& step [[buffer(9)]],                                          \
        constant uint& flags [[buffer(10)]],                                        \
        uint2 tgid [[threadgroup_position_in_grid]],                                \
        uint tid [[thread_index_in_threadgroup]]) {                                 \
        threadgroup StDecTSmem sm;                                                  \
        st_dec_t_body<FMT>(                                                         \
            sm, x, w, out, batches, rows, n, k, block, first, step, flags, tgid, tid); \
    }

#define DEFINE_ST_MMA(NAME, SWAPPED, FMT, BM, BN, WM, WN)                           \
    kernel void NAME(                                                               \
        const device uchar* x [[buffer(0)]],                                        \
        const device uchar* w [[buffer(1)]],                                        \
        device uchar* out [[buffer(2)]],                                            \
        constant uint& batches [[buffer(3)]],                                       \
        constant uint& rows [[buffer(4)]],                                          \
        constant uint& n [[buffer(5)]],                                             \
        constant uint& k [[buffer(6)]],                                             \
        constant uint& block [[buffer(7)]],                                         \
        constant uint& first [[buffer(8)]],                                         \
        constant uint& step [[buffer(9)]],                                          \
        constant uint& flags [[buffer(10)]],                                        \
        uint2 tgid [[threadgroup_position_in_grid]],                                \
        uint tid [[thread_index_in_threadgroup]]) {                                 \
        threadgroup StMmaSmem<SWAPPED, BM, BN> sm;                                  \
        st_mma_body<SWAPPED, FMT, BM, BN, WM, WN>(                                  \
            sm, x, w, out, batches, rows, n, k, block, first, step, flags, tgid, tid); \
    }

DEFINE_ST_DEC_NT(st_dec_nt_f32, 0u)
DEFINE_ST_DEC_NT(st_dec_nt_bf16, 1u)
DEFINE_ST_DEC_NT(st_dec_nt_f16, 2u)
DEFINE_ST_DEC_NT(st_dec_nt_fp8, 3u)

DEFINE_ST_DEC_T(st_dec_t_f32, 0u)
DEFINE_ST_DEC_T(st_dec_t_bf16, 1u)
DEFINE_ST_DEC_T(st_dec_t_f16, 2u)
DEFINE_ST_DEC_T(st_dec_t_fp8, 3u)

DEFINE_ST_MMA(st_pf16_nt_f32, false, 0u, 16u, 32u, 1u, 4u)
DEFINE_ST_MMA(st_pf16_nt_bf16, false, 1u, 16u, 32u, 1u, 4u)
DEFINE_ST_MMA(st_pf16_nt_f16, false, 2u, 16u, 32u, 1u, 4u)
DEFINE_ST_MMA(st_pf16_nt_fp8, false, 3u, 16u, 32u, 1u, 4u)

DEFINE_ST_MMA(st_pf16_t_f32, true, 0u, 16u, 32u, 1u, 4u)
DEFINE_ST_MMA(st_pf16_t_bf16, true, 1u, 16u, 32u, 1u, 4u)
DEFINE_ST_MMA(st_pf16_t_f16, true, 2u, 16u, 32u, 1u, 4u)
DEFINE_ST_MMA(st_pf16_t_fp8, true, 3u, 16u, 32u, 1u, 4u)

DEFINE_ST_MMA(st_pf32_nt_f32, false, 0u, 32u, 64u, 2u, 4u)
DEFINE_ST_MMA(st_pf32_nt_bf16, false, 1u, 32u, 64u, 2u, 4u)
DEFINE_ST_MMA(st_pf32_nt_f16, false, 2u, 32u, 64u, 2u, 4u)
DEFINE_ST_MMA(st_pf32_nt_fp8, false, 3u, 32u, 64u, 2u, 4u)

DEFINE_ST_MMA(st_pf32_t_f32, true, 0u, 32u, 64u, 2u, 4u)
DEFINE_ST_MMA(st_pf32_t_bf16, true, 1u, 32u, 64u, 2u, 4u)
DEFINE_ST_MMA(st_pf32_t_f16, true, 2u, 32u, 64u, 2u, 4u)
DEFINE_ST_MMA(st_pf32_t_fp8, true, 3u, 32u, 64u, 2u, 4u)
