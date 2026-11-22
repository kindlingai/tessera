// Warp-cooperative GEMV, linear dispatcher, norms, gating, and residual add.

static __device__ __forceinline__ u32 warp_start(u64 cursor, u32 wid, u32 wstep) {
    u32 rem = static_cast<u32>(cursor % wstep);
    return wid >= rem ? (wid - rem) : (wid + wstep - rem);
}

static __device__ __forceinline__ void store_warp(float acc[8], u32 n, u32 r0, u32 o, float* y,
                                                  u32 ys, u32 lane) {
    for (u32 r = 0; r < n; r++) {
        float s = warp_sum(acc[r]);
        if (lane == 0u) y[u64(r0 + r) * ys + o] = s;
    }
}

template <bool IS_BF16>
static __device__ __forceinline__ float4 load_w4(const u8* row_ptr, u32 i) {
    if (IS_BF16) {
        uint2 h = *reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(row_ptr) + i);
        return make_float4(__uint_as_float(h.x << 16), __uint_as_float(h.x & 0xFFFF0000u),
                           __uint_as_float(h.y << 16), __uint_as_float(h.y & 0xFFFF0000u));
    }
    return *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(row_ptr) + i);
}

template <bool IS_BF16>
static __device__ __forceinline__ float load_w1(const u8* row_ptr, u32 i) {
    if (IS_BF16) {
        return __uint_as_float(u32(reinterpret_cast<const u16*>(row_ptr)[i]) << 16);
    }
    return reinterpret_cast<const float*>(row_ptr)[i];
}

template <bool IS_BF16>
static __device__ __noinline__ void gemv_dense(const mat& w, const float* x, u32 xs, float* y,
                                               u32 ys, u32 rows, u32 out, u64 cursor) {
    constexpr u32 elem_bytes = IS_BF16 ? 2u : 4u;
    constexpr u64 align_mask = IS_BF16 ? 7ull : 15ull;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 chunks = (rows + 7u) >> 3;
    u32 vec_end = w.in & ~3u;
    bool x_vec = (reinterpret_cast<u64>(x) & 15u) == 0u && (xs & 3u) == 0u;
    for (u32 u = warp_start(cursor, wid, wstep); u < out * chunks; u += wstep) {
        u32 o = u % out, r0 = (u / out) * 8u, n = min(8u, rows - r0);
        const u8* wr = w.p + u64(o) * w.in * elem_bytes;
        bool use_vec = x_vec && (reinterpret_cast<u64>(wr) & align_mask) == 0ull && vec_end > 0u;
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (use_vec) {
            for (u32 i = lane << 2; i < vec_end; i += 128u) {
                float4 wv = load_w4<IS_BF16>(wr, i);
                for (u32 r = 0; r < n; r++) {
                    float4 xv = *reinterpret_cast<const float4*>(x + u64(r0 + r) * xs + i);
                    acc[r] = fmaf(wv.x, xv.x, acc[r]);
                    acc[r] = fmaf(wv.y, xv.y, acc[r]);
                    acc[r] = fmaf(wv.z, xv.z, acc[r]);
                    acc[r] = fmaf(wv.w, xv.w, acc[r]);
                }
            }
            for (u32 i = vec_end + lane; i < w.in; i += 32u) {
                float wv = load_w1<IS_BF16>(wr, i);
                for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + i], acc[r]);
            }
        } else {
            for (u32 i = lane; i < w.in; i += 32u) {
                float wv = load_w1<IS_BF16>(wr, i);
                for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + i], acc[r]);
            }
        }
        store_warp(acc, n, r0, o, y, ys, lane);
    }
}

template <bool IS_MXFP8>
static __device__ __noinline__ void gemv_q32(const mat& w, const float* x, u32 xs, float* y,
                                             u32 ys, u32 rows, u32 out, u64 cursor) {
    constexpr u32 blk_bytes = IS_MXFP8 ? 33u : 18u;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 chunks = (rows + 7u) >> 3;
    u32 blocks = w.in >> 5;
    bool aligned = (w.in & 31u) == 0u;
    u32 nib_byte = 2u + (lane >> 1), nib_shift = (lane & 1u) << 2;
    for (u32 u = warp_start(cursor, wid, wstep); u < out * chunks; u += wstep) {
        u32 o = u % out, r0 = (u / out) * 8u, n = min(8u, rows - r0);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (aligned) {
            u64 row_blk = u64(o) * blocks;
            for (u32 blk = 0; blk < blocks; blk++) {
                const u8* b = w.p + (row_blk + blk) * blk_bytes;
                float sc = 0.0f;
                if (lane == 0u) {
                    sc = IS_MXFP8 ? mxfp8_scale(b[0]) : trellis::f16_to_f32(b[0] | (u32(b[1]) << 8));
                }
                sc = __shfl_sync(0xFFFFFFFFu, sc, 0);
                float q = IS_MXFP8 ? trellis::e4m3_to_f32(b[1u + lane])
                                   : (float((b[nib_byte] >> nib_shift) & 15u) - 8.0f);
                float wv = sc * q;
                u32 col = (blk << 5) + lane;
                for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + col], acc[r]);
            }
        } else {
            for (u32 i = lane; i < w.in; i += 32u) {
                float wv = elem(w, 0u, o, i);
                for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + i], acc[r]);
            }
        }
        store_warp(acc, n, r0, o, y, ys, lane);
    }
}

static __device__ void gemv_any(const mat& w, const float* x, u32 xs, float* y, u32 ys, u32 rows,
                                u32 out, u64 cursor) {
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 chunks = (rows + 7u) >> 3;
    for (u32 u = warp_start(cursor, wid, wstep); u < out * chunks; u += wstep) {
        u32 o = u % out, r0 = (u / out) * 8u, n = min(8u, rows - r0);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (u32 i = lane; i < w.in; i += 32u) {
            float wv = elem(w, 0u, o, i);
            for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + i], acc[r]);
        }
        store_warp(acc, n, r0, o, y, ys, lane);
    }
}

static __device__ void times_plain(const mat& w, const float* x, u32 xs, float* y, u32 ys,
                                   u32 rows, u32 out, u64 cursor) {
    switch (w.klass) {
    case K_F32:
        gemv_dense<false>(w, x, xs, y, ys, rows, out, cursor);
        return;
    case K_BF16:
        gemv_dense<true>(w, x, xs, y, ys, rows, out, cursor);
        return;
    case K_INT4:
        gemv_q32<false>(w, x, xs, y, ys, rows, out, cursor);
        return;
    case K_MXFP8:
        gemv_q32<true>(w, x, xs, y, ys, rows, out, cursor);
        return;
    default:
        gemv_any(w, x, xs, y, ys, rows, out, cursor);
        return;
    }
}

// Computes y = x w^T using trellis MMA or warp GEMV.
static __device__ bool linear(const ctx& c, const mat& w, const float* x, u32 xs, float* y, u32 ys,
                              u32 rows, u32 out) {
    bool fast_t = (w.in & 127u) == 0u && (w.out & 127u) == 0u && out == w.out;
    if (w.klass == K_TURNED) {
        PHASE(turn_rows(x, xs, w.suh, nullptr, c.p.t1, w.in, rows, w.in));
        if (fast_t) {
            PHASE(trellis_run(c, w, c.p.t1, w.in, y, ys, rows, true));
            return true;
        }
        PHASE(gemv_any(w, c.p.t1, w.in, c.p.t2, w.out, rows, w.out, 0ull));
        PHASE(turn_rows(c.p.t2, w.out, nullptr, w.svh, y, ys, rows, w.out));
        return true;
    }
    if (w.klass == K_TRELLIS && fast_t) {
        PHASE(trellis_run(c, w, x, xs, y, ys, rows, false));
        return true;
    }
    PHASE(times_plain(w, x, xs, y, ys, rows, out, 0ull));
    return true;
}

// Dual projections sharing x, fused when both are plain GEMV.
static __device__ bool linear_pair(const ctx& c, const mat& w0, const mat& w1, const float* x,
                                   u32 xs, float* y0, u32 ys0, u32 out0, float* y1, u32 ys1,
                                   u32 out1, u32 rows) {
    if (w0.klass <= K_MXFP8 && w1.klass <= K_MXFP8) {
        u64 n0 = u64(out0) * ((rows + 7u) >> 3);
        PHASE(times_plain(w0, x, xs, y0, ys0, rows, out0, 0ull);
              times_plain(w1, x, xs, y1, ys1, rows, out1, n0));
        return true;
    }
    return linear(c, w0, x, xs, y0, ys0, rows, out0)
           && linear(c, w1, x, xs, y1, ys1, rows, out1);
}

// Block-parallel RMS norm with optional mean centering.
static __device__ void rms(const ctx& c, const float* x, u32 xs, const mat& w, float* y, u32 ys,
                           u32 rows, u32 n, const u32* from, bool centered) {
    bool vec_n = (n & 3u) == 0u;
    for (u32 r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* row = x + u64(from ? from[r] : r) * xs;
        float* dst = y + u64(r) * ys;
        bool r_vec = vec_n && (reinterpret_cast<u64>(row) & 15u) == 0u;
        bool d_vec = vec_n && (reinterpret_cast<u64>(dst) & 15u) == 0u;
        float mean = 0.0f;
        if (centered) {
            float sum = 0.0f;
            if (r_vec) {
                for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
                    float4 v = *reinterpret_cast<const float4*>(row + i);
                    sum += v.x + v.y + v.z + v.w;
                }
            } else {
                for (u32 i = threadIdx.x; i < n; i += blockDim.x) sum += row[i];
            }
            mean = block_sum(c, sum) / float(n);
        }
        float sq = 0.0f;
        if (r_vec) {
            for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
                float4 v = *reinterpret_cast<const float4*>(row + i);
                float a0 = v.x - mean, a1 = v.y - mean, a2 = v.z - mean, a3 = v.w - mean;
                sq = fmaf(a0, a0, sq);
                sq = fmaf(a1, a1, sq);
                sq = fmaf(a2, a2, sq);
                sq = fmaf(a3, a3, sq);
            }
        } else {
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
                float d = row[i] - mean;
                sq = fmaf(d, d, sq);
            }
        }
        float inv = 1.0f / sqrtf(block_sum(c, sq) / float(n) + c.s.eps);
        if (r_vec && d_vec) {
            for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
                float4 v = *reinterpret_cast<const float4*>(row + i);
                float4 out = make_float4((v.x - mean) * (inv * vec_elem(w, 0u, i)),
                                         (v.y - mean) * (inv * vec_elem(w, 0u, i + 1u)),
                                         (v.z - mean) * (inv * vec_elem(w, 0u, i + 2u)),
                                         (v.w - mean) * (inv * vec_elem(w, 0u, i + 3u)));
                *reinterpret_cast<float4*>(dst + i) = out;
            }
        } else {
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
                dst[i] = (row[i] - mean) * (inv * vec_elem(w, 0u, i));
            }
        }
    }
}

static __device__ void gated(const float* a, const float* b, float* y, u32 n) {
    line l = threads();
    bool vec = (reinterpret_cast<u64>(a) & 15u) == 0u && (reinterpret_cast<u64>(b) & 15u) == 0u
               && (reinterpret_cast<u64>(y) & 15u) == 0u;
    u32 vec_end = vec ? (n & ~3u) : 0u;
    for (u32 i = l.at << 2; i < vec_end; i += l.step << 2) {
        float4 va = *reinterpret_cast<const float4*>(a + i);
        float4 vb = *reinterpret_cast<const float4*>(b + i);
        *reinterpret_cast<float4*>(y + i) = make_float4(
            va.x / (1.0f + expf(-va.x)) * vb.x, va.y / (1.0f + expf(-va.y)) * vb.y,
            va.z / (1.0f + expf(-va.z)) * vb.z, va.w / (1.0f + expf(-va.w)) * vb.w);
    }
    for (u32 i = vec_end + l.at; i < n; i += l.step) y[i] = a[i] / (1.0f + expf(-a[i])) * b[i];
}

// Residual add, with cross-rank all-reduce when split.
static __device__ bool add(const ctx& c, bool ranks) {
    line l = threads();
    u32 n = c.s.rows * c.s.hidden;
    if (!ranks) {
        bool vec = (reinterpret_cast<u64>(c.p.x) & 15u) == 0u
                   && (reinterpret_cast<u64>(c.p.part) & 15u) == 0u;
        u32 vec_end = vec ? (n & ~3u) : 0u;
        for (u32 i = l.at << 2; i < vec_end; i += l.step << 2) {
            float4 vx = *reinterpret_cast<const float4*>(c.p.x + i);
            float4 vp = *reinterpret_cast<const float4*>(c.p.part + i);
            *reinterpret_cast<float4*>(c.p.x + i) =
                make_float4(vx.x + vp.x, vx.y + vp.y, vx.z + vp.z, vx.w + vp.w);
        }
        for (u32 i = vec_end + l.at; i < n; i += l.step) c.p.x[i] += c.p.part[i];
        return sync(c.t);
    }
    const lane* ln = get<const lane>(c.t, c.t->lanes);
    bool bf16 = ln->elem == BF16;
    u32 room = detail::pools(c.t)->pools[ln->pool].send_bytes >> (bf16 ? 1u : 2u);
    for (u32 done = 0; done < n; done += room) {
        u32 len = min(room, n - done);
        ticket r = prepare(c.se, 0u);
        if (bf16) {
            u16* sent = send_buffer<u16>(r);
            if (sent == nullptr) return false;
            for (u32 i = l.at; i < len; i += l.step) {
                sent[i] = trellis::f32_to_bf16(c.p.part[done + i]);
            }
        } else {
            float* sent = send_buffer<float>(r);
            if (sent == nullptr) return false;
            for (u32 i = l.at; i < len; i += l.step) sent[i] = c.p.part[done + i];
        }
        if (!trigger(r, len) || !wait(r)) return false;
        parts got = received(r);
        for (u32 i = l.at; i < len; i += l.step) c.p.x[done + i] += got.sum(i);
    }
    return sync(c.t);
}
