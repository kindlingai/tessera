// Trellis MMA tile loader, codebook decode, and fused GEMM.

struct bit_pos {
    u32 w0, w1, w2, bit;
};

template <u32 K>
static __device__ __forceinline__ bit_pos lane_map(u32 lane) {
    constexpr u32 total_words = 8u * K;
    constexpr u32 span = (16u + K - 1u) / K;
    u32 start = (((lane << 3) + 257u - span) & 255u) * K + (span * K - 16u);
    u32 w = start >> 5;
    return {w, (w + 1u) % total_words, (w + 2u) % total_words, start & 31u};
}

static __device__ __forceinline__ u32 shf_left(u32 lo, u32 hi, u32 shift) {
    u32 out;
    asm("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(lo), "r"(hi), "r"(shift));
    return out;
}

static __device__ __forceinline__ u32 byte_perm(u32 a, u32 b, u32 sel) {
    u32 out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
    return out;
}

static __device__ __forceinline__ u32 ld_nc_u32(const u8* p) {
    u32 v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ void ld_nc_v4(const u8* p, u32 dst[4]) {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                 : "l"(p));
}

template <u32 BOOK>
static __device__ __forceinline__ u32 book_pair(u32 s0, u32 s1) {
    if (BOOK == 0u) {
        u32 p0 = ((s0 * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        u32 p1 = ((s1 * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        u32 sum;
        asm("add.rn.f16x2 %0, %1, %2;"
            : "=r"(sum)
            : "r"(byte_perm(p0, p1, 0x5410u)), "r"(byte_perm(p0, p1, 0x7632u)));
        return narrowed(sum);
    }
    constexpr float mul_s = 1774.0f / 262144.0f;
    constexpr float mul_b = (-1329.0f / 128.0f) - 8388608.0f * mul_s;
    u32 acc0, acc1;
    asm("dp4a.u32.u32 %0, %1, %2, %3;"
        : "=r"(acc0) : "r"(s0 * MUL1), "r"(0x01010101u), "r"(0x4B000400u));
    asm("dp4a.u32.u32 %0, %1, %2, %3;"
        : "=r"(acc1) : "r"(s1 * MUL1), "r"(0x01010101u), "r"(0x4B000400u));
    float f0 = fmaf(__uint_as_float(acc0), mul_s, mul_b);
    float f1 = fmaf(__uint_as_float(acc1), mul_s, mul_b);
    return narrowed(halves(f0, f1));
}

template <u32 K, u32 BOOK>
static __device__ __forceinline__ void tile_decode(const u32* words, bit_pos pos, u32 frag[4]) {
    u32 a = words[pos.w0], b = words[pos.w1], c = words[pos.w2];
    u32 hi = shf_left(b, a, pos.bit);
    u32 lo = shf_left(c, b, pos.bit);
    u32 st[8];
#pragma unroll
    for (u32 idx = 0; idx < 8; idx++) st[idx] = shf_left(lo, hi, K * idx) >> 16;
    frag[0] = book_pair<BOOK>(st[0], st[1]);
    frag[1] = book_pair<BOOK>(st[4], st[5]);
    frag[2] = book_pair<BOOK>(st[2], st[3]);
    frag[3] = book_pair<BOOK>(st[6], st[7]);
}

static __device__ __forceinline__ void mma_tile(float d[4], const u32 a[4], const u32 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Pointer to 8 tiles at input `ti`, output block `ob`.
template <u32 K>
static __device__ __forceinline__ const u8* tile_ptr(const mat& w, u32 m, u32 ti, u32 ob,
                                                     bool input_side) {
    u32 tile = input_side
                   ? (ti >> 3) * (w.out >> 1) + (ob << 6) + ((ti & 7u) << 3)
                   : ob * (w.in >> 1) + (ti << 3);
    u64 mat_tiles = u64(w.in >> 4) * (w.out >> 4);
    return w.p + HEAD + (u64(m) * mat_tiles + tile) * (32u * K);
}

template <u32 K>
static __device__ __forceinline__ void tile_load(const u8* ptr, u32 lane, bool wide, u32 reg[8]) {
    if (wide) {
        ld_nc_v4(ptr + (lane << 4), reg);
        if (K > 2u && lane < (16u * K - 32u)) ld_nc_v4(ptr + 512u + (lane << 4), reg + 4);
        return;
    }
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) {
        reg[mt] = lane < (8u * K) ? ld_nc_u32(ptr + ((mt * 8u * K + lane) << 2)) : 0u;
    }
}

template <u32 K>
static __device__ __forceinline__ void tile_store(const u32 reg[8], u32 lane, bool wide,
                                                  u32* smem) {
    if (wide) {
        uint4* dst = reinterpret_cast<uint4*>(smem);
        dst[lane] = make_uint4(reg[0], reg[1], reg[2], reg[3]);
        if (K > 2u && lane < (16u * K - 32u)) {
            dst[32u + lane] = make_uint4(reg[4], reg[5], reg[6], reg[7]);
        }
        return;
    }
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) {
        if (lane < (8u * K)) smem[mt * 8u * K + lane] = reg[mt];
    }
}

// Packs hi and lo f16 B fragment for 4 rows.
static __device__ __forceinline__ void act_frag(const float* buf, u32 n, u32 ks, u32 lane,
                                                u32 b[2]) {
    u32 col = lane >> 2, r = col >> 1;
    if (r >= n) {
        b[0] = 0u;
        b[1] = 0u;
        return;
    }
    const float* src = buf + (r << 7) + (ks << 4) + ((lane & 3u) << 1);
    float x0 = src[0], x1 = src[1], x2 = src[8], x3 = src[9];
    u32 h0 = halves(x0, x1);
    u32 h1 = halves(x2, x3);
    if ((col & 1u) == 0u) {
        b[0] = h0;
        b[1] = h1;
        return;
    }
    b[0] = halves(x0 - half(h0 & 0xFFFFu), x1 - half(h0 >> 16));
    b[1] = halves(x2 - half(h1 & 0xFFFFu), x3 - half(h1 >> 16));
}

// Writes one warp's 8x(16x4) MMA accumulators into shared buffer.
static __device__ __forceinline__ void store_acc(const float acc[8][4], u32 n, u32 lane,
                                                 float* buf) {
    u32 r = lane & 3u, g = lane >> 2;
    if (r >= n) return;
    float* dst = buf + (r << 7) + g;
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) {
        dst[mt * 16u] = acc[mt][0] + acc[mt][1];
        dst[mt * 16u + 8u] = acc[mt][2] + acc[mt][3];
    }
}

template <u32 K, u32 BOOK>
static __device__ __noinline__ void trellis_gemm(const ctx& c, const mat& w, const float* x,
                                                 u32 xs, float* y, u32 ys, u32 rows,
                                                 bool fuse_out) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    bit_pos pos = lane_map<K>(lane);
    u32 row_groups = (rows + ROW_TILE - 1u) / ROW_TILE;
    u32 out_blocks = w.out >> 7;
    u32 in_blocks = w.in >> 7;
    u32 total = row_groups * out_blocks;
    bool wide = (reinterpret_cast<u64>(w.p + HEAD) & 15u) == 0u;
    bool input_side = word(w.p, 8) == INPUT_SIDE;
    bool x_vec = (reinterpret_cast<u64>(x) & 15u) == 0u && (xs & 3u) == 0u;
    bool y_vec = (reinterpret_cast<u64>(y) & 15u) == 0u && (ys & 3u) == 0u;
    bool sv_vec = fuse_out && w.svh != nullptr && (reinterpret_cast<u64>(w.svh) & 15u) == 0u;
    u32 my_ibs = in_blocks > warp ? ((in_blocks - 1u - warp) / WARPS + 1u) : 0u;
    u32 steps = my_ibs << 3;
    float* wbuf = &c.sm.lin.buf[warp][0][0];
    u32* wtiles = c.sm.lin.tiles[warp];

    for (u32 job = blockIdx.x; job < total; job += gridDim.x) {
        u32 rg = job / out_blocks, ob = job % out_blocks;
        u32 r0 = rg * ROW_TILE, n = min(ROW_TILE, rows - r0);
        float acc[8][4];
#pragma unroll
        for (u32 mt = 0; mt < 8; mt++) {
            acc[mt][0] = 0.0f;
            acc[mt][1] = 0.0f;
            acc[mt][2] = 0.0f;
            acc[mt][3] = 0.0f;
        }
        if (steps > 0u) {
            u32 reg[8];
            tile_load<K>(tile_ptr<K>(w, 0u, warp << 3, ob, input_side), lane, wide, reg);
            for (u32 st = 0; st < steps; st++) {
                u32 ib = warp + (st >> 3) * WARPS;
                u32 ks = st & 7u;
                if (ks == 0u) {
                    __syncwarp();
                    for (u32 r = 0; r < n; r++) {
                        float v[4];
                        load_vec4(x + u64(r0 + r) * xs + (ib << 7) + (lane << 2), x_vec, v);
                        *reinterpret_cast<float4*>(wbuf + (r << 7) + (lane << 2)) =
                            make_float4(v[0], v[1], v[2], v[3]);
                    }
                    __syncwarp();
                }
                __syncwarp();
                tile_store<K>(reg, lane, wide, wtiles);
                __syncwarp();
                if (st + 1u < steps) {
                    u32 nib = warp + ((st + 1u) >> 3) * WARPS;
                    u32 nks = (st + 1u) & 7u;
                    tile_load<K>(tile_ptr<K>(w, 0u, (nib << 3) + nks, ob, input_side), lane, wide,
                                 reg);
                }
                u32 fb[2];
                act_frag(wbuf, n, ks, lane, fb);
#pragma unroll
                for (u32 mt = 0; mt < 8; mt++) {
                    u32 fa[4];
                    tile_decode<K, BOOK>(wtiles + mt * 8u * K, pos, fa);
                    mma_tile(acc[mt], fa, fb);
                }
            }
        }
        __syncwarp();
        store_acc(acc, n, lane, wbuf);
        __syncthreads();
        if (warp < n) {
            u32 base = (ob << 7) + (lane << 2);
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (u32 widx = 0; widx < WARPS; widx++) {
                float4 p = *reinterpret_cast<const float4*>(&c.sm.lin.buf[widx][warp][lane << 2]);
                v[0] += p.x;
                v[1] += p.y;
                v[2] += p.z;
                v[3] += p.w;
            }
            if (fuse_out) {
                wht128_warp(v, INV_SQRT128);
                if (w.svh != nullptr) {
                    float sc[4];
                    load_vec4(w.svh + base, sv_vec, sc);
                    for (u32 k = 0; k < 4; k++) v[k] *= sc[k];
                }
            }
            store_vec4(y + u64(r0 + warp) * ys + base, y_vec, v);
        }
        __syncthreads();
    }
}

template <u32 BOOK>
static __device__ void trellis_book(const ctx& c, const mat& w, const float* x, u32 xs, float* y,
                                    u32 ys, u32 rows, bool fuse_out) {
    if (w.k == 2u) {
        trellis_gemm<2u, BOOK>(c, w, x, xs, y, ys, rows, fuse_out);
    } else if (w.k == 3u) {
        trellis_gemm<3u, BOOK>(c, w, x, xs, y, ys, rows, fuse_out);
    } else {
        trellis_gemm<4u, BOOK>(c, w, x, xs, y, ys, rows, fuse_out);
    }
}

static __device__ void trellis_run(const ctx& c, const mat& w, const float* x, u32 xs, float* y,
                                   u32 ys, u32 rows, bool fuse_out) {
    if (w.book == 1u) {
        trellis_book<1u>(c, w, x, xs, y, ys, rows, fuse_out);
    } else {
        trellis_book<0u>(c, w, x, xs, y, ys, rows, fuse_out);
    }
}
