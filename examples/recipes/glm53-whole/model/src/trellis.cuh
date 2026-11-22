// Fused trellis GEMM using shared trellis primitives.

static __device__ __forceinline__ void zero_acc(float acc[8][4]) {
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) {
        acc[mt][0] = 0.0f;
        acc[mt][1] = 0.0f;
        acc[mt][2] = 0.0f;
        acc[mt][3] = 0.0f;
    }
}

// Stores tiles, prefetches next if present, and runs MMA.
template <u32 K, u32 BOOK>
static __device__ __forceinline__ void trellis_step(u32 reg[8], const u8* next, u32 lane,
                                                    bool wide, u32* wtiles, const float* act,
                                                    u32 n, u32 ks, const trellis::lane_map& pos,
                                                    float acc[8][4]) {
    __syncwarp();
    trellis::tile_store<K>(reg, lane, wide, wtiles);
    __syncwarp();
    if (next != nullptr) trellis::tile_load<K>(next, lane, wide, reg);
    u32 fb[2];
    trellis::act_frag(act, n, ks, lane, fb);
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) {
        u32 fa[4];
        trellis::decode_frag<K, BOOK>(wtiles + mt * 8u * K, pos, fa);
        trellis::mma_m16n8k16(acc[mt], fa, fb);
    }
}

template <u32 K, u32 BOOK>
static __device__ __noinline__ void trellis_gemm(const ctx& c, const mat& w, const float* x,
                                                 u32 xs, float* y, u32 ys, u32 rows,
                                                 bool fuse_out) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    trellis::lane_map pos = trellis::map_lane<K, BOOK>(lane);
    u32 row_groups = (rows + ROW_TILE - 1u) / ROW_TILE;
    u32 out_blocks = w.out >> 7;
    u32 in_blocks = w.in >> 7;
    u32 total = row_groups * out_blocks;
    bool wide = (reinterpret_cast<u64>(w.p + trellis::HEAD) & 15u) == 0u;
    bool input_side = word(w.p, 8) == trellis::INPUT_SIDE;
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
        zero_acc(acc);
        if (steps > 0u) {
            u32 reg[8];
            trellis::tile_load<K>(
                trellis::tile_ptr<K>(w.p, w.in, w.out, 0u, warp << 3, ob, input_side), lane, wide,
                reg);
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
                const u8* next = nullptr;
                if (st + 1u < steps) {
                    u32 nti = ((warp + ((st + 1u) >> 3) * WARPS) << 3) + ((st + 1u) & 7u);
                    next = trellis::tile_ptr<K>(w.p, w.in, w.out, 0u, nti, ob, input_side);
                }
                trellis_step<K, BOOK>(reg, next, lane, wide, wtiles, wbuf, n, ks, pos, acc);
            }
        }
        __syncwarp();
        trellis::store_acc(acc, n, lane, wbuf);
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
                trellis::wht128(v, trellis::INV_SQRT128);
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
    if (w.book == trellis::BOOK_PMCGX) {
        trellis_book<trellis::BOOK_PMCGX>(c, w, x, xs, y, ys, rows, fuse_out);
    } else if (w.book == trellis::BOOK_MUL1) {
        trellis_book<trellis::BOOK_MUL1>(c, w, x, xs, y, ys, rows, fuse_out);
    } else {
        trellis_book<trellis::BOOK_MCG>(c, w, x, xs, y, ys, rows, fuse_out);
    }
}
