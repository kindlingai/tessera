// MoE router, fused trellis expert pipeline, SwiGLU, and decoder layer.

static __device__ __forceinline__ bool local_pick(const ctx& c, u32 rk, u32* m) {
    u32 e = c.p.ids[rk];
    *m = e - c.s.held_first;
    return e >= c.s.held_first && e < c.s.held_first + c.s.held;
}

// Warp-parallel SwiGLU MLP into `part`.
static __device__ bool swiglu(const ctx& c, const place& pl, u32 gate, u32 inner) {
    const sizes& s = c.s;
    u32 r = s.rows;
    if (!linear(c, field_mat(c, pl, gate), c.p.h, s.hidden, c.p.a, inner, r, inner)) return false;
    if (!linear(c, field_mat(c, pl, gate + 1u), c.p.h, s.hidden, c.p.b, inner, r, inner)) {
        return false;
    }
    PHASE(gated(c.p.a, c.p.b, c.p.c, r * inner));
    return linear(c, field_mat(c, pl, gate + 2u), c.p.c, inner, c.p.part, s.hidden, r, s.hidden);
}

// Fused router sigmoid, bias top-k, gate norm, and `d` zeroing.
static __device__ bool route(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    u32 r_cnt = s.rows, e = s.experts;
    if (!linear(c, field_mat(c, pl, ROUTER), c.p.h, s.hidden, c.p.a, e, r_cnt, e)) return false;
    mat bias = field_mat(c, pl, R_BIAS);
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    for (u32 row = wid; row < r_cnt; row += wstep) {
        float* scores = c.p.a + u64(row) * e;
        for (u32 idx = lane; idx < e; idx += 32u) scores[idx] = 1.0f / (1.0f + expf(-scores[idx]));
        __syncwarp();
        u64 taken = 0ull;
        float sum = 0.0f;
        u32* out_ids = c.p.ids + u64(row) * s.topk;
        float* out_gates = c.p.gates + u64(row) * s.topk;
        for (u32 k = 0; k < s.topk; k++) {
            u32 best = e;
            float most = __int_as_float(0xFF800000u);
            for (u32 i = 0, idx = lane; idx < e; i++, idx += 32u) {
                bool used = i < 64u ? (((taken >> i) & 1ull) != 0ull) : false;
                if (e > 2048u && !used) {
                    for (u32 p = 0; p < k; p++) {
                        if (out_ids[p] == idx) used = true;
                    }
                }
                if (used) continue;
                float v = scores[idx] + vec_elem(bias, 0u, idx);
                if (best == e || v > most) {
                    best = idx;
                    most = v;
                }
            }
#pragma unroll
            for (u32 off = 16; off > 0; off >>= 1) {
                float pv = __shfl_xor_sync(0xFFFFFFFFu, most, off);
                u32 pe = __shfl_xor_sync(0xFFFFFFFFu, best, off);
                bool win = pe < e && (best == e || pv > most || (pv == most && pe < best));
                if (win) {
                    best = pe;
                    most = pv;
                }
            }
            if ((best & 31u) == lane && (best >> 5) < 64u) taken |= 1ull << (best >> 5);
            if (lane == 0u) {
                out_ids[k] = best;
                sum += scores[best];
            }
            __syncwarp();
        }
        sum = __shfl_sync(0xFFFFFFFFu, sum, 0);
        for (u32 k = lane; k < s.topk; k += 32u) {
            out_gates[k] = scores[out_ids[k]] / sum * s.route;
        }
    }
    line l = threads();
    u64 d_len = u64(r_cnt) * s.topk * s.hidden;
    bool d_vec = (reinterpret_cast<u64>(c.p.d) & 15u) == 0u;
    u64 vec_end = d_vec ? (d_len & ~3ull) : 0ull;
    for (u64 i = u64(l.at) << 2; i < vec_end; i += u64(l.step) << 2) {
        *reinterpret_cast<float4*>(c.p.d + i) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (u64 i = vec_end + l.at; i < d_len; i += l.step) c.p.d[i] = 0.0f;
    return sync(c.t);
}

// One warp's gate or up trellis MMA pass over hidden.
template <u32 K, u32 BOOK>
static __device__ __noinline__ void routed_gu(const ctx& c, const mat& w, const mat& su,
                                              const float* in_signs, u32 m, u32 e, u32 ch, u32 n) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    u32 pair_id = warp >> 1, h = c.s.hidden;
    u32 in_blocks = h >> 7;
    u32 my_ibs = in_blocks > pair_id ? ((in_blocks - 1u - pair_id) / (WARPS >> 1) + 1u) : 0u;
    u32 steps = my_ibs << 3;
    bit_pos pos = lane_map<K>(lane);
    bool wide = (reinterpret_cast<u64>(w.p + HEAD) & 15u) == 0u;
    bool input_side = word(w.p, 8) == INPUT_SIDE;
    bool h_vec = (reinterpret_cast<u64>(c.p.h) & 15u) == 0u;
    bool s_vec = (reinterpret_cast<u64>(in_signs) & 15u) == 0u;
    float* wbuf = &c.sm.moe.buf[warp][0][0];
    u32* wtiles = c.sm.moe.tiles[warp];
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
        tile_load<K>(tile_ptr<K>(w, m, pair_id << 3, ch, input_side), lane, wide, reg);
        for (u32 st = 0; st < steps; st++) {
            u32 ib = pair_id + (st >> 3) * (WARPS >> 1);
            u32 ks = st & 7u;
            if (ks == 0u) {
                __syncwarp();
                u32 base = (ib << 7) + (lane << 2);
                float sg[4], sc[4];
                load_vec4(in_signs + base, s_vec, sg);
                for (u32 q = 0; q < 4; q++) sc[q] = sg[q] * vec_elem(su, e, base + q);
                for (u32 r = 0; r < n; r++) {
                    float v[4];
                    load_vec4(c.p.h + u64(c.sm.moe.rows[r]) * h + base, h_vec, v);
                    for (u32 q = 0; q < 4; q++) v[q] *= sc[q];
                    wht128_warp(v, INV_SQRT128);
                    *reinterpret_cast<float4*>(wbuf + (r << 7) + (lane << 2)) =
                        make_float4(v[0], v[1], v[2], v[3]);
                }
                __syncwarp();
            }
            __syncwarp();
            tile_store<K>(reg, lane, wide, wtiles);
            __syncwarp();
            if (st + 1u < steps) {
                u32 nib = pair_id + ((st + 1u) >> 3) * (WARPS >> 1);
                u32 nks = (st + 1u) & 7u;
                tile_load<K>(tile_ptr<K>(w, m, (nib << 3) + nks, ch, input_side), lane, wide, reg);
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
}

// One warp's down trellis MMA pass, accumulating into `d`.
template <u32 K, u32 BOOK>
static __device__ __noinline__ void routed_down(const ctx& c, const mat& w, u32 m, u32 ch,
                                                u32 chunks, u32 n) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    u32 h = c.s.hidden, out_blocks = h >> 7;
    u32 my_obs = out_blocks > warp ? ((out_blocks - 1u - warp) / WARPS + 1u) : 0u;
    u32 steps = my_obs << 3;
    if (steps == 0u) return;
    bit_pos pos = lane_map<K>(lane);
    bool wide = (reinterpret_cast<u64>(w.p + HEAD) & 15u) == 0u;
    bool input_side = word(w.p, 8) == INPUT_SIDE;
    u32* wtiles = c.sm.moe.tiles[warp];
    u32 reg[8];
    tile_load<K>(tile_ptr<K>(w, m, ch << 3, warp, input_side), lane, wide, reg);
    float out[8][4];
    u32 r = lane & 3u, g = lane >> 2;
    for (u32 st = 0; st < steps; st++) {
        u32 ob = warp + (st >> 3) * WARPS;
        u32 ks = st & 7u;
        if (ks == 0u) {
#pragma unroll
            for (u32 mt = 0; mt < 8; mt++) {
                out[mt][0] = 0.0f;
                out[mt][1] = 0.0f;
                out[mt][2] = 0.0f;
                out[mt][3] = 0.0f;
            }
        }
        __syncwarp();
        tile_store<K>(reg, lane, wide, wtiles);
        __syncwarp();
        if (st + 1u < steps) {
            u32 nob = warp + ((st + 1u) >> 3) * WARPS;
            u32 nks = (st + 1u) & 7u;
            tile_load<K>(tile_ptr<K>(w, m, (ch << 3) + nks, nob, input_side), lane, wide, reg);
        }
        u32 fb[2];
        act_frag(&c.sm.moe.vec[0][0], n, ks, lane, fb);
#pragma unroll
        for (u32 mt = 0; mt < 8; mt++) {
            u32 fa[4];
            tile_decode<K, BOOK>(wtiles + mt * 8u * K, pos, fa);
            mma_tile(out[mt], fa, fb);
        }
        if (ks == 7u && r < n) {
            float* dst = c.p.d + u64(c.sm.moe.picks[r]) * h + (ob << 7) + g;
#pragma unroll
            for (u32 mt = 0; mt < 8; mt++) {
                float v0 = out[mt][0] + out[mt][1];
                float v1 = out[mt][2] + out[mt][3];
                if (chunks == 1u) {
                    dst[mt * 16u] = v0;
                    dst[mt * 16u + 8u] = v1;
                } else {
                    atomicAdd(&dst[mt * 16u], v0);
                    atomicAdd(&dst[mt * 16u + 8u], v1);
                }
            }
        }
    }
}

static __device__ void call_gu(const ctx& c, const mat& w, const mat& su, const float* in_signs,
                               u32 m, u32 e, u32 ch, u32 n) {
    if (w.book == 1u) {
        if (w.k == 2u) routed_gu<2u, 1u>(c, w, su, in_signs, m, e, ch, n);
        else if (w.k == 3u) routed_gu<3u, 1u>(c, w, su, in_signs, m, e, ch, n);
        else routed_gu<4u, 1u>(c, w, su, in_signs, m, e, ch, n);
    } else {
        if (w.k == 2u) routed_gu<2u, 0u>(c, w, su, in_signs, m, e, ch, n);
        else if (w.k == 3u) routed_gu<3u, 0u>(c, w, su, in_signs, m, e, ch, n);
        else routed_gu<4u, 0u>(c, w, su, in_signs, m, e, ch, n);
    }
}

static __device__ void call_down(const ctx& c, const mat& w, u32 m, u32 ch, u32 chunks, u32 n) {
    if (w.book == 1u) {
        if (w.k == 2u) routed_down<2u, 1u>(c, w, m, ch, chunks, n);
        else if (w.k == 3u) routed_down<3u, 1u>(c, w, m, ch, chunks, n);
        else routed_down<4u, 1u>(c, w, m, ch, chunks, n);
    } else {
        if (w.k == 2u) routed_down<2u, 0u>(c, w, m, ch, chunks, n);
        else if (w.k == 3u) routed_down<3u, 0u>(c, w, m, ch, chunks, n);
        else routed_down<4u, 0u>(c, w, m, ch, chunks, n);
    }
}

// Routed MoE phase 1: active expert chunks into `d`.
static __device__ void routed_step(const ctx& c, const mat codes[3], const mat su[3],
                                   const mat sv[3], const float* in_signs) {
    const sizes& s = c.s;
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    u32 groups = (s.rows + ROW_TILE - 1u) / ROW_TILE;
    u32 chunks = s.effn >> 7;
    u32 cursor = 0u;
    for (u32 gr = 0; gr < groups; gr++) {
        u32 base = gr * ROW_TILE, nr = min(ROW_TILE, s.rows - base);
        for (u32 i = threadIdx.x; i < 64u; i += blockDim.x) c.sm.moe.mask[i] = 0u;
        __syncthreads();
        for (u32 i = threadIdx.x; i < nr * s.topk; i += blockDim.x) {
            u32 e = c.p.ids[(base + i / s.topk) * s.topk + (i % s.topk)];
            if (e >= s.held_first && e < s.held_first + s.held) {
                atomicOr(&c.sm.moe.mask[e >> 5], 1u << (e & 31u));
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            u32 tot = 0u;
            for (u32 i = 0; i < 64u; i++) tot += __popc(c.sm.moe.mask[i]);
            c.sm.moe.count = tot;
        }
        __syncthreads();
        u32 jobs = c.sm.moe.count * chunks;
        u32 rem = cursor % gridDim.x;
        u32 first = blockIdx.x >= rem ? (blockIdx.x - rem) : (blockIdx.x + gridDim.x - rem);
        for (u32 job = first; job < jobs; job += gridDim.x) {
            u32 slot = job / chunks, ch = job % chunks;
            if (threadIdx.x == 0u) {
                u32 left = slot, exp_id = s.experts;
                for (u32 i = 0; i < 64u && exp_id == s.experts; i++) {
                    u32 bits = c.sm.moe.mask[i], have = __popc(bits);
                    if (left < have) {
                        for (u32 t = 0; t < left; t++) bits &= bits - 1u;
                        exp_id = (i << 5) + u32(__ffs(bits) - 1);
                    } else {
                        left -= have;
                    }
                }
                u32 cnt = 0u;
                for (u32 r = 0; r < nr; r++) {
                    for (u32 k = 0; k < s.topk; k++) {
                        u32 pick = (base + r) * s.topk + k;
                        if (c.p.ids[pick] == exp_id && cnt < ROW_TILE) {
                            c.sm.moe.rows[cnt] = base + r;
                            c.sm.moe.picks[cnt] = pick;
                            cnt++;
                        }
                    }
                }
                c.sm.moe.chosen = exp_id | (cnt << 16);
            }
            __syncthreads();
            u32 e = c.sm.moe.chosen & 0xFFFFu;
            u32 n = c.sm.moe.chosen >> 16;
            u32 m = e - s.held_first;
            u32 side = warp & 1u;
            call_gu(c, codes[side], su[side], in_signs, m, e, ch, n);
            __syncthreads();
            if (warp < n) {
                float gv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float uv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (u32 widx = 0; widx < WARPS; widx += 2u) {
                    float4 pg =
                        *reinterpret_cast<const float4*>(&c.sm.moe.buf[widx][warp][lane << 2]);
                    float4 pu = *reinterpret_cast<const float4*>(
                        &c.sm.moe.buf[widx + 1u][warp][lane << 2]);
                    gv[0] += pg.x;
                    gv[1] += pg.y;
                    gv[2] += pg.z;
                    gv[3] += pg.w;
                    uv[0] += pu.x;
                    uv[1] += pu.y;
                    uv[2] += pu.z;
                    uv[3] += pu.w;
                }
                wht128_warp(gv, INV_SQRT128);
                wht128_warp(uv, INV_SQRT128);
                u32 o_base = (ch << 7) + (lane << 2);
                float act[4];
                for (u32 q = 0; q < 4; q++) {
                    u32 o = o_base + q;
                    float gval = gv[q] * vec_elem(sv[0], e, o);
                    float uval = uv[q] * vec_elem(sv[1], e, o);
                    act[q] = (gval / (1.0f + expf(-gval))) * uval * vec_elem(su[2], e, o);
                }
                wht128_warp(act, INV_SQRT128);
                *reinterpret_cast<float4*>(&c.sm.moe.vec[warp][lane << 2]) =
                    make_float4(act[0], act[1], act[2], act[3]);
            }
            __syncthreads();
            call_down(c, codes[2], m, ch, chunks, n);
            __syncthreads();
        }
        cursor += jobs;
        __syncthreads();
    }
}

// Routed MoE phase 2: WHT, scales, and `part` add.
static __device__ void routed_out(const ctx& c, const mat& down_sv, const float* out_signs) {
    const sizes& s = c.s;
    u32 h = s.hidden, ob_cnt = h >> 7;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    bool d_vec = (reinterpret_cast<u64>(c.p.d) & 15u) == 0u;
    bool p_vec = (reinterpret_cast<u64>(c.p.part) & 15u) == 0u;
    bool s_vec = (reinterpret_cast<u64>(out_signs) & 15u) == 0u;
    for (u32 u = wid; u < s.rows * ob_cnt; u += wstep) {
        u32 r = u / ob_cnt, ob = u % ob_cnt;
        u32 base = (ob << 7) + (lane << 2);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float sg[4];
        load_vec4(out_signs + base, s_vec, sg);
        for (u32 k = 0; k < s.topk; k++) {
            u32 pick = r * s.topk + k, m;
            if (!local_pick(c, pick, &m)) continue;
            u32 e = m + s.held_first;
            float v[4];
            load_vec4(c.p.d + u64(pick) * h + base, d_vec, v);
            wht128_warp(v, INV_SQRT128);
            float g = c.p.gates[pick];
            for (u32 q = 0; q < 4; q++) {
                acc[q] = fmaf(g, v[q] * vec_elem(down_sv, e, base + q) * sg[q], acc[q]);
            }
        }
        float* dst = c.p.part + u64(r) * h + base;
        float cur[4];
        load_vec4(dst, p_vec, cur);
        for (u32 q = 0; q < 4; q++) cur[q] += acc[q];
        store_vec4(dst, p_vec, cur);
    }
}

// Scalar fallback for non-trellis or unaligned routed experts.
static __device__ void turn_picks(const ctx& c, const float* x, const float* sign, const mat& by,
                                  const mat& after, const float* also, float* y, u32 n) {
    line l = threads();
    for (u32 u = l.at; u < c.s.rows * c.s.topk * n; u += l.step) {
        u32 rk = u / n, i = u % n, m;
        if (!local_pick(c, rk, &m)) continue;
        u32 e = m + c.s.held_first, base = i & ~127u, o = i & 127u;
        const float* row = x + u64(sign ? rk / c.s.topk : rk) * n;
        float sum = 0.0f;
        for (u32 j = 0; j < 128u; j++) {
            float v = row[base + j] * (by.p ? vec_elem(by, e, base + j) : 1.0f);
            if (sign) v *= sign[base + j];
            sum += (__popc(o & j) & 1) ? -v : v;
        }
        float v = sum * INV_SQRT128;
        if (after.p) v *= vec_elem(after, e, i);
        y[u] = also ? v * also[i] : v;
    }
}

static __device__ void picked_mat(const ctx& c, const mat& w, const float* x, float* y, u32 out,
                                  u32 in) {
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    for (u32 u = wid; u < c.s.rows * c.s.topk * out; u += wstep) {
        u32 rk = u / out, o = u % out, m;
        if (!local_pick(c, rk, &m)) continue;
        float acc = 0.0f;
        for (u32 i = lane; i < in; i += 32u) acc = fmaf(elem(w, m, o, i), x[u64(rk) * in + i], acc);
        float sum = warp_sum(acc);
        if (lane == 0u) y[u] = sum;
    }
}

static __device__ bool routed_slow(const ctx& c, const mat codes[3], const mat su[3],
                                   const mat sv[3], const float* in_signs, const float* out_signs) {
    const sizes& s = c.s;
    line l = threads();
    u32 h = s.hidden, f = s.effn, rk = s.rows * s.topk;
    mat none = {nullptr, K_F32, 0u, 0u, 0u, 0u, nullptr, nullptr};
    PHASE(turn_picks(c, c.p.h, in_signs, su[0], none, nullptr, c.p.b, h);
          turn_picks(c, c.p.h, in_signs, su[1], none, nullptr, c.p.c, h));
    PHASE(picked_mat(c, codes[0], c.p.b, c.p.d, f, h);
          picked_mat(c, codes[1], c.p.c, c.p.t1, f, h));
    PHASE(turn_picks(c, c.p.d, nullptr, none, sv[0], nullptr, c.p.b, f);
          turn_picks(c, c.p.t1, nullptr, none, sv[1], nullptr, c.p.c, f));
    for (u32 u = l.at; u < rk * f; u += l.step) {
        u32 m;
        if (!local_pick(c, u / f, &m)) continue;
        float g = c.p.b[u];
        c.p.t2[u] = g / (1.0f + expf(-g)) * c.p.c[u];
    }
    if (!sync(c.t)) return false;
    PHASE(turn_picks(c, c.p.t2, nullptr, su[2], none, nullptr, c.p.b, f));
    PHASE(picked_mat(c, codes[2], c.p.b, c.p.d, h, f));
    PHASE(turn_picks(c, c.p.d, nullptr, none, sv[2], out_signs, c.p.c, h));
    for (u32 u = l.at; u < s.rows * h; u += l.step) {
        u32 r = u / h, o = u % h, m;
        float sum = 0.0f;
        for (u32 k = 0; k < s.topk; k++) {
            u32 pick = r * s.topk + k;
            if (local_pick(c, pick, &m)) sum = fmaf(c.p.gates[pick], c.p.c[u64(pick) * h + o], sum);
        }
        c.p.part[u] += sum;
    }
    return sync(c.t);
}

static __device__ bool routed(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    const float* in_signs = reinterpret_cast<const float*>(field_mat(c, pl, SIGNS_IN).p);
    const float* out_signs = reinterpret_cast<const float*>(field_mat(c, pl, SIGNS_OUT).p);
    mat su[3], sv[3], codes[3];
    for (u32 i = 0; i < 3; i++) {
        u32 base = E_GATE + 3u * i;
        codes[i] = field_mat(c, pl, base);
        su[i] = field_mat(c, pl, base + 1u);
        sv[i] = field_mat(c, pl, base + 2u);
    }
    bool fast = codes[0].klass == K_TRELLIS && codes[1].klass == K_TRELLIS
                && codes[2].klass == K_TRELLIS && (s.hidden & 127u) == 0u && (s.effn & 127u) == 0u
                && s.experts <= 2048u;
    if (!fast) return routed_slow(c, codes, su, sv, in_signs, out_signs);
    PHASE(routed_step(c, codes, su, sv, in_signs));
    PHASE(routed_out(c, sv[2], out_signs));
    return true;
}

static __device__ sizes for_layer(const sizes& s, u32 l) {
    sizes ls = s;
    if (!bit(s.attn_split, l)) ls.heads = s.heads_all;
    if (!bit(s.mlp_split, l)) {
        ls.dense_ffn = s.dense_all;
        ls.ffn = s.ffn_all;
        ls.held_first = 0u;
        ls.held = s.experts;
    }
    return ls;
}

static __device__ bool layer(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    bool split = bit(s.mlp_split, pl.layer);
    bool mark = pl.layer == 0u || pl.layer == s.dense;
    if (mark) {
        TS_MARK();
    }
    if (!attention(c, pl)) return false;
    if (mark) {
        TS_MARK();
    }
    PHASE(rms(c, c.p.x, s.hidden, field_mat(c, pl, MLP_NORM), c.p.h, s.hidden, s.rows, s.hidden,
              nullptr, false));
    if (pl.layer < s.dense) {
        bool ok = swiglu(c, pl, D_GATE, s.dense_ffn) && add(c, split);
        if (mark) {
            TS_MARK();
        }
        return ok;
    }
    if (!route(c, pl)) return false;
    if (mark) {
        TS_MARK();
    }
    if (!swiglu(c, pl, S_GATE, s.ffn)) return false;
    if (mark) {
        TS_MARK();
    }
    if (!routed(c, pl)) return false;
    if (mark) {
        TS_MARK();
    }
    return add(c, split);
}
