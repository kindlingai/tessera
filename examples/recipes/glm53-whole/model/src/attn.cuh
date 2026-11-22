// MLA indexer, RoPE, KV absorb, block-parallel attend, and value projection.

static __device__ __forceinline__ u32 float_key(float v) {
    if (v == 0.0f) v = 0.0f;
    u32 b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}

static __device__ __forceinline__ u32 block_count(const ctx& c, u32 local) {
    u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
#pragma unroll
    for (u32 off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xFFFFFFFFu, local, off);
    if (lane == 0u) c.sm.sel.cnt[warp] = local;
    __syncthreads();
    u32 sum = 0u, nw = blockDim.x >> 5;
    for (u32 w = 0; w < nw; w++) sum += c.sm.sel.cnt[w];
    __syncthreads();
    return sum;
}

static __device__ __forceinline__ void block_prefix(const ctx& c, bool flag, u32* pos_out,
                                                    u32* tot_out) {
    u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    u32 mask = __ballot_sync(0xFFFFFFFFu, flag);
    u32 lane_off = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0u) c.sm.sel.cnt[warp] = __popc(mask);
    __syncthreads();
    u32 warp_off = 0u, total = 0u, nw = blockDim.x >> 5;
    for (u32 w = 0; w < nw; w++) {
        u32 wc = c.sm.sel.cnt[w];
        if (w < warp) warp_off += wc;
        total += wc;
    }
    __syncthreads();
    *pos_out = warp_off + lane_off;
    *tot_out = total;
}

// Centered RMS norm plus bias, stored into kept index table.
static __device__ void index_keys(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    const kept* kt = kept_table(c.t, pl.kept + 2u);
    mat knorm = field_mat(c, pl, K_NORM);
    mat kbias = field_mat(c, pl, K_BIAS);
    for (u32 r = blockIdx.x; r < s.rows; r += gridDim.x) {
        const float* row = c.p.b + u64(r) * s.idim;
        float sum = 0.0f;
        for (u32 d = threadIdx.x; d < s.idim; d += blockDim.x) sum += row[d];
        float mean = block_sum(c, sum) / float(s.idim);
        float sq = 0.0f;
        for (u32 d = threadIdx.x; d < s.idim; d += blockDim.x) {
            float v = row[d] - mean;
            sq = fmaf(v, v, sq);
        }
        float inv = 1.0f / sqrtf(block_sum(c, sq) / float(s.idim) + s.eps);
        u8* dst = kept_row(c.t, kt, slot[r], pos[r]);
        for (u32 d = threadIdx.x; d < s.idim; d += blockDim.x) {
            float out = (row[d] - mean) * (inv * vec_elem(knorm, 0u, d)) + vec_elem(kbias, 0u, d);
            kv_store(dst, kt->elem, d, out);
        }
    }
}

// Warp-parallel causal indexer scores for rows exceeding keep.
static __device__ void index_scores(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    const kept* kt = kept_table(c.t, pl.kept + 2u);
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 cursor = 0u;
    for (u32 r = 0; r < s.rows; r++) {
        u32 lim = min(pos[r] + 1u, s.context);
        if (lim <= s.keep) continue;
        u32 rem = cursor % wstep;
        u32 first = wid >= rem ? (wid - rem) : (wid + wstep - rem);
        const float* qr = c.p.a + u64(r) * s.ih * s.idim;
        const float* wr = c.p.c + u64(r) * s.ih;
        float* dr = c.p.d + u64(r) * s.context;
        u32 sl = slot[r];
        for (u32 j = first; j < lim; j += wstep) {
            const u8* krow = kept_row(c.t, kt, sl, j);
            float score = 0.0f;
            if (s.idim <= 128u) {
                float kr[4];
#pragma unroll
                for (u32 t = 0; t < 4; t++) {
                    u32 d = lane + (t << 5);
                    kr[t] = d < s.idim ? kv_load(krow, kt->elem, d) : 0.0f;
                }
                for (u32 h = 0; h < s.ih; h++) {
                    const float* qh = qr + u64(h) * s.idim;
                    float hit = 0.0f;
#pragma unroll
                    for (u32 t = 0; t < 4; t++) {
                        u32 d = lane + (t << 5);
                        if (d < s.idim) hit = fmaf(qh[d], kr[t], hit);
                    }
                    hit = warp_sum(hit);
                    if (lane == 0u) score = fmaf(fmaxf(hit, 0.0f), wr[h], score);
                }
            } else {
                for (u32 h = 0; h < s.ih; h++) {
                    const float* qh = qr + u64(h) * s.idim;
                    float hit = 0.0f;
                    for (u32 d = lane; d < s.idim; d += 32u) {
                        hit = fmaf(qh[d], kv_load(krow, kt->elem, d), hit);
                    }
                    hit = warp_sum(hit);
                    if (lane == 0u) score = fmaf(fmaxf(hit, 0.0f), wr[h], score);
                }
            }
            if (lane == 0u) dr[j] = score;
        }
        cursor += lim;
    }
}

// Block-parallel exact top-k selection with lower-index tie break.
static __device__ void select_row(const ctx& c, const float* row, u32 lim, u32 keep, u32 ctx_len,
                                  u32* out) {
    if (lim <= keep) {
        for (u32 j = threadIdx.x; j < keep; j += blockDim.x) out[j] = j < lim ? j : ctx_len;
        return;
    }
    float neg_inf = __int_as_float(0xFF800000u);
    u32 my_valid = 0u;
    for (u32 j = threadIdx.x; j < lim; j += blockDim.x) {
        if (row[j] > neg_inf) my_valid++;
    }
    u32 valid_total = block_count(c, my_valid);
    if (valid_total <= keep) {
        u32 off = 0u;
        for (u32 base = 0; base < lim; base += blockDim.x) {
            u32 j = base + threadIdx.x;
            bool ok = j < lim && row[j] > neg_inf;
            u32 p, tot;
            block_prefix(c, ok, &p, &tot);
            if (ok) out[off + p] = j;
            off += tot;
        }
        for (u32 j = valid_total + threadIdx.x; j < keep; j += blockDim.x) out[j] = ctx_len;
        return;
    }
    u32 prefix = 0u;
    for (int b = 31; b >= 0; b--) {
        u32 cand = prefix | (1u << b);
        u32 cnt = 0u;
        for (u32 j = threadIdx.x; j < lim; j += blockDim.x) {
            float v = row[j];
            if (v > neg_inf && float_key(v) >= cand) cnt++;
        }
        if (block_count(c, cnt) >= keep) prefix = cand;
    }
    u32 my_gt = 0u;
    for (u32 j = threadIdx.x; j < lim; j += blockDim.x) {
        float v = row[j];
        if (v > neg_inf && float_key(v) > prefix) my_gt++;
    }
    u32 gt_total = block_count(c, my_gt);
    u32 eq_need = keep - gt_total;
    u32 gt_off = 0u, eq_off = 0u;
    for (u32 base = 0; base < lim; base += blockDim.x) {
        u32 j = base + threadIdx.x;
        float v = j < lim ? row[j] : neg_inf;
        bool ok = j < lim && v > neg_inf;
        u32 key = ok ? float_key(v) : 0u;
        bool is_gt = ok && key > prefix;
        bool is_eq = ok && key == prefix;
        u32 pg, tg, pe, te;
        block_prefix(c, is_gt, &pg, &tg);
        block_prefix(c, is_eq, &pe, &te);
        if (is_gt) out[gt_off + pg] = j;
        if (is_eq && eq_off + pe < eq_need) out[gt_total + eq_off + pe] = j;
        gt_off += tg;
        eq_off += te;
    }
}

static __device__ void select_top(const ctx& c) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    for (u32 r = blockIdx.x; r < s.rows; r += gridDim.x) {
        u32 lim = min(pos[r] + 1u, s.context);
        select_row(c, c.p.d + u64(r) * s.context, lim, s.keep, s.context,
                   c.p.picked + u64(r) * s.keep);
    }
}

static __device__ void fill_direct(const ctx& c) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    line l = threads();
    for (u32 u = l.at; u < s.rows * s.keep; u += l.step) {
        u32 r = u / s.keep, j = u % s.keep;
        u32 lim = min(pos[r] + 1u, s.context);
        c.p.picked[u] = j < lim ? j : s.context;
    }
}

static __device__ bool pick_keys(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    u32 r = s.rows;
    if (!linear(c, field_mat(c, pl, WQ), c.p.qr, s.q_latent, c.p.a, s.ih * s.idim, r,
                s.ih * s.idim)) {
        return false;
    }
    if (!linear(c, field_mat(c, pl, WK), c.p.h, s.hidden, c.p.b, s.idim, r, s.idim)) return false;
    if (!linear(c, field_mat(c, pl, I_WEIGHTS), c.p.h, s.hidden, c.p.c, s.ih, r, s.ih)) {
        return false;
    }
    PHASE(index_keys(c, pl));
    const u32* pos = at<const u32>(c.t, 1);
    bool need_sel = false;
    for (u32 row = 0; row < r; row++) {
        if (min(pos[row] + 1u, s.context) > s.keep) need_sel = true;
    }
    if (!need_sel) {
        PHASE(fill_direct(c));
        return true;
    }
    PHASE(index_scores(c, pl));
    PHASE(select_top(c));
    return true;
}

// Parallel RoPE on queries and key, plus kept stores.
static __device__ void turn_keys(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    line l = threads();
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    const kept* kt_l = kept_table(c.t, pl.kept);
    const kept* kt_r = kept_table(c.t, pl.kept + 1u);
    u32 pairs = s.rope >> 1, q_stride = s.nope + s.rope, kv_stride = s.latent + s.rope;
    for (u32 u = l.at; u < s.rows * s.heads * pairs; u += l.step) {
        u32 j = u % pairs, rh = u / pairs, r = rh / s.heads;
        float angle = float(pos[r]) * powf(s.theta, -float(j << 1) / float(s.rope));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        float* v = c.p.a + u64(rh) * q_stride + s.nope + (j << 1);
        float a = v[0], b = v[1];
        v[0] = a * cs - b * sn;
        v[1] = a * sn + b * cs;
    }
    for (u32 u = l.at; u < s.rows * pairs; u += l.step) {
        u32 r = u / pairs, j = u % pairs;
        float angle = float(pos[r]) * powf(s.theta, -float(j << 1) / float(s.rope));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        float* kv = c.p.kv + u64(r) * kv_stride + s.latent + (j << 1);
        float a = kv[0], b = kv[1];
        float r0 = a * cs - b * sn, r1 = a * sn + b * cs;
        kv[0] = r0;
        kv[1] = r1;
        u8* kr = kept_row(c.t, kt_r, slot[r], pos[r]);
        kv_store(kr, kt_r->elem, j << 1, r0);
        kv_store(kr, kt_r->elem, (j << 1) + 1u, r1);
    }
    for (u32 u = l.at; u < s.rows * s.latent; u += l.step) {
        u32 r = u / s.latent, d = u % s.latent;
        u8* kl = kept_row(c.t, kt_l, slot[r], pos[r]);
        kv_store(kl, kt_l->elem, d, c.p.kv[u64(r) * kv_stride + d]);
    }
}

// Block-parallel query absorption into latent space, coalesced along latent.
static __device__ void absorb(const ctx& c, const mat& kv_b) {
    const sizes& s = c.s;
    u32 q_stride = s.nope + s.rope, width = s.latent + s.rope;
    bool stage = s.nope <= ATTN_STAGE;
    for (u32 rh = blockIdx.x; rh < s.rows * s.heads; rh += gridDim.x) {
        u32 h = rh % s.heads;
        const float* qa = c.p.a + u64(rh) * q_stride;
        float* qb = c.p.b + u64(rh) * width;
        if (stage) {
            for (u32 n = threadIdx.x; n < s.nope; n += blockDim.x) c.sm.attn.q[n] = qa[n];
            __syncthreads();
        }
        const float* q_src = stage ? c.sm.attn.q : qa;
        for (u32 d = threadIdx.x; d < s.rope; d += blockDim.x) qb[s.latent + d] = qa[s.nope + d];
        u32 base_row = h * (s.nope + s.hd);
        for (u32 i = threadIdx.x; i < s.latent; i += blockDim.x) {
            float acc = 0.0f;
            for (u32 n = 0; n < s.nope; n++) {
                acc = fmaf(q_src[n], vec_elem(kv_b, base_row + n, i), acc);
            }
            qb[i] = acc;
        }
        if (stage) __syncthreads();
    }
}

// Block-parallel online softmax attention over picked kept rows.
static __device__ void attend(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    const kept* kt_l = kept_table(c.t, pl.kept);
    const kept* kt_r = kept_table(c.t, pl.kept + 1u);
    u32 width = s.latent + s.rope;
    bool stage = width <= ATTN_STAGE;
    bool in_reg = s.latent <= 1024u;
    for (u32 rh = blockIdx.x; rh < s.rows * s.heads; rh += gridDim.x) {
        u32 r = rh / s.heads;
        u32 pr = pos[r], sl = slot[r];
        const float* qg = c.p.b + u64(rh) * width;
        float* og = c.p.c + u64(rh) * s.latent;
        if (stage) {
            for (u32 d = threadIdx.x; d < width; d += blockDim.x) c.sm.attn.q[d] = qg[d];
            __syncthreads();
        }
        const float* q = stage ? c.sm.attn.q : qg;
        float o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (!in_reg) {
            for (u32 d = threadIdx.x; d < s.latent; d += blockDim.x) og[d] = 0.0f;
        }
        float most = __int_as_float(0xFF800000u), total = 0.0f;
        for (u32 n = 0; n < s.keep; n++) {
            u32 j = c.p.picked[u64(r) * s.keep + n];
            if (j >= s.context || j > pr) continue;
            const u8* kl = kept_row(c.t, kt_l, sl, j);
            const u8* kr = kept_row(c.t, kt_r, sl, j);
            float dot = 0.0f, kv_reg[4];
            if (in_reg) {
#pragma unroll
                for (u32 t = 0; t < 4; t++) {
                    u32 d = threadIdx.x + (t << 8);
                    if (d < s.latent) {
                        kv_reg[t] = kv_load(kl, kt_l->elem, d);
                        dot = fmaf(q[d], kv_reg[t], dot);
                    }
                }
            } else {
                for (u32 d = threadIdx.x; d < s.latent; d += blockDim.x) {
                    dot = fmaf(q[d], kv_load(kl, kt_l->elem, d), dot);
                }
            }
            for (u32 d = threadIdx.x; d < s.rope; d += blockDim.x) {
                dot = fmaf(q[s.latent + d], kv_load(kr, kt_r->elem, d), dot);
            }
            float score = block_sum(c, dot) * s.scale;
            float topmost = fmaxf(most, score);
            float shrink = expf(most - topmost), e = expf(score - topmost);
            total = total * shrink + e;
            most = topmost;
            if (in_reg) {
#pragma unroll
                for (u32 t = 0; t < 4; t++) {
                    u32 d = threadIdx.x + (t << 8);
                    if (d < s.latent) o_reg[t] = fmaf(o_reg[t], shrink, e * kv_reg[t]);
                }
            } else {
                for (u32 d = threadIdx.x; d < s.latent; d += blockDim.x) {
                    og[d] = fmaf(og[d], shrink, e * kv_load(kl, kt_l->elem, d));
                }
            }
        }
        if (in_reg) {
#pragma unroll
            for (u32 t = 0; t < 4; t++) {
                u32 d = threadIdx.x + (t << 8);
                if (d < s.latent) og[d] = o_reg[t] / total;
            }
        } else {
            for (u32 d = threadIdx.x; d < s.latent; d += blockDim.x) og[d] /= total;
        }
        if (stage) __syncthreads();
    }
}

// Warp-parallel value projection, coalesced along latent.
static __device__ void head_values(const ctx& c, const mat& kv_b) {
    const sizes& s = c.s;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 vec_end = s.latent & ~3u;
    bool o_vec = (reinterpret_cast<u64>(c.p.c) & 15u) == 0u && (s.latent & 3u) == 0u;
    for (u32 u = wid; u < s.rows * s.heads * s.hd; u += wstep) {
        u32 rh = u / s.hd, d = u % s.hd, h = rh % s.heads;
        u32 row_idx = h * (s.nope + s.hd) + s.nope + d;
        const float* o = c.p.c + u64(rh) * s.latent;
        float acc = 0.0f;
        if (kv_b.klass == K_BF16 && o_vec) {
            const u16* wr = reinterpret_cast<const u16*>(kv_b.p) + u64(row_idx) * kv_b.in;
            if ((reinterpret_cast<u64>(wr) & 7u) == 0u && vec_end > 0u) {
                for (u32 i = lane << 2; i < vec_end; i += 128u) {
                    uint2 b = *reinterpret_cast<const uint2*>(wr + i);
                    float4 ov = *reinterpret_cast<const float4*>(o + i);
                    acc = fmaf(__uint_as_float(b.x << 16), ov.x, acc);
                    acc = fmaf(__uint_as_float(b.x & 0xFFFF0000u), ov.y, acc);
                    acc = fmaf(__uint_as_float(b.y << 16), ov.z, acc);
                    acc = fmaf(__uint_as_float(b.y & 0xFFFF0000u), ov.w, acc);
                }
                for (u32 i = vec_end + lane; i < s.latent; i += 32u) {
                    acc = fmaf(__uint_as_float(u32(wr[i]) << 16), o[i], acc);
                }
            } else {
                for (u32 i = lane; i < s.latent; i += 32u) {
                    acc = fmaf(__uint_as_float(u32(wr[i]) << 16), o[i], acc);
                }
            }
        } else if (kv_b.klass == K_F32 && o_vec) {
            const float* wr = reinterpret_cast<const float*>(kv_b.p) + u64(row_idx) * kv_b.in;
            if ((reinterpret_cast<u64>(wr) & 15u) == 0u && vec_end > 0u) {
                for (u32 i = lane << 2; i < vec_end; i += 128u) {
                    float4 wv = *reinterpret_cast<const float4*>(wr + i);
                    float4 ov = *reinterpret_cast<const float4*>(o + i);
                    acc = fmaf(wv.x, ov.x, acc);
                    acc = fmaf(wv.y, ov.y, acc);
                    acc = fmaf(wv.z, ov.z, acc);
                    acc = fmaf(wv.w, ov.w, acc);
                }
                for (u32 i = vec_end + lane; i < s.latent; i += 32u) acc = fmaf(wr[i], o[i], acc);
            } else {
                for (u32 i = lane; i < s.latent; i += 32u) acc = fmaf(wr[i], o[i], acc);
            }
        } else {
            for (u32 i = lane; i < s.latent; i += 32u) {
                acc = fmaf(vec_elem(kv_b, row_idx, i), o[i], acc);
            }
        }
        float sum = warp_sum(acc);
        if (lane == 0u) c.p.d[u] = sum;
    }
}

static __device__ bool attention(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    u32 r = s.rows, kv = s.latent + s.rope, q = s.heads * (s.nope + s.rope);
    PHASE(rms(c, c.p.x, s.hidden, field_mat(c, pl, A_NORM), c.p.h, s.hidden, r, s.hidden, nullptr,
              false));
    if (!linear(c, field_mat(c, pl, Q_A), c.p.h, s.hidden, c.p.qr, s.q_latent, r, s.q_latent)) {
        return false;
    }
    if (!linear(c, field_mat(c, pl, KV_A), c.p.h, s.hidden, c.p.kv, kv, r, kv)) return false;
    PHASE(rms(c, c.p.qr, s.q_latent, field_mat(c, pl, Q_NORM), c.p.qr, s.q_latent, r, s.q_latent,
              nullptr, false));
    PHASE(rms(c, c.p.kv, kv, field_mat(c, pl, KV_NORM), c.p.kv, kv, r, s.latent, nullptr, false));
    if (indexes(s, pl.layer) && !pick_keys(c, pl)) return false;
    if (!linear(c, field_mat(c, pl, Q_B), c.p.qr, s.q_latent, c.p.a, q, r, q)) return false;
    PHASE(turn_keys(c, pl));
    mat kv_b = field_mat(c, pl, KV_B);
    PHASE(absorb(c, kv_b));
    PHASE(attend(c, pl));
    PHASE(head_values(c, kv_b));
    if (!linear(c, field_mat(c, pl, O), c.p.d, s.heads * s.hd, c.p.part, s.hidden, r, s.hidden)) {
        return false;
    }
    return add(c, bit(s.attn_split, pl.layer));
}
