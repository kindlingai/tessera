// A small decoder's step in one persistent grid, sums inside.
#include <tessera/1/all_reduce>
#include <tessera/1/grid>

using namespace tessera;

// Items: ids, embed and norm, layer weights, kept rows.
#define TOKENS 0
#define POSITIONS 1
#define SLOTS 2
#define PICKS 3
#define EMBED 4
#define NORM 5
#define LAYER 6
#define FIELDS 11
#define WARPS 32

// Layer fields, in item order.
enum field { LN1, Q, Q_NORM, K, K_NORM, V, O, LN2, GATE, UP, DOWN };

// Values, in the order the module counts them.
struct sizes {
    u32 rows, picks, vocab, hidden, layers, heads, kv, width, ffn;
    u32 first, group, kv_first, attn_sum, mlp_sum, layout, page;
    float eps, theta, scale;
    u32 kept;
};

static __device__ sizes read(const table* t) {
    sizes s;
    u32* w = reinterpret_cast<u32*>(&s);
    for (u32 i = 0; i < sizeof(sizes) / 4; i++) w[i] = value(t, i);
    return s;
}

// Every thread of the grid, as one line.
struct line {
    u32 at;
    u32 step;
};

static __device__ line threads() {
    return {blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x};
}

static __device__ const float* weight(const table* t, u32 layer, field f) {
    return at<const float>(t, LAYER + layer * FIELDS + f);
}

// A grid barrier after this thread's writes.
static __device__ bool sync(const table* t) {
    __threadfence();
    return grid_sync(t);
}

// The block's sum of `v`, in a fixed order.
static __device__ float block_sum(float v) {
    __shared__ float warps[WARPS];
    for (u32 o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffffu, v, o);
    u32 lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warps[warp] = v;
    __syncthreads();
    float all = 0.0f;
    for (u32 i = 0; i < blockDim.x / 32; i++) all += warps[i];
    __syncthreads();
    return all;
}

// Rows of `x` normed by `w`, one block per row.
static __device__ void rms(const float* x, const float* w, float* out, const u32* from, u32 rows,
                            u32 n, float eps) {
    for (u32 r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* row = x + u64(from ? from[r] : r) * n;
        float sq = 0.0f;
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) sq += row[i] * row[i];
        float inv = rsqrtf(block_sum(sq) / n + eps);
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) out[u64(r) * n + i] = row[i] * inv * w[i];
    }
}

// Each row of `x` times each row of `w`.
static __device__ void times(const float* x, const float* w, float* out, u32 rows, u32 n, u32 k) {
    line l = threads();
    for (u32 i = l.at; i < rows * n; i += l.step) {
        const float* a = x + u64(i / n) * k;
        const float* b = w + u64(i % n) * k;
        float s = 0.0f;
        for (u32 j = 0; j < k; j++) s += a[j] * b[j];
        out[i] = s;
    }
}

// One head of `v`, normed by `w`, turned at `pos`.
static __device__ void turn(float* v, const float* w, u32 width, u32 pos, const sizes& s) {
    float sq = 0.0f;
    for (u32 i = 0; i < width; i++) sq += v[i] * v[i];
    float inv = rsqrtf(sq / width + s.eps);
    for (u32 i = 0; i < width; i++) v[i] *= inv * w[i];
    u32 half = width / 2;
    for (u32 j = 0; j < half; j++) {
        float angle = float(pos) * powf(s.theta, -float(2 * j) / float(width));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        float a = v[j], b = v[j + half];
        v[j] = a * cs - b * sn;
        v[j + half] = a * sn + b * cs;
    }
}

// The kept row of `pos` in `slot`, by the layout.
static __device__ u64 kept_row(const sizes& s, const u32* pages, u32 slot, u32 pos) {
    if (s.layout == 0) return u64(slot) * s.kept + pos;
    if (s.layout == 1) return u64(pos) * s.kept + slot;
    u32 shift = __ffs(s.page) - 1;
    u64 page = pages[slot * s.kept + (pos >> shift)];
    return page << shift | (pos & (s.page - 1));
}

// Adds `part`, summed over ranks if `ranks`, into `x`.
static __device__ bool add(const table* t, session& se, const float* part, float* x, u32 n,
                           bool ranks) {
    line l = threads();
    if (!ranks) {
        for (u32 i = l.at; i < n; i += l.step) x[i] += part[i];
        return sync(t);
    }
    u32 room = detail::pools(t)->pools[0].send_bytes / 4;
    for (u32 done = 0; done < n; done += room) {
        u32 len = min(room, n - done);
        ticket r = prepare(se, 0);
        float* sent = send_buffer<float>(r);
        if (sent == nullptr) return false;
        for (u32 i = l.at; i < len; i += l.step) sent[i] = part[done + i];
        if (!trigger(r, len) || !wait(r)) return false;
        parts got = received(r);
        for (u32 i = l.at; i < len; i += l.step) x[done + i] += got.sum(i);
    }
    return sync(t);
}

// Scratch spans, in floats per row.
struct spans {
    float *x, *h, *q, *k, *v, *att, *mlp, *part;
};

static __device__ spans lay_out(const table* t, const sizes& s) {
    float* base = at<float>(t, t->items_len - 1);
    u32 r = s.rows, q = s.heads * s.width, kv = s.kv * s.width;
    spans p;
    p.x = base;
    p.h = p.x + r * s.hidden;
    p.part = p.h + r * s.hidden;
    p.q = p.part + r * s.hidden;
    p.att = p.q + r * q;
    p.k = p.att + r * q;
    p.v = p.k + r * kv;
    p.mlp = p.v + r * kv;
    return p;
}

// Norms and turns queries and keys. Appends keys, values.
static __device__ void keys(const table* t, const sizes& s, const spans& p, u32 layer) {
    line l = threads();
    const u32* pos = at<const u32>(t, POSITIONS);
    const u32* slot = at<const u32>(t, SLOTS);
    const u32* pages = s.layout == 2 ? at<const u32>(t, t->items_len - 2) : nullptr;
    u32 base = LAYER + s.layers * FIELDS + 2 * layer;
    float* kept_k = at<float>(t, base);
    float* kept_v = at<float>(t, base + 1);
    u32 heads = s.heads + s.kv;
    for (u32 i = l.at; i < s.rows * heads; i += l.step) {
        u32 r = i / heads, h = i % heads;
        if (h < s.heads) {
            turn(p.q + (u64(r) * s.heads + h) * s.width, weight(t, layer, Q_NORM), s.width, pos[r], s);
            continue;
        }
        h -= s.heads;
        u64 off = (u64(r) * s.kv + h) * s.width;
        turn(p.k + off, weight(t, layer, K_NORM), s.width, pos[r], s);
        u64 row = kept_row(s, pages, slot[r], pos[r]) * s.kv * s.width + h * s.width;
        for (u32 d = 0; d < s.width; d++) {
            kept_k[row + d] = p.k[off + d];
            kept_v[row + d] = p.v[off + d];
        }
    }
}

// Each row's heads over its slot's kept rows, causal.
static __device__ void attend(const table* t, const sizes& s, const spans& p, u32 layer) {
    line l = threads();
    const u32* pos = at<const u32>(t, POSITIONS);
    const u32* slot = at<const u32>(t, SLOTS);
    const u32* pages = s.layout == 2 ? at<const u32>(t, t->items_len - 2) : nullptr;
    u32 base = LAYER + s.layers * FIELDS + 2 * layer;
    const float* kept_k = at<const float>(t, base);
    const float* kept_v = at<const float>(t, base + 1);
    for (u32 i = l.at; i < s.rows * s.heads; i += l.step) {
        u32 r = i / s.heads, h = i % s.heads;
        u32 g = (s.first + h) / s.group - s.kv_first;
        const float* q = p.q + (u64(r) * s.heads + h) * s.width;
        float* o = p.att + (u64(r) * s.heads + h) * s.width;
        float most = __int_as_float(0xff800000u), total = 0.0f;
        for (u32 d = 0; d < s.width; d++) o[d] = 0.0f;
        for (u32 j = 0; j <= pos[r]; j++) {
            u64 row = kept_row(s, pages, slot[r], j) * s.kv * s.width + g * s.width;
            float score = 0.0f;
            for (u32 d = 0; d < s.width; d++) score += q[d] * kept_k[row + d];
            score *= s.scale;
            float top = fmaxf(most, score);
            float shrink = expf(most - top), e = expf(score - top);
            total = total * shrink + e;
            for (u32 d = 0; d < s.width; d++) o[d] = o[d] * shrink + e * kept_v[row + d];
            most = top;
        }
        for (u32 d = 0; d < s.width; d++) o[d] /= total;
    }
}

// SiLU of the gate times the up rows.
static __device__ void gated(float* mlp, const float* up, u32 n) {
    line l = threads();
    for (u32 i = l.at; i < n; i += l.step) {
        float g = mlp[i];
        mlp[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

static __device__ bool layer(const table* t, session& se, const sizes& s, const spans& p, u32 i) {
    u32 r = s.rows, q = s.heads * s.width, kv = s.kv * s.width;
    rms(p.x, weight(t, i, LN1), p.h, nullptr, r, s.hidden, s.eps);
    if (!sync(t)) return false;
    times(p.h, weight(t, i, Q), p.q, r, q, s.hidden);
    times(p.h, weight(t, i, K), p.k, r, kv, s.hidden);
    times(p.h, weight(t, i, V), p.v, r, kv, s.hidden);
    if (!sync(t)) return false;
    keys(t, s, p, i);
    if (!sync(t)) return false;
    attend(t, s, p, i);
    if (!sync(t)) return false;
    times(p.att, weight(t, i, O), p.part, r, s.hidden, q);
    if (!sync(t) || !add(t, se, p.part, p.x, r * s.hidden, s.attn_sum)) return false;
    rms(p.x, weight(t, i, LN2), p.h, nullptr, r, s.hidden, s.eps);
    if (!sync(t)) return false;
    times(p.h, weight(t, i, GATE), p.mlp, r, s.ffn, s.hidden);
    times(p.h, weight(t, i, UP), p.att, r, s.ffn, s.hidden);
    if (!sync(t)) return false;
    gated(p.mlp, p.att, r * s.ffn);
    if (!sync(t)) return false;
    times(p.mlp, weight(t, i, DOWN), p.part, r, s.hidden, s.ffn);
    return sync(t) && add(t, se, p.part, p.x, r * s.hidden, s.mlp_sum);
}

// Values: see `sizes`. Items: see the defines, then out.
extern "C" __global__ void resident_decoder(const table* t) {
    if (!check(t)) return;
    sizes s = read(t);
    spans p = lay_out(t, s);
    session se = begin(t);
    const u32* tokens = at<const u32>(t, TOKENS);
    const float* embed = at<const float>(t, EMBED);
    line l = threads();
    for (u32 i = l.at; i < s.rows * s.hidden; i += l.step) {
        p.x[i] = embed[u64(tokens[i / s.hidden]) * s.hidden + i % s.hidden];
    }
    if (!sync(t)) return;
    for (u32 i = 0; i < s.layers; i++) {
        if (!layer(t, se, s, p, i)) return;
    }
    const u32* picks = at<const u32>(t, PICKS);
    rms(p.x, at<const float>(t, NORM), p.h, picks, s.picks, s.hidden, s.eps);
    if (!sync(t)) return;
    u32 out = LAYER + s.layers * (FIELDS + 2);
    times(p.h, embed, at<float>(t, out), s.picks, s.vocab, s.hidden);
    end(se);
}
