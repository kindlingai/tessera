// GLM-5.3's whole step in one persistent grid, sums inside.
#include <tessera/1/all_reduce>
#include <tessera/1/grid>
#include <tessera/1/kept>

using namespace tessera;

// Items: ids, then fields by layer, kept rows, out.
#define IDS 4
// Trellis header bytes, and the side word of input entries.
#define HEAD 16
#define INPUT_SIDE 2u
#define MCG 0xCBAC1FEDu
#define MUL1 0x83DCD12Du

// Field numbers, in the module's order.
enum field : u32 {
    EMBED, NORM, LM_HEAD,
    A_NORM, Q_A, Q_NORM, Q_B, KV_A, KV_NORM, KV_B, O, MLP_NORM,
    WQ, WK, K_NORM, K_BIAS, I_WEIGHTS,
    D_GATE, D_UP, D_DOWN,
    ROUTER, R_BIAS, SIGNS_IN, SIGNS_OUT, E_GATE, GATE_SU, GATE_SV, E_UP, UP_SU, UP_SV,
    E_DOWN, DOWN_SU, DOWN_SV, S_GATE, S_UP, S_DOWN,
    FIELD_END
};

// Form classes, read from `core.h` item and form metadata.
enum klass : u32 { K_F32, K_BF16, K_INT4, K_MXFP8, K_TRELLIS, K_TURNED };

// Values, in the module's order.
struct sizes {
    u32 rows, picks, vocab, hidden, layers, dense, every;
    u32 heads, heads_all, nope, rope, hd, q_latent, latent;
    u32 ih, idim, keep, context, dense_ffn, dense_all, ffn, ffn_all, effn;
    u32 experts, held_first, held, topk;
    // Per layer: whether attention and the MLP split.
    u32 attn_split[3], mlp_split[3];
    u32 width;
    float eps, theta, route, scale;
};

static __device__ sizes read(const table* t) {
    sizes s;
    u32* w = reinterpret_cast<u32*>(&s);
    for (u32 i = 0; i < sizeof(sizes) / 4; i++) w[i] = value(t, i);
    return s;
}

static __device__ bool bit(const u32* words, u32 l) {
    return (words[l / 32] >> (l % 32)) & 1u;
}

static __device__ bool indexes(const sizes& s, u32 l) {
    return l < s.dense || (l + 1 - s.dense) % s.every == 0;
}

// Whether `item` is a rotated packed form.
static __device__ bool is_turned(const table* t, u32 item) {
    const tessera::item* it = item_at(t, item);
    return it->elem == FORM && form_of(t, item, 0)->suh.arena != NONE;
}

// A layer's items, and its first kept table.
struct place {
    u32 item[FIELD_END];
    u32 next, kept, layer;
};

// Places fields `from..to` at `*at`. Rotated codes take three.
static __device__ void lay_fields(const table* t, place& pl, u32 from, u32 to, u32* at) {
    for (u32 f = from; f < to; f++) {
        pl.item[f] = *at;
        *at += is_turned(t, *at) ? 3 : 1;
    }
}

// Layer `l`'s places, from item `first`, at its sizes `s`.
static __device__ place walk(const table* t, const sizes& s, u32 l, u32 first, u32 kept) {
    place pl;
    pl.kept = kept;
    pl.layer = l;
    u32 at = first;
    lay_fields(t, pl, A_NORM, WQ, &at);
    if (indexes(s, l)) lay_fields(t, pl, WQ, D_GATE, &at);
    if (l < s.dense) {
        lay_fields(t, pl, D_GATE, ROUTER, &at);
    } else {
        lay_fields(t, pl, ROUTER, FIELD_END, &at);
    }
    pl.next = at;
    return pl;
}

// Every thread of the grid, as one line.
struct line {
    u32 at;
    u32 step;
};

static __device__ line threads() {
    return {blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x};
}

static __device__ bool sync(const table* t) {
    __threadfence();
    return grid_sync(t);
}

static __device__ float half(u32 h) {
    float r;
    unsigned short b = (unsigned short)h;
    asm("{.reg .b16 t; mov.b16 t, %1; cvt.f32.f16 %0, t;}" : "=f"(r) : "h"(b));
    return r;
}

// `v` rounded to f16, then to E4M3, saturated at 448.
static __device__ float narrow(float v) {
    u32 h, e;
    asm("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(h) : "f"(v));
    asm("{.reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}"
        : "=r"(e) : "r"(h));
    return half(e & 0xFFFFu);
}

static __device__ float e4m3(u32 c) {
    u32 e = (c >> 3) & 15u, m = c & 7u;
    if (e == 15 && m == 7) return __int_as_float(0x7FC00000);
    float v = e == 0 ? ldexpf(float(m), -9) : ldexpf(float(8 + m), int(e) - 10);
    return (c & 0x80u) ? -v : v;
}

// One weight matrix, or a set, in its form.
struct mat {
    const u8* p;
    u32 klass, in, out, k, book;
    const float* suh;
    const float* svh;
};

static __device__ u32 word(const u8* p, u32 at) {
    return *reinterpret_cast<const u32*>(p + at);
}

// Weight at `item`, decoded from `core.h` item and form metadata.
static __device__ mat weight(const table* t, u32 item) {
    const tessera::item* it = item_at(t, item);
    u32 out = it->dims >= 2 ? it->shape[it->dims - 2] : 1;
    u32 in = it->dims >= 1 ? it->shape[it->dims - 1] : 0;
    mat w = {at<const u8>(t, item), K_F32, in, out, 0, 0, nullptr, nullptr};
    if (it->elem == F32) return w;
    if (it->elem == BF16) {
        w.klass = K_BF16;
        return w;
    }
    const tessera::form* fm = form_of(t, item, 0);
    if (fm->kind == FORM_INT4_G32) {
        w.klass = K_INT4;
        return w;
    }
    if (fm->kind == FORM_MXFP8) {
        w.klass = K_MXFP8;
        return w;
    }
    bool turned = fm->suh.arena != NONE;
    w.klass = turned ? K_TURNED : K_TRELLIS;
    w.in = word(w.p, 0);
    w.out = word(w.p, 4);
    bool mul1 = fm->kind >= FORM_TRELLIS_K2_MUL1;
    w.k = (mul1 ? fm->kind - FORM_TRELLIS_K2_MUL1 : fm->kind - FORM_TRELLIS_K2) + 2;
    w.book = mul1 ? 1 : 0;
    if (turned) {
        w.suh = get<const float>(t, fm->suh);
        w.svh = get<const float>(t, fm->svh);
    }
    return w;
}

static __device__ u32 tile_bit(const u8* tile, u32 b) {
    u32 at = 2 * ((b / 16) ^ 1);
    u32 w = tile[at] | u32(tile[at + 1]) << 8;
    return (w >> (15 - b % 16)) & 1u;
}

// Trellis element of matrix `m` at input `row`, output `col`.
static __device__ float trellis(const mat& w, u64 m, u32 row, u32 col) {
    u32 r = row % 16, c = col % 16;
    u32 tile = word(w.p, 8) == INPUT_SIDE
                   ? (row / 128) * (w.out / 2) + 64 * (col / 128) + 8 * (row % 128 / 16) + col % 128 / 16
                   : (col / 128) * (w.in / 2) + 8 * (row / 16) + col % 128 / 16;
    u64 first = m * (u64(w.in) * w.out / 256);
    const u8* bits = w.p + HEAD + (first + tile) * (32 * w.k);
    u32 j = 8 * (4 * (c % 8) + (r % 8) / 2) + 4 * (c >= 8) + 2 * (r >= 8) + r % 2;
    u32 state = 0;
    for (u32 n = 0; n < (16 + w.k - 1) / w.k; n++) {
        u32 at = ((j + 256 - n) % 256) * w.k, code = 0;
        for (u32 b = 0; b < w.k; b++) code = code << 1 | tile_bit(bits, at + b);
        state |= code << (n * w.k);
    }
    state &= 0xFFFFu;
    if (w.book == 1) {
        u32 x = state * MUL1;
        u32 sum = (x & 255u) + (x >> 8 & 255u) + (x >> 16 & 255u) + (x >> 24);
        return narrow(__fadd_rn(__fmul_rn(float(1024 + sum), half(0x1EEE)), half(0xC931)));
    }
    u32 x = ((state * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    return narrow(__fadd_rn(half(x & 0xFFFFu), half(x >> 16)));
}

// Element `(o, i)` of matrix `m`, in any form.
static __device__ float elem(const mat& w, u64 m, u32 o, u32 i) {
    u64 n = (m * w.out + o) * w.in + i;
    switch (w.klass) {
    case K_F32:
        return reinterpret_cast<const float*>(w.p)[n];
    case K_BF16:
        return __uint_as_float(u32(reinterpret_cast<const u16*>(w.p)[n]) << 16);
    case K_INT4: {
        const u8* b = w.p + n / 32 * 18;
        u32 j = n % 32;
        return half(b[0] | u32(b[1]) << 8) * (float((b[2 + j / 2] >> (4 * (j % 2))) & 15u) - 8.0f);
    }
    case K_MXFP8: {
        const u8* b = w.p + n / 32 * 33;
        u32 e = b[0];
        float scale = e == 255 ? __int_as_float(0x7FC00000) : ldexpf(1.0f, int(e) - 127);
        return scale * e4m3(b[1 + n % 32]);
    }
    default:
        return trellis(w, m, i, o);
    }
}

// The scaled Hadamard transform of 128-runs, at element `i`.
static __device__ float hadamard(const float* v, const float* by, u32 i) {
    u32 base = i & ~127u, o = i & 127u;
    float sum = 0.0f;
    for (u32 j = 0; j < 128; j++) {
        float x = v[base + j] * (by ? by[base + j] : 1.0f);
        sum += (__popc(o & j) & 1) ? -x : x;
    }
    return sum * (1.0f / sqrtf(128.0f));
}

// Scratch spans. `ids` and `picked` hold u32.
struct spans_t {
    float *x, *h, *part, *qr, *kv, *a, *b, *c, *d, *t1, *t2, *gates;
    u32 *picked, *ids;
};

static __device__ spans_t lay_out(const table* t, const sizes& s) {
    float* p = at<float>(t, t->items_len - 1);
    u64 r = s.rows, w = r * s.width;
    spans_t q;
    float** wide[6] = {&q.a, &q.b, &q.c, &q.d, &q.t1, &q.t2};
    q.x = p;
    q.h = q.x + r * s.hidden;
    q.part = q.h + r * s.hidden;
    q.qr = q.part + r * s.hidden;
    q.kv = q.qr + r * s.q_latent;
    p = q.kv + r * (s.latent + s.rope);
    for (u32 i = 0; i < 6; i++, p += w) *wide[i] = p;
    q.picked = reinterpret_cast<u32*>(p);
    q.ids = q.picked + r * s.keep;
    q.gates = reinterpret_cast<float*>(q.ids + r * s.topk);
    return q;
}

// Everything one phase reads: the table, sizes and scratch.
struct ctx {
    const table* t;
    const sizes& s;
    const spans_t& p;
    session& se;
};

// Runs a phase over the grid, then a barrier.
#define PHASE(call)                    \
    do {                               \
        call;                          \
        if (!sync(c.t)) return false;  \
    } while (0)

// y = x w^T for `rows` rows of `x`, plain codes.
static __device__ void times(const ctx& c, const mat& w, const float* x, u32 xs, float* y, u32 ys,
                             u32 rows, u32 out) {
    line l = threads();
    u32 chunks = (rows + 7) / 8;
    for (u32 u = l.at; u < out * chunks; u += l.step) {
        u32 o = u % out, r0 = u / out * 8, n = min(8u, rows - r0);
        float acc[8] = {};
        for (u32 i = 0; i < w.in; i++) {
            float v = elem(w, 0, o, i);
            for (u32 r = 0; r < n; r++) acc[r] += x[u64(r0 + r) * xs + i] * v;
        }
        for (u32 r = 0; r < n; r++) y[u64(r0 + r) * ys + o] = acc[r];
    }
}

// Each row's 128-runs turned, scaled by `by` then `after`.
static __device__ void turn_rows(const float* x, u32 xs, const float* by, const float* after,
                                 float* y, u32 ys, u32 rows, u32 n) {
    line l = threads();
    for (u32 u = l.at; u < rows * n; u += l.step) {
        u32 r = u / n, i = u % n;
        float v = hadamard(x + u64(r) * xs, by, i);
        y[u64(r) * ys + i] = after ? v * after[i] : v;
    }
}

// y = x w^T. Rotated codes turn between `t1` and `t2`.
static __device__ bool linear(const ctx& c, const mat& w, const float* x, u32 xs, float* y, u32 ys,
                              u32 rows, u32 out) {
    if (w.klass != K_TURNED) {
        PHASE(times(c, w, x, xs, y, ys, rows, out));
        return true;
    }
    PHASE(turn_rows(x, xs, w.suh, nullptr, c.p.t1, w.in, rows, w.in));
    PHASE(times(c, w, c.p.t1, w.in, c.p.t2, w.out, rows, w.out));
    PHASE(turn_rows(c.p.t2, w.out, nullptr, w.svh, y, ys, rows, w.out));
    return true;
}

// The block sum of `v`, over a power of two.
static __device__ float block_sum(float v) {
    __shared__ float all[1024];
    all[threadIdx.x] = v;
    __syncthreads();
    for (u32 half = blockDim.x / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) all[threadIdx.x] += all[threadIdx.x + half];
        __syncthreads();
    }
    float sum = all[0];
    __syncthreads();
    return sum;
}

// Rows normed by `w`, one block per row, maybe centered.
static __device__ void rms(const ctx& c, const float* x, u32 xs, const mat& w, float* y, u32 ys,
                           u32 rows, u32 n, const u32* from, bool centered) {
    for (u32 r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* row = x + u64(from ? from[r] : r) * xs;
        float mean = 0.0f;
        if (centered) {
            float sum = 0.0f;
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) sum += row[i];
            mean = block_sum(sum) / n;
        }
        float sq = 0.0f;
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) sq += (row[i] - mean) * (row[i] - mean);
        float inv = 1.0f / sqrtf(block_sum(sq) / n + c.s.eps);
        for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
            y[u64(r) * ys + i] = (row[i] - mean) * (inv * elem(w, 0, 0, i));
        }
    }
}

// Kept row `(slot, pos)` via the KV convention.
static __device__ float* kv_row(const ctx& c, u32 index, u32 slot, u32 pos) {
    return reinterpret_cast<float*>(kept_row(c.t, kept_table(c.t, index), slot, pos));
}

// Pair `j` of a row `width` wide, turned to `pos`.
static __device__ void rope(float* v, u32 width, u32 pos, float theta) {
    for (u32 j = 0; j < width / 2; j++) {
        float angle = float(pos) * powf(theta, -float(2 * j) / float(width));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        float a = v[2 * j], b = v[2 * j + 1];
        v[2 * j] = a * cs - b * sn;
        v[2 * j + 1] = a * sn + b * cs;
    }
}

// Adds `part`, summed over ranks if `ranks`, into `x`.
static __device__ bool add(const ctx& c, bool ranks) {
    line l = threads();
    u32 n = c.s.rows * c.s.hidden;
    if (!ranks) {
        for (u32 i = l.at; i < n; i += l.step) c.p.x[i] += c.p.part[i];
        return sync(c.t);
    }
    u32 room = detail::pools(c.t)->pools[0].send_bytes / 4;
    for (u32 done = 0; done < n; done += room) {
        u32 len = min(room, n - done);
        ticket r = prepare(c.se, 0);
        float* sent = send_buffer<float>(r);
        if (sent == nullptr) return false;
        for (u32 i = l.at; i < len; i += l.step) sent[i] = c.p.part[done + i];
        if (!trigger(r, len) || !wait(r)) return false;
        parts got = received(r);
        for (u32 i = l.at; i < len; i += l.step) c.p.x[done + i] += got.sum(i);
    }
    return sync(c.t);
}

static __device__ mat field_mat(const ctx& c, const place& pl, u32 f) {
    return weight(c.t, pl.item[f]);
}

// Index keys: centred, normed, biased, then kept.
static __device__ void index_keys(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    rms(c, c.p.b, s.idim, field_mat(c, pl, K_NORM), c.p.b, s.idim, s.rows, s.idim,
        nullptr, true);
}

// Each row's kept-key scores, causal, then its best `keep`.
static __device__ void index_scores(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    line l = threads();
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    for (u32 u = l.at; u < s.rows * s.context; u += l.step) {
        u32 r = u / s.context, j = u % s.context;
        float score = __int_as_float(0xff800000u);
        if (j <= pos[r]) {
            const float* k = kv_row(c, pl.kept + 2, slot[r], j);
            score = 0.0f;
            for (u32 h = 0; h < s.ih; h++) {
                const float* q = c.p.a + (u64(r) * s.ih + h) * s.idim;
                float hit = 0.0f;
                for (u32 d = 0; d < s.idim; d++) hit += q[d] * k[d];
                score += fmaxf(hit, 0.0f) * c.p.c[u64(r) * s.ih + h];
            }
        }
        c.p.d[u64(r) * s.context + j] = score;
    }
}

// Rank `j` of `v`: best first, ties to lower index.
static __device__ u32 rank(const float* v, u32 n, u32 j) {
    u32 better = 0;
    for (u32 i = 0; i < n; i++) better += v[i] > v[j] || (v[i] == v[j] && i < j);
    return better;
}

// Each row's top `k` of `n`. Spare slots hold `n`.
static __device__ void top(const float* v, u32 rows, u32 n, u32 k, u32* ids, bool fill) {
    line l = threads();
    for (u32 u = l.at; u < rows * (fill ? k : n); u += l.step) {
        if (fill) {
            ids[u] = n;
            continue;
        }
        u32 r = u / n, j = u % n;
        const float* row = v + u64(r) * n;
        if (!(row[j] > __int_as_float(0xff800000u))) continue;
        u32 at = rank(row, n, j);
        if (at < k) ids[u64(r) * k + at] = j;
    }
}

// The indexer: picks each row's best kept keys.
static __device__ bool pick_keys(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    u32 r = s.rows;
    mat wq = field_mat(c, pl, WQ);
    if (!linear(c, wq, c.p.qr, s.q_latent, c.p.a, s.ih * s.idim, r, s.ih * s.idim)) return false;
    mat wk = field_mat(c, pl, WK);
    if (!linear(c, wk, c.p.h, s.hidden, c.p.b, s.idim, r, s.idim)) return false;
    mat by = field_mat(c, pl, I_WEIGHTS);
    if (!linear(c, by, c.p.h, s.hidden, c.p.c, s.ih, r, s.ih)) return false;
    PHASE(index_keys(c, pl));
    line l = threads();
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    mat bias = field_mat(c, pl, K_BIAS);
    for (u32 u = l.at; u < r * s.idim; u += l.step) {
        u32 row = u / s.idim, d = u % s.idim;
        kv_row(c, pl.kept + 2, slot[row], pos[row])[d] = c.p.b[u] + elem(bias, 0, 0, d);
    }
    if (!sync(c.t)) return false;
    PHASE(index_scores(c, pl));
    PHASE(top(c.p.d, r, s.context, s.keep, c.p.picked, true));
    PHASE(top(c.p.d, r, s.context, s.keep, c.p.picked, false));
    return true;
}

// Turns queries and the shared key. Keeps latents and keys.
static __device__ void turn_keys(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    line l = threads();
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    u32 units = s.heads + 1, q = s.nope + s.rope;
    for (u32 u = l.at; u < s.rows * units; u += l.step) {
        u32 r = u / units, h = u % units;
        if (h < s.heads) {
            rope(c.p.a + (u64(r) * s.heads + h) * q + s.nope, s.rope, pos[r], s.theta);
            continue;
        }
        float* kv = c.p.kv + u64(r) * (s.latent + s.rope);
        rope(kv + s.latent, s.rope, pos[r], s.theta);
        float* latent = kv_row(c, pl.kept, slot[r], pos[r]);
        float* keys = kv_row(c, pl.kept + 1, slot[r], pos[r]);
        for (u32 d = 0; d < s.latent; d++) latent[d] = kv[d];
        for (u32 d = 0; d < s.rope; d++) keys[d] = kv[s.latent + d];
    }
}

// Queries in latent space: `kv_b`'s key rows folded in.
static __device__ void absorb(const ctx& c, const mat& kv_b) {
    const sizes& s = c.s;
    line l = threads();
    u32 width = s.latent + s.rope, q = s.nope + s.rope;
    for (u32 u = l.at; u < s.rows * s.heads * width; u += l.step) {
        u32 rh = u / width, i = u % width, h = rh % s.heads;
        const float* nope = c.p.a + u64(rh) * q;
        float v = 0.0f;
        if (i >= s.latent) {
            v = nope[s.nope + i - s.latent];
        } else {
            for (u32 n = 0; n < s.nope; n++) v += nope[n] * elem(kv_b, 0, h * (s.nope + s.hd) + n, i);
        }
        c.p.b[u] = v;
    }
}

// Each row's heads over its picked kept rows, causal.
static __device__ void attend(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    line l = threads();
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    u32 width = s.latent + s.rope;
    for (u32 u = l.at; u < s.rows * s.heads; u += l.step) {
        u32 r = u / s.heads;
        const float* q = c.p.b + u64(u) * width;
        float* o = c.p.c + u64(u) * s.latent;
        float most = __int_as_float(0xff800000u), total = 0.0f;
        for (u32 d = 0; d < s.latent; d++) o[d] = 0.0f;
        for (u32 n = 0; n < s.keep; n++) {
            u32 j = c.p.picked[u64(r) * s.keep + n];
            if (j >= s.context || j > pos[r]) continue;
            const float* kl = kv_row(c, pl.kept, slot[r], j);
            const float* kr = kv_row(c, pl.kept + 1, slot[r], j);
            float score = 0.0f;
            for (u32 d = 0; d < s.latent; d++) score += q[d] * kl[d];
            for (u32 d = 0; d < s.rope; d++) score += q[s.latent + d] * kr[d];
            score *= s.scale;
            float topmost = fmaxf(most, score);
            float shrink = expf(most - topmost), e = expf(score - topmost);
            total = total * shrink + e;
            for (u32 d = 0; d < s.latent; d++) o[d] = o[d] * shrink + e * kl[d];
            most = topmost;
        }
        for (u32 d = 0; d < s.latent; d++) o[d] /= total;
    }
}

// Head outputs: `kv_b`'s value rows over latent outputs.
static __device__ void head_values(const ctx& c, const mat& kv_b) {
    const sizes& s = c.s;
    line l = threads();
    for (u32 u = l.at; u < s.rows * s.heads * s.hd; u += l.step) {
        u32 rh = u / s.hd, d = u % s.hd, h = rh % s.heads;
        const float* o = c.p.c + u64(rh) * s.latent;
        float v = 0.0f;
        for (u32 i = 0; i < s.latent; i++) v += o[i] * elem(kv_b, 0, h * (s.nope + s.hd) + s.nope + d, i);
        c.p.d[u] = v;
    }
}

static __device__ bool attention(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    u32 r = s.rows, kv = s.latent + s.rope, q = s.heads * (s.nope + s.rope);
    PHASE(rms(c, c.p.x, s.hidden, field_mat(c, pl, A_NORM), c.p.h, s.hidden, r,
              s.hidden, nullptr, false));
    mat q_a = field_mat(c, pl, Q_A);
    if (!linear(c, q_a, c.p.h, s.hidden, c.p.qr, s.q_latent, r, s.q_latent)) return false;
    mat kv_a = field_mat(c, pl, KV_A);
    if (!linear(c, kv_a, c.p.h, s.hidden, c.p.kv, kv, r, kv)) return false;
    PHASE(rms(c, c.p.qr, s.q_latent, field_mat(c, pl, Q_NORM), c.p.qr, s.q_latent,
              r, s.q_latent, nullptr, false));
    PHASE(rms(c, c.p.kv, kv, field_mat(c, pl, KV_NORM), c.p.kv, kv, r, s.latent,
              nullptr, false));
    if (indexes(s, pl.layer) && !pick_keys(c, pl)) return false;
    mat q_b = field_mat(c, pl, Q_B);
    if (!linear(c, q_b, c.p.qr, s.q_latent, c.p.a, q, r, q)) return false;
    PHASE(turn_keys(c, pl));
    mat kv_b = field_mat(c, pl, KV_B);
    PHASE(absorb(c, kv_b));
    PHASE(attend(c, pl));
    PHASE(head_values(c, kv_b));
    mat o = field_mat(c, pl, O);
    if (!linear(c, o, c.p.d, s.heads * s.hd, c.p.part, s.hidden, r, s.hidden)) return false;
    return add(c, bit(s.attn_split, pl.layer));
}

// SiLU of `a` times `b`, into `y`.
static __device__ void gated(const float* a, const float* b, float* y, u32 n) {
    line l = threads();
    for (u32 i = l.at; i < n; i += l.step) y[i] = a[i] / (1.0f + expf(-a[i])) * b[i];
}

// `gate`, `up` and `down` fields as one SwiGLU into `part`.
static __device__ bool swiglu(const ctx& c, const place& pl, u32 gate, u32 inner) {
    const sizes& s = c.s;
    u32 r = s.rows;
    mat g = field_mat(c, pl, gate);
    mat u = field_mat(c, pl, gate + 1);
    mat d = field_mat(c, pl, gate + 2);
    if (!linear(c, g, c.p.h, s.hidden, c.p.a, inner, r, inner)) return false;
    if (!linear(c, u, c.p.h, s.hidden, c.p.b, inner, r, inner)) return false;
    PHASE(gated(c.p.a, c.p.b, c.p.c, r * inner));
    return linear(c, d, c.p.c, inner, c.p.part, s.hidden, r, s.hidden);
}

// Router scores, then each row's picks and gates.
static __device__ bool route(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    line l = threads();
    u32 r = s.rows, e = s.experts;
    mat router = field_mat(c, pl, ROUTER);
    if (!linear(c, router, c.p.h, s.hidden, c.p.a, e, r, e)) return false;
    mat bias = field_mat(c, pl, R_BIAS);
    for (u32 u = l.at; u < r * e; u += l.step) {
        float v = 1.0f / (1.0f + expf(-c.p.a[u]));
        c.p.a[u] = v;
        c.p.b[u] = v + elem(bias, 0, 0, u % e);
    }
    if (!sync(c.t)) return false;
    PHASE(top(c.p.b, r, e, s.topk, c.p.ids, false));
    for (u32 row = l.at; row < r; row += l.step) {
        const u32* ids = c.p.ids + u64(row) * s.topk;
        float sum = 0.0f;
        for (u32 k = 0; k < s.topk; k++) sum += c.p.a[u64(row) * e + ids[k]];
        for (u32 k = 0; k < s.topk; k++) {
            c.p.gates[u64(row) * s.topk + k] = c.p.a[u64(row) * e + ids[k]] / sum * s.route;
        }
    }
    return sync(c.t);
}

// Whether pick `rk`'s expert is local, and its number.
static __device__ bool local_pick(const ctx& c, u32 rk, u32* m) {
    u32 e = c.p.ids[rk];
    *m = e - c.s.held_first;
    return e >= c.s.held_first && e < c.s.held_first + c.s.held;
}

// `y[rk] = codes[e] x[rk]` for each held pick.
static __device__ void picked(const ctx& c, const mat& w, const float* x, float* y, u32 out,
                              u32 in) {
    line l = threads();
    for (u32 u = l.at; u < c.s.rows * c.s.topk * out; u += l.step) {
        u32 rk = u / out, o = u % out, m;
        if (!local_pick(c, rk, &m)) continue;
        float v = 0.0f;
        for (u32 i = 0; i < in; i++) v += elem(w, m, o, i) * x[u64(rk) * in + i];
        y[u] = v;
    }
}

// Each held pick's rows turned, with per-expert scales.
static __device__ void turn_picks(const ctx& c, const float* x, const float* sign, const mat& by,
                                  const mat& after, const float* also, float* y, u32 n) {
    line l = threads();
    for (u32 u = l.at; u < c.s.rows * c.s.topk * n; u += l.step) {
        u32 rk = u / n, i = u % n, m;
        if (!local_pick(c, rk, &m)) continue;
        u32 e = m + c.s.held_first, base = i & ~127u, o = i & 127u;
        const float* row = x + u64(sign ? rk / c.s.topk : rk) * n;
        float sum = 0.0f;
        for (u32 j = 0; j < 128; j++) {
            float v = row[base + j] * (by.p ? elem(by, 0, e, base + j) : 1.0f);
            if (sign) v *= sign[base + j];
            sum += (__popc(o & j) & 1) ? -v : v;
        }
        float v = sum * (1.0f / sqrtf(128.0f));
        if (after.p) v *= elem(after, 0, e, i);
        y[u] = also ? v * also[i] : v;
    }
}

static __device__ bool routed(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    line l = threads();
    u32 h = s.hidden, f = s.effn, e = s.experts, rk = s.rows * s.topk;
    mat none = {nullptr, K_F32, 0, 0, 0, 0, nullptr, nullptr};
    mat signs_in = field_mat(c, pl, SIGNS_IN);
    mat signs_out = field_mat(c, pl, SIGNS_OUT);
    const float* in_signs = reinterpret_cast<const float*>(signs_in.p);
    const float* out_signs = reinterpret_cast<const float*>(signs_out.p);
    mat su[3], sv[3], codes[3];
    for (u32 i = 0; i < 3; i++) {
        u32 base = E_GATE + 3 * i;
        codes[i] = field_mat(c, pl, base);
        su[i] = field_mat(c, pl, base + 1);
        sv[i] = field_mat(c, pl, base + 2);
    }
    PHASE(turn_picks(c, c.p.h, in_signs, su[0], none, nullptr, c.p.b, h);
          turn_picks(c, c.p.h, in_signs, su[1], none, nullptr, c.p.c, h));
    PHASE(picked(c, codes[0], c.p.b, c.p.d, f, h); picked(c, codes[1], c.p.c, c.p.t1, f, h));
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
    PHASE(picked(c, codes[2], c.p.b, c.p.d, h, f));
    PHASE(turn_picks(c, c.p.d, nullptr, none, sv[2], out_signs, c.p.c, h));
    for (u32 u = l.at; u < s.rows * h; u += l.step) {
        u32 r = u / h, o = u % h, m;
        float sum = 0.0f;
        for (u32 k = 0; k < s.topk; k++) {
            u32 pick = r * s.topk + k;
            if (local_pick(c, pick, &m)) sum += c.p.gates[pick] * c.p.c[u64(pick) * h + o];
        }
        c.p.part[u] = sum + c.p.part[u];
    }
    return sync(c.t);
}

// Layer `l`'s sizes: split parts, or whole widths.
static __device__ sizes for_layer(const sizes& s, u32 l) {
    sizes ls = s;
    if (!bit(s.attn_split, l)) ls.heads = s.heads_all;
    if (!bit(s.mlp_split, l)) {
        ls.dense_ffn = s.dense_all;
        ls.ffn = s.ffn_all;
        ls.held_first = 0;
        ls.held = s.experts;
    }
    return ls;
}

static __device__ bool layer(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    bool split = bit(s.mlp_split, pl.layer);
    if (!attention(c, pl)) return false;
    PHASE(rms(c, c.p.x, s.hidden, field_mat(c, pl, MLP_NORM), c.p.h, s.hidden,
              s.rows, s.hidden, nullptr, false));
    if (pl.layer < s.dense) return swiglu(c, pl, D_GATE, s.dense_ffn) && add(c, split);
    if (!route(c, pl) || !swiglu(c, pl, S_GATE, s.ffn) || !routed(c, pl)) return false;
    return add(c, split);
}

// Values: see `sizes`. Items: see the defines, then out.
extern "C" __global__ void glm53_whole(const table* t) {
    if (!check(t)) return;
    sizes s = read(t);
    spans_t p = lay_out(t, s);
    session se = begin(t);
    ctx c = {t, s, p, se};
    u32 norm = IDS + (is_turned(t, IDS) ? 3 : 1);
    u32 head = norm + (is_turned(t, norm) ? 3 : 1);
    u32 first = head + (is_turned(t, head) ? 3 : 1);
    u32 reads = first, kept = 0;
    for (u32 l = 0; l < s.layers; l++) {
        reads = walk(t, for_layer(s, l), l, reads, kept).next;
        kept += indexes(s, l) ? 3 : 2;
    }
    const u32* tokens = at<const u32>(t, 0);
    mat embed = weight(t, IDS);
    line l = threads();
    for (u32 i = l.at; i < s.rows * s.hidden; i += l.step) {
        p.x[i] = elem(embed, 0, tokens[i / s.hidden], i % s.hidden);
    }
    if (!sync(t)) return;
    for (u32 n = 0, from = first, rows = 0; n < s.layers; n++) {
        sizes ls = for_layer(s, n);
        ctx lc = {t, ls, p, se};
        place pl = walk(t, ls, n, from, rows);
        if (!layer(lc, pl)) return;
        from = pl.next;
        rows += indexes(s, n) ? 3 : 2;
    }
    rms(c, p.x, s.hidden, weight(t, norm), p.h, s.hidden, s.picks, s.hidden,
        at<const u32>(t, 3), false);
    if (!sync(t)) return;
    times(c, weight(t, head), p.h, s.hidden, at<float>(t, reads + kept), s.vocab,
          s.picks, s.vocab);
    end(se);
}
