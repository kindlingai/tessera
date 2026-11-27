// Shared types, metadata readers, kept access, and row WHT.

using namespace tessera;
using trellis::word;

#define IDS 4u
#ifndef WARPS
#define WARPS 8u
#endif
#ifndef ROW_TILE
#define ROW_TILE 4u
#endif
#ifndef ATTN_STAGE
#define ATTN_STAGE 2048u
#endif

#ifndef TS_MARK
#define TS_MARK() ((void)0)
#endif

// Field numbers, in module order.
enum field : u32 {
    EMBED, NORM, LM_HEAD,
    A_NORM, Q_A, Q_NORM, Q_B, KV_A, KV_NORM, KV_B, O, MLP_NORM,
    WQ, WK, K_NORM, K_BIAS, I_WEIGHTS,
    D_GATE, D_UP, D_DOWN,
    ROUTER, R_BIAS, SIGNS_IN, SIGNS_OUT, E_GATE, GATE_SU, GATE_SV, E_UP, UP_SU, UP_SV,
    E_DOWN, DOWN_SU, DOWN_SV, S_GATE, S_UP, S_DOWN,
    FIELD_END
};

// Weight form classes from item and form metadata.
enum klass : u32 { K_F32, K_BF16, K_INT4, K_MXFP8, K_TRELLIS, K_TURNED };

// Launch scalars, in module order.
struct sizes {
    u32 rows, picks, vocab, hidden, layers, dense, every;
    u32 heads, heads_all, nope, rope, hd, q_latent, latent;
    u32 ih, idim, keep, context, dense_ffn, dense_all, ffn, ffn_all, effn;
    u32 experts, held_first, held, topk;
    u32 attn_split[3], mlp_split[3];
    u32 width;
    float eps, theta, route, scale;
};

struct place {
    u32 item[FIELD_END];
    u32 next, kept, layer;
};

struct mat {
    const u8* p;
    u32 klass, in, out, k, book;
    const float* suh;
    const float* svh;
};

struct spans_t {
    float *x, *h, *part, *qr, *kv, *a, *b, *c, *d, *t1, *t2, *gates;
    u32 *picked, *ids;
};

// Unified block shared memory, under 32 KiB.
struct alignas(16) Smem {
    float red[WARPS];
    union {
        struct {
            u32 tiles[WARPS][256];
            float buf[WARPS][ROW_TILE][128];
        } lin;
        struct {
            u32 tiles[WARPS][256];
            float buf[WARPS][ROW_TILE][128];
            float vec[ROW_TILE][128];
            u32 rows[ROW_TILE];
            u32 picks[ROW_TILE];
            u32 mask[64];
            u32 chosen;
            u32 count;
        } moe;
        struct {
            float q[ATTN_STAGE];
        } attn;
        struct {
            u32 cnt[WARPS];
            u32 thresh;
            u32 gt;
            u32 valid;
        } sel;
    };
};

struct ctx {
    const table* t;
    const sizes& s;
    const spans_t& p;
    session& se;
    Smem& sm;
};

#define PHASE(call)                    \
    do {                               \
        call;                          \
        if (!sync(c.t)) return false;  \
    } while (0)

static __device__ __forceinline__ bool bit(const u32* words, u32 l) {
    return (words[l >> 5] >> (l & 31u)) & 1u;
}

static __device__ __forceinline__ bool indexes(const sizes& s, u32 l) {
    return l < s.dense || (l + 1 - s.dense) % s.every == 0;
}

// Kept tables of layer `l`: keys, then indexer keys.
static __device__ __forceinline__ u32 kept_count(const sizes& s, u32 l) {
    return indexes(s, l) ? 2u : 1u;
}

static __device__ __forceinline__ bool is_turned(const table* t, u32 item) {
    const tessera::item* it = item_at(t, item);
    return it->elem == FORM && form_of(t, item, 0)->suh.arena != NONE;
}

static __device__ void lay_fields(const table* t, place& pl, u32 from, u32 to, u32* at) {
    for (u32 f = from; f < to; f++) {
        pl.item[f] = *at;
        *at += is_turned(t, *at) ? 3u : 1u;
    }
}

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

static __device__ __forceinline__ mat weight(const table* t, u32 item) {
    const tessera::item* it = item_at(t, item);
    u32 out = it->dims >= 2 ? it->shape[it->dims - 2] : 1u;
    u32 in = it->dims >= 1 ? it->shape[it->dims - 1] : 0u;
    mat w = {at<const u8>(t, item), K_F32, in, out, 0u, 0u, nullptr, nullptr};
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
    trellis::from_form_kind(fm->kind, &w.k, &w.book);
    if (turned) {
        w.suh = get<const float>(t, fm->suh);
        w.svh = get<const float>(t, fm->svh);
    }
    return w;
}

static __device__ __forceinline__ mat field_mat(const ctx& c, const place& pl, u32 f) {
    return weight(c.t, pl.item[f]);
}

static __device__ spans_t lay_out(const table* t, const sizes& s) {
    float* p = scratch<float>(t);
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

static __device__ __forceinline__ float mxfp8_scale(u32 e) {
    if (e == 0u) return __int_as_float(0x00400000);
    if (e == 255u) return __int_as_float(0x7FC00000);
    return __uint_as_float(e << 23);
}

// Native F32, BF16, F16, and E4M3 kept row access.
static __device__ __forceinline__ float kv_load(const u8* row, u32 elem_type, u32 d) {
    if (elem_type == F32) return reinterpret_cast<const float*>(row)[d];
    if (elem_type == BF16) return __uint_as_float(u32(reinterpret_cast<const u16*>(row)[d]) << 16);
    if (elem_type == F16) return trellis::f16_to_f32(reinterpret_cast<const u16*>(row)[d]);
    return trellis::e4m3_to_f32(row[d]);
}

static __device__ __forceinline__ void kv_load4(const u8* row, u32 elem_type, u32 d, float v[4]) {
    if (elem_type == F32 && (reinterpret_cast<u64>(row + (u64(d) << 2)) & 15u) == 0u) {
        float4 f = *reinterpret_cast<const float4*>(row + (u64(d) << 2));
        v[0] = f.x;
        v[1] = f.y;
        v[2] = f.z;
        v[3] = f.w;
        return;
    }
    if (elem_type == BF16 && (reinterpret_cast<u64>(row + (u64(d) << 1)) & 7u) == 0u) {
        uint2 h = *reinterpret_cast<const uint2*>(row + (u64(d) << 1));
        v[0] = __uint_as_float(h.x << 16);
        v[1] = __uint_as_float(h.x & 0xFFFF0000u);
        v[2] = __uint_as_float(h.y << 16);
        v[3] = __uint_as_float(h.y & 0xFFFF0000u);
        return;
    }
    if (elem_type == F16 && (reinterpret_cast<u64>(row + (u64(d) << 1)) & 7u) == 0u) {
        uint2 h = *reinterpret_cast<const uint2*>(row + (u64(d) << 1));
        v[0] = trellis::f16_to_f32(h.x & 0xFFFFu);
        v[1] = trellis::f16_to_f32(h.x >> 16);
        v[2] = trellis::f16_to_f32(h.y & 0xFFFFu);
        v[3] = trellis::f16_to_f32(h.y >> 16);
        return;
    }
    if (elem_type == E4M3 && (reinterpret_cast<u64>(row + d) & 3u) == 0u) {
        u32 b = *reinterpret_cast<const u32*>(row + d);
        float2 p0 = trellis::e4m3x2_to_f32x2(b & 0xFFFFu);
        float2 p1 = trellis::e4m3x2_to_f32x2(b >> 16);
        v[0] = p0.x;
        v[1] = p0.y;
        v[2] = p1.x;
        v[3] = p1.y;
        return;
    }
#pragma unroll
    for (u32 k = 0; k < 4; k++) v[k] = kv_load(row, elem_type, d + k);
}

static __device__ __forceinline__ void kv_store(u8* row, u32 elem_type, u32 d, float v) {
    if (elem_type == F32) {
        reinterpret_cast<float*>(row)[d] = v;
    } else if (elem_type == BF16) {
        reinterpret_cast<u16*>(row)[d] = trellis::f32_to_bf16(v);
    } else if (elem_type == F16) {
        reinterpret_cast<u16*>(row)[d] =
            static_cast<u16>(trellis::f32x2_to_f16x2(v, 0.0f) & 0xFFFFu);
    } else {
        row[d] = trellis::f32_to_e4m3(v);
    }
}

static __device__ float elem(const mat& w, u64 m, u32 o, u32 i) {
    u64 n = (m * w.out + o) * w.in + i;
    switch (w.klass) {
    case K_F32:
        return reinterpret_cast<const float*>(w.p)[n];
    case K_BF16:
        return __uint_as_float(u32(reinterpret_cast<const u16*>(w.p)[n]) << 16);
    case K_INT4: {
        const u8* b = w.p + (n >> 5) * 18u;
        u32 j = static_cast<u32>(n & 31u);
        return trellis::f16_to_f32(b[0] | (u32(b[1]) << 8))
               * (float((b[2 + (j >> 1)] >> (4u * (j & 1u))) & 15u) - 8.0f);
    }
    case K_MXFP8: {
        const u8* b = w.p + (n >> 5) * 33u;
        return mxfp8_scale(b[0]) * trellis::e4m3_to_f32(b[1 + (n & 31u)]);
    }
    default:
        return trellis::scalar_elem(w.p, w.in, w.out, w.k, w.book, m, i, o);
    }
}

static __device__ __forceinline__ float vec_elem(const mat& w, u32 o, u32 i) {
    u64 n = u64(o) * w.in + i;
    if (w.klass == K_F32) return reinterpret_cast<const float*>(w.p)[n];
    if (w.klass == K_BF16) return __uint_as_float(u32(reinterpret_cast<const u16*>(w.p)[n]) << 16);
    return elem(w, 0, o, i);
}

static __device__ __forceinline__ float block_sum(const ctx& c, float v) {
    u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0u) c.sm.red[warp] = v;
    __syncthreads();
    u32 nw = blockDim.x >> 5;
    float sum = 0.0f;
    for (u32 w = 0; w < nw; w++) sum += c.sm.red[w];
    __syncthreads();
    return sum;
}

static __device__ __forceinline__ void load_vec4(const float* p, bool aligned, float v[4]) {
    if (aligned) {
        float4 f = *reinterpret_cast<const float4*>(p);
        v[0] = f.x;
        v[1] = f.y;
        v[2] = f.z;
        v[3] = f.w;
        return;
    }
#pragma unroll
    for (u32 k = 0; k < 4; k++) v[k] = p[k];
}

static __device__ __forceinline__ void store_vec4(float* p, bool aligned, const float v[4]) {
    if (aligned) {
        *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
        return;
    }
#pragma unroll
    for (u32 k = 0; k < 4; k++) p[k] = v[k];
}

// Warp-parallel 128-point WHT over rows, with optional scales.
static __device__ void turn_rows(const float* x, u32 xs, const float* by, const float* after,
                                 float* y, u32 ys, u32 rows, u32 n) {
    u32 blocks = n >> 7;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    bool xv = (reinterpret_cast<u64>(x) & 15u) == 0u && (xs & 3u) == 0u;
    bool yv = (reinterpret_cast<u64>(y) & 15u) == 0u && (ys & 3u) == 0u;
    bool bv = by != nullptr && (reinterpret_cast<u64>(by) & 15u) == 0u;
    bool av = after != nullptr && (reinterpret_cast<u64>(after) & 15u) == 0u;
    for (u32 u = wid; u < rows * blocks; u += wstep) {
        u32 r = u / blocks, blk = u % blocks;
        u32 base = (blk << 7) + (lane << 2);
        float v[4], sc[4];
        load_vec4(x + u64(r) * xs + base, xv, v);
        if (by != nullptr) {
            load_vec4(by + base, bv, sc);
            for (u32 k = 0; k < 4; k++) v[k] *= sc[k];
        }
        trellis::wht128(v, trellis::INV_SQRT128);
        if (after != nullptr) {
            load_vec4(after + base, av, sc);
            for (u32 k = 0; k < 4; k++) v[k] *= sc[k];
        }
        store_vec4(y + u64(r) * ys + base, yv, v);
    }
}
