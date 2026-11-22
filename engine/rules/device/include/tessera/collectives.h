// Collectives and fetches inside one kernel, in three phases.

// prepare wakes the proxy, trigger sends, wait receives.

// Rule: every thread runs the same prepares, in order.

// Rule: trigger every prepare. Trigger and wait in ordinal order.

// Rule: two-way pools read o before triggering o + depth/2.

// Rule: fetched rows last until this rank triggers o + depth.

// Rule: a one-way sender sends at most depth before done.

// Rule: every block calls send_buffer and wait, block 0 too.

#ifndef TESSERA_CAP_COLLECTIVES
#error "import a collective, e.g. <tessera/1/all_reduce>"
#endif

#ifndef TESSERA_COLLECTIVES_H
#define TESSERA_COLLECTIVES_H

namespace tessera {

enum op : u32 {
    SUM = 0,
    GATHER = 1,
    ALL_TO_ALL = 2,
    SEND = 3,
    RECV = 4,
    FETCH = 5,
};

// In the layout the NIC or proxy writes.
struct alignas(16) flag {
    u64 ordinal;
    u64 count;
};

struct lane {
    u32 what;
    u32 elem;
    u32 group_size;
    u32 group_rank;
    u32 block;
    u32 pool;
    u32 unused[2];
};

struct pool {
    u32 depth;
    u32 one_way;
    u32 send_bytes;
    u32 recv_bytes;
    u32 unused[4];
    ref send[MAX_DEPTH];
    ref recv[MAX_DEPTH * MAX_GROUP];
    ref arrived;
    ref credit;
    ref sizes;
};

// Device memory. Block 0 copies host state here for others.
struct counters {
    u64 next[MAX_POOLS];
    u64 writes;
    u64 arrived[MAX_POOLS][MAX_DEPTH];
    u64 free[MAX_POOLS];
};

struct pool_set {
    u32 count;
    u32 unused;
    ref state;
    pool pools[MAX_POOLS];
};

namespace detail {

TESSERA_FN u64* page(const table* t) {
    return get<u64>(t, t->doorbell);
}

// The triggered word, then the done word.
TESSERA_FN u64* slot_words(const table* t, u32 pool, u32 slot) {
    return page(t) + DOORBELL_SLOTS / 8 + (pool * MAX_DEPTH + slot) * 2;
}

TESSERA_FN const pool_set* pools(const table* t) {
    return get<const pool_set>(t, t->pools);
}

TESSERA_FN counters* shared(const table* t) {
    return get<counters>(t, pools(t)->state);
}

}  // namespace detail

struct session {
    const table* t;
    u64 next[MAX_POOLS];
    u64 writes;

    // Every thread calls this. Block 0's leader alone stores.
    TESSERA_FN void bump() {
        if (prim::leader()) prim::store_release_sys(detail::page(t), writes);
        writes++;
    }
};

// Every thread calls it once, at kernel start.
TESSERA_FN session begin(const table* t) {
    const counters* c = detail::shared(t);
    session s;
    s.t = t;
    for (u32 p = 0; p < MAX_POOLS; p++) s.next[p] = c->next[p];
    s.writes = c->writes;
    return s;
}

// Every thread calls it once, before a normal return.
TESSERA_FN void end(const session& s) {
    if (!prim::leader()) return;
    counters* c = detail::shared(s.t);
    for (u32 p = 0; p < MAX_POOLS; p++) c->next[p] = s.next[p];
    c->writes = s.writes;
}

struct ticket {
    session* s;
    const lane* l;
    const pool* p;
    u32 index;
    u32 slot;
    u64 ordinal;
    u32 elements;
};

// Takes the pool's next ordinal and wakes the proxy.
TESSERA_FN ticket prepare(session& s, u32 index) {
    const table* t = s.t;
    const lane* l = get<const lane>(t, t->lanes) + index;
    const pool* p = &detail::pools(t)->pools[l->pool];
    u64 ordinal = s.next[l->pool]++;
    u32 slot = u32(ordinal % p->depth);
    s.bump();
    return {&s, l, p, index, slot, ordinal, 0};
}

namespace detail {

// Block 0 waits until the proxy sent this slot.
TESSERA_FN bool slot_free(const ticket& r, u64 need) {
    const table* t = r.s->t;
    const u64* sent = page(t) + DOORBELL_SENT / 8 + r.l->pool;
    const u64* credit = get<const u64>(t, r.p->credit);
    u32 peer = 1 - r.l->group_rank;
    for (u32 spin = 1;; spin++) {
        bool free = prim::load_acquire_sys(sent) >= need;
        if (free && r.p->one_way) free = prim::load_acquire_sys(&credit[peer]) >= need;
        if (free) return true;
        if (failed(t, spin, r.l->pool, r.ordinal)) return false;
        prim::pause();
    }
}

// Other blocks wait on block 0's copy in device memory.
TESSERA_FN bool mirror(const ticket& r, const u64* word, u64 want, bool exact) {
    for (u32 spin = 1;; spin++) {
        u64 got = prim::load_acquire_gpu64(word);
        if (exact ? got == want : got >= want) return true;
        if (failed(r.s->t, spin, r.l->pool, r.ordinal)) return false;
        prim::pause();
    }
}

}  // namespace detail

// This rank's part, or null if the kernel must stop.
template <typename T>
TESSERA_FN T* send_buffer(const ticket& r) {
    const table* t = r.s->t;
    __shared__ int ok;
    if (prim::leader_thread()) {
        ok = 1;
        if (r.ordinal > r.p->depth) {
            u64 need = r.ordinal - r.p->depth;
            u64* free = &detail::shared(t)->free[r.l->pool];
            if (!prim::leader_block()) {
                ok = detail::mirror(r, free, need, false);
            } else if (detail::slot_free(r, need)) {
                prim::store_release_gpu64(free, need);
            } else {
                ok = 0;
            }
        }
    }
    prim::block_sync();
    return ok ? get<T>(t, r.p->send[r.slot]) : nullptr;
}

// All-to-all only: elements for group rank k.
TESSERA_FN void send_size(const ticket& r, u32 k, u32 elements) {
    get<u32>(r.s->t, r.p->sizes)[r.slot * MAX_GROUP + k] = elements;
}

// Every thread of the grid calls it after its writes.
TESSERA_FN bool trigger(ticket& r, u32 elements, bool wrote = true) {
    const table* t = r.s->t;
    r.elements = elements;
    if (wrote) prim::fence_sys();
    if (!grid_sync(t)) return false;
    if (prim::leader()) {
        u64 word = 1ull << 63 | u64(r.index) << 48 | (r.ordinal & 0xFFFFull) << 32 | elements;
        prim::store_relaxed_sys(detail::slot_words(t, r.l->pool, r.slot), word);
    }
    r.s->bump();
    return true;
}

TESSERA_FN const flag* flags(const ticket& r) {
    return get<const flag>(r.s->t, r.p->arrived) + r.slot * MAX_GROUP;
}

// Whether every part this rank needs has this ordinal.
TESSERA_FN bool ready(const ticket& r) {
    if (r.l->what == SEND) return true;
    const flag* f = flags(r);
    if (r.l->what == FETCH) return prim::load_acquire_sys(&f[0].ordinal) == r.ordinal;
    for (u32 k = 0; k < r.l->group_size; k++) {
        if (k == r.l->group_rank) continue;
        if (prim::load_acquire_sys(&f[k].ordinal) != r.ordinal) return false;
    }
    return true;
}

// Waits in this block until ready. False means stop.
TESSERA_FN bool wait(const ticket& r, bool read = true) {
    __shared__ int ok;
    const table* t = r.s->t;
    u64* seen = &detail::shared(t)->arrived[r.l->pool][r.slot];
    if (prim::leader_thread()) {
        ok = 1;
        bool none = r.l->what == SEND;
        if (none) {
            // A send waits for nothing.
        } else if (prim::leader_block()) {
            for (u32 spin = 1; !ready(r); spin++) {
                if (failed(t, spin, r.l->pool, r.ordinal)) {
                    ok = 0;
                    break;
                }
                prim::pause();
            }
            if (ok) prim::store_release_gpu64(seen, r.ordinal);
        } else if (!detail::mirror(r, seen, r.ordinal, true)) {
            ok = 0;
        } else if (read && !ready(r)) {
            // One system-scope read orders this block's data reads.
            fail(t, PROTOCOL, r.l->pool, r.ordinal);
            ok = 0;
        }
    }
    prim::block_sync();
    if (read) prim::fence_sys();
    return ok != 0;
}

// One-way receivers: every thread calls it after its reads.
TESSERA_FN bool done(ticket& r) {
    const table* t = r.s->t;
    if (!grid_sync(t)) return false;
    if (prim::leader()) {
        prim::store_relaxed_sys(detail::slot_words(t, r.l->pool, r.slot) + 1, r.ordinal);
    }
    r.s->bump();
    return true;
}

// Copies `bytes`, a multiple of 4, past every cache.
TESSERA_FN void copy_fresh(void* to, const void* from, u32 bytes) {
    u32* d = static_cast<u32*>(to);
    const u32* s = static_cast<const u32*>(from);
    for (u32 i = 0; i < bytes / 4; i++) d[i] = prim::load_fresh(s + i);
}

// The parts of a finished collective.
struct parts {
    const void* part[MAX_GROUP];
    u32 size[MAX_GROUP];
    u32 count;
    u32 elem;

    // Element `i` of the sum, in rank order.
    TESSERA_FN float sum(u32 i) const {
        float s = 0.0f;
        for (u32 k = 0; k < count; k++) s += load(k, i);
        return s;
    }

    // Four aligned elements of the sum, in rank order.
    TESSERA_FN float4 sum4(u32 i) const {
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (elem == BF16) {
            for (u32 k = 0; k < count; k++) {
                uint2 h = prim::load_fresh4(static_cast<const u16*>(part[k]) + i);
                s.x += prim::from_bits(h.x << 16);
                s.y += prim::from_bits(h.x & 0xFFFF0000u);
                s.z += prim::from_bits(h.y << 16);
                s.w += prim::from_bits(h.y & 0xFFFF0000u);
            }
            return s;
        }
        for (u32 k = 0; k < count; k++) {
            float4 v = prim::load_fresh4(static_cast<const float*>(part[k]) + i);
            s.x += v.x;
            s.y += v.y;
            s.z += v.z;
            s.w += v.w;
        }
        return s;
    }

    TESSERA_FN float load(u32 k, u32 i) const {
        if (elem == BF16) {
            u16 h = prim::load_fresh(static_cast<const u16*>(part[k]) + i);
            return prim::from_bits(u32(h) << 16);
        }
        return prim::load_fresh(static_cast<const float*>(part[k]) + i);
    }
};

// After wait. Send has no parts, receive and fetch one.
TESSERA_FN parts received(const ticket& r) {
    const table* t = r.s->t;
    const lane* l = r.l;
    const flag* f = flags(r);
    parts p;
    p.elem = l->elem;
    p.count = 0;
    if (l->what == SEND) return p;
    if (l->what == FETCH) {
        p.part[0] = get<const void>(t, r.p->recv[r.slot * MAX_GROUP]);
        p.size[0] = u32(prim::load_acquire_sys(&f[0].count));
        p.count = 1;
        return p;
    }
    u32 width = l->elem == BF16 ? 2 : 4;
    for (u32 k = 0; k < l->group_size; k++) {
        if (k != l->group_rank) {
            p.part[p.count] = get<const void>(t, r.p->recv[r.slot * MAX_GROUP + k]);
            p.size[p.count] = l->what == SUM ? r.elements : u32(prim::load_acquire_sys(&f[k].count));
            p.count++;
            continue;
        }
        if (l->what == RECV) continue;
        // This rank's own part stays in its send buffer.
        const u8* own = get<const u8>(t, r.p->send[r.slot]);
        bool split = l->what == ALL_TO_ALL;
        p.part[p.count] = own + (split ? l->block * k * width : 0);
        const u32* sizes = get<const u32>(t, r.p->sizes);
        p.size[p.count] = split ? sizes[r.slot * MAX_GROUP + k] : r.elements;
        p.count++;
    }
    return p;
}

}  // namespace tessera

#endif  // TESSERA_COLLECTIVES_H
