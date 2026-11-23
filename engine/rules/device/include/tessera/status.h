// The fault record, deadlines and timing marks.

#ifndef TESSERA_STATUS_H
#define TESSERA_STATUS_H

namespace tessera {

// Records the first failure. Later ones are dropped.
TESSERA_FN void fail(const table* t, status_code code, u32 pool = NONE, u64 ordinal = 0) {
    fault* f = get<fault>(t, t->status);
    if (prim::cas(&f->code, OK, code) != OK) return;
    f->pool = pool;
    f->ordinal = ordinal;
    u64* page = get<u64>(t, t->doorbell);
    if (page != nullptr) prim::store_release_sys(page + DOORBELL_FAULT / 8, u64(code));
}

// Reads the code alone, from device memory.
TESSERA_FN bool status_ok(const table* t) {
    return *static_cast<volatile u32*>(&get<fault>(t, t->status)->code) == OK;
}

// For spin loops: checks the clock every 1024th spin.

// Only block 0 reads the abort word in host memory.
TESSERA_FN bool failed(const table* t, u32 spin, u32 pool = NONE, u64 ordinal = 0) {
    if (!status_ok(t)) return true;
    if ((spin & 1023) != 0) return false;
    const u64* page = get<const u64>(t, t->doorbell);
    bool host = prim::leader_block() && page != nullptr;
    if (host && prim::load_acquire_sys(page + DOORBELL_ABORT / 8) != 0) {
        fail(t, PEER_LOST, pool, ordinal);
        return true;
    }
    u64 deadline = now(t)->deadline;
    if (deadline != 0 && prim::clock_ticks() > deadline) {
        fail(t, TIMEOUT, pool, ordinal);
        return true;
    }
    return false;
}

// Call first. False means stop.
TESSERA_FN bool check(const table* t) {
    if (t->edition == EDITION) return true;
    if (prim::leader()) fail(t, ABI_MISMATCH);
    return false;
}

// Mark `n` from block 0: capacity first, then pairs.
TESSERA_FN void mark(const table* t, u32 id, u32 n) {
#if defined(TESSERA_MARKS)
    if (!prim::leader()) return;
    u64* m = get<u64>(t, t->marks);
    if (m == nullptr || n >= m[0]) return;
    m[1 + 2 * n] = id;
    m[2 + 2 * n] = prim::clock_ticks();
#else
    (void)t;
    (void)id;
    (void)n;
#endif
}

}  // namespace tessera

#endif  // TESSERA_STATUS_H
