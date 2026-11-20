// Kept rows that grow by position, in pages, in tiers.

// Tiers: device, host (read in place), absent (fetch), unmapped.

// Without TESSERA_REMOTE_KV the fetch calls are empty stubs.

// Rule: the kernel never changes a page table.

// Rule: each slot's last page is on the device.

#ifndef TESSERA_CAP_KEPT
#error "import <tessera/1/kept>"
#endif

#ifndef TESSERA_KEPT_H
#define TESSERA_KEPT_H

namespace tessera {

constexpr u32 TIER_SHIFT = 30;
constexpr u32 TIER_DEVICE = 0;
constexpr u32 TIER_HOST = 1;
constexpr u32 TIER_ABSENT = 2;
constexpr u32 TIER_UNMAPPED = 3;
constexpr u32 PAGE_INDEX = (1u << TIER_SHIFT) - 1;
constexpr u32 NO_PAGE = 0xFFFFFFFFu;

struct kept {
    ref base;
    ref host;
    ref pages;
    ref len;
    u32 pages_per_slot;
    u32 page_rows;
    u32 row_bytes;
    u32 elem;
};

struct kv_tables {
    u32 count;
    u32 unused;
    ref tables;
};

// One fetch request. Kept NONE means skip.
struct alignas(16) request {
    u32 kept;
    u32 slot;
    u32 pos;
    u32 unused;
};

#if !defined(__METAL_VERSION__)

TESSERA_FN bool unmapped(u32 entry) {
    return (entry >> TIER_SHIFT) == TIER_UNMAPPED;
}

TESSERA_FN const kept* kept_table(const table* t, u32 index) {
    return get<const kept>(t, uses<kv_tables>(t, KV)->tables) + index;
}

TESSERA_FN u32 kept_len(const table* t, const kept* k, u32 slot) {
    return get<const u32>(t, k->len)[slot];
}

TESSERA_FN u32 page_entry(const table* t, const kept* k, u32 slot, u32 pos) {
    return get<const u32>(t, k->pages)[slot * k->pages_per_slot + pos / k->page_rows];
}

TESSERA_FN u8* page_row(const table* t, ref storage, const kept* k, u32 page, u32 pos) {
    u64 row = u64(page) * k->page_rows + pos % k->page_rows;
    return get<u8>(t, storage) + row * k->row_bytes;
}

#if defined(TESSERA_REMOTE_KV)

constexpr bool REMOTE_KV = true;

// The device or host row, else null.
TESSERA_FN u8* kept_row(const table* t, const kept* k, u32 slot, u32 pos) {
    u32 entry = page_entry(t, k, slot, pos);
    u32 tier = entry >> TIER_SHIFT;
    u32 page = entry & PAGE_INDEX;
    if (tier == TIER_DEVICE) return page_row(t, k->base, k, page, pos);
    if (tier == TIER_HOST) return page_row(t, k->host, k, page, pos);
    return nullptr;
}

struct fetch {
    ticket r;
    const table* t;
    const kept* k;
    u32 index;
    request* req;
    const u8* staging;
};

// Every thread of the grid calls it, for one table.
template <typename S>
TESSERA_FN fetch fetch_begin(S& s, u32 lane, const table* t, u32 index) {
    ticket r = prepare(s, lane);
    return {r, t, kept_table(t, index), index, send_buffer<request>(r), nullptr};
}

// The row if readable now, else request `i` for it.
TESSERA_FN u8* row_or_request(fetch& f, u32 slot, u32 pos, u32 i) {
    u8* row = kept_row(f.t, f.k, slot, pos);
    bool missing = row == nullptr && unmapped(page_entry(f.t, f.k, slot, pos));
    if (missing) fail(f.t, BAD_INPUT);
    bool ask = row == nullptr && !missing;
    if (f.req != nullptr) f.req[i] = {ask ? f.index : NONE, slot, pos, 0};
    return row;
}

TESSERA_FN bool fetch_trigger(fetch& f, u32 requests) {
    if (f.req == nullptr) return false;
    return trigger(f.r, requests);
}

TESSERA_FN bool fetch_wait(fetch& f) {
    if (!wait(f.r)) return false;
    f.staging = static_cast<const u8*>(received(f.r).part[0]);
    return true;
}

// Request `i`'s row, after fetch_wait. Read it with load_fresh.
TESSERA_FN const u8* fetched_row(const fetch& f, u32 i) {
    return f.staging + u64(i) * f.k->row_bytes;
}

#else

constexpr bool REMOTE_KV = false;

#if !defined(TESSERA_COLLECTIVES_H)
struct session {};

TESSERA_FN session begin(const table*) {
    return {};
}

TESSERA_FN void end(const session&) {}
#endif

// The device row, or null if unmapped.
TESSERA_FN u8* kept_row(const table* t, const kept* k, u32 slot, u32 pos) {
    u32 entry = page_entry(t, k, slot, pos);
    if (unmapped(entry)) return nullptr;
    return page_row(t, k->base, k, entry, pos);
}

struct fetch {
    const table* t;
    const kept* k;
};

template <typename S>
TESSERA_FN fetch fetch_begin(S&, u32, const table* t, u32 index) {
    return {t, kept_table(t, index)};
}

TESSERA_FN u8* row_or_request(fetch& f, u32 slot, u32 pos, u32) {
    u8* row = kept_row(f.t, f.k, slot, pos);
    if (row == nullptr) fail(f.t, BAD_INPUT);
    return row;
}

TESSERA_FN bool fetch_trigger(fetch&, u32) {
    return true;
}

TESSERA_FN bool fetch_wait(fetch&) {
    return true;
}

TESSERA_FN const u8* fetched_row(const fetch&, u32) {
    return nullptr;
}

#endif

#endif

}  // namespace tessera

#endif  // TESSERA_KEPT_H
