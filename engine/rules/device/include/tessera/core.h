// Layouts every backend shares, and plain access.

// No atomics or fences here. Places are arena offsets.

#ifndef TESSERA_CORE_H
#define TESSERA_CORE_H

#if defined(__METAL_VERSION__)
#define TESSERA_FN inline
namespace tessera {
using u8 = uchar;
using u16 = ushort;
using u32 = uint;
using u64 = ulong;
}
#else
#define TESSERA_FN __device__ __forceinline__
namespace tessera {
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
}
#endif

namespace tessera {

constexpr u32 EDITION = 1;
constexpr u32 ARENAS = 16;
constexpr u32 MAX_GROUP = 8;
constexpr u32 MAX_POOLS = 4;
constexpr u32 MAX_DEPTH = 4;
constexpr u32 VALUES = 58;
constexpr u32 NONE = 0xFFFFFFFFu;

// An arena and a byte offset, or arena NONE.
struct ref {
    u32 arena;
    u32 unused;
    u64 offset;
};

enum dtype : u32 {
    F32 = 0,
    BF16 = 1,
    F16 = 2,
    E4M3 = 3,
    U32 = 4,
    FORM = 5,
    BYTES = 6,
};

enum item_kind : u32 {
    INPUT = 0,
    OUTPUT = 1,
    SCRATCH = 2,
};

// One item the module declared, in its order.
struct item {
    ref at;
    u32 kind;
    u32 elem;
    u32 dims;
    u32 shape[4];
    u32 share[4];
    u32 form;
    u32 forms;
    u32 unused;
};

// One packed weight matrix and its rotation parts.
struct form {
    ref codes;
    ref suh;
    ref svh;
    u32 kind;
    u32 rows;
    u32 cols;
    u32 unused;
};

// Values the engine writes before each launch.
struct launch {
    u64 seq;
    u64 deadline;
    u32 values[VALUES];
};

enum convention : u32 {
    LLM = 0,
    KV = 1,
    STATE = 2,
    IMAGE = 3,
    CONVENTIONS = 16,
};

struct convention_ref {
    ref at;
    u32 version;
    u32 bytes;
};

enum status_code : u32 {
    OK = 0,
    ABI_MISMATCH = 1,
    PEER_LOST = 2,
    BAD_INPUT = 3,
    TIMEOUT = 4,
    PROTOCOL = 5,
};

// The first failure. The engine reads it after the launch.
struct fault {
    u32 code;
    u32 pool;
    u64 ordinal;
};

// Doorbell page offsets, in bytes.
constexpr u32 DOORBELL_SLOTS = 64;
constexpr u32 DOORBELL_ABORT = 4032;
constexpr u32 DOORBELL_SENT = 4040;

struct table {
    u32 edition;
    u32 rank;
    u32 ranks;
    u32 items_len;
    u32 forms_len;
    u32 lanes_len;
    u32 values_len;
    u32 unused;
    u64 arenas[ARENAS];
    ref now;
    ref items;
    ref forms;
    convention_ref conventions[CONVENTIONS];
    ref lanes;
    ref pools;
    ref doorbell;
    ref barrier;
    ref status;
    ref marks;
};

#if !defined(__METAL_VERSION__)

// Null for a missing place.
template <typename T>
TESSERA_FN T* get(const table* t, ref r) {
    if (r.arena >= ARENAS) return nullptr;
    return reinterpret_cast<T*>(t->arenas[r.arena] + r.offset);
}

TESSERA_FN const launch* now(const table* t) {
    return get<const launch>(t, t->now);
}

TESSERA_FN u32 value(const table* t, u32 index) {
    return now(t)->values[index];
}

TESSERA_FN const item* item_at(const table* t, u32 index) {
    return get<const item>(t, t->items) + index;
}

template <typename T>
TESSERA_FN T* at(const table* t, u32 index) {
    return get<T>(t, item_at(t, index)->at);
}

// Form `i` of a FORM item.
TESSERA_FN const form* form_of(const table* t, u32 index, u32 i) {
    return get<const form>(t, t->forms) + item_at(t, index)->form + i;
}

// Null if the kernel did not import it.
template <typename T>
TESSERA_FN const T* uses(const table* t, convention id) {
    return get<const T>(t, t->conventions[id].at);
}

#endif

}  // namespace tessera

#endif  // TESSERA_CORE_H
