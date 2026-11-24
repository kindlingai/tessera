// Plain access to the shared layouts.

// No atomics or fences here. Places are arena offsets.

#ifndef TESSERA_CORE_H
#define TESSERA_CORE_H

namespace tessera {

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
