// Fixed-size state per sequence, for recurrent layers.

#ifndef TESSERA_CAP_STATE
#error "import <tessera/1/state>"
#endif

#ifndef TESSERA_STATE_H
#define TESSERA_STATE_H

namespace tessera {

#if !defined(__METAL_VERSION__)

TESSERA_FN const held* held_table(const table* t, u32 index) {
    return get<const held>(t, uses<state_tables>(t, STATE)->tables) + index;
}

// A slot's state. Treat it as zero when fresh.
TESSERA_FN u8* held_slot(const table* t, const held* h, u32 slot) {
    return get<u8>(t, h->base) + u64(slot) * h->slot_bytes;
}

TESSERA_FN bool held_fresh(const table* t, const held* h, u32 slot) {
    return get<const u32>(t, h->fresh)[slot] != 0;
}

#endif

}  // namespace tessera

#endif  // TESSERA_STATE_H
