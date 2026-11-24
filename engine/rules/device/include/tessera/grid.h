// A grid barrier inside one kernel.

// Rule: the whole grid fits on the device at once.

// Rule: the kernel runs alone, on a zeroed barrier.

#ifndef TESSERA_CAP_GRID
#error "import <tessera/1/grid>"
#endif

#ifndef TESSERA_GRID_H
#define TESSERA_GRID_H

namespace tessera {

// Every thread of the grid calls it. False means stop.
TESSERA_FN bool grid_sync(const table* t) {
    __shared__ int ok;
    prim::block_sync();
    if (prim::leader_thread()) {
        barrier* b = get<barrier>(t, t->barrier);
        u32 round = prim::load_acquire_gpu(&b->round);
        u32 arrived = prim::fetch_add_gpu(&b->count, 1) + 1;
        if (arrived == prim::blocks()) {
            prim::store_relaxed_gpu(&b->count, 0);
            prim::fetch_add_gpu(&b->round, 1);
        } else {
            for (u32 spin = 1; prim::load_acquire_gpu(&b->round) == round; spin++) {
                if (failed(t, spin)) break;
                prim::pause();
            }
        }
        ok = status_ok(t) ? 1 : 0;
    }
    prim::block_sync();
    return ok != 0;
}

}  // namespace tessera

#endif  // TESSERA_GRID_H
