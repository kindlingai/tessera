#include <tessera/1/all_reduce>
#include <tessera/1/grid>

using namespace tessera;

// The 64-bit address in values `i` and `i + 1`.
template <typename T>
__device__ T* address(const table* t, u32 i) {
    return reinterpret_cast<T*>(u64(value(t, i)) | u64(value(t, i + 1)) << 32);
}

// One sum the hand kernel passes to the grid.
struct job {
    u64 from;
    u64 to;
    u64 seq;
    u64 done;
    u32 n;
    // Nonzero: a barrier round first, and ticks summed below.
    u32 timed;
    u64 waited;
    u64 moved;
};

// Values: floats per round, job, first seq. Zero floats stop.
extern "C" __global__ void tessera_sums(const table* t) {
    if (!check(t)) return;
    u32 room = value(t, 0);
    job* j = address<job>(t, 1);
    u32 first = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = blockDim.x * prim::blocks();
    __shared__ job now;
    __shared__ int live;
    __shared__ u64 seen;
    session s = begin(t);
    for (u64 seq = u64(value(t, 3)) | u64(value(t, 4)) << 32;; seq++) {
        if (prim::leader_thread()) {
            for (u32 spin = 1; prim::load_acquire_gpu64(&j->seq) < seq; spin++) {
                if (failed(t, spin)) break;
                prim::pause();
            }
            seen = prim::clock_ticks();
            live = status_ok(t) ? 1 : 0;
            now.from = j->from;
            now.to = j->to;
            now.n = j->n;
            now.timed = j->timed;
        }
        prim::block_sync();
        if (live == 0) return;
        if (now.n == 0) break;
        if (now.timed != 0) {
            ticket b = prepare(s, 0);
            float* part = send_buffer<float>(b);
            if (part == nullptr) return;
            if (first == 0) part[0] = 0.0f;
            if (!trigger(b, 1)) return;
            if (!wait(b)) return;
        }
        u64 met = prim::clock_ticks();
        const float* from = reinterpret_cast<const float*>(now.from);
        float* to = reinterpret_cast<float*>(now.to);
        for (u32 done = 0; done < now.n; done += room) {
            u32 len = min(room, now.n - done);
            ticket r = prepare(s, 0);
            float* part = send_buffer<float>(r);
            if (part == nullptr) return;
            for (u32 i = first; i < len; i += stride) part[i] = from[done + i];
            if (!trigger(r, len)) return;
            if (!wait(r)) return;
            parts got = received(r);
            for (u32 i = first; i < len; i += stride) to[done + i] = got.sum(i);
        }
        u64 summed = prim::clock_ticks();
        __threadfence();
        if (!grid_sync(t)) return;
        if (prim::leader()) {
            if (now.timed != 0) {
                j->waited += met - seen;
                j->moved += summed - met;
            }
            prim::store_release_gpu64(&j->done, seq);
        }
    }
    end(s);
}

// On the main stream: passes one sum, then waits.
extern "C" __global__ void tessera_hand(const table* t, job* j, u64 from, u64 to, u32 n,
                                        u32 timed, u64 seq, u64 deadline) {
    j->from = from;
    j->to = to;
    j->n = n;
    j->timed = timed;
    prim::store_release_gpu64(&j->seq, seq);
    if (n == 0) return;
    for (u32 spin = 1; prim::load_acquire_gpu64(&j->done) < seq; spin++) {
        if (failed(t, spin)) return;
        if ((spin & 1023) == 0 && prim::clock_ticks() > deadline) {
            fail(t, TIMEOUT);
            return;
        }
        prim::pause();
    }
}
