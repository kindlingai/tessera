#include <tessera/1/all_reduce>

using namespace tessera;

// Values: floats per part, then rounds. Item 0 holds sums.
extern "C" __global__ void all_reduce_stub(const table* t) {
    if (!check(t)) return;
    u32 n = value(t, 0);
    u32 rounds = value(t, 1);
    float* out = at<float>(t, 0);
    u32 first = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = blockDim.x * prim::blocks();
    session s = begin(t);
    for (u32 round = 0; round < rounds; round++) {
        ticket r = prepare(s, 0);
        float* part = send_buffer<float>(r);
        if (part == nullptr) return;
        for (u32 i = first; i < n; i += stride) part[i] = float((t->rank + 1) * (i % 97) + round);
        if (!trigger(r, n)) return;
        if (!wait(r)) return;
        parts got = received(r);
        float* row = out + u64(round) * n;
        for (u32 i = first; i < n; i += stride) row[i] = got.sum(i);
    }
    end(s);
}
