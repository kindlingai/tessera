#include <tessera/1/all_reduce>

using namespace tessera;

// The 64-bit address in values `i` and `i + 1`.
template <typename T>
__device__ T* address(const table* t, u32 i) {
    return reinterpret_cast<T*>(u64(value(t, i)) | u64(value(t, i + 1)) << 32);
}

// Values: floats, floats per round, read address, write address.
extern "C" __global__ void tessera_sum(const table* t) {
    if (!check(t)) return;
    u32 n = value(t, 0);
    u32 room = value(t, 1);
    const float* from = address<const float>(t, 2);
    float* to = address<float>(t, 4);
    u32 first = blockIdx.x * blockDim.x + threadIdx.x;
    u32 stride = blockDim.x * prim::blocks();
    session s = begin(t);
    for (u32 done = 0; done < n; done += room) {
        u32 len = min(room, n - done);
        ticket r = prepare(s, 0);
        float* part = send_buffer<float>(r);
        if (part == nullptr) return;
        for (u32 i = first; i < len; i += stride) part[i] = from[done + i];
        if (!trigger(r, len)) return;
        if (!wait(r)) return;
        parts got = received(r);
        for (u32 i = first; i < len; i += stride) to[done + i] = got.sum(i);
    }
    end(s);
}
