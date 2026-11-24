#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

// pmcgx pairs, then mcg and mul1 pairs `(s, 65535 - s)`.
extern "C" __global__ void trellis_states(u32* out) {
    u32 s = blockIdx.x * blockDim.x + threadIdx.x;
    out[s] = trellis::decode_pmcgx(s);
    if (s < 65536u) {
        out[1048576u + s] = trellis::decode_pair<trellis::BOOK_MCG>(s, 65535u - s);
        out[1114112u + s] = trellis::decode_pair<trellis::BOOK_MUL1>(s, 65535u - s);
    }
}
