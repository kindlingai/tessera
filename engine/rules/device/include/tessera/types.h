// Integer names and the function prefix, per backend.

#ifndef TESSERA_TYPES_H
#define TESSERA_TYPES_H

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

#endif  // TESSERA_TYPES_H
