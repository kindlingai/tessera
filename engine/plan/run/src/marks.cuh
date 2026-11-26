// The runner puts this before kernel text it marks.

#define TESSERA_MARK_BLOCKS 4096u
#define TESSERA_MARK_ROOM 256u

// Counts per block, then 16-byte records. Zero records nothing.
__device__ unsigned long long tessera_marks;

__device__ __forceinline__ unsigned long long tessera_clock_ticks() {
#if defined(__HIPCC_RTC__) || defined(__HIP__)
  return wall_clock64();
#else
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  return now;
#endif
}

__device__ __forceinline__ void tessera_stamp_step(
    unsigned long long* starts,
    unsigned long long* accum,
    unsigned slot,
    unsigned mode) {
  unsigned long long now = tessera_clock_ticks();
  if (mode == 0u) {
    starts[0] = now;
    starts[slot] = now;
  } else if (mode == 1u) {
    accum[slot] += now - starts[0];
    starts[0] = now;
  } else if (mode == 2u) {
    accum[slot] += now - starts[slot];
  } else if (mode == 3u) {
    starts[slot] = now;
  } else if (mode == 4u) {
    accum[slot] += now - starts[slot];
  } else if (mode == 5u) {
    accum[slot] += now - starts[0];
    accum[slot + 1u] += now - starts[slot + 1u];
  }
}

__device__ __forceinline__ void tessera_mark(unsigned line) {
  unsigned long long now = tessera_clock_ticks();
  if ((threadIdx.x | threadIdx.y | threadIdx.z) != 0 || tessera_marks == 0) {
    return;
  }
  unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  if (block >= TESSERA_MARK_BLOCKS) {
    return;
  }
  unsigned* counts = (unsigned*)tessera_marks;
  unsigned at = counts[block]++;
  if (at >= TESSERA_MARK_ROOM) {
    return;
  }
  unsigned long long* records = (unsigned long long*)(counts + TESSERA_MARK_BLOCKS);
  unsigned long long* record = records + 2ull * (block * TESSERA_MARK_ROOM + at);
  record[0] = line;
  record[1] = now;
}

// Notes the line and the clock, once per block.
#define TS_MARK() tessera_mark(__LINE__)
#line 1
