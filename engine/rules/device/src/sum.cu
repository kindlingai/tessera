#include <tessera/1/all_gather>
#include <tessera/1/all_reduce>
#include <tessera/1/grid>

using namespace tessera;

// The 64-bit address in values `i` and `i + 1`.
template <typename T>
__device__ T* address(const table* t, u32 i) {
    return reinterpret_cast<T*>(u64(value(t, i)) | u64(value(t, i + 1)) << 32);
}

// One sum or gather the hand kernel passes.
struct job {
    u64 from;
    u64 to;
    u64 seq;
    u64 done;
    u32 n;
    // Nonzero: a barrier round first, and ticks summed below.
    u32 timed;
    // Nonzero: gather `rows` rows into `total` floats.
    u32 rows;
    u32 total;
    // Nonzero: parts cross as F32, else as BF16.
    u32 wide;
    u64 waited;
    u64 moved;
};

// Copies `len` floats into the send buffer.
__device__ bool stage_send(const float* src, float* dst, u32 len, u32 first, u32 stride) {
    u64 align = reinterpret_cast<u64>(src) | reinterpret_cast<u64>(dst);
    bool vec4 = (len & 3u) == 0u && (align & 15ull) == 0ull;
    if (vec4) {
        for (u32 i = first << 2; i < len; i += stride << 2) {
            *reinterpret_cast<float4*>(dst + i) = *reinterpret_cast<const float4*>(src + i);
        }
        return ((blockIdx.x * blockDim.x) << 2) < len;
    }
    for (u32 i = first; i < len; i += stride) dst[i] = src[i];
    return (blockIdx.x * blockDim.x) < len;
}

// Packs `len` floats as BF16 into the send buffer.
__device__ bool stage_bf16(const float* src, u16* dst, u32 len, u32 first, u32 stride) {
    bool vec4 = (len & 3u) == 0u && (reinterpret_cast<u64>(src) & 15ull) == 0ull
                && (reinterpret_cast<u64>(dst) & 7ull) == 0ull;
    if (vec4) {
        for (u32 i = first << 2; i < len; i += stride << 2) {
            float4 v = *reinterpret_cast<const float4*>(src + i);
            uint2 h = make_uint2(prim::to_bf16x2(v.x, v.y), prim::to_bf16x2(v.z, v.w));
            *reinterpret_cast<uint2*>(dst + i) = h;
        }
        return ((blockIdx.x * blockDim.x) << 2) < len;
    }
    for (u32 i = first; i < len; i += stride) dst[i] = prim::to_bf16(src[i]);
    return (blockIdx.x * blockDim.x) < len;
}

// Lane 0 sums BF16 parts, lane 2 F32 parts.
template <bool BF16>
__device__ bool run_sum(session& s, const float* from, float* to, u32 n, u32 room, u32 first,
                        u32 stride) {
    u32 sum_room = BF16 ? room << 1 : room;
    for (u32 done = 0; done < n; done += sum_room) {
        u32 len = min(sum_room, n - done);
        ticket r = prepare(s, BF16 ? 0 : 2);
        void* part = BF16 ? static_cast<void*>(send_buffer<u16>(r))
                          : static_cast<void*>(send_buffer<float>(r));
        if (part == nullptr) return false;
        bool active = BF16 ? stage_bf16(from + done, static_cast<u16*>(part), len, first, stride)
                           : stage_send(from + done, static_cast<float*>(part), len, first, stride);
        u64 align = BF16 ? 7ull : 15ull;
        bool vec4 = (len & 3u) == 0u && (reinterpret_cast<u64>(to + done) & 15ull) == 0ull
                    && (reinterpret_cast<u64>(part) & align) == 0ull;
        bool reads = vec4 ? ((blockIdx.x * blockDim.x) << 2) < len
                          : (blockIdx.x * blockDim.x) < len;
        if (!trigger(r, len, active)) return false;
        if (!wait(r, reads)) return false;
        if (reads) {
            parts got = received(r);
            if (vec4) {
                for (u32 i = first << 2; i < len; i += stride << 2) {
                    *reinterpret_cast<float4*>(to + done + i) = got.sum4(i);
                }
            } else {
                for (u32 i = first; i < len; i += stride) to[done + i] = got.sum(i);
            }
        }
    }
    return true;
}

__device__ bool run_gather(session& s, const float* from, float* to, u32 n, u32 rows, u32 total,
                           u32 room, u32 first, u32 stride) {
    u32 cols = total / rows;
    u32 local_w = n / rows;
    u32 step_rows = max(1u, room / cols);
    for (u32 r0 = 0; r0 < rows; r0 += step_rows) {
        u32 r_cnt = min(step_rows, rows - r0);
        u32 len = r_cnt * local_w;
        ticket r = prepare(s, 1);
        float* part = send_buffer<float>(r);
        if (part == nullptr) return false;
        bool active = stage_send(from + u64(r0) * local_w, part, len, first, stride);
        if (!trigger(r, len, active)) return false;
        if (!wait(r, true)) return false;
        parts got = received(r);
        u32 off = 0;
        for (u32 k = 0; k < got.count; k++) {
            u32 rk_w = got.size[k] / r_cnt;
            const float* src = static_cast<const float*>(got.part[k]);
            u64 out_base = reinterpret_cast<u64>(to + u64(r0) * cols + off);
            bool vec4 = (rk_w & 3u) == 0u && (cols & 3u) == 0u && (out_base & 15ull) == 0ull
                        && (reinterpret_cast<u64>(src) & 15ull) == 0ull;
            if (vec4) {
                u32 q_w = rk_w >> 2;
                u32 q_tot = r_cnt * q_w;
                for (u32 q = first; q < q_tot; q += stride) {
                    u32 row = q / q_w, c4 = (q % q_w) << 2;
                    *reinterpret_cast<float4*>(to + u64(r0 + row) * cols + off + c4) =
                        prim::load_fresh4(src + u64(row) * rk_w + c4);
                }
            } else {
                u32 e_tot = r_cnt * rk_w;
                for (u32 e = first; e < e_tot; e += stride) {
                    u32 row = e / rk_w, c = e % rk_w;
                    to[u64(r0 + row) * cols + off + c] = prim::load_fresh(src + e);
                }
            }
            off += rk_w;
        }
    }
    return true;
}

// Values: floats per round, job. Zero floats stop.
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
    u64 seq = prim::load_acquire_gpu64(&j->done) + 1;
    for (;; seq++) {
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
            now.rows = j->rows;
            now.total = j->total;
            now.wide = j->wide;
        }
        prim::block_sync();
        if (live == 0) return;
        if (now.n == 0) {
            if (!grid_sync(t)) return;
            if (prim::leader()) prim::store_release_gpu64(&j->done, seq);
            break;
        }
        if (now.timed != 0) {
            ticket b = prepare(s, 0);
            u16* part = send_buffer<u16>(b);
            if (part == nullptr) return;
            if (first == 0) part[0] = 0u;
            if (!trigger(b, 1, first == 0)) return;
            if (!wait(b, false)) return;
        }
        u64 met = prim::clock_ticks();
        const float* from = reinterpret_cast<const float*>(now.from);
        float* to = reinterpret_cast<float*>(now.to);
        bool ok = now.rows > 0
                      ? run_gather(s, from, to, now.n, now.rows, now.total, room, first, stride)
                  : now.wide != 0 ? run_sum<false>(s, from, to, now.n, room, first, stride)
                                  : run_sum<true>(s, from, to, now.n, room, first, stride);
        if (!ok) return;
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

// On the main stream: passes one job, then waits.
extern "C" __global__ void tessera_hand(const table* t, job* j, u64 from, u64 to, u32 n,
                                        u32 timed, u32 rows, u32 total, u32 wide, u64 timeout) {
    u64 seq = prim::load_acquire_gpu64(&j->seq) + 1;
    j->from = from;
    j->to = to;
    j->n = n;
    j->timed = timed;
    j->rows = rows;
    j->total = total;
    j->wide = wide;
    prim::store_release_gpu64(&j->seq, seq);
    if (n == 0) return;
    u64 deadline = prim::clock_ticks() + timeout;
    for (u32 spin = 1; prim::load_acquire_gpu64(&j->done) < seq; spin++) {
        if (failed(t, spin)) {
            u64* page = get<u64>(t, t->doorbell);
            u32 code = *static_cast<volatile u32*>(&get<fault>(t, t->status)->code);
            if (page != nullptr && code != OK) {
                prim::store_release_sys(page + DOORBELL_FAULT / 8, u64(code));
            }
            return;
        }
        if ((spin & 1023) == 0 && prim::clock_ticks() > deadline) {
            fail(t, TIMEOUT);
            return;
        }
        prim::pause();
    }
}
