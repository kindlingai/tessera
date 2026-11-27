// Warp argmax, lower expert wins ties.
static __device__ inline void warp_argmax(u32 experts, u32& best, float& most, float& best_raw) {
  for (u32 off = 16u; off > 0u; off >>= 1u) {
    float v = __shfl_xor_sync(0xFFFFFFFFu, most, off), r = __shfl_xor_sync(0xFFFFFFFFu, best_raw, off);
    u32 e = __shfl_xor_sync(0xFFFFFFFFu, best, off);
    if (e < experts && (best == experts || v > most || (v == most && e < best))) { best = e; most = v; best_raw = r; }
  }
}

// Scales the first `picks` gates to sum to ROUTE_SCALE.
static __device__ inline void normalise_gates(float* gates, u32 lane, u32 n, float sum) {
  __syncwarp();
  if (lane < n) gates[lane] *= ROUTE_SCALE / sum;
  __syncwarp();
}

// One warp's picks and gates for 256 experts in registers.
static __device__ void route_256(
    const float* scores, const float* bias, u32 row, u32 rows, u32 rks, bool fused, u16* ids,
    float* gates) {
  u32 lane = threadIdx.x & 31u, e0 = lane << 2u;
  float raw[8];
  if (fused) {
    u64 re = (u64)rows * 256ull;
    float4 t0 = make_float4(0.f, 0.f, 0.f, 0.f), t1 = t0;
    for (u32 sp = 0u; sp < rks; sp++) {
      const float* base = scores + (u64)sp * re + (u64)row * 256ull;
      float4 v0 = prim::load_fresh4(base + e0), v1 = prim::load_fresh4(base + e0 + 128u);
      t0.x += v0.x; t0.y += v0.y; t0.z += v0.z; t0.w += v0.w;
      t1.x += v1.x; t1.y += v1.y; t1.z += v1.z; t1.w += v1.w;
    }
    t0 = sigmoid4(t0); t1 = sigmoid4(t1);
    raw[0] = t0.x; raw[1] = t0.y; raw[2] = t0.z; raw[3] = t0.w;
    raw[4] = t1.x; raw[5] = t1.y; raw[6] = t1.z; raw[7] = t1.w;
  } else {
    const float* base = scores + (u64)row * 256ull;
    float4 v0 = (rks == 1u) ? prim::load_fresh4(base + e0) : *reinterpret_cast<const float4*>(base + e0);
    float4 v1 = (rks == 1u) ? prim::load_fresh4(base + e0 + 128u) : *reinterpret_cast<const float4*>(base + e0 + 128u);
    raw[0] = v0.x; raw[1] = v0.y; raw[2] = v0.z; raw[3] = v0.w;
    raw[4] = v1.x; raw[5] = v1.y; raw[6] = v1.z; raw[7] = v1.w;
  }
  float4 b0 = *reinterpret_cast<const float4*>(bias + e0), b1 = *reinterpret_cast<const float4*>(bias + e0 + 128u);
  float biased[8] = {raw[0] + b0.x, raw[1] + b0.y, raw[2] + b0.z, raw[3] + b0.w,
                     raw[4] + b1.x, raw[5] + b1.y, raw[6] + b1.z, raw[7] + b1.w};
  u32 taken = 0u;
  float sum = 0.0f;
  for (u32 p = 0u; p < PICKS; p++) {
    u32 best = 256u; float most = 0.0f, best_raw = 0.0f;
    for (u32 i = 0u; i < 8u; i++) {
      u32 e = (i < 4u) ? (e0 + i) : (e0 + 124u + i);
      if (!(taken & (1u << i)) && (best == 256u || biased[i] > most)) { best = e; most = biased[i]; best_raw = raw[i]; }
    }
    warp_argmax(256u, best, most, best_raw);
    if (((best & 127u) >> 2u) == lane) taken |= 1u << (((best >> 7u) << 2u) | (best & 3u));
    if (lane == 0u) { ids[p] = (u16)best; gates[p] = best_raw; }
    sum += best_raw;
  }
  normalise_gates(gates, lane, PICKS, sum);
}

// One warp's picks and gates. Ties take lower expert.
static __device__ void route(
    const float* score, const float* bias, u32 experts, u32 picks, bool fresh, u16* ids, float* gates) {
  u32 lane = threadIdx.x & 31u; u64 taken = 0ull; float sum = 0.0f;
  for (u32 p = 0u; p < picks; p++) {
    u32 best = experts; float most = 0.0f, best_raw = 0.0f;
    for (u32 i = 0u, e = lane; e < experts; i++, e += 32u) {
      float s = fresh ? prim::load_fresh(score + e) : score[e], v = s + bias[e];
      if (!((taken >> i) & 1ull) && (best == experts || v > most)) { best = e; most = v; best_raw = s; }
    }
    warp_argmax(experts, best, most, best_raw);
    if ((best & 31u) == lane) taken |= 1ull << (best >> 5u);
    if (lane == 0u) { ids[p] = (u16)best; gates[p] = best_raw; }
    sum += best_raw;
  }
  normalise_gates(gates, lane, picks, sum);
}

// Four router weights from `at`, f32 or bf16.
static __device__ inline float4 weights4(const void* router, u64 at, u32 bf16) {
  if (!bf16) return *(const float4*)((const float*)router + at);
  uint2 h = *(const uint2*)((const unsigned short*)router + at);
  return make_float4(__uint_as_float(h.x << 16u), __uint_as_float(h.x & 0xFFFF0000u),
      __uint_as_float(h.y << 16u), __uint_as_float(h.y & 0xFFFF0000u));
}

struct PlanSmem { u16 ids[8][PICKS], slot_of[128]; float gates[8][PICKS]; u32 emask[ROW_WORDS][128], wsum[8], total; };
struct RouteTcSmem { u32 sxh[4][16][36], sxl[4][16][36]; union { struct { u32 h[8][8][36], l[8][8][36]; } sw; float4 red[4][2][32]; } u; };
struct RoutePfSmem { u32 sx[64][36], sw[64][36]; };
union RouteBfSmem {
  struct { u32 sxh[32][36], sxl[32][36], sw[64][36]; } k;
  struct { float sc[32][256]; u16 ids[8][PICKS]; float gates[8][PICKS]; } p;
};
union alignas(16) RouteSmem { float wsum[2][8][8]; RouteTcSmem tc; RoutePfSmem pf; RouteBfSmem bf; PlanSmem plan; };

static __device__ inline void split_f32x4(float4 v, uint2& hi, uint2& lo) {
  hi.x = prim::to_bf16x2(v.x, v.y); hi.y = prim::to_bf16x2(v.z, v.w);
  lo.x = prim::to_bf16x2(v.x - __uint_as_float(hi.x << 16u), v.y - __uint_as_float(hi.x & 0xFFFF0000u));
  lo.y = prim::to_bf16x2(v.z - __uint_as_float(hi.y << 16u), v.w - __uint_as_float(hi.y & 0xFFFF0000u));
}

static __device__ inline void load_w_tc(
    const void* router, u64 off, u32 bf16, bool aligned, uint4& hi, uint4& lo) {
  if (bf16) { hi = load_bf16x8(reinterpret_cast<const u16*>(router) + off, aligned); lo = make_uint4(0u, 0u, 0u, 0u); return; }
  const float* fp = reinterpret_cast<const float*>(router) + off;
  uint2 h0, l0, h1, l1;
  split_f32x4(*reinterpret_cast<const float4*>(fp), h0, l0);
  split_f32x4(*reinterpret_cast<const float4*>(fp + 4u), h1, l1);
  hi = make_uint4(h0.x, h0.y, h1.x, h1.y); lo = make_uint4(l0.x, l0.y, l1.x, l1.y);
}

static __device__ inline void stage_x_bf16(
    const float* x, u32 row, u32 rows, u32 hidden, u32 col, u32* dst_h, u32* dst_l) {
  uint2 hi = make_uint2(0u, 0u), lo = make_uint2(0u, 0u);
  if (row < rows) split_f32x4(*reinterpret_cast<const float4*>(x + (u64)row * hidden + col), hi, lo);
  *reinterpret_cast<uint2*>(dst_h) = hi; *reinterpret_cast<uint2*>(dst_l) = lo;
}

// Streaming 8-BF16 decode router GEMV, 2 experts per CTA.
static __device__ __noinline__ void route_gemv(
    const float* x, const u16* router, float* scores, float wsum[2][8][8],
    u32 hidden, u32 experts) {
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, grp = warp >> 2u, gw = warp & 3u, e = blockIdx.x * 2u + grp;
  if (e >= experts) return;
  bool aligned = (reinterpret_cast<u64>(router) & 15ull) == 0ull;
  const u16* wr = router + (u64)e * hidden;
  u32 stid = (gw << 5u) | lane;
  float acc = 0.0f;
  for (u32 i = stid << 3u; i < hidden; i += 1024u) {
    uint4 h = load_bf16x8(wr + i, aligned);
    float4 x0 = *reinterpret_cast<const float4*>(x + i), x1 = *reinterpret_cast<const float4*>(x + i + 4u);
    acc = fmaf(__uint_as_float(h.x << 16u), x0.x, fmaf(__uint_as_float(h.x & 0xFFFF0000u), x0.y, fmaf(__uint_as_float(h.y << 16u), x0.z, fmaf(__uint_as_float(h.y & 0xFFFF0000u), x0.w, acc))));
    acc = fmaf(__uint_as_float(h.z << 16u), x1.x, fmaf(__uint_as_float(h.z & 0xFFFF0000u), x1.y, fmaf(__uint_as_float(h.w << 16u), x1.z, fmaf(__uint_as_float(h.w & 0xFFFF0000u), x1.w, acc))));
  }
#pragma unroll
  for (u32 off = 16u; off > 0u; off >>= 1u) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0u) wsum[grp][0][gw] = acc;
  prim::block_sync();
  if (lane == 0u && gw == 0u) scores[e] = sigmoid(wsum[grp][0][0] + wsum[grp][0][1] + wsum[grp][0][2] + wsum[grp][0][3]);
}

static __device__ inline void load_w_tf32(
    const void* router, u64 off, u32 bf16, bool aligned, uint4& w0, uint4& w1) {
  if (bf16) {
    uint4 h = load_bf16x8(reinterpret_cast<const u16*>(router) + off, aligned);
    w0 = make_uint4(h.x << 16u, h.x & 0xFFFF0000u, h.y << 16u, h.y & 0xFFFF0000u);
    w1 = make_uint4(h.z << 16u, h.z & 0xFFFF0000u, h.w << 16u, h.w & 0xFFFF0000u);
    return;
  }
  const float* fp = reinterpret_cast<const float*>(router) + off;
  float4 v0 = *reinterpret_cast<const float4*>(fp), v1 = *reinterpret_cast<const float4*>(fp + 4u);
  w0 = make_uint4(f32_to_tf32(v0.x), f32_to_tf32(v0.y), f32_to_tf32(v0.z), f32_to_tf32(v0.w));
  w1 = make_uint4(f32_to_tf32(v1.x), f32_to_tf32(v1.y), f32_to_tf32(v1.z), f32_to_tf32(v1.w));
}

// Fused 32x256 BF16 Tensor-Core router with in-smem parallel top-8.
static __device__ __noinline__ void route_pf_bf16(
    const float* x, const u16* router, const float* bias, float* plan_out,
    RouteBfSmem& sm, u32 rows, u32 hidden, u32 picks, u32 first, u32 count) {
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
  u32 r0 = blockIdx.y << 5u, xr = tid >> 4u, xc = (tid & 15u) << 2u, xw = (tid & 15u) << 1u;
  u32 wr = tid >> 3u, wc = (tid & 7u) << 3u;
  bool aligned = (reinterpret_cast<u64>(router) & 15ull) == 0ull;
  float acc[2][4][4] = {};
  uint4 nw[2] = {load_bf16x8(router + (u64)wr * hidden + wc, aligned), load_bf16x8(router + (u64)(wr + 32u) * hidden + wc, aligned)};
  for (u32 k0 = 0u; k0 < hidden; k0 += 64u) {
    stage_x_bf16(x, r0 + xr, rows, hidden, k0 + xc, &sm.k.sxh[xr][xw], &sm.k.sxl[xr][xw]);
    stage_x_bf16(x, r0 + xr + 16u, rows, hidden, k0 + xc, &sm.k.sxh[xr + 16u][xw], &sm.k.sxl[xr + 16u][xw]);
    for (u32 h = 0u; h < 4u; h++) {
      *reinterpret_cast<uint4*>(&sm.k.sw[wr][wc >> 1u]) = nw[0]; *reinterpret_cast<uint4*>(&sm.k.sw[wr + 32u][wc >> 1u]) = nw[1];
      prim::block_sync();
      u32 nh = (h + 1u) & 3u, nk0 = (h == 3u) ? (k0 + 64u) : k0;
      if (nk0 < hidden) { u32 eb = (nh << 6u) + wr; nw[0] = load_bf16x8(router + (u64)eb * hidden + nk0 + wc, aligned); nw[1] = load_bf16x8(router + (u64)(eb + 32u) * hidden + nk0 + wc, aligned); }
#pragma unroll
      for (u32 ks = 0u; ks < 4u; ks++) {
        u32 o = (ks << 3u) + t, e0 = (warp << 3u) + g, b[2] = {sm.k.sw[e0][o], sm.k.sw[e0][o + 4u]};
#pragma unroll
        for (u32 mi = 0u; mi < 2u; mi++) {
          u32 mr = mi << 4u;
          u32 ah[4] = {sm.k.sxh[mr + g][o], sm.k.sxh[mr + g + 8u][o], sm.k.sxh[mr + g][o + 4u], sm.k.sxh[mr + g + 8u][o + 4u]};
          u32 al[4] = {sm.k.sxl[mr + g][o], sm.k.sxl[mr + g + 8u][o], sm.k.sxl[mr + g][o + 4u], sm.k.sxl[mr + g + 8u][o + 4u]};
          mma_bf16_m16n8k16(acc[mi][h], ah, b); mma_bf16_m16n8k16(acc[mi][h], al, b);
        }
      }
      prim::block_sync();
    }
  }
  for (u32 mi = 0u; mi < 2u; mi++) for (u32 h = 0u; h < 4u; h++) {
    u32 col = (h << 6u) + (warp << 3u) + (t << 1u), rr = (mi << 4u) + g;
    float* a = acc[mi][h];
    *reinterpret_cast<float2*>(&sm.p.sc[rr][col]) = make_float2(sigmoid(a[0]), sigmoid(a[1]));
    *reinterpret_cast<float2*>(&sm.p.sc[rr + 8u][col]) = make_float2(sigmoid(a[2]), sigmoid(a[3]));
  }
  prim::block_sync();
  Plan* plan = reinterpret_cast<Plan*>(plan_out);
  for (u32 dr = warp; dr < 32u && r0 + dr < rows; dr += 8u) {
    route_256(&sm.p.sc[0][0], bias, dr, 32u, 0u, false, sm.p.ids[warp], sm.p.gates[warp]);
    if (lane < picks) { u32 e = sm.p.ids[warp][lane]; plan->tok_expert[r0 + dr][lane] = (e >= first && e < first + count) ? e : 0xFFFFFFFFu; plan->tok_w[r0 + dr][lane] = sm.p.gates[warp][lane]; }
    __syncwarp();
  }
}

// TF32 Tensor-Core prefill router projection, RMx64 per CTA.
template <u32 RM>
static __device__ __noinline__ void route_pf(
    const float* x, const void* router, float* scores, RoutePfSmem& sm,
    u32 rows, u32 hidden, u32 experts, u32 bf16, u32 nb, u32 rks) {
  constexpr u32 MT = RM / 32u;
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
  u32 wr = warp >> 2u, wn = warp & 3u, r0 = blockIdx.y * RM;
  u32 ob = blockIdx.x % nb, sp = blockIdx.x / nb, n_base = ob << 6u;
  u32 ch_per = (hidden >> 5u) / rks, k_start = (sp * ch_per) << 5u, k_end = k_start + (ch_per << 5u);
  u32 xr = tid >> 3u, xc = (tid & 7u) << 2u, wl = tid >> 2u, wc = (tid & 3u) << 3u;
  bool aligned = (reinterpret_cast<u64>(router) & 15ull) == 0ull;
  float acc[MT][2][4] = {};
  uint4 w0, w1;
  load_w_tf32(router, (u64)(n_base + wl) * hidden + k_start + wc, bf16, aligned, w0, w1);
  for (u32 k0 = k_start; k0 < k_end; k0 += 32u) {
    for (u32 dr = 0u; dr < RM; dr += 32u) {
      uint4 xv = make_uint4(0u, 0u, 0u, 0u);
      if (r0 + xr + dr < rows) {
        u32 d[4];
        trellis::ld_nc_v4(reinterpret_cast<const u8*>(x + (u64)(r0 + xr + dr) * hidden + k0 + xc), d);
        xv = make_uint4(f32_to_tf32(__uint_as_float(d[0])), f32_to_tf32(__uint_as_float(d[1])),
                        f32_to_tf32(__uint_as_float(d[2])), f32_to_tf32(__uint_as_float(d[3])));
      }
      *reinterpret_cast<uint4*>(&sm.sx[xr + dr][xc]) = xv;
    }
    *reinterpret_cast<uint4*>(&sm.sw[wl][wc]) = w0;
    *reinterpret_cast<uint4*>(&sm.sw[wl][wc + 4u]) = w1;
    prim::block_sync();
    if (k0 + 32u < k_end) load_w_tf32(router, (u64)(n_base + wl) * hidden + k0 + 32u + wc, bf16, aligned, w0, w1);
#pragma unroll
    for (u32 ks = 0u; ks < 4u; ks++) {
      u32 o = (ks << 3u) + t, e0 = (wn << 4u) + g;
      u32 b0[2] = {sm.sw[e0][o], sm.sw[e0][o + 4u]}, b1[2] = {sm.sw[e0 + 8u][o], sm.sw[e0 + 8u][o + 4u]};
      for (u32 mt = 0u; mt < MT; mt++) {
        u32 rr = (wr * MT + mt) * 16u + g;
        u32 a[4] = {sm.sx[rr][o], sm.sx[rr + 8u][o], sm.sx[rr][o + 4u], sm.sx[rr + 8u][o + 4u]};
        mma_tf32_m16n8k8(acc[mt][0], a, b0); mma_tf32_m16n8k8(acc[mt][1], a, b1);
      }
    }
    prim::block_sync();
  }
  float* dst = scores + (u64)sp * rows * experts;
  for (u32 mt = 0u; mt < MT; mt++) {
    u32 rr = r0 + (wr * MT + mt) * 16u + g;
    for (u32 nt = 0u; nt < 2u; nt++) {
      u32 col = n_base + (wn << 4u) + (nt << 3u) + (t << 1u);
      float v0 = acc[mt][nt][0], v1 = acc[mt][nt][1], v2 = acc[mt][nt][2], v3 = acc[mt][nt][3];
      if (rks == 1u) {
        v0 = sigmoid(v0); v1 = sigmoid(v1); v2 = sigmoid(v2); v3 = sigmoid(v3);
      }
      if (rr < rows) *reinterpret_cast<float2*>(dst + (u64)rr * experts + col) = make_float2(v0, v1);
      if (rr + 8u < rows) *reinterpret_cast<float2*>(dst + (u64)(rr + 8u) * experts + col) = make_float2(v2, v3);
    }
  }
}

// Tensor-Core split-K router projection, 16x16 per CTA.
static __device__ __noinline__ void route_tc(
    const float* x, const void* router, float* scores, RouteTcSmem& sm,
    u32 rows, u32 hidden, u32 experts, u32 bf16, u32 nb, u32 rks) {
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
  u32 wt = warp & 1u, kp = warp >> 1u, ob = blockIdx.x % nb, p = blockIdx.x / nb, r0 = blockIdx.y << 4u, n0 = ob << 4u;
  u32 ch_per = (hidden >> 8u) / rks, k_start = (p * ch_per) << 8u, k_end = k_start + (ch_per << 8u);
  bool aligned = (reinterpret_cast<u64>(router) & 15ull) == 0ull;
  u64 wrow = (u64)(n0 + (wt << 3u) + g) * hidden + (kp << 6u);
  u32 xs = tid >> 6u, xr = (tid >> 4u) & 3u, xc = (tid & 15u) << 2u, xw = (tid & 15u) << 1u;
  float *dst = scores + (u64)p * rows * experts, acc[4] = {};
  uint4 c0h, c0l, c1h, c1l;
  load_w_tc(router, wrow + k_start + (t << 3u), bf16, aligned, c0h, c0l);
  load_w_tc(router, wrow + k_start + 32u + (t << 3u), bf16, aligned, c1h, c1l);
  for (u32 k0 = k_start; k0 < k_end; k0 += 256u) {
#pragma unroll
    for (u32 dr = 0u; dr < 16u; dr += 4u)
      stage_x_bf16(x, r0 + xr + dr, rows, hidden, k0 + (xs << 6u) + xc, &sm.sxh[xs][xr + dr][xw], &sm.sxl[xs][xr + dr][xw]);
    *reinterpret_cast<uint4*>(&sm.u.sw.h[warp][g][t << 2u]) = c0h;
    *reinterpret_cast<uint4*>(&sm.u.sw.h[warp][g][16u + (t << 2u)]) = c1h;
    if (!bf16) { *reinterpret_cast<uint4*>(&sm.u.sw.l[warp][g][t << 2u]) = c0l; *reinterpret_cast<uint4*>(&sm.u.sw.l[warp][g][16u + (t << 2u)]) = c1l; }
    prim::block_sync();
    if (k0 + 256u < k_end) { u64 nk = wrow + k0 + 256u + (t << 3u); load_w_tc(router, nk, bf16, aligned, c0h, c0l); load_w_tc(router, nk + 32u, bf16, aligned, c1h, c1l); }
#pragma unroll
    for (u32 ks = 0u; ks < 4u; ks++) {
      u32 o = (ks << 3u) + t, b[2] = {sm.u.sw.h[warp][g][o], sm.u.sw.h[warp][g][o + 4u]};
      u32 ah[4] = {sm.sxh[kp][g][o], sm.sxh[kp][g + 8u][o], sm.sxh[kp][g][o + 4u], sm.sxh[kp][g + 8u][o + 4u]};
      mma_bf16_m16n8k16(acc, ah, b);
      u32 al[4] = {sm.sxl[kp][g][o], sm.sxl[kp][g + 8u][o], sm.sxl[kp][g][o + 4u], sm.sxl[kp][g + 8u][o + 4u]};
      mma_bf16_m16n8k16(acc, al, b);
      if (!bf16) { u32 bl[2] = {sm.u.sw.l[warp][g][o], sm.u.sw.l[warp][g][o + 4u]}; mma_bf16_m16n8k16(acc, ah, bl); }
    }
    prim::block_sync();
  }
  sm.u.red[kp][wt][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  prim::block_sync();
  if (kp == 0u) {
    float4 s = sm.u.red[0][wt][lane], q1 = sm.u.red[1][wt][lane], q2 = sm.u.red[2][wt][lane], q3 = sm.u.red[3][wt][lane];
    float v0 = s.x + q1.x + q2.x + q3.x, v1 = s.y + q1.y + q2.y + q3.y, v2 = s.z + q1.z + q2.z + q3.z, v3 = s.w + q1.w + q2.w + q3.w;
    if (rks == 1u) { v0 = sigmoid(v0); v1 = sigmoid(v1); v2 = sigmoid(v2); v3 = sigmoid(v3); }
    u32 col = n0 + (wt << 3u) + (t << 1u);
    if (r0 + g < rows) *reinterpret_cast<float2*>(dst + (u64)(r0 + g) * experts + col) = make_float2(v0, v1);
    if (r0 + g + 8u < rows) *reinterpret_cast<float2*>(dst + (u64)(r0 + g + 8u) * experts + col) = make_float2(v2, v3);
  }
}
