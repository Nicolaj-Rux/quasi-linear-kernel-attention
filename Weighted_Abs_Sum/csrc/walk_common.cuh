#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Shared device helpers for the sorted merge-walk kernels.

#define FULL_MASK 0xffffffffu


// first n in [0, N) with a[n] >= val
__device__ __forceinline__ int lower_bound(
    const float* __restrict__ a, const int N, const float val
){
    int lo = 0, hi = N;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (a[mid] < val) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// k-range start of q-tile t: 0 for tile 0, N once tiles are past M
__device__ __forceinline__ int walk_start(
    const float* __restrict__ kseg, const float* __restrict__ qseg,
    const int t, const int Bm, const int N, const int M
){
    if (t == 0)          return 0;
    if (t * Bm >= M)     return N;
    return lower_bound(kseg, N, qseg[t * Bm]);
}

// addr[0] += x, addr[1] += y — one 8B reduction op on sm_90+, two 4B below
__device__ __forceinline__ void atomic_add2(float* addr, const float x, const float y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(addr), "f"(x), "f"(y) : "memory");
#else
    atomicAdd(addr,     x);
    atomicAdd(addr + 1, y);
#endif
}

// lane l caches stream element base+l (value + perm), guarded by limit
template <typename IdxT>
__device__ __forceinline__ void refill(
    const float* __restrict__ vals, const IdxT* __restrict__ perm,
    const int base, const int limit, const int lane,
    float& val_buf, int& perm_buf
){
    int i = base + lane;
    val_buf  = (i < limit) ? vals[i] : 0.0f;
    perm_buf = (i < limit) ? (int)perm[i] : 0;
}
