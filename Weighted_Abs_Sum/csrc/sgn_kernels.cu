#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

#include "walk_common.cuh"


// Weighted sgn sum (backward walks and the public sgn op), queries q (P,D,M), sorted stream k (P,D,N),
// streamed weights w (P,N,C), emit-side matrix g (P,M,C):
//   out[p,d,m] = sum_c g[p,m,c] * (2*pref_w[c](rank of q_m) - w_total[p,c])
//              = sum_n sgn(q[p,d,m] - k[p,d,n]) * <w[p,n,:], g[p,m,:]>
// grad_q = sgn(queries=q, stream=k, w=v, g=g_w); grad_k = sgn(queries=k, stream=q, w=g_w, g=v); sgn op = -grad_q.
// With q-tiling each (p,d,m) is emitted by one warp, so the output is a plain store.
// FMODE (g_u terms of the full backward, added by lane 0): 0 none, 1 out += uq[p,m] * (2*rank - N),
// 2 out += 2*pref_us - us_total[p] (pref_us: running sum of us[perm], tile-started from chunk_us).

template <typename IdxT, int NV, int FMODE>
__global__ void sgn_prefix(
    const float*  __restrict__ q_sorted,  // shape (P, D, M)   queries
    const IdxT*   __restrict__ q_perm,    // shape (P, D, M)
    const float*  __restrict__ k_sorted,  // shape (P, D, N)   stream
    const IdxT*   __restrict__ k_perm,    // shape (P, D, N)
    const float2* __restrict__ w,         // shape (P, N, C/2) streamed weights
    const float2* __restrict__ w_total,   // shape (P, C/2)    w.sum(dim=1)
    const float2* __restrict__ g,         // shape (P, M, C/2) emit-side matrix
    const float2* __restrict__ chunk_w,   // shape (P, D, T, C/2)
    const float*  __restrict__ uq,        // shape (P, M)      FMODE==1 only
    const float*  __restrict__ us,        // shape (P, N)      FMODE==2 only
    const float*  __restrict__ us_total,  // shape (P)         FMODE==2 only
    const float*  __restrict__ chunk_us,  // shape (P, D, T)   FMODE==2 only
          float*  __restrict__ out,       // shape (P, D, M)
    const int D, const int C, const int N, const int M,
    const int T, const int Bm
){
    int l = threadIdx.x;                        // lane; group j -> channels 2l+64j, 2l+64j+1
    int d = blockIdx.x * blockDim.y + threadIdx.y;
    int t = blockIdx.y;
    int p = blockIdx.z;
    if (d >= D) return;

    int m0 = t * Bm;
    if (m0 >= M) return;
    int m1 = min(m0 + Bm, M);

    bool on[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) on[j] = 2 * l + 64 * j < C;

    int64_t   pd  = (int64_t)p * D + d;
    int64_t   pdn = pd * N;
    int64_t   pdm = pd * M;
    int64_t   pn  = (int64_t)p * N;
    int64_t   pm  = (int64_t)p * M;
    const int C2  = C >> 1;

    const float* kseg  = k_sorted + pdn;
    const IdxT*  kperm = k_perm + pdn;
    const float* qseg  = q_sorted + pdm;
    const IdxT*  qperm = q_perm + pdm;

    int n0 = walk_start(kseg, qseg, t,     Bm, N, M);
    int n1 = walk_start(kseg, qseg, t + 1, Bm, N, M);

    // w_total is loaded once per lane (d-independent)
    float2 wt[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++)
        wt[j] = on[j] ? w_total[p * C2 + l + 32 * j] : make_float2(0.0f, 0.0f);

    float us_tot = 0.0f;
    if constexpr (FMODE == 2) us_tot = us_total[p];

    float2 pref_v[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) pref_v[j] = make_float2(0.0f, 0.0f);
    float   pref_us = 0.0f;                     // FMODE==2: warp-uniform scalar prefix
    int64_t cbase   = pd * T * (int64_t)C2 + l;
    for (int tt = 0; tt < t; tt++) {
        #pragma unroll
        for (int j = 0; j < NV; j++) {
            if (on[j]) {
                float2 cv = chunk_w[cbase + tt * C2 + 32 * j];
                pref_v[j].x += cv.x;   pref_v[j].y += cv.y;
            }
        }
        if constexpr (FMODE == 2) pref_us += chunk_us[pd * T + tt];
    }

    // warp stream buffers (n, m and all branch conditions are warp-uniform)
    int   kbase = n0, qbase = m0;
    float kv, qv; int kp, qp;
    refill(kseg, kperm, kbase, n1, l, kv, kp);
    refill(qseg, qperm, qbase, m1, l, qv, qp);

    int   n   = n0;                             // merge position == rank of the emitted q
    int   m   = m0;
    float q_m = __shfl_sync(FULL_MASK, qv, 0);

    auto load_v = [&](int row, float2* dst) {
        const float2* vrow = w + (pn + row) * C2;
        #pragma unroll
        for (int j = 0; j < NV; j++) dst[j] = on[j] ? vrow[l + 32 * j] : make_float2(0.0f, 0.0f);
    };
    auto emit = [&](int q_idx) {
        const float2* grow = g + (pm + q_idx) * C2;
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < NV; j++) {
            if (on[j]) {
                float2 gv = grow[l + 32 * j];
                acc += gv.x * (2.0f * pref_v[j].x - wt[j].x)
                     + gv.y * (2.0f * pref_v[j].y - wt[j].y);
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(FULL_MASK, acc, off);
        if (l == 0) {
            if constexpr (FMODE == 1) acc += uq[pm + q_idx] * (2.0f * (float)n - (float)N);
            if constexpr (FMODE == 2) acc += 2.0f * pref_us - us_tot;
            out[pdm + q_idx] = acc;             // single writer per (p,d,m): plain store
        }
    };

    // prime k/w (and us) for n0 so one row load is always in flight
    float  k_n = 0.0f, us_nxt = 0.0f, us_cur = 0.0f;
    float2 v_nxt[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) v_nxt[j] = make_float2(0.0f, 0.0f);
    if (n0 < n1) {
        k_n     = __shfl_sync(FULL_MASK, kv, 0);
        int row = __shfl_sync(FULL_MASK, kp, 0);
        load_v(row, v_nxt);
        if constexpr (FMODE == 2) us_nxt = us[pn + row];
    }

    while (n < n1 && m < m1) {
        float  k_cur = k_n;
        float2 v_cur[NV];
        #pragma unroll
        for (int j = 0; j < NV; j++) v_cur[j] = v_nxt[j];
        if constexpr (FMODE == 2) us_cur = us_nxt;

        int nn = n + 1;                         // prefetch next w row before emitting
        if (nn < n1) {
            int idx = nn - kbase;
            if (idx == 32) {
                kbase = nn; idx = 0;
                refill(kseg, kperm, kbase, n1, l, kv, kp);
            }
            k_n     = __shfl_sync(FULL_MASK, kv, idx);
            int row = __shfl_sync(FULL_MASK, kp, idx);
            load_v(row, v_nxt);
            if constexpr (FMODE == 2) us_nxt = us[pn + row];
        }

        while (m < m1 && q_m <= k_cur) {
            int q_idx = __shfl_sync(FULL_MASK, qp, m - qbase);
            emit(q_idx);
            m++;
            if (m < m1) {
                int idx = m - qbase;
                if (idx == 32) {
                    qbase = m; idx = 0;
                    refill(qseg, qperm, qbase, m1, l, qv, qp);
                }
                q_m = __shfl_sync(FULL_MASK, qv, idx);
            }
        }

        #pragma unroll
        for (int j = 0; j < NV; j++) {
            pref_v[j].x += v_cur[j].x;   pref_v[j].y += v_cur[j].y;
        }
        if constexpr (FMODE == 2) pref_us += us_cur;
        n++;
    }

    while (m < m1) {                            // trailing queries (rank == n1, full prefix)
        int q_idx = __shfl_sync(FULL_MASK, qp, m - qbase);
        emit(q_idx);
        m++;
        if (m < m1) {
            int idx = m - qbase;
            if (idx == 32) {
                qbase = m; idx = 0;
                refill(qseg, qperm, qbase, m1, l, qv, qp);
            }
            q_m = __shfl_sync(FULL_MASK, qv, idx);
        }
    }
}

template <typename IdxT, int NV>
static void launch_sgn_prefix_nv(
    const float* d_q_sorted, const void* d_q_perm,
    const float* d_k_sorted, const void* d_k_perm,
    const float* d_w, const float* d_w_total, const float* d_g, const float* d_chunk_w,
    const float* d_uq, const float* d_us, const float* d_us_total, const float* d_chunk_us,
    float* d_out,
    const int P, const int D, const int C, const int N, const int M,
    const int T, const int Bm, const int fmode, cudaStream_t stream
){
    dim3 grid((D + 3) / 4, T, P), block(32, 4, 1);
    auto run = [&](auto kernel) {
        kernel<<<grid, block, 0, stream>>>(
            d_q_sorted, (const IdxT*)d_q_perm, d_k_sorted, (const IdxT*)d_k_perm,
            (const float2*)d_w, (const float2*)d_w_total, (const float2*)d_g,
            (const float2*)d_chunk_w, d_uq, d_us, d_us_total, d_chunk_us,
            d_out, D, C, N, M, T, Bm);
    };
    if      (fmode == 0) run(sgn_prefix<IdxT, NV, 0>);
    else if (fmode == 1) run(sgn_prefix<IdxT, NV, 1>);
    else                 run(sgn_prefix<IdxT, NV, 2>);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_sgn_prefix(
    const float* d_q_sorted,
    const void*  d_q_perm,
    const float* d_k_sorted,
    const void*  d_k_perm,
    const float* d_w,
    const float* d_w_total,
    const float* d_g,
    const float* d_chunk_w,
    const float* d_uq,         // nullptr unless fmode == 1
    const float* d_us,         // nullptr unless fmode == 2
    const float* d_us_total,   // nullptr unless fmode == 2
    const float* d_chunk_us,   // nullptr unless fmode == 2
          float* d_out,
    const int P, const int D, const int C, const int N, const int M,
    const int T, const int Bm, const bool idx32, const int NV, const int fmode
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (idx32) {
        if (NV == 1) launch_sgn_prefix_nv<uint32_t, 1>(d_q_sorted, d_q_perm, d_k_sorted, d_k_perm, d_w, d_w_total, d_g, d_chunk_w, d_uq, d_us, d_us_total, d_chunk_us, d_out, P, D, C, N, M, T, Bm, fmode, stream);
        else         launch_sgn_prefix_nv<uint32_t, 2>(d_q_sorted, d_q_perm, d_k_sorted, d_k_perm, d_w, d_w_total, d_g, d_chunk_w, d_uq, d_us, d_us_total, d_chunk_us, d_out, P, D, C, N, M, T, Bm, fmode, stream);
    } else {
        if (NV == 1) launch_sgn_prefix_nv<uint16_t, 1>(d_q_sorted, d_q_perm, d_k_sorted, d_k_perm, d_w, d_w_total, d_g, d_chunk_w, d_uq, d_us, d_us_total, d_chunk_us, d_out, P, D, C, N, M, T, Bm, fmode, stream);
        else         launch_sgn_prefix_nv<uint16_t, 2>(d_q_sorted, d_q_perm, d_k_sorted, d_k_perm, d_w, d_w_total, d_g, d_chunk_w, d_uq, d_us, d_us_total, d_chunk_us, d_out, P, D, C, N, M, T, Bm, fmode, stream);
    }
}
