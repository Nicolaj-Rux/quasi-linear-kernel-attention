#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_radix_sort.cuh>
#include <cstdint>

#include "walk_common.cuh"


// Weighted abs sum forward:
// out[p,m,c] = sum_{d,n} |k[p,d,n] - q[p,d,m]| * v[p,n,c]
//            = sum_d 2*(q*pref_v - pref_k)             (pref over k < q, sorted merge walk: abs_prefix)
//            + k_total[p,c] - qsum[p,m]*vsum[p,c]      (correction, pre-written by abs_init)
// Each (p,d) walk is split into T q-tiles; abs_chunk pre-sums each tile's k-range so a tile starts from the
// right prefix. IdxT: uint16 perms for max(N,M) <= 65536, else uint32. NV float2 channel groups per lane:
// lane l, group j covers channels 2l+64j, 2l+64j+1 (C is padded to even in binding.cpp).
// Each warp buffers 32 stream elements (one per lane) and broadcasts them via __shfl_sync.
// FULL also emits out_u[p,m] = sum_{d,n} |k - q| = 2*(q*rank - PKraw_rank) + (ksum_raw - q*N), with rank = merge
// position n and PKraw = running sum of sorted k; lane 0 writes it.


// correction as out's initial value: out[p,m,c] = k_total[p,c] - qsum[p,m]*vsum[p,c]
// FULL: out_u[p,m] = ksum_raw[p] - N*qsum[p,m]  (the v==1 correction)
template <bool FULL>
__global__ void abs_init(
    const float2* __restrict__ k_total,   // shape (P, C/2)
    const float2* __restrict__ vsum,      // shape (P, C/2)
    const float*  __restrict__ qsum,      // shape (P, M)
    const float*  __restrict__ ksum_raw,  // shape (P)      FULL only
          float2* __restrict__ out,       // shape (P, M, C/2)
          float*  __restrict__ out_u,     // shape (P, M)   FULL only
    const int64_t total2, const int M, const int C2, const float Nk
){
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total2) return;

    int     c2 = (int)(i % C2);
    int64_t pm = i / C2;
    int     p  = (int)(pm / M);

    float2 kt = k_total[p * C2 + c2];
    float2 vs = vsum[p * C2 + c2];
    float  qs = qsum[pm];
    out[i] = make_float2(kt.x - qs * vs.x, kt.y - qs * vs.y);
    if constexpr (FULL) {
        if (c2 == 0) out_u[pm] = ksum_raw[p] - Nk * qs;
    }
}

void launch_abs_init(
    const float* d_k_total,
    const float* d_vsum,
    const float* d_qsum,
    const float* d_ksum_raw,   // nullptr unless full
          float* d_out,
          float* d_out_u,      // nullptr unless full
    const int P, const int M, const int C, const int Nk, const bool full
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int64_t total2 = (int64_t)P * M * (C >> 1);
    int     blocks = (int)((total2 + 255) >> 8);
    if (full)
        abs_init<true><<<blocks, 256, 0, stream>>>(
            (const float2*)d_k_total, (const float2*)d_vsum, d_qsum, d_ksum_raw,
            (float2*)d_out, d_out_u, total2, M, C >> 1, (float)Nk);
    else
        abs_init<false><<<blocks, 256, 0, stream>>>(
            (const float2*)d_k_total, (const float2*)d_vsum, d_qsum, nullptr,
            (float2*)d_out, nullptr, total2, M, C >> 1, (float)Nk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// per-tile sums over the tile's k-range; consumed by later tiles' walks.
// WITH_K: also sum v*k (needed by the abs walks, not by the sgn walks).
// SCAL:   extra warp-uniform scalar per (p,d,tile) — 0: none,
//         1: sum of sorted-key values      (chunk_u, FULL abs walk),
//         2: sum of u[perm] scalar weights (chunk_u, FULL sgn grad_k walk).
template <typename IdxT, int NV, bool WITH_K, int SCAL>
__global__ void abs_chunk(
    const float*  __restrict__ q_sorted,  // shape (P, D, M)
    const float*  __restrict__ k_sorted,  // shape (P, D, N)
    const IdxT*   __restrict__ k_perm,    // shape (P, D, N)
    const float2* __restrict__ v,         // shape (P, N, C/2)
    const float*  __restrict__ u,         // shape (P, N)        SCAL==2 only
          float2* __restrict__ chunk_v,   // shape (P, D, T, C/2)
          float2* __restrict__ chunk_k,   // shape (P, D, T, C/2) WITH_K only
          float*  __restrict__ chunk_u,   // shape (P, D, T)     SCAL>0 only
    const int D, const int C, const int N, const int M,
    const int T, const int Bm
){
    int l = threadIdx.x;                        // lane; group j -> channels 2l+64j, 2l+64j+1
    int d = blockIdx.x * blockDim.y + threadIdx.y;
    int t = blockIdx.y;
    int p = blockIdx.z;
    if (d >= D) return;

    bool on[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) on[j] = 2 * l + 64 * j < C;

    int64_t   pd  = (int64_t)p * D + d;
    int64_t   pdn = pd * N;
    int64_t   pn  = (int64_t)p * N;
    const int C2  = C >> 1;

    const float* kseg  = k_sorted + pdn;
    const IdxT*  kperm = k_perm + pdn;
    const float* qseg  = q_sorted + pd * M;

    int n0 = walk_start(kseg, qseg, t,     Bm, N, M);
    int n1 = walk_start(kseg, qseg, t + 1, Bm, N, M);

    float2 sv[NV], sk[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) {
        sv[j] = make_float2(0.0f, 0.0f);
        sk[j] = make_float2(0.0f, 0.0f);
    }
    float su = 0.0f;                            // SCAL: warp-uniform, redundant in all lanes

    for (int base = n0; base < n1; base += 32) {
        float kv; int kp;
        refill(kseg, kperm, base, n1, l, kv, kp);
        int end = min(n1 - base, 32);
        for (int s = 0; s < end; s++) {
            float         k_s  = __shfl_sync(FULL_MASK, kv, s);
            int           row  = __shfl_sync(FULL_MASK, kp, s);
            const float2* vrow = v + (pn + row) * C2;
            #pragma unroll
            for (int j = 0; j < NV; j++) {
                float2 v_s = on[j] ? vrow[l + 32 * j] : make_float2(0.0f, 0.0f);
                sv[j].x += v_s.x;          sv[j].y += v_s.y;
                if constexpr (WITH_K) {
                    sk[j].x += v_s.x * k_s;    sk[j].y += v_s.y * k_s;
                }
            }
            if constexpr (SCAL == 1) su += k_s;
            if constexpr (SCAL == 2) su += u[pn + row];
        }
    }
    int64_t cb = (pd * T + t) * (int64_t)C2 + l;
    #pragma unroll
    for (int j = 0; j < NV; j++) {
        if (on[j]) {
            chunk_v[cb + 32 * j] = sv[j];
            if constexpr (WITH_K) chunk_k[cb + 32 * j] = sk[j];
        }
    }
    if constexpr (SCAL != 0) {
        if (l == 0) chunk_u[pd * T + t] = su;
    }
}

template <typename IdxT, int NV>
static void launch_abs_chunk_nv(
    const float* d_q_sorted, const float* d_k_sorted, const void* d_k_perm,
    const float* d_v, const float* d_u,
    float* d_chunk_v, float* d_chunk_k, float* d_chunk_u,
    const int P, const int D, const int C, const int N, const int M,
    const int T, const int Bm, const bool with_k, const int scal, cudaStream_t stream
){
    dim3 grid((D + 3) / 4, T - 1, P), block(32, 4, 1);
    auto run = [&](auto kernel) {
        kernel<<<grid, block, 0, stream>>>(
            d_q_sorted, d_k_sorted, (const IdxT*)d_k_perm, (const float2*)d_v, d_u,
            (float2*)d_chunk_v, (float2*)d_chunk_k, d_chunk_u, D, C, N, M, T, Bm);
    };
    if (with_k) {
        if      (scal == 0) run(abs_chunk<IdxT, NV, true, 0>);
        else if (scal == 1) run(abs_chunk<IdxT, NV, true, 1>);
        else                run(abs_chunk<IdxT, NV, true, 2>);
    } else {                                    // sgn walks only need the v sums
        run(abs_chunk<IdxT, NV, false, 0>);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_abs_chunk(
    const float* d_q_sorted,
    const float* d_k_sorted,
    const void*  d_k_perm,
    const float* d_v,
    const float* d_u,          // nullptr unless scal == 2
          float* d_chunk_v,
          float* d_chunk_k,    // nullptr unless with_k
          float* d_chunk_u,    // nullptr unless scal > 0
    const int P, const int D, const int C, const int N, const int M,
    const int T, const int Bm, const bool idx32, const int NV,
    const bool with_k, const int scal
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (idx32) {
        if (NV == 1) launch_abs_chunk_nv<uint32_t, 1>(d_q_sorted, d_k_sorted, d_k_perm, d_v, d_u, d_chunk_v, d_chunk_k, d_chunk_u, P, D, C, N, M, T, Bm, with_k, scal, stream);
        else         launch_abs_chunk_nv<uint32_t, 2>(d_q_sorted, d_k_sorted, d_k_perm, d_v, d_u, d_chunk_v, d_chunk_k, d_chunk_u, P, D, C, N, M, T, Bm, with_k, scal, stream);
    } else {
        if (NV == 1) launch_abs_chunk_nv<uint16_t, 1>(d_q_sorted, d_k_sorted, d_k_perm, d_v, d_u, d_chunk_v, d_chunk_k, d_chunk_u, P, D, C, N, M, T, Bm, with_k, scal, stream);
        else         launch_abs_chunk_nv<uint16_t, 2>(d_q_sorted, d_k_sorted, d_k_perm, d_v, d_u, d_chunk_v, d_chunk_k, d_chunk_u, P, D, C, N, M, T, Bm, with_k, scal, stream);
    }
}


template <typename IdxT, int NV, bool FULL>
__global__ void abs_prefix(
    const float*  __restrict__ q_sorted,  // shape (P, D, M)
    const IdxT*   __restrict__ q_perm,    // shape (P, D, M)
    const float*  __restrict__ k_sorted,  // shape (P, D, N)
    const IdxT*   __restrict__ k_perm,    // shape (P, D, N)
    const float2* __restrict__ v,         // shape (P, N, C/2)
    const float2* __restrict__ chunk_v,   // shape (P, D, T, C/2)
    const float2* __restrict__ chunk_k,   // shape (P, D, T, C/2)
    const float*  __restrict__ chunk_ku,  // shape (P, D, T)  FULL only
          float*  __restrict__ out,       // shape (P, M, C)
          float*  __restrict__ out_u,     // shape (P, M)     FULL only
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

    float2 pref_v[NV], pref_k[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) {
        pref_v[j] = make_float2(0.0f, 0.0f);
        pref_k[j] = make_float2(0.0f, 0.0f);
    }
    float   pref_ku = 0.0f;                     // FULL: warp-uniform sum of walked k values
    int64_t cbase   = pd * T * (int64_t)C2 + l;
    for (int tt = 0; tt < t; tt++) {
        #pragma unroll
        for (int j = 0; j < NV; j++) {
            if (on[j]) {
                float2 cv = chunk_v[cbase + tt * C2 + 32 * j];
                float2 ck = chunk_k[cbase + tt * C2 + 32 * j];
                pref_v[j].x += cv.x;   pref_v[j].y += cv.y;
                pref_k[j].x += ck.x;   pref_k[j].y += ck.y;
            }
        }
        if constexpr (FULL) pref_ku += chunk_ku[pd * T + tt];
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
        const float2* vrow = v + (pn + row) * C2;
        #pragma unroll
        for (int j = 0; j < NV; j++) dst[j] = on[j] ? vrow[l + 32 * j] : make_float2(0.0f, 0.0f);
    };
    auto emit = [&](int q_idx) {
        float* orow = out + (pm + q_idx) * C + 2 * l;
        #pragma unroll
        for (int j = 0; j < NV; j++) {
            if (on[j]) atomic_add2(orow + 64 * j,
                                   2.0f * (pref_v[j].x * q_m - pref_k[j].x),
                                   2.0f * (pref_v[j].y * q_m - pref_k[j].y));
        }
        if constexpr (FULL) {
            if (l == 0) atomicAdd(&out_u[pm + q_idx], 2.0f * ((float)n * q_m - pref_ku));
        }
    };

    // prime k/v for n0 so one v-row load is always in flight
    float  k_n = 0.0f;
    float2 v_nxt[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) v_nxt[j] = make_float2(0.0f, 0.0f);
    if (n0 < n1) {
        k_n     = __shfl_sync(FULL_MASK, kv, 0);
        int row = __shfl_sync(FULL_MASK, kp, 0);
        load_v(row, v_nxt);
    }

    while (n < n1 && m < m1) {
        float  k_cur = k_n;
        float2 v_cur[NV];
        #pragma unroll
        for (int j = 0; j < NV; j++) v_cur[j] = v_nxt[j];

        int nn = n + 1;                         // prefetch next v row before emitting
        if (nn < n1) {
            int idx = nn - kbase;
            if (idx == 32) {
                kbase = nn; idx = 0;
                refill(kseg, kperm, kbase, n1, l, kv, kp);
            }
            k_n     = __shfl_sync(FULL_MASK, kv, idx);
            int row = __shfl_sync(FULL_MASK, kp, idx);
            load_v(row, v_nxt);
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
            pref_v[j].x += v_cur[j].x;              pref_v[j].y += v_cur[j].y;
            pref_k[j].x += v_cur[j].x * k_cur;      pref_k[j].y += v_cur[j].y * k_cur;
        }
        if constexpr (FULL) pref_ku += k_cur;
        n++;
    }

    while (m < m1) {                            // trailing queries, prefix complete (rank == n1)
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
static void launch_abs_prefix_nv(
    const float* d_q_sorted, const void* d_q_perm,
    const float* d_k_sorted, const void* d_k_perm,
    const float* d_v, const float* d_chunk_v, const float* d_chunk_k,
    const float* d_chunk_ku, float* d_out, float* d_out_u,
    const int P, const int D, const int C, const int N, const int M,
    const int T, const int Bm, const bool full, cudaStream_t stream
){
    dim3 grid((D + 3) / 4, T, P), block(32, 4, 1);
    auto run = [&](auto kernel) {
        kernel<<<grid, block, 0, stream>>>(
            d_q_sorted, (const IdxT*)d_q_perm, d_k_sorted, (const IdxT*)d_k_perm,
            (const float2*)d_v, (const float2*)d_chunk_v, (const float2*)d_chunk_k,
            d_chunk_ku, d_out, d_out_u, D, C, N, M, T, Bm);
    };
    if (full) run(abs_prefix<IdxT, NV, true>);
    else      run(abs_prefix<IdxT, NV, false>);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_abs_prefix(
    const float* d_q_sorted,
    const void*  d_q_perm,
    const float* d_k_sorted,
    const void*  d_k_perm,
    const float* d_v,
    const float* d_chunk_v,
    const float* d_chunk_k,
    const float* d_chunk_ku,   // nullptr unless full
          float* d_out,
          float* d_out_u,      // nullptr unless full
    const int P, const int D, const int C, const int N, const int M,
    const int T, const int Bm, const bool idx32, const int NV, const bool full
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (idx32) {
        if (NV == 1) launch_abs_prefix_nv<uint32_t, 1>(d_q_sorted, d_q_perm, d_k_sorted, d_k_perm, d_v, d_chunk_v, d_chunk_k, d_chunk_ku, d_out, d_out_u, P, D, C, N, M, T, Bm, full, stream);
        else         launch_abs_prefix_nv<uint32_t, 2>(d_q_sorted, d_q_perm, d_k_sorted, d_k_perm, d_v, d_chunk_v, d_chunk_k, d_chunk_ku, d_out, d_out_u, P, D, C, N, M, T, Bm, full, stream);
    } else {
        if (NV == 1) launch_abs_prefix_nv<uint16_t, 1>(d_q_sorted, d_q_perm, d_k_sorted, d_k_perm, d_v, d_chunk_v, d_chunk_k, d_chunk_ku, d_out, d_out_u, P, D, C, N, M, T, Bm, full, stream);
        else         launch_abs_prefix_nv<uint16_t, 2>(d_q_sorted, d_q_perm, d_k_sorted, d_k_perm, d_v, d_chunk_v, d_chunk_k, d_chunk_ku, d_out, d_out_u, P, D, C, N, M, T, Bm, full, stream);
    }
}


// Fused path (max(N,M) <= 4096): one block per (p,d) sorts the q/k segments in shared memory (cub BlockRadixSort,
// keys padded with +inf), then runs the chunk pass and the abs_prefix walk in-block; warp w owns q-tile w.
// The sort temp is aliased with the sorted arrays: results stay in registers until both sorts are done.
// Indices are uint16; the launcher returns false when the bucket's smem does not fit the device.

// union region: cub temp during the sorts, the four sorted arrays afterwards
template <int THREADS, int IPT>
static constexpr size_t fused_union_bytes() {
    using Sorter  = cub::BlockRadixSort<float, THREADS, IPT, uint16_t>;
    size_t temp   = sizeof(typename Sorter::TempStorage);
    size_t arrays = (size_t)THREADS * IPT * 12;     // ks,qs (4B) + kp,qp (2B)
    return ((temp > arrays ? temp : arrays) + 15) & ~(size_t)15;
}

// dynamic smem layout — must match fused_union_bytes(): [ temp / ks|qs|kp|qp ] [csv] [csk] [csu]
template <int THREADS, int IPT, int NV, bool FULL>
__global__ void __launch_bounds__(THREADS) abs_fused(
    const float* __restrict__ q,     // (P, D, M)
    const float* __restrict__ k,     // (P, D, N)
    const float* __restrict__ v,     // (P, N, C)
          float* __restrict__ out,   // (P, M, C)
          float* __restrict__ out_u, // (P, M)  FULL only
    const int D, const int C, const int N, const int M
){
    using Sorter = cub::BlockRadixSort<float, THREADS, IPT, uint16_t>;
    constexpr int B     = THREADS * IPT;
    constexpr int WARPS = THREADS / 32;

    extern __shared__ char smem[];
    auto&     temp = *reinterpret_cast<typename Sorter::TempStorage*>(smem);
    float*    ks   = (float*)smem;
    float*    qs   = ks + B;
    uint16_t* kp   = (uint16_t*)(qs + B);
    uint16_t* qp   = kp + B;
    float2*   csv  = (float2*)(smem + fused_union_bytes<THREADS, IPT>());
    float2*   csk  = csv + WARPS * 32 * NV;
    float*    csu  = (float*)(csk + WARPS * 32 * NV);   // FULL only

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int w    = tid >> 5;
    int d    = blockIdx.x;
    int p    = blockIdx.y;

    bool on[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) on[j] = 2 * lane + 64 * j < C;

    int64_t       pd  = (int64_t)p * D + d;
    int64_t       pn  = (int64_t)p * N;
    int64_t       pm  = (int64_t)p * M;
    const int     C2  = C >> 1;
    const float2* vf2 = (const float2*)v;

    {   // sort k then q; pads (+inf keys) land at the end, never walked.
        // Results stay in registers until both sorts released the temp,
        // then the stores overwrite the temp region (aliasing).
        float kkey[IPT]; uint16_t kval[IPT];
        for (int i = 0; i < IPT; i++) {
            int idx = tid * IPT + i;
            kkey[i] = (idx < N) ? k[pd * N + idx] : __int_as_float(0x7f800000);
            kval[i] = (uint16_t)idx;
        }
        Sorter(temp).Sort(kkey, kval);
        __syncthreads();                        // temp reusable for the q sort

        float qkey[IPT]; uint16_t qval[IPT];
        for (int i = 0; i < IPT; i++) {
            int idx = tid * IPT + i;
            qkey[i] = (idx < M) ? q[pd * M + idx] : __int_as_float(0x7f800000);
            qval[i] = (uint16_t)idx;
        }
        Sorter(temp).Sort(qkey, qval);
        __syncthreads();                        // temp done -> arrays may overwrite it

        for (int i = 0; i < IPT; i++) {
            int idx = tid * IPT + i;
            ks[idx] = kkey[i];  kp[idx] = kval[i];
            qs[idx] = qkey[i];  qp[idx] = qval[i];
        }
        __syncthreads();                        // arrays visible
    }

    int Bm = (M + WARPS - 1) / WARPS;
    int m0 = w * Bm;
    int m1 = min(m0 + Bm, M);
    int n0 = walk_start(ks, qs, w,     Bm, N, M);
    int n1 = walk_start(ks, qs, w + 1, Bm, N, M);

    auto load_v = [&](int row, float2* dst) {
        const float2* vrow = vf2 + (pn + row) * C2;
        #pragma unroll
        for (int j = 0; j < NV; j++) dst[j] = on[j] ? vrow[lane + 32 * j] : make_float2(0.0f, 0.0f);
    };

    // block-local chunk pass: per-warp k-range sums of v and k*v
    float2 sv[NV], sk[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) {
        sv[j] = make_float2(0.0f, 0.0f);
        sk[j] = make_float2(0.0f, 0.0f);
    }
    float su = 0.0f;                            // FULL: warp-uniform k-range sum
    for (int base = n0; base < n1; base += 32) {
        float kv; int kpb;
        refill(ks, kp, base, n1, lane, kv, kpb);
        int end = min(n1 - base, 32);
        for (int s = 0; s < end; s++) {
            float         k_s  = __shfl_sync(FULL_MASK, kv, s);
            int           row  = __shfl_sync(FULL_MASK, kpb, s);
            const float2* vrow = vf2 + (pn + row) * C2;
            #pragma unroll
            for (int j = 0; j < NV; j++) {
                float2 v_s = on[j] ? vrow[lane + 32 * j] : make_float2(0.0f, 0.0f);
                sv[j].x += v_s.x;          sv[j].y += v_s.y;
                sk[j].x += v_s.x * k_s;    sk[j].y += v_s.y * k_s;
            }
            if constexpr (FULL) su += k_s;
        }
    }
    #pragma unroll
    for (int j = 0; j < NV; j++) {
        csv[j * (WARPS * 32) + w * 32 + lane] = sv[j];
        csk[j * (WARPS * 32) + w * 32 + lane] = sk[j];
    }
    if constexpr (FULL) {
        if (lane == 0) csu[w] = su;
    }
    __syncthreads();

    if (m0 >= M) return;                        // after the last barrier

    float2 pref_v[NV], pref_k[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) {
        pref_v[j] = make_float2(0.0f, 0.0f);
        pref_k[j] = make_float2(0.0f, 0.0f);
    }
    float pref_ku = 0.0f;
    for (int tt = 0; tt < w; tt++) {            // idle lanes stored zeros
        #pragma unroll
        for (int j = 0; j < NV; j++) {
            float2 cv = csv[j * (WARPS * 32) + tt * 32 + lane];
            float2 ck = csk[j * (WARPS * 32) + tt * 32 + lane];
            pref_v[j].x += cv.x;   pref_v[j].y += cv.y;
            pref_k[j].x += ck.x;   pref_k[j].y += ck.y;
        }
        if constexpr (FULL) pref_ku += csu[tt];
    }

    // walk — same as abs_prefix, streams from smem
    int   kbase = n0, qbase = m0;
    float kv, qv; int kpb, qpb;
    refill(ks, kp, kbase, n1, lane, kv, kpb);
    refill(qs, qp, qbase, m1, lane, qv, qpb);

    int   n   = n0;                             // merge position == rank of the emitted q
    int   m   = m0;
    float q_m = __shfl_sync(FULL_MASK, qv, 0);

    auto emit = [&](int q_idx) {
        float* orow = out + (pm + q_idx) * C + 2 * lane;
        #pragma unroll
        for (int j = 0; j < NV; j++) {
            if (on[j]) atomic_add2(orow + 64 * j,
                                   2.0f * (pref_v[j].x * q_m - pref_k[j].x),
                                   2.0f * (pref_v[j].y * q_m - pref_k[j].y));
        }
        if constexpr (FULL) {
            if (lane == 0) atomicAdd(&out_u[pm + q_idx], 2.0f * ((float)n * q_m - pref_ku));
        }
    };

    float  k_n = 0.0f;
    float2 v_nxt[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) v_nxt[j] = make_float2(0.0f, 0.0f);
    if (n0 < n1) {
        k_n     = __shfl_sync(FULL_MASK, kv, 0);
        int row = __shfl_sync(FULL_MASK, kpb, 0);
        load_v(row, v_nxt);
    }

    while (n < n1 && m < m1) {
        float  k_cur = k_n;
        float2 v_cur[NV];
        #pragma unroll
        for (int j = 0; j < NV; j++) v_cur[j] = v_nxt[j];

        int nn = n + 1;
        if (nn < n1) {
            int idx = nn - kbase;
            if (idx == 32) {
                kbase = nn; idx = 0;
                refill(ks, kp, kbase, n1, lane, kv, kpb);
            }
            k_n     = __shfl_sync(FULL_MASK, kv, idx);
            int row = __shfl_sync(FULL_MASK, kpb, idx);
            load_v(row, v_nxt);
        }

        while (m < m1 && q_m <= k_cur) {
            int q_idx = __shfl_sync(FULL_MASK, qpb, m - qbase);
            emit(q_idx);
            m++;
            if (m < m1) {
                int idx = m - qbase;
                if (idx == 32) {
                    qbase = m; idx = 0;
                    refill(qs, qp, qbase, m1, lane, qv, qpb);
                }
                q_m = __shfl_sync(FULL_MASK, qv, idx);
            }
        }

        #pragma unroll
        for (int j = 0; j < NV; j++) {
            pref_v[j].x += v_cur[j].x;              pref_v[j].y += v_cur[j].y;
            pref_k[j].x += v_cur[j].x * k_cur;      pref_k[j].y += v_cur[j].y * k_cur;
        }
        if constexpr (FULL) pref_ku += k_cur;
        n++;
    }

    while (m < m1) {                            // trailing queries (rank == n1)
        int q_idx = __shfl_sync(FULL_MASK, qpb, m - qbase);
        emit(q_idx);
        m++;
        if (m < m1) {
            int idx = m - qbase;
            if (idx == 32) {
                qbase = m; idx = 0;
                refill(qs, qp, qbase, m1, lane, qv, qpb);
            }
            q_m = __shfl_sync(FULL_MASK, qv, idx);
        }
    }
}

template <int THREADS, int IPT, int NV, bool FULL>
static bool launch_abs_fused_impl(
    const float* d_q, const float* d_k, const float* d_v, float* d_out, float* d_out_u,
    const int P, const int D, const int C, const int N, const int M,
    const size_t max_smem, cudaStream_t stream
){
    constexpr int WARPS = THREADS / 32;
    size_t bytes = fused_union_bytes<THREADS, IPT>()
                 + (size_t)WARPS * 32 * NV * sizeof(float2) * 2
                 + (FULL ? (size_t)WARPS * sizeof(float) : 0);
    if (bytes > max_smem) return false;
    if (bytes > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            abs_fused<THREADS, IPT, NV, FULL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    }
    abs_fused<THREADS, IPT, NV, FULL><<<dim3(D, P), THREADS, bytes, stream>>>(
        d_q, d_k, d_v, d_out, d_out_u, D, C, N, M
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

template <int NV, bool FULL>
static bool launch_abs_fused_nv(
    const float* d_q, const float* d_k, const float* d_v, float* d_out, float* d_out_u,
    const int P, const int D, const int C, const int N, const int M,
    const size_t max_smem, cudaStream_t stream
){
    int mx = N > M ? N : M;
    if      (mx <= 512)  return launch_abs_fused_impl< 512, 1, NV, FULL>(d_q, d_k, d_v, d_out, d_out_u, P, D, C, N, M, max_smem, stream);
    else if (mx <= 1024) return launch_abs_fused_impl< 512, 2, NV, FULL>(d_q, d_k, d_v, d_out, d_out_u, P, D, C, N, M, max_smem, stream);
    else if (mx <= 2048) return launch_abs_fused_impl< 512, 4, NV, FULL>(d_q, d_k, d_v, d_out, d_out_u, P, D, C, N, M, max_smem, stream);
    else                 return launch_abs_fused_impl<1024, 4, NV, FULL>(d_q, d_k, d_v, d_out, d_out_u, P, D, C, N, M, max_smem, stream);
}

// false -> bucket does not fit max_smem, caller must take the global path
bool launch_abs_fused(
    const float* d_q, const float* d_k, const float* d_v, float* d_out, float* d_out_u,
    const int P, const int D, const int C, const int N, const int M,
    const int NV, const bool full, const size_t max_smem
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (NV == 1) {
        if (full) return launch_abs_fused_nv<1, true >(d_q, d_k, d_v, d_out, d_out_u, P, D, C, N, M, max_smem, stream);
        else      return launch_abs_fused_nv<1, false>(d_q, d_k, d_v, d_out, d_out_u, P, D, C, N, M, max_smem, stream);
    } else {
        if (full) return launch_abs_fused_nv<2, true >(d_q, d_k, d_v, d_out, d_out_u, P, D, C, N, M, max_smem, stream);
        else      return launch_abs_fused_nv<2, false>(d_q, d_k, d_v, d_out, d_out_u, P, D, C, N, M, max_smem, stream);
    }
}
