#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

// ---------------- WEIGHTED LAPLACE SUM — forward ----------------
// out[p,m,c] = sum_{d,n} exp(-|k[p,d,n] - q[p,d,m]|) * v[p,n,c]
// out_u[p,m] = sum_{d,n} exp(-|k[p,d,n] - q[p,d,m]|)            (the v == 1 sum)
//
// Per (p,d), with keys sorted ascending, the sum over keys below a query is the
// contractive recurrence  L <- exp(-(k_n - k_ref)) L + v_n  (k_ref <- k_n) and
// the query gets exp(-(q - k_ref)) L; the sum over keys above is the mirrored
// suffix recurrence R. Every exponent argument is <= 0, so nothing overflows.
// Ties k == q go to the R side.
//
// The sorted keys of each (p,d) are cut into tiles of BN keys; tile t owns the
// queries q with k[n0-1] < q <= k[n1-1] (n0, n1 its key range). Three kernels:
//   laplace_chunk  per tile: the L recurrence over the tile (relative to its
//                  last key) and the R sum (relative to its first key), v read
//                  once for both.
//   laplace_scan   per (p,d): exclusive prefix (L) and suffix (R) over tiles,
//                  in place, so a tile starts from its own row only.
//   laplace_walk   per tile, one warp: register tiles of K keys (v rows and
//                  keys in registers, lane l owns channels l+32j, j < NV).
//                  Pass A walks the tile backwards to get R at every register
//                  tile start, pass B walks forward, recomputes R per key from
//                  those and emits each query once with both sides summed:
//                  one atomic per (query, d) into out. The v == 1 sums are
//                  warp-uniform scalars, emitted by lane 0.
// Every branch below is warp-uniform; the sorted streams are read 32 at a time
// and broadcast with shuffles.

#define FULL_MASK 0xffffffffu

constexpr int K  = 16;        // keys per register tile
constexpr int J  = 16;        // register tiles per tile
constexpr int BN = K * J;     // keys per tile (chunk/scan granularity)
constexpr int NS = 4;         // scalars per tile: L_u, R_u, ref_f (last key), ref_b (first key)


// first i in [0, n) with a[i] > x
__device__ __forceinline__ int upper_bound(const float* __restrict__ a, const int n, const float x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (a[mid] <= x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ float decay(const float x) { return expf(-x); }   // x >= 0 always

// lane l caches stream element base+l (value + perm), guarded by limit
template <typename IdxT>
__device__ __forceinline__ void refill(const float* __restrict__ vals, const IdxT* __restrict__ perm,
                                       const int base, const int limit, const int lane,
                                       float& val_buf, int& perm_buf) {
    int i = base + lane;
    val_buf  = (i < limit) ? vals[i] : 0.0f;
    perm_buf = (i < limit) ? (int)perm[i] : 0;
}


// ---------------- chunk ----------------

template <typename IdxT, int NV>
__global__ void laplace_chunk(
    const float* __restrict__ k_sorted,   // (P, D, N)
    const IdxT*  __restrict__ k_perm,     // (P, D, N)
    const float* __restrict__ v,          // (P, N, C)
          float* __restrict__ chunk_f,    // (P, D, T, C)
          float* __restrict__ chunk_b,    // (P, D, T, C)
          float* __restrict__ chunk_s,    // (P, D, T, NS)
    const int D, const int C, const int N, const int T
){
    int l = threadIdx.x;
    int d = blockIdx.x * blockDim.y + threadIdx.y;
    int t = blockIdx.y;
    int p = blockIdx.z;
    if (d >= D) return;

    bool on[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) on[j] = l + 32 * j < C;

    int64_t      pd    = (int64_t)p * D + d;
    int64_t      pn    = (int64_t)p * N;
    const float* kseg  = k_sorted + pd * N;
    const IdxT*  kperm = k_perm + pd * N;

    int n0 = t * BN;
    int n1 = min(n0 + BN, N);

    float sf[NV], sb[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) { sf[j] = 0.0f; sb[j] = 0.0f; }
    float sfu = 0.0f, sbu = 0.0f;
    float kfirst = kseg[n0], kref = kfirst;

    for (int base = n0; base < n1; base += 32) {
        float kv; int kp;
        refill(kseg, kperm, base, n1, l, kv, kp);
        int end = min(n1 - base, 32);
        for (int s = 0; s < end; s++) {
            float        k_s  = __shfl_sync(FULL_MASK, kv, s);
            int          row  = __shfl_sync(FULL_MASK, kp, s);
            const float* vrow = v + (pn + row) * C;
            float        df   = decay(k_s - kref);
            float        wb   = decay(k_s - kfirst);
            #pragma unroll
            for (int j = 0; j < NV; j++) {
                float v_s = on[j] ? vrow[l + 32 * j] : 0.0f;
                sf[j] = df * sf[j] + v_s;
                sb[j] += wb * v_s;
            }
            sfu  = df * sfu + 1.0f;
            sbu += wb;
            kref = k_s;
        }
    }

    int64_t ct = pd * T + t;
    #pragma unroll
    for (int j = 0; j < NV; j++) {
        if (on[j]) { chunk_f[ct * C + l + 32 * j] = sf[j]; chunk_b[ct * C + l + 32 * j] = sb[j]; }
    }
    if (l == 0) { chunk_s[ct * NS] = sfu; chunk_s[ct * NS + 1] = sbu; chunk_s[ct * NS + 2] = kref; chunk_s[ct * NS + 3] = kfirst; }
}


// ---------------- scan ----------------
// in place: chunk_f[t] <- L state at the start of tile t (relative to k[n0-1]),
// chunk_b[t] <- R state after tile t (relative to k[n1]); same for the scalars.

template <int NV>
__global__ void laplace_scan(
    float* __restrict__ chunk_f, float* __restrict__ chunk_b, float* __restrict__ chunk_s,
    const int PD, const int C, const int T
){
    int l  = threadIdx.x;
    int pd = blockIdx.x * blockDim.y + threadIdx.y;
    if (pd >= PD) return;

    bool on[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) on[j] = l + 32 * j < C;

    float S[NV], Su = 0.0f, ref = 0.0f;
    #pragma unroll
    for (int j = 0; j < NV; j++) S[j] = 0.0f;
    for (int t = 0; t < T; t++) {
        int64_t ct  = (int64_t)pd * T + t;
        float   cu  = chunk_s[ct * NS];
        float   rf  = chunk_s[ct * NS + 2];
        float   dec = (t > 0) ? decay(rf - ref) : 0.0f;
        #pragma unroll
        for (int j = 0; j < NV; j++) {
            if (on[j]) {
                float c = chunk_f[ct * C + l + 32 * j];
                chunk_f[ct * C + l + 32 * j] = S[j];
                S[j] = dec * S[j] + c;
            }
        }
        if (l == 0) chunk_s[ct * NS] = Su;
        Su  = dec * Su + cu;
        ref = rf;
    }

    Su = 0.0f;
    #pragma unroll
    for (int j = 0; j < NV; j++) S[j] = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        int64_t ct  = (int64_t)pd * T + t;
        float   cu  = chunk_s[ct * NS + 1];
        float   rb  = chunk_s[ct * NS + 3];
        float   dec = (t < T - 1) ? decay(ref - rb) : 0.0f;
        #pragma unroll
        for (int j = 0; j < NV; j++) {
            if (on[j]) {
                float c = chunk_b[ct * C + l + 32 * j];
                chunk_b[ct * C + l + 32 * j] = S[j];
                S[j] = dec * S[j] + c;
            }
        }
        if (l == 0) chunk_s[ct * NS + 1] = Su;
        Su  = dec * Su + cu;
        ref = rb;
    }
}


// ---------------- walk ----------------

// 4-byte async copy global -> shared (sm_80+): no register staging, so a whole
// register tile of v rows is in flight while the warp computes
__device__ __forceinline__ void cp_async4(float* smem, const float* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;" ::: "memory");
}

// a[j] for runtime j from a register array (predicated moves, no local memory)
template <int NV>
__device__ __forceinline__ void select_row(const float (&a)[J][NV], const int j, float (&out)[NV]) {
    #pragma unroll
    for (int jj = 0; jj < J; jj++) {
        #pragma unroll
        for (int c = 0; c < NV; c++) if (j == jj) out[c] = a[jj][c];
    }
}
__device__ __forceinline__ float select_scalar(const float (&a)[J], const int j) {
    float r = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < J; jj++) if (j == jj) r = a[jj];
    return r;
}

template <typename IdxT, int NV>
__global__ void __launch_bounds__(128) laplace_walk(
    const float* __restrict__ q_sorted,   // (P, D, M)
    const IdxT*  __restrict__ q_perm,     // (P, D, M)
    const float* __restrict__ k_sorted,   // (P, D, N)
    const IdxT*  __restrict__ k_perm,     // (P, D, N)
    const float* __restrict__ v,          // (P, N, C)
    const float* __restrict__ chunk_f,    // (P, D, T, C)   scanned
    const float* __restrict__ chunk_b,    // (P, D, T, C)   scanned
    const float* __restrict__ chunk_s,    // (P, D, T, NS)  scanned
          float* __restrict__ out,        // (P, M, C)
          float* __restrict__ out_u,      // (P, M)
    const int D, const int C, const int N, const int M, const int T
){
    __shared__ float vt_s[4][K][NV * 32];   // per warp: the register tile's v rows
    float (*vt)[NV * 32] = vt_s[threadIdx.y];

    int l = threadIdx.x;
    int d = blockIdx.x * blockDim.y + threadIdx.y;
    int t = blockIdx.y;
    int p = blockIdx.z;
    if (d >= D) return;

    int64_t      pd    = (int64_t)p * D + d;
    int64_t      pn    = (int64_t)p * N;
    int64_t      pm    = (int64_t)p * M;
    const float* kseg  = k_sorted + pd * N;
    const IdxT*  kperm = k_perm + pd * N;
    const float* qseg  = q_sorted + pd * M;
    const IdxT*  qperm = q_perm + pd * M;

    int n0  = t * BN;
    int n1  = min(n0 + BN, N);
    int cnt = n1 - n0;
    int nj  = (cnt + K - 1) / K;
    int m0  = (t == 0)     ? 0 : upper_bound(qseg, M, kseg[n0 - 1]);
    int m1  = (t == T - 1) ? M : upper_bound(qseg, M, kseg[n1 - 1]);
    if (m0 >= m1) return;

    bool on[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) on[j] = l + 32 * j < C;

    int64_t ct = pd * T + t;
    float L[NV], R0[NV];
    #pragma unroll
    for (int j = 0; j < NV; j++) {
        L[j]  = on[j] ? chunk_f[ct * C + l + 32 * j] : 0.0f;
        R0[j] = on[j] ? chunk_b[ct * C + l + 32 * j] : 0.0f;
    }
    float Lu    = chunk_s[ct * NS];
    float R0u   = chunk_s[ct * NS + 1];
    // t == 0: L == 0, so kref only has to keep q - kref >= 0 and k - kref >= 0; with the lowest query alone,
    // a key more than ~88.7 below every query gives exp(+x) = inf and inf * 0 = NaN
    float kref  = (t > 0)     ? kseg[n0 - 1] : fminf(qseg[m0], kseg[n0]);
    float knext = (t < T - 1) ? kseg[n1]     : kseg[n1 - 1];   // last tile: R0 == 0, keeps knext - k >= 0

    // register tile j: keys kv (lane r holds key r and kd = exp(-(k_r - k_{r-1}))),
    // rows vt[r][.] (async), row count cj
    float kv, kd; int kp;
    auto load_tile = [&](int j, int& cj) {
        int base = n0 + j * K;
        cj = min(K, n1 - base);
        refill(kseg, kperm, base, n1, l, kv, kp);
        kd = decay(kv - __shfl_up_sync(FULL_MASK, kv, 1));
        #pragma unroll
        for (int r = 0; r < K; r++) {
            if (r < cj) {
                const float* vrow = v + (pn + __shfl_sync(FULL_MASK, kp, r)) * C;
                #pragma unroll
                for (int jj = 0; jj < NV; jj++) if (on[jj]) cp_async4(&vt[r][l + 32 * jj], vrow + l + 32 * jj);
            }
        }
        cp_async_wait();
    };

    // pass A: R at the start of every register tile (relative to its first key)
    float Rb[J][NV], Rbu[J];
    {
        float R[NV], Ru = R0u, kn = knext;
        #pragma unroll
        for (int jj = 0; jj < NV; jj++) R[jj] = R0[jj];
        for (int j = nj - 1; j >= 0; j--) {
            int cj;
            load_tile(j, cj);
            float dn = decay(kn - __shfl_sync(FULL_MASK, kv, cj - 1));
            #pragma unroll
            for (int r = K - 1; r >= 0; r--) {
                if (r < cj) {
                    float dec = (r == cj - 1) ? dn : __shfl_sync(FULL_MASK, kd, r + 1);
                    #pragma unroll
                    for (int jj = 0; jj < NV; jj++) R[jj] = dec * R[jj] + vt[r][l + 32 * jj];
                    Ru = dec * Ru + 1.0f;
                }
            }
            kn = __shfl_sync(FULL_MASK, kv, 0);
            #pragma unroll
            for (int jj = 0; jj < J; jj++) {
                if (j == jj) {
                    #pragma unroll
                    for (int c = 0; c < NV; c++) Rb[jj][c] = R[c];
                    Rbu[jj] = Ru;
                }
            }
            __syncwarp();
        }
    }

    // pass B: forward merge with emits
    int   qbase = m0, m = m0;
    float qv; int qp;
    refill(qseg, qperm, qbase, m1, l, qv, qp);
    float q_m = __shfl_sync(FULL_MASK, qv, 0);
    auto next_q = [&]() {
        m++;
        if (m < m1) {
            int idx = m - qbase;
            if (idx == 32) {
                qbase = m; idx = 0;
                refill(qseg, qperm, qbase, m1, l, qv, qp);
            }
            q_m = __shfl_sync(FULL_MASK, qv, idx);
        }
    };
    auto emit = [&](const float* Rr, float Rru, float k_r) {
        int    q_idx = __shfl_sync(FULL_MASK, qp, m - qbase);
        float  dq    = decay(q_m - kref);
        float  dr    = decay(k_r - q_m);
        float* orow  = out + (pm + q_idx) * C + l;
        #pragma unroll
        for (int jj = 0; jj < NV; jj++) {
            if (on[jj]) atomicAdd(orow + 32 * jj, dq * L[jj] + dr * Rr[jj]);
        }
        if (l == 0) atomicAdd(&out_u[pm + q_idx], dq * Lu + dr * Rru);
    };

    for (int j = 0; j < nj; j++) {
        int cj;
        load_tile(j, cj);

        // R at every key of this register tile, from the next tile's start state
        float R[K][NV], ru_mine = 0.0f;
        {
            bool  has_next = j + 1 < nj;
            float Rn[NV], Rnu, kn;
            select_row<NV>(Rb, j + 1, Rn);
            Rnu = select_scalar(Rbu, j + 1);
            if (!has_next) {
                #pragma unroll
                for (int jj = 0; jj < NV; jj++) Rn[jj] = R0[jj];
                Rnu = R0u;
            }
            kn = has_next ? kseg[n0 + (j + 1) * K] : knext;
            float dn = decay(kn - __shfl_sync(FULL_MASK, kv, cj - 1));
            #pragma unroll
            for (int r = K - 1; r >= 0; r--) {
                if (r < cj) {
                    float dec = (r == cj - 1) ? dn : __shfl_sync(FULL_MASK, kd, r + 1);
                    #pragma unroll
                    for (int jj = 0; jj < NV; jj++) { Rn[jj] = dec * Rn[jj] + vt[r][l + 32 * jj]; R[r][jj] = Rn[jj]; }
                    Rnu = dec * Rnu + 1.0f;
                    if (l == r) ru_mine = Rnu;
                }
            }
        }
        float d0 = decay(__shfl_sync(FULL_MASK, kv, 0) - kref);

        #pragma unroll
        for (int r = 0; r < K; r++) {
            if (r < cj) {
                float k_r = __shfl_sync(FULL_MASK, kv, r);
                float Rru = __shfl_sync(FULL_MASK, ru_mine, r);
                while (m < m1 && q_m <= k_r) {
                    emit(R[r], Rru, k_r);
                    next_q();
                }
                float dec = (r == 0) ? d0 : __shfl_sync(FULL_MASK, kd, r);
                #pragma unroll
                for (int jj = 0; jj < NV; jj++) L[jj] = dec * L[jj] + vt[r][l + 32 * jj];
                Lu   = dec * Lu + 1.0f;
                kref = k_r;
            }
        }
        __syncwarp();
    }

    while (m < m1) {                    // last tile only: queries above every key, R == 0
        emit(R0, 0.0f, q_m);
        next_q();
    }
}


// ---------------- launchers ----------------

template <typename IdxT, int NV>
static void launch_nv(
    const float* q_sorted, const void* q_perm, const float* k_sorted, const void* k_perm,
    const float* v, float* chunk_f, float* chunk_b, float* chunk_s, float* out, float* out_u,
    const int P, const int D, const int C, const int N, const int M, const int T, cudaStream_t stream
){
    dim3 block(32, 4, 1);
    laplace_chunk<IdxT, NV><<<dim3((D + 3) / 4, T, P), block, 0, stream>>>(
        k_sorted, (const IdxT*)k_perm, v, chunk_f, chunk_b, chunk_s, D, C, N, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    laplace_scan<NV><<<dim3((P * D + 3) / 4), block, 0, stream>>>(chunk_f, chunk_b, chunk_s, P * D, C, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    laplace_walk<IdxT, NV><<<dim3((D + 3) / 4, T, P), block, 0, stream>>>(
        q_sorted, (const IdxT*)q_perm, k_sorted, (const IdxT*)k_perm, v,
        chunk_f, chunk_b, chunk_s, out, out_u, D, C, N, M, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int laplace_tiles(const int N) { return (N + BN - 1) / BN; }

void launch_laplace(
    const float* q_sorted, const void* q_perm, const float* k_sorted, const void* k_perm,
    const float* v, float* chunk_f, float* chunk_b, float* chunk_s, float* out, float* out_u,
    const int P, const int D, const int C, const int N, const int M, const int T,
    const bool idx32, const int NV
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto run = [&](auto fn) { fn(q_sorted, q_perm, k_sorted, k_perm, v, chunk_f, chunk_b, chunk_s, out, out_u, P, D, C, N, M, T, stream); };
    if (idx32) {
        if      (NV == 1) run(launch_nv<uint32_t, 1>);
        else if (NV == 2) run(launch_nv<uint32_t, 2>);
        else              run(launch_nv<uint32_t, 4>);
    } else {
        if      (NV == 1) run(launch_nv<uint16_t, 1>);
        else if (NV == 2) run(launch_nv<uint16_t, 2>);
        else              run(launch_nv<uint16_t, 4>);
    }
}
