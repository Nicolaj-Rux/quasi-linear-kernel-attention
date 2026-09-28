
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <pybind11/pybind11.h>
namespace py = pybind11;


void launch_abs_init(
    const float* d_k_total, const float* d_vsum, const float* d_qsum,
    const float* d_ksum_raw, float* d_out, float* d_out_u,
    const int P, const int M, const int C, const int Nk, const bool full);

void launch_abs_chunk(
    const float* d_q_sorted, const float* d_k_sorted, const void* d_k_perm,
    const float* d_v, const float* d_u,
    float* d_chunk_v, float* d_chunk_k, float* d_chunk_u,
    const int P, const int D, const int C, const int N, const int M,
    const int T, const int Bm, const bool idx32, const int NV,
    const bool with_k, const int scal);

void launch_abs_prefix(
    const float* d_q_sorted, const void* d_q_perm,
    const float* d_k_sorted, const void* d_k_perm,
    const float* d_v, const float* d_chunk_v, const float* d_chunk_k,
    const float* d_chunk_ku, float* d_out, float* d_out_u,
    const int P, const int D, const int C, const int N, const int M,
    const int T, const int Bm, const bool idx32, const int NV, const bool full);

bool launch_abs_fused(   // false -> bucket does not fit max_smem, use the global path
    const float* d_q, const float* d_k, const float* d_v, float* d_out, float* d_out_u,
    const int P, const int D, const int C, const int N, const int M,
    const int NV, const bool full, const size_t max_smem);

void launch_sgn_prefix(
    const float* d_q_sorted, const void* d_q_perm,
    const float* d_k_sorted, const void* d_k_perm,
    const float* d_w, const float* d_w_total, const float* d_g, const float* d_chunk_w,
    const float* d_uq, const float* d_us, const float* d_us_total, const float* d_chunk_us,
    float* d_out,
    const int P, const int D, const int C, const int N, const int M,
    const int T, const int Bm, const bool idx32, const int NV, const int fmode);

size_t cub_sort_temp_bytes(const int num_segs, const int seg_len, const bool idx32);

void launch_iota(void* d_out, const int total, const int seg_len, const bool idx32, cudaStream_t stream);

void launch_cub_sort(
    const float* d_in, float* d_sorted, void* d_iota, void* d_perm,
    void* d_temp, size_t temp_bytes,
    const int num_segs, const int seg_len, const bool idx32, cudaStream_t stream);


// T q-tiles per (p,d) walk from device properties: T_occ fills 3/4 of the resident-warp capacity at small P*D
// (cap 64, the chunk-prefix reads grow O(T^2)); T_l2 keeps the active-p window of v+out inside L2 at large N (cap 256).
static int walk_tiles(const int PD, const int C, const int mx) {
    const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    const int     warp_cap = prop->multiProcessorCount * (prop->maxThreadsPerMultiProcessor / 32);
    const int64_t l2       = prop->l2CacheSize > 0 ? (int64_t)prop->l2CacheSize : (int64_t)(48 << 20);
    const int     T_occ    = (3 * warp_cap / 4 + PD - 1) / PD;
    const int     T_l2     = (int)(((int64_t)mx * C * 1536 + l2 - 1) / l2);
    return std::max(1, std::max(std::min(T_occ, 64), std::min(T_l2, 256)));
}


static void check_qkv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, and v must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, and v must be on the same CUDA device");
    TORCH_CHECK(q.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32 && v.dtype() == torch::kFloat32, "q, k, and v must be float32");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, and v must be 3D tensors");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "Batch size of q, k, and v must match");
    TORCH_CHECK(q.size(1) == k.size(1), "q and k must have same feature dimension");
    TORCH_CHECK(k.size(2) == v.size(1), "k and v must have same length");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k, and v must be contiguous");

    const int64_t P = q.size(0), D = q.size(1), N = k.size(2), C = v.size(2), M = q.size(2);
    TORCH_CHECK(M > 0 && N > 0 && D > 0, "N, M and D must be > 0");
    TORCH_CHECK(C >= 1 && C <= 128, "C (v.size(2)) must be in [1, 128]");
    TORCH_CHECK(N <= 131072 && M <= 131072, "N and M must be <= 131072");
    TORCH_CHECK(P * D * std::max(N, M) <= (int64_t)INT32_MAX, "P*D*max(N,M) must fit int32 (cub sort offsets)");
}

// pad the channel dim with one zero column when C is odd (kernels see even C only)
static torch::Tensor pad_even_c(const torch::Tensor& x, const int C) {
    return (C & 1) ? at::constant_pad_nd(x, {0, 1}) : x;
}

struct SortedQK {
    torch::Tensor k_sorted, k_perm, q_sorted, q_perm;
};

// sort k and q into (sorted, perm) with one shared CUB temp + iota buffer from the torch caching allocator
static SortedQK sort_qk(
    const torch::Tensor& q, const torch::Tensor& k,
    const int P, const int D, const int N, const int M,
    const bool idx32, cudaStream_t stream)
{
    auto idx = q.options().dtype(idx32 ? torch::kUInt32 : torch::kUInt16);

    SortedQK s;
    s.k_sorted = torch::empty_like(k);
    s.k_perm   = torch::empty_like(k, idx);
    s.q_sorted = torch::empty_like(q);
    s.q_perm   = torch::empty_like(q, idx);

    size_t temp_bytes = std::max(cub_sort_temp_bytes(P * D, N, idx32), cub_sort_temp_bytes(P * D, M, idx32));
    auto   temp       = torch::empty({(int64_t)std::max<size_t>(temp_bytes, 1)}, q.options().dtype(torch::kUInt8));
    auto   iota       = torch::empty({(int64_t)P * D * std::max(N, M)}, idx);

    launch_iota(iota.data_ptr(), P * D * N, N, idx32, stream);
    launch_cub_sort(
        k.data_ptr<float>(), s.k_sorted.data_ptr<float>(),
        iota.data_ptr(), s.k_perm.data_ptr(),
        temp.data_ptr(), temp_bytes, P * D, N, idx32, stream
    );

    if (M != N) launch_iota(iota.data_ptr(), P * D * M, M, idx32, stream);
    launch_cub_sort(
        q.data_ptr<float>(), s.q_sorted.data_ptr<float>(),
        iota.data_ptr(), s.q_perm.data_ptr(),
        temp.data_ptr(), temp_bytes, P * D, M, idx32, stream
    );
    return s;   // temp + iota freed here
}


// Forward: out = abs_init (rank-1 correction) + abs_chunk/abs_prefix (sorted walk), or the fused kernel for
// max(N,M) <= 4096; full also returns out_u (P,M). path: -1 auto, 0 global, 1 fused (raises if unavailable).
static std::tuple<torch::Tensor, torch::Tensor> abs_forward_impl(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const bool full, const int path)
{
    check_qkv(q, k, v);
    TORCH_CHECK(path >= -1 && path <= 1, "path must be -1 (auto), 0 (global) or 1 (fused)");

    const int P = q.size(0);
    const int D = q.size(1);
    const int N = k.size(2);
    const int C = v.size(2);
    const int M = q.size(2);

    torch::Tensor vp = pad_even_c(v, C);
    const int     Ce = vp.size(2);

    const int  NV    = Ce <= 64 ? 1 : 2;             // float2 channel groups per lane
    const bool idx32 = std::max(N, M) > 65536;       // uint16 perms fit, else uint32

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // v_total is d-independent, so sum_d q[p,d,m]*v_total[p,d,c] = qsum[p,m]*vsum[p,c]
    auto vsum    = vp.sum(1);                                          // (P, Ce)
    auto qsum    = q.sum(1);                                           // (P, M)
    auto k_total = torch::bmm(k.sum(1).unsqueeze(1), vp).squeeze(1);   // (P, Ce)

    torch::Tensor ksum_raw;
    if (full) ksum_raw = k.sum({1, 2});                                // (P)

    // allocated after the sort scratch (CUB temp + iota) is freed, lowering the peak by one out-size
    torch::Tensor out, out_u;
    auto init_out = [&] {
        out = torch::empty({P, M, Ce}, q.options());
        if (full) out_u = torch::empty({P, M}, q.options());
        launch_abs_init(
            k_total.data_ptr<float>(), vsum.data_ptr<float>(), qsum.data_ptr<float>(),
            full ? ksum_raw.data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            full ? out_u.data_ptr<float>() : nullptr,
            P, M, Ce, N, full
        );
    };

    // fused path; falls through to the global path when the bucket's smem does not fit the device
    bool fused = false;
    if (path != 0 && std::max(N, M) <= 4096) {
        const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
        init_out();
        fused = launch_abs_fused(
            q.data_ptr<float>(), k.data_ptr<float>(), vp.data_ptr<float>(),
            out.data_ptr<float>(), full ? out_u.data_ptr<float>() : nullptr,
            P, D, Ce, N, M, NV, full, prop->sharedMemPerBlockOptin
        );
    }
    TORCH_CHECK(!(path == 1 && !fused),
                "fused path pinned but unavailable: needs max(N,M) <= 4096 and the "
                "bucket's shared memory to fit this device");

    if (!fused) {
        const int T  = walk_tiles(P * D, Ce, std::max(N, M));
        const int Bm = (M + T - 1) / T;

        auto s = sort_qk(q, k, P, D, N, M, idx32, stream);
        if (!out.defined()) init_out();

        auto chunk_v  = torch::empty({(int64_t)P * D * T * Ce}, q.options());
        auto chunk_k  = torch::empty({(int64_t)P * D * T * Ce}, q.options());
        torch::Tensor chunk_ku;
        if (full) chunk_ku = torch::empty({(int64_t)P * D * T}, q.options());

        if (T > 1) {
            launch_abs_chunk(
                s.q_sorted.data_ptr<float>(),
                s.k_sorted.data_ptr<float>(), s.k_perm.data_ptr(),
                vp.data_ptr<float>(), nullptr,
                chunk_v.data_ptr<float>(), chunk_k.data_ptr<float>(),
                full ? chunk_ku.data_ptr<float>() : nullptr,
                P, D, Ce, N, M, T, Bm, idx32, NV,
                /*with_k=*/true, /*scal=*/full ? 1 : 0
            );
        }
        launch_abs_prefix(
            s.q_sorted.data_ptr<float>(), s.q_perm.data_ptr(),
            s.k_sorted.data_ptr<float>(), s.k_perm.data_ptr(),
            vp.data_ptr<float>(),
            chunk_v.data_ptr<float>(), chunk_k.data_ptr<float>(),
            full ? chunk_ku.data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            full ? out_u.data_ptr<float>() : nullptr,
            P, D, Ce, N, M, T, Bm, idx32, NV, full
        );
    }

    if (C & 1) out = out.narrow(2, 0, C).contiguous();
    return std::make_tuple(out, out_u);
}

torch::Tensor weighted_abs_sum_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, int64_t path)
{
    return std::get<0>(abs_forward_impl(q, k, v, /*full=*/false, (int)path));
}

std::tuple<torch::Tensor, torch::Tensor> weighted_abs_sum_full_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, int64_t path)
{
    return abs_forward_impl(q, k, v, /*full=*/true, (int)path);
}


// Backward, one sort pass for all three:
// grad_v = abs walk with roles swapped (queries=k, stream=q, weights=g_w)
// grad_q = sgn walk (queries=q, stream=k, w=v,   g=g_w)   [+ g_u rank term, FMODE 1]
// grad_k = sgn walk (queries=k, stream=q, w=g_w, g=v)     [+ g_u prefix term, FMODE 2]
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> abs_backward_impl(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& g_w, const torch::Tensor& g_u, const bool full)
{
    check_qkv(q, k, v);
    TORCH_CHECK(g_w.is_cuda() && g_w.device() == q.device(), "g_w must be on q's CUDA device");
    TORCH_CHECK(g_w.dtype() == torch::kFloat32, "g_w must be float32");
    TORCH_CHECK(g_w.dim() == 3, "g_w must be 3D");
    TORCH_CHECK(g_w.size(0) == q.size(0) && g_w.size(1) == q.size(2) && g_w.size(2) == v.size(2),
                "g_w must have shape (P, M, C)");
    TORCH_CHECK(g_w.is_contiguous(), "g_w must be contiguous");
    if (full) {
        TORCH_CHECK(g_u.is_cuda() && g_u.device() == q.device(), "g_u must be on q's CUDA device");
        TORCH_CHECK(g_u.dtype() == torch::kFloat32, "g_u must be float32");
        TORCH_CHECK(g_u.dim() == 2 && g_u.size(0) == q.size(0) && g_u.size(1) == q.size(2),
                    "g_u must have shape (P, M)");
        TORCH_CHECK(g_u.is_contiguous(), "g_u must be contiguous");
    }

    const int P = q.size(0);
    const int D = q.size(1);
    const int N = k.size(2);
    const int C = v.size(2);
    const int M = q.size(2);

    torch::Tensor vp = pad_even_c(v, C);
    torch::Tensor gp = pad_even_c(g_w, C);
    const int     Ce = vp.size(2);

    const int  NV    = Ce <= 64 ? 1 : 2;
    const bool idx32 = std::max(N, M) > 65536;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int T    = walk_tiles(P * D, Ce, std::max(N, M));
    const int Bm_q = (M + T - 1) / T;   // q-tiles (grad_q walk)
    const int Bm_k = (N + T - 1) / T;   // k-tiles (grad_k / grad_v walks)

    auto s = sort_qk(q, k, P, D, N, M, idx32, stream);

    auto vsum  = vp.sum(1).contiguous();    // (P, Ce) w_total of the grad_q walk
    auto gwsum = gp.sum(1).contiguous();    // (P, Ce) w_total of the grad_k walk + grad_v init
    torch::Tensor gu_total;
    if (full) gu_total = g_u.sum(1).contiguous();   // (P)

    auto grad_q = torch::empty({P, D, M}, q.options());
    auto grad_k = torch::empty({P, D, N}, q.options());
    auto grad_v = torch::empty({P, N, Ce}, q.options());

    {   // grad_q: chunk of v over k-ranges (per q-tile), then the sgn walk
        auto chunk_v = torch::empty({(int64_t)P * D * T * Ce}, q.options());
        if (T > 1) {
            launch_abs_chunk(
                s.q_sorted.data_ptr<float>(),
                s.k_sorted.data_ptr<float>(), s.k_perm.data_ptr(),
                vp.data_ptr<float>(), nullptr,
                chunk_v.data_ptr<float>(), nullptr, nullptr,
                P, D, Ce, N, M, T, Bm_q, idx32, NV,
                /*with_k=*/false, /*scal=*/0
            );
        }
        launch_sgn_prefix(
            s.q_sorted.data_ptr<float>(), s.q_perm.data_ptr(),
            s.k_sorted.data_ptr<float>(), s.k_perm.data_ptr(),
            vp.data_ptr<float>(), vsum.data_ptr<float>(), gp.data_ptr<float>(),
            chunk_v.data_ptr<float>(),
            full ? g_u.data_ptr<float>() : nullptr, nullptr, nullptr, nullptr,
            grad_q.data_ptr<float>(),
            P, D, Ce, N, M, T, Bm_q, idx32, NV, /*fmode=*/full ? 1 : 0
        );
    }

    {   // grad_k + grad_v share the swapped chunk pass (g_w over q-ranges, per k-tile)
        auto chunk_gw  = torch::empty({(int64_t)P * D * T * Ce}, q.options());
        auto chunk_gwq = torch::empty({(int64_t)P * D * T * Ce}, q.options());
        torch::Tensor chunk_gu;
        if (full) chunk_gu = torch::empty({(int64_t)P * D * T}, q.options());

        if (T > 1) {
            launch_abs_chunk(
                s.k_sorted.data_ptr<float>(),
                s.q_sorted.data_ptr<float>(), s.q_perm.data_ptr(),
                gp.data_ptr<float>(), full ? g_u.data_ptr<float>() : nullptr,
                chunk_gw.data_ptr<float>(), chunk_gwq.data_ptr<float>(),
                full ? chunk_gu.data_ptr<float>() : nullptr,
                P, D, Ce, /*N=*/M, /*M=*/N, T, Bm_k, idx32, NV,
                /*with_k=*/true, /*scal=*/full ? 2 : 0
            );
        }
        launch_sgn_prefix(
            s.k_sorted.data_ptr<float>(), s.k_perm.data_ptr(),
            s.q_sorted.data_ptr<float>(), s.q_perm.data_ptr(),
            gp.data_ptr<float>(), gwsum.data_ptr<float>(), vp.data_ptr<float>(),
            chunk_gw.data_ptr<float>(),
            nullptr,
            full ? g_u.data_ptr<float>() : nullptr,
            full ? gu_total.data_ptr<float>() : nullptr,
            full ? chunk_gu.data_ptr<float>() : nullptr,
            grad_k.data_ptr<float>(),
            P, D, Ce, /*N=*/M, /*M=*/N, T, Bm_k, idx32, NV, /*fmode=*/full ? 2 : 0
        );

        // grad_v: rank-1 correction as initial value, then the swapped abs walk
        auto q_total = torch::bmm(q.sum(1).unsqueeze(1), gp).squeeze(1);   // (P, Ce)
        auto ksum_d  = k.sum(1).contiguous();                              // (P, N)
        launch_abs_init(
            q_total.data_ptr<float>(), gwsum.data_ptr<float>(), ksum_d.data_ptr<float>(),
            nullptr, grad_v.data_ptr<float>(), nullptr,
            P, /*M=*/N, Ce, /*Nk=*/M, /*full=*/false
        );
        launch_abs_prefix(
            s.k_sorted.data_ptr<float>(), s.k_perm.data_ptr(),
            s.q_sorted.data_ptr<float>(), s.q_perm.data_ptr(),
            gp.data_ptr<float>(),
            chunk_gw.data_ptr<float>(), chunk_gwq.data_ptr<float>(), nullptr,
            grad_v.data_ptr<float>(), nullptr,
            P, D, Ce, /*N=*/M, /*M=*/N, T, Bm_k, idx32, NV, /*full=*/false
        );
    }

    if (C & 1) grad_v = grad_v.narrow(2, 0, C).contiguous();
    return std::make_tuple(grad_q, grad_k, grad_v);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> weighted_abs_sum_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g_w)
{
    return abs_backward_impl(q, k, v, g_w, torch::Tensor(), /*full=*/false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> weighted_abs_sum_full_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g_w, torch::Tensor g_u)
{
    return abs_backward_impl(q, k, v, g_w, g_u, /*full=*/true);
}


// out[p,d,m] = sum_n sgn(k[p,d,n] - q[p,d,m]) * dot(v[p,n,:], g[p,m,:]) = the negated grad_q walk
torch::Tensor weighted_sgn_sum_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g)
{
    check_qkv(q, k, v);
    TORCH_CHECK(g.is_cuda() && g.device() == q.device(), "g must be on q's CUDA device");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be float32");
    TORCH_CHECK(g.dim() == 3, "g must be 3D");
    TORCH_CHECK(g.size(0) == q.size(0) && g.size(1) == q.size(2) && g.size(2) == v.size(2),
                "g must have shape (P, M, C)");
    TORCH_CHECK(g.is_contiguous(), "g must be contiguous");

    const int P = q.size(0);
    const int D = q.size(1);
    const int N = k.size(2);
    const int C = v.size(2);
    const int M = q.size(2);

    torch::Tensor vp = pad_even_c(v, C);
    torch::Tensor gpad = pad_even_c(g, C);
    const int     Ce = vp.size(2);

    const int  NV    = Ce <= 64 ? 1 : 2;
    const bool idx32 = std::max(N, M) > 65536;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int T  = walk_tiles(P * D, Ce, std::max(N, M));
    const int Bm = (M + T - 1) / T;

    auto s    = sort_qk(q, k, P, D, N, M, idx32, stream);
    auto vsum = vp.sum(1).contiguous();     // (P, Ce)

    auto out     = torch::empty({P, D, M}, q.options());
    auto chunk_v = torch::empty({(int64_t)P * D * T * Ce}, q.options());
    if (T > 1) {
        launch_abs_chunk(
            s.q_sorted.data_ptr<float>(),
            s.k_sorted.data_ptr<float>(), s.k_perm.data_ptr(),
            vp.data_ptr<float>(), nullptr,
            chunk_v.data_ptr<float>(), nullptr, nullptr,
            P, D, Ce, N, M, T, Bm, idx32, NV,
            /*with_k=*/false, /*scal=*/0
        );
    }
    launch_sgn_prefix(
        s.q_sorted.data_ptr<float>(), s.q_perm.data_ptr(),
        s.k_sorted.data_ptr<float>(), s.k_perm.data_ptr(),
        vp.data_ptr<float>(), vsum.data_ptr<float>(), gpad.data_ptr<float>(),
        chunk_v.data_ptr<float>(),
        nullptr, nullptr, nullptr, nullptr,
        out.data_ptr<float>(),
        P, D, Ce, N, M, T, Bm, idx32, NV, /*fmode=*/0
    );

    return out.neg_();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_abs_sum_forward",       &weighted_abs_sum_forward,       "Weighted abs sum forward",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("path") = -1);
    m.def("weighted_abs_sum_full_forward",  &weighted_abs_sum_full_forward,  "Weighted abs sum full forward (out_w, out_u)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("path") = -1);
    m.def("weighted_abs_sum_backward",      &weighted_abs_sum_backward,      "Weighted abs sum backward",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g_w"));
    m.def("weighted_abs_sum_full_backward", &weighted_abs_sum_full_backward, "Weighted abs sum full backward",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g_w"), py::arg("g_u"));
    m.def("weighted_sgn_sum_forward",       &weighted_sgn_sum_forward,       "Weighted sgn sum forward",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"));
}
