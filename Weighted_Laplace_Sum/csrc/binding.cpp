#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <tuple>

// -------- CUDA declarations (laplace_kernels.cu / sort_utils.cu) --------

void launch_laplace(
    const float* q_sorted, const void* q_perm, const float* k_sorted, const void* k_perm,
    const float* v, float* chunk_f, float* chunk_b, float* chunk_s, float* out, float* out_u,
    const int P, const int D, const int C, const int N, const int M, const int T,
    const bool idx32, const int NV);

int laplace_tiles(const int N);

size_t cub_sort_temp_bytes(const int num_segs, const int seg_len, const bool idx32);

void launch_iota(void* d_out, const int total, const int seg_len, const bool idx32, cudaStream_t stream);

void launch_cub_sort(
    const float* d_in, float* d_sorted, void* d_iota, void* d_perm,
    void* d_temp, size_t temp_bytes,
    const int num_segs, const int seg_len, const bool idx32, cudaStream_t stream);


// sort k and q into (sorted, perm) with one shared CUB temp + iota buffer
struct SortedQK {
    torch::Tensor k_sorted, k_perm, q_sorted, q_perm;
};

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
    launch_cub_sort(k.data_ptr<float>(), s.k_sorted.data_ptr<float>(), iota.data_ptr(), s.k_perm.data_ptr(),
                    temp.data_ptr(), temp_bytes, P * D, N, idx32, stream);

    if (M != N) launch_iota(iota.data_ptr(), P * D * M, M, idx32, stream);
    launch_cub_sort(q.data_ptr<float>(), s.q_sorted.data_ptr<float>(), iota.data_ptr(), s.q_perm.data_ptr(),
                    temp.data_ptr(), temp_bytes, P * D, M, idx32, stream);
    return s;   // temp + iota freed here
}


// out_w[p,m,c] = sum_{d,n} exp(-|k[p,d,n] - q[p,d,m]|) v[p,n,c]
// out_u[p,m]   = sum_{d,n} exp(-|k[p,d,n] - q[p,d,m]|)
std::tuple<torch::Tensor, torch::Tensor> weighted_laplace_sum_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k and v must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k and v must be on one device");
    TORCH_CHECK(q.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32 && v.dtype() == torch::kFloat32, "q, k and v must be float32");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k and v must be 3D");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch size must match");
    TORCH_CHECK(q.size(1) == k.size(1), "q and k must have the same feature dimension");
    TORCH_CHECK(k.size(2) == v.size(1), "k and v must have the same length");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k and v must be contiguous");

    const int64_t P = q.size(0), D = q.size(1), N = k.size(2), C = v.size(2), M = q.size(2);
    TORCH_CHECK(M > 0 && N > 0 && D > 0, "N, M and D must be > 0");
    TORCH_CHECK(C >= 1 && C <= 128, "C (v.size(2)) must be in [1, 128]");
    TORCH_CHECK(N <= 131072 && M <= 131072, "N and M must be <= 131072");
    TORCH_CHECK(P * D * std::max(N, M) <= (int64_t)INT32_MAX, "P*D*max(N,M) must fit int32 (cub sort offsets)");

    const int  NV    = C <= 32 ? 1 : (C <= 64 ? 2 : 4);   // channels per lane
    const bool idx32 = std::max(N, M) > 65536;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int T = laplace_tiles(N);

    auto s = sort_qk(q, k, P, D, N, M, idx32, stream);   // sort scratch is freed before out exists

    auto out     = torch::zeros({P, M, C}, q.options());
    auto out_u   = torch::zeros({P, M}, q.options());
    auto chunk_f = torch::empty({P * D * T * C}, q.options());
    auto chunk_b = torch::empty({P * D * T * C}, q.options());
    auto chunk_s = torch::empty({P * D * T * 4}, q.options());

    launch_laplace(
        s.q_sorted.data_ptr<float>(), s.q_perm.data_ptr(), s.k_sorted.data_ptr<float>(), s.k_perm.data_ptr(),
        v.data_ptr<float>(), chunk_f.data_ptr<float>(), chunk_b.data_ptr<float>(), chunk_s.data_ptr<float>(),
        out.data_ptr<float>(), out_u.data_ptr<float>(), P, D, C, N, M, T, idx32, NV);
    return std::make_tuple(out, out_u);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_laplace_sum_forward", &weighted_laplace_sum_forward, "Weighted Laplace sum forward",
          py::arg("q"), py::arg("k"), py::arg("v"));
}
