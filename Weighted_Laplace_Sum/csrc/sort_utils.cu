#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cub/cub.cuh>



// ---------------- CUB SEGMENTED SORT ----------------
// Permutation values templated on IdxT: uint16 for seg_len <= 65536 (25% less
// sort traffic than int32), uint32 above (up to 128k+; binding.cpp dispatches).
// Custom policy: 8-bit digits -> 4 radix passes over the 32-bit keys instead of
// cub's segmented default (6-bit -> 6 passes). RADIX_RANK_MATCH keeps smem small.

struct StrideBy {
    int N;
    __host__ __device__ int operator()(int i) const { return i * N; }
};

using OffsetIter = cub::TransformInputIterator<int, StrideBy, cub::CountingInputIterator<int>>;

// Tile (128*IPT) should cover the segment or divide it evenly; 128 threads keeps the
// upsweep smem counters at 2^RADIX_BITS * BLOCK_THREADS = 32 KB (256 threads -> 64 KB, too big).
template <int IPT>
struct SegSortPolicy {
    struct Policy : cub::ChainedPolicy<900, Policy, Policy> {   // self-reference ends the chain -> applies to every arch
        using SegmentedPolicy    = cub::AgentRadixSortDownsweepPolicy<
            128, IPT, float,
            cub::BLOCK_LOAD_WARP_TRANSPOSE, cub::LOAD_DEFAULT,
            cub::RADIX_RANK_MATCH, cub::BLOCK_SCAN_WARP_SCANS, 8>;
        using AltSegmentedPolicy = cub::AgentRadixSortDownsweepPolicy<   // unused (32 % 8 == 0)
            128, IPT, float,
            cub::BLOCK_LOAD_WARP_TRANSPOSE, cub::LOAD_DEFAULT,
            cub::RADIX_RANK_MATCH, cub::BLOCK_SCAN_WARP_SCANS, 7>;
    };
    using MaxPolicy = Policy;
};

template <int IPT, typename IdxT>
static cudaError_t seg_sort_with(
    void* d_temp, size_t& temp_bytes,
    const float* d_in, float* d_sorted, IdxT* d_iota, IdxT* d_perm,
    const int num_segs, const int seg_len,
    OffsetIter beg, OffsetIter end, cudaStream_t stream
) {
    cub::DoubleBuffer<float> keys(const_cast<float*>(d_in), d_sorted);
    cub::DoubleBuffer<IdxT>  vals(d_iota, d_perm);

    return cub::DispatchSegmentedRadixSort<
        false, float, IdxT, OffsetIter, OffsetIter, int, SegSortPolicy<IPT>>::Dispatch(
        d_temp, temp_bytes, keys, vals,
        num_segs * seg_len, num_segs, beg, end,
        0, 32, false, stream                        // overwrite not allowed -> result in d_sorted/d_perm
    );
}

// 8-bit 4-pass policies with tile matched to seg_len; cub default (6-bit) for long
// segments where fine digits hurt scatter coalescing more than 2 saved passes gain.
template <typename IdxT>
static cudaError_t seg_sort(
    void* d_temp, size_t& temp_bytes,
    const float* d_in, float* d_sorted, IdxT* d_iota, IdxT* d_perm,
    const int num_segs, const int seg_len, cudaStream_t stream
) {
    OffsetIter beg(cub::CountingInputIterator<int>(0), StrideBy{seg_len});
    OffsetIter end(cub::CountingInputIterator<int>(1), StrideBy{seg_len});

    if (seg_len <= 512)
        return seg_sort_with<4> (d_temp, temp_bytes, d_in, d_sorted, d_iota, d_perm, num_segs, seg_len, beg, end, stream);
    if (seg_len <= 1024)
        return seg_sort_with<8> (d_temp, temp_bytes, d_in, d_sorted, d_iota, d_perm, num_segs, seg_len, beg, end, stream);
    if (seg_len <= 2048)
        return seg_sort_with<16>(d_temp, temp_bytes, d_in, d_sorted, d_iota, d_perm, num_segs, seg_len, beg, end, stream);
    if (seg_len <= 16384)
        return seg_sort_with<32>(d_temp, temp_bytes, d_in, d_sorted, d_iota, d_perm, num_segs, seg_len, beg, end, stream);

    return cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp, temp_bytes,
        d_in, d_sorted, d_iota, d_perm,
        num_segs * seg_len, num_segs, beg, end, 0, 32, stream
    );
}

// Temp bytes CUB needs to sort (num_segs) segments of length seg_len.
size_t cub_sort_temp_bytes(const int num_segs, const int seg_len, const bool idx32) {
    size_t temp_bytes = 0;
    if (idx32) seg_sort<uint32_t>(nullptr, temp_bytes, nullptr, nullptr, nullptr, nullptr, num_segs, seg_len, 0);
    else       seg_sort<uint16_t>(nullptr, temp_bytes, nullptr, nullptr, nullptr, nullptr, num_segs, seg_len, 0);
    return temp_bytes;
}

// Sort (num_segs) contiguous segments of length seg_len.
// Keys: float32 ascending. Values: uint16/uint32 permutation indices (iota -> perm).
// d_temp must hold at least cub_sort_temp_bytes(num_segs, seg_len, idx32).
void launch_cub_sort(
    const float* d_in,
    float*       d_sorted,
    void*        d_iota,
    void*        d_perm,
    void*        d_temp,
    size_t       temp_bytes,
    const int    num_segs,
    const int    seg_len,
    const bool   idx32,
    cudaStream_t stream
) {
    if (idx32) seg_sort<uint32_t>(d_temp, temp_bytes, d_in, d_sorted, (uint32_t*)d_iota, (uint32_t*)d_perm, num_segs, seg_len, stream);
    else       seg_sort<uint16_t>(d_temp, temp_bytes, d_in, d_sorted, (uint16_t*)d_iota, (uint16_t*)d_perm, num_segs, seg_len, stream);
}



// ---------------- IOTA ----------------

// out[i] = i % seg_len — permutation values input for the segmented sort.
template <typename IdxT>
__global__ void iota_seg(IdxT* __restrict__ out, const int total, const int seg_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = (IdxT)(i % seg_len);
}

void launch_iota(void* d_out, const int total, const int seg_len, const bool idx32, cudaStream_t stream) {
    int blocks = (total + 255) / 256;
    if (idx32) iota_seg<uint32_t><<<blocks, 256, 0, stream>>>((uint32_t*)d_out, total, seg_len);
    else       iota_seg<uint16_t><<<blocks, 256, 0, stream>>>((uint16_t*)d_out, total, seg_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
