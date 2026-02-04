#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/macros/Macros.h>

#include "base/base_cuda.cuh"
#include "torch_extensions/inplace_topk/inplace_topk_cuda_kernels.h"

#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void inplace_topk_cuda_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> top_values,
    torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> top_indices,
    torch::PackedTensorAccessor64<int, 1, torch::RestrictPtrTraits> min_top_indices,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> sums,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> square_sums,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> candidate_values,
    const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> candidate_indices
)
{
    size_t i;  // Element index
    
    size_t K = top_values.size(1);
    size_t N = candidate_values.size(1);
    scalar_t sums_(0);
    scalar_t square_sums_(0);

    // Iterate over each batch
    if (thread_index_expand(top_values.size(0), i))
    {
        int min_top_index = min_top_indices[i]; // Use external min index for this batch

        for (size_t n = 0; n < N; ++n) 
        {
            scalar_t candidate_value = candidate_values[i][n];
            long candidate_index = candidate_indices[n];

            sums_ += candidate_value;
            square_sums_ += candidate_value * candidate_value;

            // If the candidate is larger than the smallest top value, replace it
            if (candidate_value > top_values[i][min_top_index]) 
            {
                top_values[i][min_top_index] = candidate_value;
                top_indices[i][min_top_index] = candidate_index;

                // Update min_top_index by scanning for the new minimum
                min_top_index = 0;
                for (size_t k = 1; k < K; ++k) 
                    if (top_values[i][k] < top_values[i][min_top_index])
                        min_top_index = k;
            }
        }
        min_top_indices[i] = min_top_index;
        sums[i] += sums_;
        square_sums[i] += square_sums_;
    }
}


void inplace_topk_cuda(
    torch::Tensor top_values,
    torch::Tensor top_indices,
    torch::Tensor min_top_indices,
    torch::Tensor sums,
    torch::Tensor square_sums,
    torch::Tensor candidate_values,
    torch::Tensor candidate_indices
)
{
    CHECK_CUDA_INPUT(top_values);
    CHECK_CUDA_INPUT(top_indices);
    CHECK_CUDA_INPUT(min_top_indices);
    CHECK_CUDA_INPUT(sums);
    CHECK_CUDA_INPUT(square_sums);
    CHECK_CUDA_INPUT(candidate_values);
    CHECK_CUDA_INPUT(candidate_indices);

    const int deviceId = top_values.device().index();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(deviceId);
    CUDA_ERRCHK(cudaSetDevice(deviceId));
    
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((top_values.size(0) + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(
        top_values.scalar_type(),
        "inplace_topk_cuda_kernel",
        [&] {
            inplace_topk_cuda_kernel
            <scalar_t>
            <<<blocks, threads, 0, stream>>>(
                top_values.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                top_indices.packed_accessor64<long, 2, torch::RestrictPtrTraits>(),
                min_top_indices.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
                sums.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                square_sums.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                candidate_values.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                candidate_indices.packed_accessor64<long, 1, torch::RestrictPtrTraits>()
            );
        }
    );
}