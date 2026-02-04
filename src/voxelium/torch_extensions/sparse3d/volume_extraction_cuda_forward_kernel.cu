#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/macros/Macros.h>

#include "base/base_cuda.cuh"
#include "torch_extensions/sparse3d/volume_extraction_cuda_kernels.h"


template <typename scalar_t, typename accscaler_t, bool do_bias>
__global__ void volume_extraction_forward_cuda_kernel(
    const torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> grid3d_index,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> bias,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> output,
    const int offset,
    const int max_r2
)
{
    size_t b, z, y, x;
    if (thread_index_expand(output.size(0), output.size(1), output.size(2), output.size(3), b, z, y, x))
    {
        const long zp = z - (output.size(1) - 1) / 2;
        const long yp = y - (output.size(2) - 1) / 2;
        const long xp = x;

        if (xp*xp + yp*yp + zp*zp < max_r2)
        {
            const long i = grid3d_index[z+offset][y+offset][x];

            for (int c = 0; c < 2; c++) // Over real and imaginary
            {
                accscaler_t v(0);
                for (int j = 0; j < input.size(1); j ++)
                    v += weight[i][j][c] * input[b][j];
                if (do_bias)
                    v += bias[i][c];
                output[b][z][y][x][c] = (scalar_t) v;
            }
        }
    }
}

void volume_extraction_forward_cuda(
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor input,
    torch::Tensor output,
    const int max_r2,
    const bool do_bias
)
{
    CHECK_CUDA_INPUT(grid3d_index)
    CHECK_CUDA_INPUT(weight)
    CHECK_CUDA_INPUT(bias)
    CHECK_CUDA_INPUT(input)
    CHECK_CUDA_INPUT(output)

    const int offset = (grid3d_index.size(0) - output.size(1)) / 2;

    const int deviceId = input.device().index();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(deviceId);
    CUDA_ERRCHK(cudaSetDevice(deviceId));

    const dim3 threads(512);
    const dim3 blocks(
        (
            output.size(0) *
            output.size(1) *
            output.size(2) *
            output.size(3) +
            threads.x - 1
        ) / threads.x
    );

    std::array<bool, 1> bargs={{do_bias}};
    dispatch_bools<1>{}(
        bargs,
        [&](auto...Bargs) {
            AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(),
                "volume_extraction_forward_cuda_kernel",
                [&] {
                    using accscalar_t = at::acc_type<scalar_t, true>;
                    volume_extraction_forward_cuda_kernel
                    <scalar_t, accscalar_t, decltype(Bargs)::value...>
                    <<<blocks, threads, 0, stream>>>(
                        /*grid3d_index*/ grid3d_index.packed_accessor64<long, 3, torch::RestrictPtrTraits>(),
                        /*weight*/ weight.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*bias*/ bias.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*input*/ input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*output*/ output.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                        /*offset*/ offset,
                        /*max_r2*/ max_r2
                    );
                }
            );
        }
    );

#ifdef DEBUG
    CUDA_ERRCHK(cudaPeekAtLastError());
    CUDA_ERRCHK(cudaDeviceSynchronize());
#endif
}
