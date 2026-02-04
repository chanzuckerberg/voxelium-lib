#include "torch_extensions/sparse3d/trilinear_projection_cuda_kernels.h"
#include "torch_extensions/sparse3d/trilinear_projection_cuda_utils.cuh"

#define FORWARD_BLOCK_SIZE 512

template <typename scalar_t, typename accscalar_t, bool do_bias>
__global__ void trilinear_projection_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid2d_coord,
    const torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> grid3d_index,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> bias,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> rot_matrix,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output,
    const int max_r2,
    const int init_offset
)
{
    int xs[2], ys[2], zs[2];
    scalar_t xp, yp, zp, fx, fy, fz;
    size_t b, i;

    if (thread_index_expand(input.size(0), grid2d_coord.size(0), b, i))
    {
        scalar_t x(grid2d_coord[i][1]), y(grid2d_coord[i][0]);
        xp = rot_matrix[b][0][0] * x + rot_matrix[b][1][0] * y;
        yp = rot_matrix[b][0][1] * x + rot_matrix[b][1][1] * y;
        zp = rot_matrix[b][0][2] * x + rot_matrix[b][1][2] * y;

        if (xp*xp + yp*yp + zp*zp < max_r2)
        {
            scalar_t conj = _helper_cube_interpolation_coordinates<scalar_t>(
                xp, yp, zp, init_offset, xs, ys, zs, fx, fy, fz
            );

            for (int c = 0; c < 2; c ++) // Over real and imaginary
            {
                // Order: 0=000 1=001 2=010 3=011 4=100 5=101 6=110 7=111
                accscalar_t v[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int k = 0; k < 8; k ++) // Over local cube vertices
                {
                    long index = grid3d_index[zs[k/4]] [ys[(k/2)%2]] [xs[k%2]];

                    for (int j = 0; j < input.size(1); j ++) // Over input vector
                        v[k] += weight[index][j][c] * input[b][j];

                    if (do_bias)
                        v[k] += bias[index][c];
                }

                // Set the interpolated value in the 2D output array
                const accscalar_t dx00 = LIN_INTERP(fx, v[0], v[1]);
                const accscalar_t dx10 = LIN_INTERP(fx, v[2], v[3]);
                const accscalar_t dx01 = LIN_INTERP(fx, v[4], v[5]);
                const accscalar_t dx11 = LIN_INTERP(fx, v[6], v[7]);
                const accscalar_t dxy0 = LIN_INTERP(fy, dx00, dx10);
                const accscalar_t dxy1 = LIN_INTERP(fy, dx01, dx11);
                if (c == 1) // Only flip sign of the imaginary component (complex conjugate)
                    output[b][i][c] = (scalar_t) conj * LIN_INTERP(fz, dxy0, dxy1);
                else
                    output[b][i][c] = (scalar_t) LIN_INTERP(fz, dxy0, dxy1);
            }
        } // If < max_r
    }
}

void trilinear_projection_forward_cuda(
    const torch::Tensor grid2d_coord,
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor rot_matrix,
    const torch::Tensor input,
    torch::Tensor output,
    const int max_r2,
    const int init_offset,
    const bool do_bias
)
{
    CHECK_CUDA_INPUT(grid2d_coord)
    CHECK_CUDA_INPUT(grid3d_index)
    CHECK_CUDA_INPUT(weight)
    CHECK_CUDA_INPUT(bias)
    CHECK_CUDA_INPUT(rot_matrix)
    CHECK_CUDA_INPUT(input)
    CHECK_CUDA_INPUT(output)

    const int deviceId = input.device().index();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(deviceId);
    CUDA_ERRCHK(cudaSetDevice(deviceId));

    const dim3 threads(FORWARD_BLOCK_SIZE);
    const dim3 blocks((input.size(0) * grid2d_coord.size(0) + threads.x - 1) / threads.x);

    std::array<bool, 1> bargs={{do_bias}};
    dispatch_bools<1>{}(
        bargs,
        [&](auto...Bargs) {
            AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(),
                "trilinear_projection_forward_cuda_kernel",
                [&] {
                    using accscalar_t = at::acc_type<scalar_t, true>;
                    trilinear_projection_forward_cuda_kernel
                    <scalar_t, accscalar_t, decltype(Bargs)::value...>
                    <<<blocks, threads, 0, stream>>>(
                        /*grid2d_coord*/ grid2d_coord.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*grid3d_index*/ grid3d_index.packed_accessor64<long, 3, torch::RestrictPtrTraits>(),
                        /*weight*/ weight.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*bias*/ bias.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*rot_matrix*/ rot_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*input*/ input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*output*/ output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*max_r2*/ max_r2,
                        /*init_offset*/ init_offset
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
