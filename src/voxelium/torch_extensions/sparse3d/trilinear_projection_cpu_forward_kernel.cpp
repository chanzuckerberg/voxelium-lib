#include "torch_extensions/sparse3d/trilinear_projection_cpu_kernels.h"
#include "torch_extensions/sparse3d/trilinear_projection_cpu_utils.h"


template <typename scalar_t, bool do_bias>
void trilinear_projection_forward_cpu_kernel(
    const torch::TensorAccessor<scalar_t, 2> grid2d_coord,
    const torch::TensorAccessor<long, 3> grid3d_index,
    const torch::TensorAccessor<scalar_t, 3> weight,
    const torch::TensorAccessor<scalar_t, 2> bias,
    const torch::TensorAccessor<scalar_t, 3> rot_matrix,
    const torch::TensorAccessor<scalar_t, 2> input,
    torch::TensorAccessor<scalar_t, 3> output,
    const int max_r2,
    const int init_offset
)
{
    int xs[2], ys[2], zs[2];
    scalar_t xp, yp, zp, fx, fy, fz;
    for (long b = 0; b < input.size(0); b ++)
    {
        for (long i = 0; i < grid2d_coord.size(0); i ++)
        {
            _helper_rotate_coordinates<scalar_t>(
                rot_matrix[b], grid2d_coord[i][1], grid2d_coord[i][0], xp, yp, zp
            );

            if (xp*xp + yp*yp + zp*zp < max_r2)
            {
                scalar_t conj = _helper_cube_interpolation_coordinates<scalar_t>(
                    xp, yp, zp, init_offset, xs, ys, zs, fx, fy, fz
                );

                for (int c = 0; c < 2; c ++) // Over real and imaginary
                {
                    // Order: 0=000 1=001 2=010 3=011 4=100 5=101 6=110 7=111
                    scalar_t v[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                    for (int k = 0; k < 8; k ++) // Over local cube vertices
                    {
                        long index = grid3d_index[zs[k/4]] [ys[(k/2)%2]] [xs[k%2]];

                        for (int j = 0; j < input.size(1); j ++) // Over input vector
                            v[k] += weight[index][j][c] * input[b][j];

                        if (do_bias)
                            v[k] += bias[index][c];
                    }

                    // Set the interpolated value in the 2D output array
                    const scalar_t dx00 = LIN_INTERP(fx, v[0], v[1]);
                    const scalar_t dx10 = LIN_INTERP(fx, v[2], v[3]);
                    const scalar_t dx01 = LIN_INTERP(fx, v[4], v[5]);
                    const scalar_t dx11 = LIN_INTERP(fx, v[6], v[7]);
                    const scalar_t dxy0 = LIN_INTERP(fy, dx00, dx10);
                    const scalar_t dxy1 = LIN_INTERP(fy, dx01, dx11);
                    if (c == 1) // Only flip sign of the imaginary component (complex conjugate)
                        output[b][i][c] = conj * LIN_INTERP(fz, dxy0, dxy1);
                    else
                        output[b][i][c] = LIN_INTERP(fz, dxy0, dxy1);
                }
            } // If < max_r
        } // 2D Point
    } // Batch
}

void trilinear_projection_forward_cpu(
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
    CHECK_CPU_INPUT(grid2d_coord)
    CHECK_CPU_INPUT(grid3d_index)
    CHECK_CPU_INPUT(weight)
    CHECK_CPU_INPUT(bias)
    CHECK_CPU_INPUT(rot_matrix)
    CHECK_CPU_INPUT(input)
    CHECK_CPU_INPUT(output)

    std::array<bool, 1> bargs={{do_bias}};
    dispatch_bools<1>{}(
        bargs,
        [&](auto...Bargs) {
            AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(),
                "trilinear_projection_forward_cpu_kernel",
                [&] {
                    trilinear_projection_forward_cpu_kernel<scalar_t, decltype(Bargs)::value...>(
                        /*grid2d_coord*/ grid2d_coord.accessor<scalar_t, 2>(),
                        /*grid3d_index*/ grid3d_index.accessor<long, 3>(),
                        /*weight*/ weight.accessor<scalar_t, 3>(),
                        /*bias*/ bias.accessor<scalar_t, 2>(),
                        /*rot_matrix*/ rot_matrix.accessor<scalar_t, 3>(),
                        /*input*/ input.accessor<scalar_t, 2>(),
                        /*output*/ output.accessor<scalar_t, 3>(),
                        /*max_r2*/ max_r2,
                        /*init_offset*/ init_offset
                    );
                }
            );
        }
    );
}
