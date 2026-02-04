#include "torch_extensions/sparse3d/volume_extraction_cpu_kernels.h"

template <typename scalar_t, bool do_bias>
void volume_extraction_forward_cpu_kernel(
    const torch::TensorAccessor<long, 3> grid3d_index,
    const torch::TensorAccessor<scalar_t, 3> weight,
    const torch::TensorAccessor<scalar_t, 2> bias,
    const torch::TensorAccessor<scalar_t, 2> input,
    torch::TensorAccessor<scalar_t, 5> output,
    const int offset,
    const int max_r2
)
{
    for (long b = 0; b < output.size(0); b ++)
        for (long z = 0; z < output.size(1); z ++)
            for (long y = 0; y < output.size(2); y ++)
                for (long x = 0; x < output.size(3); x ++)
                {
                    const long zp = z - (output.size(1) - 1) / 2;
                    const long yp = y - (output.size(2) - 1) / 2;
                    const long xp = x;
                    if (xp*xp + yp*yp + zp*zp < max_r2)
                    {
                        const long i = grid3d_index[z+offset][y+offset][x];

                        for (int c = 0; c < 2; c++) // Over real and imaginary
                        {
                            for (int j = 0; j < input.size(1); j ++)
                                output[b][z][y][x][c] += weight[i][j][c] * input[b][j];
                            if (do_bias)
                                output[b][z][y][x][c] += bias[i][c];
                        }
                    }
                }
}

void volume_extraction_forward_cpu(
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor input,
    torch::Tensor output,
    const int max_r2,
    const bool do_bias
)
{
    CHECK_CPU_INPUT(grid3d_index)
    CHECK_CPU_INPUT(weight)
    CHECK_CPU_INPUT(bias)
    CHECK_CPU_INPUT(input)
    CHECK_CPU_INPUT(output)

    const int offset = (grid3d_index.size(0) - output.size(1)) / 2;

    std::array<bool, 1> bargs={{do_bias}};
    dispatch_bools<1>{}(
        bargs,
        [&](auto...Bargs) {
            AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(),
                "volume_extraction_forward_cpu_kernel",
                [&] {
                    volume_extraction_forward_cpu_kernel<scalar_t, decltype(Bargs)::value...>(
                        /*grid3d_index*/ grid3d_index.accessor<long, 3>(),
                        /*weight*/ weight.accessor<scalar_t, 3>(),
                        /*bias*/ bias.accessor<scalar_t, 2>(),
                        /*input*/ input.accessor<scalar_t, 2>(),
                        /*output*/ output.accessor<scalar_t, 5>(),
                        /*offset*/ offset,
                        /*max_r2*/ max_r2
                    );
                }
            );
        }
    );
}
