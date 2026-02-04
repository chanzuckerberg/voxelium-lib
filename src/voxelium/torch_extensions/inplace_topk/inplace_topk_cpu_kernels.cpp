#include "torch_extensions/inplace_topk/inplace_topk_cpu_kernels.h"

template <typename scalar_t>
void inplace_topk_cpu_kernel(
    torch::TensorAccessor<scalar_t, 2> top_values,
    torch::TensorAccessor<long, 2> top_indices,
    torch::TensorAccessor<int, 1> min_top_indices,
    torch::TensorAccessor<scalar_t, 1> sums,
    torch::TensorAccessor<scalar_t, 1> square_sums,
    const torch::TensorAccessor<scalar_t, 2> candidate_values,
    const torch::TensorAccessor<long, 1> candidate_indices
) 
{
    size_t idx = top_values.size(0);
    size_t K = top_values.size(1);
    size_t N = candidate_values.size(1);

    // Iterate over each element
    for (size_t i = 0; i < idx; ++i)
    {
        scalar_t sums_(0);
        scalar_t square_sums_(0);
    
        int& min_top_index = min_top_indices[i]; // Use external min index for this element

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
        sums[i] += sums_;
        square_sums[i] += square_sums_;
    }
}

void inplace_topk_cpu(
    torch::Tensor top_values,
    torch::Tensor top_indices,
    torch::Tensor min_top_indices,
    torch::Tensor sums,
    torch::Tensor square_sums,
    torch::Tensor candidate_values,
    torch::Tensor candidate_indices
)
{
    CHECK_CPU_INPUT(top_values)
    CHECK_CPU_INPUT(top_indices)
    CHECK_CPU_INPUT(min_top_indices)
    CHECK_CPU_INPUT(sums)
    CHECK_CPU_INPUT(square_sums)
    CHECK_CPU_INPUT(candidate_values)
    CHECK_CPU_INPUT(candidate_indices)

    AT_DISPATCH_FLOATING_TYPES(
        top_values.scalar_type(),
        "inplace_topk_cpu_kernel",
        [&] {
            inplace_topk_cpu_kernel<scalar_t>(
                top_values.accessor<scalar_t, 2>(),
                top_indices.accessor<long, 2>(),
                min_top_indices.accessor<int, 1>(),
                sums.accessor<scalar_t, 1>(),
                square_sums.accessor<scalar_t, 1>(),
                candidate_values.accessor<scalar_t, 2>(),
                candidate_indices.accessor<long, 1>()
            );
        }
    );
}
