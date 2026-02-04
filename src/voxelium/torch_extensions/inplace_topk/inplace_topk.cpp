#include "torch_extensions/inplace_topk/inplace_topk.h"

#include <iostream>


void inplace_topk(
    torch::Tensor top_values,
    torch::Tensor top_indices,
    torch::Tensor min_top_indices,
    torch::Tensor sums,
    torch::Tensor square_sums,
    torch::Tensor candidate_values,
    torch::Tensor candidate_indices
)
{
    CHECK_DIM(top_values, 2)

    const long numel = top_values.size(0);
    const int k = top_values.size(1);

    CHECK_DIM(top_indices, 2)
    CHECK_SIZE_DIM0(top_indices, numel)
    CHECK_SIZE_DIM1(top_indices, k)

    CHECK_DIM(min_top_indices, 1)
    CHECK_SIZE_DIM0(min_top_indices, numel)

    CHECK_DIM(sums, 1)
    CHECK_SIZE_DIM0(sums, numel)

    CHECK_DIM(square_sums, 1)
    CHECK_SIZE_DIM0(square_sums, numel)

    CHECK_DIM(candidate_values, 2)
    const int n = candidate_values.size(1);
    CHECK_SIZE_DIM0(candidate_values, numel)

    CHECK_DIM(candidate_indices, 1)
    CHECK_SIZE_DIM0(candidate_indices, n)

    if (top_values.device().type() == torch::kCPU)
    {
        inplace_topk_cpu(
            top_values,
            top_indices,
            min_top_indices,
            sums,
            square_sums,
            candidate_values,
            candidate_indices
        );
    }
    else if (top_values.device().type() == torch::kCUDA)
    {
        inplace_topk_cuda(
            top_values,
            top_indices,
            min_top_indices,
            sums,
            square_sums,
            candidate_values,
            candidate_indices
        );
    }
    else
        throw std::logic_error("Support for device not implemented");
}