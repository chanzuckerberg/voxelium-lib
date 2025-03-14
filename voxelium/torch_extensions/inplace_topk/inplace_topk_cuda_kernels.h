#pragma once
#ifndef INPLACE_TOPK_CUDA_KERNELS_H
#define INPLACE_TOPK_CUDA_KERNELS_H

#include <cmath>
#include <vector>
#include <stdexcept>

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

void inplace_topk_cuda(
    torch::Tensor top_values,
    torch::Tensor top_indices,
    torch::Tensor min_top_values,
    torch::Tensor sums,
    torch::Tensor square_sums,
    torch::Tensor candidate_values,
    torch::Tensor candidate_indices
);

#endif // INPLACE_TOPK_CUDA_KERNELS_H