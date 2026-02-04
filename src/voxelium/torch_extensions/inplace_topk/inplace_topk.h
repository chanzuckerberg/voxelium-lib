#pragma once
#ifndef INPLACE_TOPK_H
#define INPLACE_TOPK_H

#include <vector>
#include <stdexcept>

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "torch_extensions/inplace_topk/inplace_topk_cpu_kernels.h"
#include "torch_extensions/inplace_topk/inplace_topk_cuda_kernels.h"

void inplace_topk(
    torch::Tensor top_values,
    torch::Tensor top_indices,
    torch::Tensor min_top_values,
    torch::Tensor sums,
    torch::Tensor square_sums,
    torch::Tensor candidate_values,
    torch::Tensor candidate_indices
);

#endif // INPLACE_TOPK_H