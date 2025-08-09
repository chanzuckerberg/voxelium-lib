#!/usr/bin/env python3

"""
Python API for the Inplace TopK extensions 
"""
import torch

try:
    from . import _C
except ImportError:
    print("Could not find Voxelium extension 'TopK'.")
    import sys
    sys.exit(1)

class InplaceTopK(torch.nn.Module):
    def __init__(self, size, k, dtype=torch.float32):
        super().__init__()

        self.size = size
        self.k = k
        self.dtype = dtype

        min_value = torch.finfo(dtype).min
        top_values = torch.full((size, k), min_value, dtype=dtype)
        self.top_values = torch.nn.Parameter(data=top_values, requires_grad=False)

        top_indices = torch.full((size, k), -1, dtype=torch.int64)
        self.top_indices = torch.nn.Parameter(data=top_indices, requires_grad=False)

        min_top_indices = torch.zeros(size, dtype=torch.int)
        self.min_top_indices = torch.nn.Parameter(data=min_top_indices, requires_grad=False)

        sums = torch.zeros(size, dtype=dtype)
        self.sums = torch.nn.Parameter(data=sums, requires_grad=False)

        squared_sums = torch.zeros(size, dtype=dtype)
        self.squared_sums = torch.nn.Parameter(data=squared_sums, requires_grad=False)

        self.sums_count = 0

    @torch.no_grad()
    def forward(self, candidate_values, candidate_indices):
        self.sums_count += candidate_indices.size(0)
        _C.inplace_topk(
            top_values=self.top_values,
            top_indices=self.top_indices,
            min_top_indices=self.min_top_indices,
            sums=self.sums,
            squared_sums=self.squared_sums,
            candidate_values=candidate_values,
            candidate_indices=candidate_indices
        )

    def get_sorted(self):
        top_values, indices = torch.sort(self.top_values.data, dim=1, descending=True)
        top_indices = torch.gather(self.top_indices.data, dim=1, index=indices)

        return top_values, top_indices


    def get_mean(self):
        return self.sums / self.sums_count
    

    def get_std(self):
        mean = self.sums / self.sums_count
        square_mean = self.squared_sums / self.sums_count

        return torch.sqrt(square_mean - mean.square())