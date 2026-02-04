#!/usr/bin/env python3

"""
Python API for the 3D reconstruction layer
"""
import torch

try:
    from . import _C
except ImportError:
    print("Could not find Voxelium extension 'sparse3d'.")
    import sys
    sys.exit(1)

from voxelium.base import grid_iterator, dt_desymmetrize, dt_symmetrize
from voxelium.base.explicit_grid_utils import radial_index_expansion_3d, size_to_maxr


class VolumeExtraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, grid3d_index, max_r):

        assert grid3d_index.shape[0] == grid3d_index.shape[1] == grid3d_index.shape[2] * 2 - 1

        output = _C.volume_extraction_forward(
            input=input,
            weight=weight,
            bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
            grid3d_index=grid3d_index,
            max_r=max_r
        )
        ctx.save_for_backward(input, weight, bias, grid3d_index)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, grid3d_index = ctx.saved_tensors

        grad_input, grad_weight, grad_bias = \
            _C.volume_extraction_backward(
                input=input,
                weight=weight,
                bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
                grad_output=grad_output,
                grid3d_index=grid3d_index
            )

        if bias is None:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None
