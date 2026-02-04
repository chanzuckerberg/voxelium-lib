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

class TrilinearProjection(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, input, weight, bias, grid3d_index,
            rot_matrices, grid2d_coord, max_r,
            backprop_eps=0, testing=False
    ):
        assert grid3d_index.shape[0] == grid3d_index.shape[1] == grid3d_index.shape[2] * 2 - 1

        output = _C.trilinear_projection_forward(
            input=input,
            weight=weight,
            bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
            rot_matrix=rot_matrices,
            grid2d_coord=grid2d_coord,
            grid3d_index=grid3d_index,
            max_r=max_r
        )

        ctx.save_for_backward(
            input,
            weight,
            bias,
            grid3d_index,
            rot_matrices,
            grid2d_coord,
            torch.Tensor([max_r]),
            torch.Tensor([backprop_eps]),
            torch.Tensor([testing])
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, grid3d_index, rot_matrices, grid2d_coord, \
            max_r, backprop_eps, testing \
            = ctx.saved_tensors

        grad_input, grad_weight, grad_bias, backprop_weight, grad_rot_matrix = \
            _C.trilinear_projection_backward(
                input=input,
                grid2d_grad=grad_output.contiguous(),
                weight=weight,
                bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
                grid3d_index=grid3d_index,
                rot_matrix=rot_matrices,
                grid2d_coord=grid2d_coord,
                max_r=max_r[0],
                return_backprop_weight=backprop_eps[0] > 0
            )

        if bias is None:
            grad_bias = None

        if backprop_eps[0] > 0:
            backprop_weight.add_(backprop_eps[0])
            grad_weight.div_(backprop_weight[:, None, None])
            if grad_bias is not None:
                grad_bias.div_(backprop_weight[:, None])

        return grad_input, grad_weight, grad_bias, None, grad_rot_matrix, None, None, None, None, None
