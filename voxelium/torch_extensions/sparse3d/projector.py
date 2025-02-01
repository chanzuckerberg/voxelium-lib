#!/usr/bin/env python3

"""
Python API for the 3D reconstruction layer
"""

import numpy as np
import torch

from voxelium.torch_extensions.sparse3d import TrilinearProjection, VolumeExtraction
from voxelium.base import grid_iterator, dt_desymmetrize, dt_symmetrize, make_explicit_grid2d, radial_index_expansion_3d, size_to_maxr, load_mrc, trim_to_threshold, make_cubic, rescale_voxelsize, pad_and_center_mass, dft, idft

class Projector(torch.nn.Module):
    def __init__(
            self,
            size,
            trim_size=None,
            dtype=torch.float32,
            index_margin=3
    ):
        super().__init__()

        if size % 2 != 0:
            raise RuntimeError("Box size must be even")
        
        self.trim_size = size if trim_size is None else trim_size

        if size % 2 == 0:
            size += 1

        self.size = size
        self.size_x = size // 2 + 1
        self.index_margin = index_margin
        self.maxr = size_to_maxr(size)
        self.dtype = dtype

        bz = self.size
        bz_2 = bz // 2
        bz_x = bz_2 + 1
        grid_mask = np.zeros((bz, bz, bz_x), dtype=bool)
        grid_indices = np.zeros((bz, bz, bz_x), dtype=int) - 1
        inverse_grid_indices = np.zeros((bz * bz * bz_x), dtype=int)
        max_r2 = size_to_maxr(self.size) ** 2
        i = 0
        for z, y, x in grid_iterator(bz, bz, bz_x):
            if (z - bz_2) ** 2 + (y - bz_2) ** 2 + x ** 2 < max_r2:
                grid_mask[z, y, x] = True
                grid_indices[z, y, x] = i
                inverse_grid_indices[i] = z * bz * bz_x + y * bz_x + x
                i += 1

        self.weight_count = i
        inverse_grid_indices = inverse_grid_indices[:i]

        # Add margin for pixel spread into voxels
        m = self.index_margin
        bz += m * 2
        bz_2 += m
        grid_indices_margin = np.zeros((bz, bz, bz_2 + 1), dtype=int) - 1
        grid_indices_margin[m:-m, m:-m, :-m] = grid_indices
        grid_indices = grid_indices_margin

        for i in range(5):
            radial_index_expansion_3d(grid_indices)

        self.grid3d_mask = torch.nn.Parameter(
            torch.tensor(grid_mask, dtype=torch.bool), requires_grad=False)

        self.grid3d_index = torch.nn.Parameter(
            torch.tensor(grid_indices, dtype=torch.long), requires_grad=False)

        self.inverse_grid3d_indices = torch.nn.Parameter(
            torch.tensor(inverse_grid_indices, dtype=torch.long), requires_grad=False)

        data_tensor = torch.zeros((self.weight_count, 1, 2), dtype=self.dtype)
        self.weight = torch.nn.Parameter(data=data_tensor, requires_grad=True)

    def forward(self, rot_matrices=None, max_r=None, grid2d_coord=None, return_ft=False, backprop_eps=True):
        max_r = self.maxr if max_r is None else min(max_r, self.maxr)
        default_device = self.weight.device

        if rot_matrices is None:
            rot_matrices = torch.eye(3).unsqueeze(0).to(default_device)

        B = rot_matrices.size(0)
        X = self.size // 2 + 1
        Y = self.size

        if grid2d_coord is None:
            coord, mask = make_explicit_grid2d(size=Y, max_r=max_r, device=default_device)
        else:
            coord = grid2d_coord

        input = torch.ones([B, 1]).to(default_device)

        p = TrilinearProjection.apply(
            input,  # input
            self.weight,  # weight
            None,  # bias
            self.grid3d_index,  # grid3d_index
            rot_matrices,  # rot_matrices
            coord,  # grid2d_coord
            max_r,  # max_r
            backprop_eps,  # backprop_eps
            False  # testing
        )
        p /= self.size - 1
        p = torch.view_as_complex(p)

        if grid2d_coord is None:
            p_ = torch.zeros([B, Y * X], device=p.device, dtype=p.dtype)
            p_[:, mask] = p
            p_ = p_.view(B, Y, X)
            p = dt_desymmetrize(p_, dim=2)

        if not return_ft:
            p = idft(p, dim=2, real_in=True)
            #TODO Apply circular mask
            if self.trim_size is not None and self.trim_size < p.size(-1):
                m = p.size(-1) // 2 - self.trim_size // 2
                p = p[:, m:m+self.trim_size, m:m+self.trim_size]

        return p

    @torch.no_grad()
    def set_model(self, grid: torch.tensor, symmetrize: bool = True):
        """
        Set FFT of the model
        """
        if symmetrize:
            grid = dt_symmetrize(grid)

        self.weight[:, 0] = torch.view_as_real(grid.flatten()[self.inverse_grid3d_indices])
        return self

    @torch.no_grad()
    def get_model(self, desymmetrize: bool = True):
        """
        Get FFT of the model.
        """
        m = self.index_margin
        indices = self.grid3d_index[m:-m, m:-m, :-m]

        grid = self.weight[indices, 0]
        grid_ = torch.view_as_complex(grid)
        grid = torch.zeros_like(grid_)
        grid[self.grid3d_mask] = grid_[self.grid3d_mask]
        if desymmetrize:
            grid = dt_desymmetrize(grid)

        return grid

    @staticmethod
    def from_file(path, voxel_size, padding=2, backgroud_threshold=None, device="cpu"):
        ref, ref_voxel_size, _ = load_mrc(path)

        ref = ref.copy()

        if backgroud_threshold is not None:
            ref = trim_to_threshold(ref, threshold=backgroud_threshold)
        
        ref, _, _ = make_cubic(ref)

        ref = torch.from_numpy(ref).to(device)
        ref = rescale_voxelsize(ref, voxel_size=ref_voxel_size, target_voxel_size=voxel_size)

        assert padding >= 1

        size = ref.shape[0]
        size_pad = size * padding
        ref = pad_and_center_mass(ref, size_pad)
        ref_ft = dft(ref, real_in=True)

        return Projector(size=size_pad, trim_size=size).to(device).set_model(ref_ft)
