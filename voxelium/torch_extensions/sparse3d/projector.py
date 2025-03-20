#!/usr/bin/env python3

"""
Python API for the 3D reconstruction layer
"""

import numpy as np
import torch

import voxelium as vx

class Projector(torch.nn.Module):
    def __init__(
            self,
            size,
            mask_radius=None,
            mask_edge=None,
            output_norm=None,
            output_size=None,
            dtype=torch.float32,
            index_margin=3,
            pixel_size=1
    ):
        super().__init__()

        if size % 2 != 0:
            raise RuntimeError("Box size must be even")
        
        self.output_size = size if output_size is None else output_size

        if size % 2 == 0:
            size += 1

        self.output_shape = (self.output_size, self.output_size)

        self.size = size
        self.size_x = size // 2 + 1
        self.index_margin = index_margin
        self.maxr = vx.size_to_maxr(size)
        self.dtype = dtype
        self.output_norm = output_norm
        self.pixel_size = pixel_size

        mask_edge = mask_edge or 5
        mask_radius = (output_size - mask_edge) / 2 if mask_radius is None else mask_radius
        self.circular_mask = torch.nn.Parameter(vx.smooth_circular_mask(output_size, mask_radius, mask_edge), requires_grad=False)

        bz = self.size
        bz_2 = bz // 2
        bz_x = bz_2 + 1
        max_r2 = self.maxr ** 2
        
        z, y, x = torch.meshgrid(
            torch.arange(bz, dtype=torch.int32),
            torch.arange(bz, dtype=torch.int32),
            torch.arange(bz_x, dtype=torch.int32),
            indexing='ij'
        )
        
        z, y = z - bz_2, y - bz_2
        
        mask = (z**2 + y**2 + x**2) < max_r2  # Boolean mask
        del z, y, x  # To save some space

        weight_count = mask.sum().item()
        
        grid_indices = torch.full((bz, bz, bz_x), -1, dtype=torch.long)
        grid_indices[mask] = torch.arange(weight_count, dtype=torch.long)
        
        # Compute inverse indices
        inverse_grid_indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]

        # Everything outside mask is mapped to a single element at the end (null element)
        null_index = weight_count
        grid_indices[~mask] = null_index

        # Store results
        self.weight_count = weight_count + 1  # Add one to store the null element

        # Add margin for pixel spread into voxels
        m = self.index_margin
        bz += m * 2
        bz_2 += m
        grid_indices_margin = torch.full((bz, bz, bz_2 + 1), null_index, dtype=torch.long)
        grid_indices_margin[m:-m, m:-m, :-m] = grid_indices
        grid_indices = grid_indices_margin

        self.grid3d_mask = torch.nn.Parameter(mask.bool(), requires_grad=False)
        self.grid3d_index = torch.nn.Parameter(grid_indices.long(), requires_grad=False)
        self.inverse_grid3d_indices = torch.nn.Parameter(inverse_grid_indices.long(), requires_grad=False)

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
            coord, mask = vx.make_explicit_grid2d(size=Y, max_r=max_r, device=default_device)
        else:
            coord = grid2d_coord

        input = torch.ones([B, 1]).to(default_device)

        p = vx.TrilinearProjection.apply(
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
            p = vx.dt_desymmetrize(p_, dim=2)

        if not return_ft:
            p = vx.idft(p, dim=2, real_in=True)
            if self.output_size < p.size(-1):
                m = p.size(-1) // 2 - self.output_size // 2
                p = p[:, m:m+self.output_size, m:m+self.output_size]
            p *= self.circular_mask.data[None]
            if self.output_norm is not None:
                p /= self.output_norm / (p.sum(0, keepdim=True) + 1e-12)

        return p

    @torch.no_grad()
    def set_model(self, grid: torch.tensor, symmetrize: bool = True):
        """
        Set FFT of the model
        """
        if symmetrize:
            grid = vx.dt_symmetrize(grid)

        # Map to everything but the null element at -1
        self.weight[:-1, 0] = torch.view_as_real(grid.flatten()[self.inverse_grid3d_indices])
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
            grid = vx.dt_desymmetrize(grid)

        return grid

    @staticmethod
    def from_file(path, voxel_size, padding=2, trim=False, normalize=True, device="cpu", *args, **kwargs):
        assert padding >= 1, "Padding must be larger than or equal to 1"

        ref, ref_voxel_size, _ = vx.load_mrc(path)
        ref = ref.copy()
        
        ref, _, _ = vx.make_cubic(ref)

        ref = torch.from_numpy(ref).to(device)
        ref = vx.rescale_voxelsize(ref, voxel_size=ref_voxel_size, target_voxel_size=voxel_size)

        size = ref.shape[0]

        ref *= vx.smooth_spherical_mask(
            grid_size=size, radius=size/2, thickness=3, device=device)

        if normalize:
            ref /= ref[ref != 0].std() + 1e-12

        size_pad = max(int(size * padding), size)
        if trim:
            ref = vx.resize_3d_grid(ref, size_pad)
        else:
            ref = vx.pad_and_center_mass(ref, size_pad, margin=0)
        ref *= size_pad / size  # Rescale to maintain same absolute values
        ref_ft = vx.dft(ref, real_in=True)

        if "mask_edge" not in kwargs and "mask_radius" not in kwargs:
            mask_edge = max(40 / voxel_size, 3)
            kwargs["mask_radius"] = size_pad / 2 - mask_edge
            kwargs["mask_edge"] = mask_edge

        if "output_size" not in kwargs:
            output_size = size if trim else size_pad

        p = Projector(size=size_pad, output_size=output_size, *args, **kwargs)
        return p.to(device).set_model(ref_ft)
