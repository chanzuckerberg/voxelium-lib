#!/usr/bin/env python

"""
Module for calculations related to grid manipulations.
This is temporary, functions should be organized in separate files.
"""

import numpy as np
import mrcfile as mrc
import torch
from typing import Tuple, Union, List

import matplotlib.pylab as plt


def grid_iterator(
        s0: Union[int, np.ndarray],
        s1: Union[int, np.ndarray],
        s2: Union[int, np.ndarray] = None
):
    if isinstance(s0, int):
        if s2 is None:  # 2D grid
            for i in range(s0):
                for j in range(s1):
                    yield i, j
        else:  # 3D grid
            for i in range(s0):
                for j in range(s1):
                    for k in range(s2):
                        yield i, j, k
    else:
        if s2 is None:  # 2D grid
            for i in range(len(s0)):
                for j in range(len(s1)):
                    yield s0[i], s1[j]
        else:  # 3D grid
            for i in range(len(s0)):
                for j in range(len(s1)):
                    for k in range(len(s2)):
                        yield s0[i], s1[j], s2[k]


def save_mrc(grid, filename, voxel_size=1, origin=0.):
    if isinstance(origin, float) or isinstance(origin, int) or origin is None:
        origin = [origin] * 3
    (z, y, x) = grid.shape
    o = mrc.new(filename, overwrite=True)
    o.header['cella'].x = x * voxel_size
    o.header['cella'].y = y * voxel_size
    o.header['cella'].z = z * voxel_size
    o.header['origin'].x = origin[0]
    o.header['origin'].y = origin[1]
    o.header['origin'].z = origin[2]
    out_box = np.reshape(grid, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.update_header_stats()
    o.flush()
    o.close()


def load_mrc(mrc_fn):
    mrc_file = mrc.open(mrc_fn, 'r')

    global_origin = mrc_file.header['origin']
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    global_origin[0] += mrc_file.header['nxstart']
    global_origin[1] += mrc_file.header['nystart']
    global_origin[2] += mrc_file.header['nzstart']

    c = mrc_file.header['mapc'] - 1
    r = mrc_file.header['mapr'] - 1
    s = mrc_file.header['maps'] - 1

    if c == 0 and r == 1 and s == 2:
        grid = mrc_file.data
    elif c == 2 and r == 1 and s == 0:
        grid = np.moveaxis(mrc_file.data, [2, 1, 0], [0, 1, 2])
    elif c == 1 and r == 0 and s == 2:
        grid = np.moveaxis(mrc_file.data, [0, 2, 1], [0, 1, 2])
    elif c == 0 and r == 2 and s == 1:
        grid = np.moveaxis(mrc_file.data, [1, 0, 2], [0, 1, 2])
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")

    voxel_size = float(mrc_file.voxel_size.x)

    return grid, voxel_size, global_origin


def make_cubic(box):
    bz = np.array(box.shape)
    s = np.max(box.shape)
    s += s % 2
    if np.all(box.shape == s):
        return box, np.zeros(3, dtype=int), bz
    nbox = np.zeros((s, s, s))
    c = np.array(nbox.shape) // 2 - bz // 2
    nbox[c[0]:c[0] + bz[0], c[1]:c[1] + bz[1], c[2]:c[2] + bz[2]] = box
    return nbox, c, c + bz


def resize_3d_grid(grid, target_size, pad_value=0):
    """
    Resize a 3D PyTorch tensor to the specified target size by either padding or cropping.

    Args:
        grid (torch.Tensor): Input 3D grid of shape (D, H, W).
        target_size (tuple): Target size as a tuple (target_D, target_H, target_W).
        pad_value (int, float): Value to use for padding when the grid is smaller than the target size.

    Returns:
        torch.Tensor: Resized 3D grid of shape target_size.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size, target_size)
    assert len(grid.shape) == 3, "Input grid must be 3D."
    assert len(target_size) == 3, "Target size must be a tuple of three integers."

    input_size = grid.shape
    diffs = [target - current for target, current in zip(target_size, input_size)]

    # Padding: Calculate padding for each dimension (left and right for D, H, W)
    padding = []
    for diff in diffs[::-1]:  # Reverse because F.pad expects (W, H, D) order
        if diff > 0:
            pad_left = diff // 2
            pad_right = diff - pad_left
            padding.extend([pad_left, pad_right])
        else:
            padding.extend([0, 0])

    if any(diff > 0 for diff in diffs):
        grid = torch.nn.functional.pad(grid, padding, value=pad_value)

    # Cropping: Slice the tensor if it exceeds the target size
    slices = []
    for dim, diff in zip(range(3), diffs):
        if diff < 0:
            crop_start = -diff // 2
            crop_end = crop_start + target_size[dim]
            slices.append(slice(crop_start, crop_end))
        else:
            slices.append(slice(None))

    resized_tensor = grid[slices[0], slices[1], slices[2]]

    return resized_tensor

def pad_and_center_mass(grid, target_size, margin=2):
    """
    Pads a 3D tensor to a new size while ensuring a minimum margin from edges 
    and centering the mass correctly.

    Args:
        grid (torch.Tensor): Input tensor of shape [D, H, W].
        target_size (tuple): Desired output shape [D_new, H_new, W_new].
        margin (int): Minimum margin from the edges.

    Returns:
        torch.Tensor: Padded tensor with mass centered.
    """
    if isinstance(target_size, int):
        target_size = [target_size, target_size, target_size]

    assert len(grid.shape) == 3, "Input tensor must have shape [D, H, W]"
    assert len(target_size) == 3, "Target size must be a tuple of three integers"

    D, H, W = grid.shape
    D_new, H_new, W_new = target_size

    assert D_new >= D + 2 * margin and H_new >= H + 2 * margin and W_new >= W + 2 * margin, \
        "Target size must be large enough to fit the original tensor with margin"

    # Compute center of mass (CoM) of the input tensor
    indices = torch.nonzero(grid, as_tuple=False)
    if len(indices) == 0:
        # If empty, just return zero-padded centered tensor
        pad_d1, pad_h1, pad_w1 = (D_new - D) // 2, (H_new - H) // 2, (W_new - W) // 2
        pad_d2, pad_h2, pad_w2 = D_new - D - pad_d1, H_new - H - pad_h1, W_new - W - pad_w1
        pad = (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2)
        return torch.nn.functional.pad(grid, pad, mode='constant', value=0)

    weights = grid[indices[:, 0], indices[:, 1], indices[:, 2]]
    com = torch.sum(indices * weights[:, None], dim=0) / torch.sum(weights)

    # Compute the new center position in the target grid
    target_center = torch.tensor([
        margin + (D_new - 2 * margin) // 2, 
        margin + (H_new - 2 * margin) // 2, 
        margin + (W_new - 2 * margin) // 2
    ], dtype=torch.float32, device=grid.device)

    # Compute required shifts
    shift = (target_center - com).round().int()

    # Compute padding, ensuring final size matches target_size
    pad_d1 = margin + shift[0].item()
    pad_h1 = margin + shift[1].item()
    pad_w1 = margin + shift[2].item()

    pad_d2 = (D_new - D) - pad_d1
    pad_h2 = (H_new - H) - pad_h1
    pad_w2 = (W_new - W) - pad_w1

    pad = (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2)

    # Apply padding
    padded_tensor = torch.nn.functional.pad(grid, pad, mode='constant', value=0)

    return padded_tensor


def get_bounds_for_threshold(grid, threshold=0.):
    """Finds the bounding box encapsulating volume segment above threshold"""
    g = threshold < grid

    # Collapse along each axis using np.any
    x_any = np.any(g, axis=(1, 2))  # Collapse y and z axes
    y_any = np.any(g, axis=(0, 2))  # Collapse x and z axes
    z_any = np.any(g, axis=(0, 1))  # Collapse x and y axes

    # Find the min and max indices where there are True values
    try:
        x_min, x_max = np.where(x_any)[0][[0, -1]]
        y_min, y_max = np.where(y_any)[0][[0, -1]]
        z_min, z_max = np.where(z_any)[0][[0, -1]]
    except IndexError:
        raise RuntimeError(f"Bounding threshold ({threshold}) is too large. Must be less than: {np.max(grid)}")

    return (x_min, y_min, z_min), (x_max, y_max, z_max)

def trim_to_threshold(grid, threshold=0.):
    l, h = get_bounds_for_threshold(grid, threshold=threshold)
    return grid[l[0]:h[0], l[1]:h[1], l[2]:h[2]]


def smooth_circular_mask(grid_size, radius, thickness, center_shift=(0,0)):
    """ Mask radius is center of edge """
    ls = torch.linspace(-grid_size / 2, grid_size / 2, grid_size)
    ls_y = ls - center_shift[1]
    ls_x = ls - center_shift[0]
    r2 = torch.sum(torch.stack(torch.meshgrid(ls_x, ls_y, indexing="ij"), -1).square(), -1)
    r = r2.sqrt()
    band_mask = (radius <= r) & (r <= radius + thickness)
    r_band_mask = r[band_mask]
    mask = torch.zeros((grid_size, grid_size))
    mask[r < radius] = 1
    mask[band_mask] = torch.cos(torch.pi * (r_band_mask - radius) / thickness) / 2 + .5
    mask[radius + thickness < r] = 0
    return mask


def smooth_spherical_mask(grid_size, radius, thickness, device="cpu"):
    ls = torch.linspace(-grid_size / 2, grid_size / 2, grid_size, device=device)
    r2 = torch.sum(torch.stack(torch.meshgrid(ls, ls, ls, indexing="ij"), -1).square(), -1)
    r = r2.sqrt()
    band_mask = (radius <= r) & (r <= radius + thickness)
    r_band_mask = r[band_mask]
    mask = torch.zeros((grid_size, grid_size, grid_size), device=device)
    mask[r < radius] = 1
    mask[band_mask] = torch.cos(torch.pi * (r_band_mask - radius) / thickness) / 2 + .5
    mask[radius + thickness < r] = 0
    return mask


def smooth_square_mask(image_size, square_side, thickness):
    square_side_2 = square_side / 2.
    y, x = np.meshgrid(
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size),
        np.linspace(-image_size // 2, image_size // 2 - 1, image_size)
    )
    p = np.max([np.abs(x), np.abs(y)], axis=0)
    band_mask = (square_side_2 <= p) & (p <= square_side_2 + thickness)
    p_band_mask = p[band_mask]
    mask = np.zeros((image_size, image_size))
    mask[p < square_side_2] = 1
    mask[band_mask] = np.cos(np.pi * (p_band_mask - square_side_2) / thickness) / 2 + .5
    mask[square_side_2 + thickness < p] = 0
    return mask


def bilinear_shift_2d(
        grid: torch.tensor,
        shift: torch.tensor,
        y_shift: torch.tensor = None
) -> torch.tensor:
    """
    Shifts a batch of 2D images
    :param grid: Batch of images [B, Y, X]
    :param shift: Either array of size [B, 2] (X, Y) or [B, 1] (X)
    :param y_shift: If 'None' assumes 'shift' contains both X and Y shifts
    :return: The shifted 2D images
    """
    if y_shift is not None:
        assert len(shift.shape) == 1 and len(y_shift.shape) == 1 and \
               shift.shape[0] == y_shift.shape[0]
        shift = torch.cat([shift[:, None], y_shift[:, None]], 1)

    assert len(shift.shape) == 2 and shift.shape[1] == 2
    int_shift = torch.floor(shift).long()

    s0 = shift - int_shift
    s1 = 1 - s0

    int_shift = int_shift.detach().cpu().numpy()
    g00 = torch.empty_like(grid)
    for i in range(len(grid)):
        g00[i] = torch.roll(grid[i], tuple(int_shift[i]), (-1, -2))

    g01 = torch.roll(g00, (0, 1), (-1, -2))
    g10 = torch.roll(g00, (1, 0), (-1, -2))
    g11 = torch.roll(g00, (1, 1), (-1, -2))

    g = g00 * s1[:, 0, None, None] * s1[:, 1, None, None] + \
        g10 * s0[:, 0, None, None] * s1[:, 1, None, None] + \
        g01 * s1[:, 0, None, None] * s0[:, 1, None, None] + \
        g11 * s0[:, 0, None, None] * s0[:, 1, None, None]

    return g


def integer_shift_2d(
        grid: torch.tensor,
        shift: torch.tensor,
        y_shift: torch.tensor = None
) -> torch.tensor:
    """
    Shifts a batch of 2D images
    :param grid: Batch of images [B, Y, X]
    :param shift: Either array of size [B, 2] (X, Y) or [B, 1] (X)
    :param y_shift: If 'None' assumes 'shift' contains both X and Y shifts
    :return: The shifted 2D images
    """
    if y_shift is not None:
        assert len(shift.shape) == 1 and len(y_shift.shape) == 1 and \
               shift.shape[0] == y_shift.shape[0]
        shift_ = torch.empty([shift.shape[0], 2])
        shift_[:, 0] = shift
        shift_[:, 1] = y_shift
        shift = shift_
    assert len(shift.shape) == 2 and shift.shape[1] == 2

    shift = shift.long().detach().cpu().numpy()
    g = torch.empty_like(grid)
    for i in range(len(grid)):
        g[i] = torch.roll(grid[i], tuple(shift[i]), (-1, -2))

    return g


def gaussian_blur(grid, sigma):
    ks = round(sigma * 3)
    ks = max(ks, 3)
    ks += 1 - ks % 2  # Make odd
    ts = np.linspace(-ks / 2, ks / 2, ks)
    gauss = np.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()

    sharp = grid
    blur = np.zeros_like(sharp)

    for i, k in enumerate(kernel):
        j = i - len(kernel) // 2
        blur[max(j, 0):len(blur) + min(j, 0)] += sharp[max(-j, 0):len(sharp) - max(j, 0)] * k

    if grid.ndim > 1:  # 2D
        sharp = blur
        blur = np.zeros_like(sharp)

        for i, k in enumerate(kernel):
            j = i - len(kernel) // 2
            blur[:, max(j, 0):len(blur) + min(j, 0)] += sharp[:, max(-j, 0):len(sharp) - max(j, 0)] * k

    if grid.ndim > 2:  # 3D
        sharp = blur
        blur = np.zeros_like(sharp)

        for i, k in enumerate(kernel):
            j = i - len(kernel) // 2
            blur[:, :, max(j, 0):len(blur) + min(j, 0)] += sharp[:, :, max(-j, 0):len(sharp) - max(j, 0)] * k

    return blur


def make_gaussian_kernel(sigma):
    ks = round(sigma * 9)
    ks = max(ks, 3)
    ks += 1 - ks % 2  # Make odd
    ts = torch.linspace(-ks / 2, ks / 2, ks)
    gauss = torch.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def fast_gaussian_filter(grid, kernel_sigma=None, kernel=None):
    if kernel is not None:
        k = kernel
    elif kernel_sigma is not None:
        k = make_gaussian_kernel(kernel_sigma).to(grid.device)
    else:
        raise RuntimeError("Either provide sigma or kernel.")
    
    shape = grid.shape
    if len(shape) == 2:
        grid = grid.view(1, 1, *shape)
    elif len(shape) == 3:
        grid = grid.view(1, *shape)

    if len(grid.shape) == 5:  # 3D
        grid = torch.nn.functional.conv3d(grid, k[None, None, :, None, None], padding='same')
        grid = torch.nn.functional.conv3d(grid, k[None, None, None, :, None], padding='same')
        grid = torch.nn.functional.conv3d(grid, k[None, None, None, None, :], padding='same')
    elif len(grid.shape) == 4:  # 2D
        grid = torch.nn.functional.conv2d(grid, k[None, None, :, None], padding='same')
        grid = torch.nn.functional.conv2d(grid, k[None, None, None, :], padding='same')
    else:
        raise NotImplementedError("Only 2D and 3D grids are supported.")

    grid = grid.view(shape)

    return grid


def local_correlation(grid1, grid2, kernel_size):
    std = torch.std(grid1) + 1e-28
    grid1 = grid1.unsqueeze(0).unsqueeze(0) / std
    grid2 = grid2.unsqueeze(0).unsqueeze(0) / std

    kernel = make_gaussian_kernel(kernel_size).to(grid1.device)

    def f(a): return fast_gaussian_filter(a, kernel=kernel)

    grid1_mean = grid1 - f(grid1)
    grid2_mean = grid2 - f(grid2)
    norm = torch.sqrt(f(grid1_mean.square()) * f(grid2_mean.square())) + 1e-12
    corr = f(grid1_mean * grid2_mean) / norm

    return corr.squeeze(0).squeeze(0)


def random_blob_on_grid(size, positive, negative, sigma, device="cpu"):
    grid = torch.zeros([size] * 3).to(device)
    if positive > 0:
        coord = torch.clip(torch.randn(3, positive) * size / 8. + size // 2, 0, size - 1).to(device).long()
        grid[coord[0], coord[1], coord[2]] += 1
    if negative > 0:
        coord = torch.clip(torch.randn(3, negative) * size / 8. + size // 2, 0, size - 1).to(device).long()
        grid[coord[0], coord[1], coord[2]] -= 1

    grid = fast_gaussian_filter(grid.unsqueeze(0).unsqueeze(0), kernel_sigma=sigma).squeeze(0).squeeze(0)
    return grid


def circular_mask(bz, radial_fraction=1.):
    ls = torch.linspace(-1, 1, bz)
    r2 = torch.sum(torch.stack(torch.meshgrid(ls, ls, indexing="ij"), -1).square(), -1)
    return r2 < radial_fraction ** 2


def spherical_mask(bz, radial_fraction=1.):
    ls = torch.linspace(-1, 1, bz)
    r2 = torch.sum(torch.stack(torch.meshgrid(ls, ls, ls, indexing="ij"), -1).square(), -1)
    return r2 < radial_fraction ** 2


def get_bounding_box(mask):
    # Find indices where values are greater than zero
    indices = torch.nonzero(mask, as_tuple=False)

    # If there are no non-zero values, handle it appropriately
    if indices.numel() == 0:
        return None
    else:
        # Get the min and max indices along each dimension
        min_indices = torch.min(indices, dim=0)[0]
        max_indices = torch.max(indices, dim=0)[0]

        # Bounding box coordinates
        return min_indices.tolist(), max_indices.tolist()


if __name__ == "__main__":
    device = "cuda:0"
    count = 1000
    size = 50
    spread = size / 15

    while True:
        with torch.no_grad():
            g = random_blob_on_grid(size, count, count // 2, spread, device=device)

        plt.imshow(g[size // 2].detach().cpu().numpy())
        plt.show()
