#!/usr/bin/env python

"""
Module for calculations related to grid manipulations.
This is temporary, functions should be organized in separate files.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union, TypeVar, List, Sequence

from .grid import make_gaussian_kernel
from .torch_utils import torch_interp

Tensor = TypeVar('torch.tensor')


def get_complex_float_type(type):
    if type == np.float16:
        return np.complex32
    elif type == np.float32:
        return np.complex64
    elif type == np.float64:
        return np.complex128
    elif type == torch.float16:
        return torch.complex32
    elif type == torch.float32:
        return torch.complex64
    elif type == torch.float64:
        return torch.complex128
    else:
        raise RuntimeError("Unknown float type")


def _dt_set_axes(shape, dim):
    if dim is None:
        return tuple((np.arange(len(shape)).astype(int)))

    return tuple((-np.arange(1, dim+1)[::-1].astype(int)))


class rfftn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=None):
        if dim is not None:
            ctx.save_for_backward(torch.Tensor([dim]).long())
        return torch.fft.rfftn(x, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = None
        if len(ctx.saved_tensors) > 0:
            dim = tuple(ctx.saved_tensors[0][0].long().cpu().tolist())
        grad_output = torch.conj(grad_output)
        grad = torch.fft.irfftn(grad_output, dim=dim, norm="forward")
        return grad, None


class irfftn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=None):
        if dim is not None:
            ctx.save_for_backward(torch.Tensor([dim]).long())
        return torch.fft.irfftn(x, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = None
        if len(ctx.saved_tensors) > 0:
            dim = tuple(ctx.saved_tensors[0][0].long().cpu().tolist())
        grad = torch.fft.rfftn(grad_output, dim=dim, norm="forward")
        grad = torch.conj(grad)
        return grad, None


def dft(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True,
        real_in: bool = False
) -> Union[Tensor, np.ndarray]:
    """
    Computes the Discrete Fourier Transform (DFT) of an input array.

    This function applies the DFT to an input grid, which can be either a NumPy array or a PyTorch tensor. 
    It supports multi-dimensional transforms and allows for optional frequency centering.

    Parameters:
        grid (Union[Tensor, np.ndarray]): The input array or tensor to be transformed.
        dim (int, optional): The dimensionality along which the transformation is applied. 
                             If None, all dimensions are transformed.
        center (bool, optional): If True, the zero-frequency component is shifted to the center of the spectrum.
        real_in (bool, optional): If True, assumes the input is real-valued and computes the Hermitian half of the spectrum.

    Returns:
        Union[Tensor, np.ndarray]: The transformed array, either as a NumPy array or a PyTorch tensor.
    """
    use_torch = torch.is_tensor(grid)
    axes = _dt_set_axes(grid.shape, dim)

    if real_in:
        grid_ft = rfftn.apply(torch.fft.fftshift(grid, dim=axes), axes) if use_torch \
            else np.fft.rfftn(np.fft.fftshift(grid, axes=axes), axes=axes)
    else:
        grid_ft = torch.fft.fftn(torch.fft.fftshift(grid, dim=axes), dim=axes) if use_torch \
            else np.fft.fftn(np.fft.fftshift(grid, axes=axes), axes=axes)

    if center:
        grid_ft = torch.fft.fftshift(grid_ft, dim=axes[:-1] if real_in else axes) if use_torch \
            else np.fft.fftshift(grid_ft, axes=axes[:-1] if real_in else axes)

    return grid_ft


def idft(
        grid_ft: Union[Tensor, np.ndarray],
        dim: int = None,
        centered: bool = True,
        real_in: bool = False
) -> Union[Tensor, np.ndarray]:
    """
    Computes the Inverse Discrete Fourier Transform (IDFT) of an input array.

    This function applies the IDFT to reconstruct the original signal from its frequency representation.
    It supports multi-dimensional inverse transforms and allows for optional frequency centering.

    Parameters:
        grid_ft (Union[Tensor, np.ndarray]): The transformed input array or tensor in the frequency domain.
        dim (int, optional): The dimensionality along which the inverse transformation is applied.
                             If None, all dimensions are transformed.
        centered (bool, optional): If True, assumes the zero-frequency component is centered and shifts it back.
        real_in (bool, optional): If True, assumes a Hermitian symmetric input (i.e., real-valued in time domain).

    Returns:
        Union[Tensor, np.ndarray]: The inverse-transformed array, either as a NumPy array or a PyTorch tensor.
    """
    use_torch = torch.is_tensor(grid_ft)
    axes = _dt_set_axes(grid_ft.shape, dim)

    if centered:
        grid_ft = torch.fft.ifftshift(grid_ft, dim=axes[:-1] if real_in else axes) if use_torch \
            else np.fft.ifftshift(grid_ft, axes=axes[:-1] if real_in else axes)

    if real_in:
        grid = torch.fft.ifftshift(irfftn.apply(grid_ft, axes), dim=axes) if use_torch \
            else np.fft.ifftshift(np.fft.irfftn(grid_ft, axes=axes), axes=axes)
    else:
        grid = torch.fft.ifftshift(torch.fft.ifftn(grid_ft, dim=axes), dim=axes) if use_torch \
            else np.fft.ifftshift(np.fft.ifftn(grid_ft, axes=axes), axes=axes)

    return grid


def rft(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True,
):
    """
    Computes the Real Fourier Transform (RFT) of an input array.

    This is a wrapper around `dft()` with `real_in=True`, meaning it assumes the input is real-valued 
    and only returns the Hermitian half of the transformed spectrum.

    Parameters:
        grid (Union[Tensor, np.ndarray]): The input array or tensor to be transformed.
        dim (int, optional): The dimensionality along which the transformation is applied.
                             If None, all dimensions are transformed.
        center (bool, optional): If True, the zero-frequency component is shifted to the center of the spectrum.

    Returns:
        Union[Tensor, np.ndarray]: The transformed array in the frequency domain.
    """
    return dft(
        grid=grid,
        dim=dim,
        center=center,
        real_in=True
    )


def irft(
        grid_ft: Union[Tensor, np.ndarray],
        dim: int = None,
        centered: bool = True,
):
    """
    Computes the Inverse Real Fourier Transform (IRFT) of an input array.

    This is a wrapper around `idft()` with `real_in=True`, reconstructing the original real-valued 
    signal from its frequency domain representation.

    Parameters:
        grid_ft (Union[Tensor, np.ndarray]): The transformed input array or tensor in the frequency domain.
        dim (int, optional): The dimensionality along which the inverse transformation is applied.
                             If None, all dimensions are transformed.
        centered (bool, optional): If True, assumes the zero-frequency component is centered and shifts it back.

    Returns:
        Union[Tensor, np.ndarray]: The inverse-transformed real-valued array.
    """
    return idft(
        grid_ft=grid_ft,
        dim=dim,
        centered=centered,
        real_in=True
    )


def rft2(
        grid: Union[Tensor, np.ndarray],
        center: bool = True,
):
    """
    Computes the 2D Real Fourier Transform (RFT2) of an input array.

    This function applies a real-valued Fourier transform specifically in two dimensions.

    Parameters:
        grid (Union[Tensor, np.ndarray]): The input 2D array or tensor to be transformed.
        center (bool, optional): If True, the zero-frequency component is shifted to the center of the spectrum.

    Returns:
        Union[Tensor, np.ndarray]: The transformed 2D array in the frequency domain.
    """
    return dft(
        grid=grid,
        dim=2,
        center=center,
        real_in=True
    )


def irft2(
        grid_ft: Union[Tensor, np.ndarray],
        centered: bool = True,
):
    """
    Computes the 2D Inverse Real Fourier Transform (IRFT2) of an input array.

    This function applies an inverse real-valued Fourier transform specifically in two dimensions.

    Parameters:
        grid_ft (Union[Tensor, np.ndarray]): The transformed 2D array or tensor in the frequency domain.
        centered (bool, optional): If True, assumes the zero-frequency component is centered and shifts it back.

    Returns:
        Union[Tensor, np.ndarray]: The inverse-transformed real-valued 2D array.
    """
    return idft(
        grid_ft=grid_ft,
        dim=2,
        centered=centered,
        real_in=True
    )


def dht(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True
) -> Union[Tensor, np.ndarray]:
    """
    Discreet Hartley transform
    :param grid: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param center: If the zeroth frequency should be centered
    :return: Transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid)
    axes = None if dim is None else _dt_set_axes(grid.shape, dim)

    grid_ht = torch.fft.fftn(torch.fft.fftshift(grid, dim=axes), dim=axes) if use_torch \
        else np.fft.fftn(np.fft.fftshift(grid, axes=axes), axes=axes)

    if center:
        grid_ht = torch.fft.fftshift(grid_ht, dim=axes) if use_torch \
            else np.fft.fftshift(grid_ht, axes=axes)

    return grid_ht.real - grid_ht.imag


def idht(
        grid_ht: Union[Tensor, np.ndarray],
        dim: int = None,
        centered: bool = True
) -> Union[Tensor, np.ndarray]:
    """
    Inverse Discreet Hartley transform
    :param grid_ht: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param centered: If the zeroth frequency should be centered
    :return: Inverse transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid_ht)
    axes = None if dim is None else _dt_set_axes(grid_ht.shape, dim)

    if centered:
        grid_ht = torch.fft.ifftshift(grid_ht, dim=axes) if use_torch \
            else np.fft.ifftshift(grid_ht, axes=axes)

    f = torch.fft.fftshift(torch.fft.fftn(grid_ht, dim=axes), dim=axes) if use_torch \
        else np.fft.fftshift(np.fft.fftn(grid_ht, axes=axes), axes=axes)

    # Adjust for FFT normalization
    if axes is None:
        f /= np.product(f.shape)
    else:
        f /= np.product(np.array(f.shape)[list(axes)])

    return f.real - f.imag


def htToFt(
        grid_ht: Union[Tensor, np.ndarray],
        dim: int = None
):
    """
    Converts a batch of Hartley transforms to Fourier transforms
    :param grid_ht: Batch of Hartley transforms
    :param dim: Data dimension
    :return: The batch of Fourier transforms
    """
    axes = tuple(np.arange(len(grid_ht.shape))) if dim is None else _dt_set_axes(grid_ht.shape, dim)
    dtype = get_complex_float_type(grid_ht.dtype)

    if torch.is_tensor(grid_ht):
        grid_ft = torch.empty(grid_ht.shape, dtype=dtype).to(grid_ht.device)
        grid_ht_ = torch.flip(grid_ht, axes)
        if grid_ht.shape[-1] % 2 == 0:
            grid_ht_ = torch.roll(grid_ht_, [1] * len(axes), axes)
        grid_ft.real = (grid_ht + grid_ht_) / 2
        grid_ft.imag = (grid_ht - grid_ht_) / 2
    else:
        grid_ft = np.empty(grid_ht.shape, dtype=dtype)
        grid_ht_ = np.flip(grid_ht, axes)
        if grid_ht.shape[-1] % 2 == 0:
            grid_ht_ = np.roll(grid_ht_, 1, axes)
        grid_ft.real = (grid_ht + grid_ht_) / 2
        grid_ft.imag = (grid_ht - grid_ht_) / 2

    return grid_ft


def dt_symmetrize(dt: Tensor, dim: int = None) -> Tensor:
    s = dt.shape
    if dim is None:
        dim = 3 if len(s) >= 3 else 2

    if s[-2] % 2 != 0:
        raise RuntimeError("Box size must be even.")

    if dim == 2:
        if s[-1] == s[-2]:
            if len(s) == 2:
                out = torch.empty((s[0] + 1, s[1] + 1), dtype=dt.dtype).to(dt.device)
            else:
                out = torch.empty((s[0], s[1] + 1, s[2] + 1), dtype=dt.dtype).to(dt.device)
            out[..., 0:-1, 0:-1] = dt
            out[..., -1, :-1] = dt[..., 0, :]
            out[..., :, -1] = out[..., :, 0]
        elif s[-1] == s[-2] // 2 + 1:
            if len(s) == 2:
                out = torch.empty((s[0] + 1, s[1]), dtype=dt.dtype).to(dt.device)
            else:
                out = torch.empty((s[0], s[1] + 1, s[2]), dtype=dt.dtype).to(dt.device)
            out[..., 0:-1, :] = dt
            out[..., -1, :] = dt[..., 0, :]
        else:
            raise RuntimeError("Dimensionality not supported")

    elif dim == 3:
        if s[-1] == s[-2]:
            if len(s) == 3:
                out = torch.empty((s[0] + 1, s[1] + 1, s[2] + 1), dtype=dt.dtype).to(dt.device)
            else:
                out = torch.empty((s[0], s[1] + 1, s[2] + 1, s[3] + 1), dtype=dt.dtype).to(dt.device)
            out[..., 0:-1, 0:-1, 0:-1] = dt
            out[...,   -1,  :-1,  :-1] = out[..., 0, :-1, :-1]
            out[...,  :,    :-1,   -1] = out[..., :, :-1,   0]
            out[...,  :,    -1,   :  ] = out[..., :,   0, :  ]
        elif s[-1] == s[-2] // 2 + 1:
            if len(s) == 3:
                out = torch.empty((s[0] + 1, s[1] + 1, s[2]), dtype=dt.dtype).to(dt.device)
            else:
                out = torch.empty((s[0], s[1] + 1, s[2] + 1, s[3]), dtype=dt.dtype).to(dt.device)
            out[..., 0:-1, 0:-1, :] = dt
            out[..., -1,    :-1, :] = out[..., 0, :-1, :]
            out[..., :,      -1, :] = out[..., :,   0, :]
        else:
            raise RuntimeError("Dimensionality not supported")
    else:
        raise RuntimeError("Dimensionality not supported")
    return out


def dt_desymmetrize(dt: Tensor, dim: int = None) -> Tensor:
    s = dt.shape
    if dim is None:
        dim = 3 if len(s) >= 3 else 2
    if dim == 2:
        if s[-2] == s[-1] * 2 - 1:
            out = dt[..., :-1, :]
            out[..., 0, :] = (dt[..., 0, :] + dt[..., -1, :]) / 2.
        else:
            out = dt[..., :-1, :-1]
            out[..., 0, :] = (dt[..., 0, :-1] + dt[..., -1, :-1]) / 2.
            out[..., :, 0] = (dt[..., :-1, 0] + dt[..., :-1, -1]) / 2.
    elif dim == 3:
        if s[-2] == s[-1] * 2 - 1:
            out = dt[..., :-1, :-1, :]
            out[..., 0, :, :] = (dt[..., 0, :-1, :] + dt[..., -1, :-1, :]) / 2.
            out[..., :, 0, :] = (dt[..., :-1, 0, :] + dt[..., :-1, -1, :]) / 2.
        else:
            out = dt[..., :-1, :-1, :-1]
            out[..., 0, :, :] = (dt[..., 0, :-1, :-1] + dt[..., -1, :-1, :-1]) / 2.
            out[..., :, 0, :] = (dt[..., :-1, 0, :-1] + dt[..., :-1, -1, :-1]) / 2.
            out[..., :, :, 0] = (dt[..., :-1, :-1, 0] + dt[..., :-1, :-1, -1]) / 2.
    else:
        raise RuntimeError("Dimensionality not supported")

    return out


def rdht(
        grid: Union[Tensor, np.ndarray],
        dim: int = None,
        center: bool = True
) -> Union[Tensor, np.ndarray]:
    """
    Discreet Hartley transform
    :param grid: Numpy array or Pytorch tensor to be transformed, can be stack
    :param dim: If stacked grids, specify the dimensionality
    :param center: If the zeroth frequency should be centered
    :return: Transformed Numpy array or Pytorch tensor
    """
    use_torch = torch.is_tensor(grid)
    axes = tuple(np.arange(len(grid.shape))) if dim is None else _dt_set_axes(grid.shape, dim)

    if use_torch:
        grid_ft = torch.fft.rfftn(torch.fft.fftshift(grid, dim=axes), dim=axes)

        grid_ht = torch.empty_like(grid)
        grid_ht[..., :grid_ft.shape[-1]] = grid_ft.real - grid_ft.imag
        hh = torch.flip(grid_ft.real[..., 1:-1] + grid_ft.imag[..., 1:-1], axes)

        hh = torch.roll(hh, [1] * (len(axes) - 1), axes[:-1])
        grid_ht[..., grid_ft.shape[-1]:] = hh
    else:
        grid_ft = np.fft.rfftn(np.fft.fftshift(grid, axes=axes), axes=axes)

        grid_ht = np.empty(grid.shape)
        grid_ht[..., :grid_ft.shape[-1]] = grid_ft.real - grid_ft.imag
        hh = np.flip(grid_ft.real[..., 1:-1] + grid_ft.imag[..., 1:-1], axes)

        hh = np.roll(hh, 1, axis=axes[:-1])
        grid_ht[..., grid_ft.shape[-1]:] = hh

    if center:
        grid_ht = torch.fft.fftshift(grid_ht, dim=axes) if use_torch \
            else np.fft.fftshift(grid_ht, axes=axes)

    return grid_ht


def ird3ht(
        grid_ht: Tensor
) -> Tensor:
    """
    Inverse Discreet Hartley transform, carried out by doing a conversion
    to a Fourier transform and using rfft. Assumes centered!!!
    :param grid_ht: Pytorch tensor with batch of 3D grids to be transformed
    :return: Inverse transformed Pytorch tensor
    """
    s = grid_ht.shape[-1]
    assert len(grid_ht.shape) == 4
    assert s == grid_ht.shape[-3] and s == grid_ht.shape[-2]
    assert s % 2 == 1

    axes = (1, 2, 3)
    grid_ft = htToFt(grid_ht, dim=3)
    grid_ft = grid_ft[..., s // 2:]
    grid_ft = torch.fft.ifftshift(grid_ft, dim=axes)
    grid = torch.fft.ifftshift(torch.fft.irfftn(grid_ft, dim=axes), dim=axes)

    return grid


def fourier_shift_2d(
        grid_ft: Union[Tensor, np.ndarray],
        shift: Union[Tensor, np.ndarray],
        y_shift: Union[Tensor, np.ndarray] = None
) -> Union[Tensor, np.ndarray]:
    """
    Shifts a batch of 2D Fourier transformed images
    :param grid_ft: Batch of Fourier transformed images [B, Y, X]
    :param shift: Either array of size [B, 2] (X, Y) or [B, 1] (X)
    :param y_shift: If 'None' assumes 'shift' contains both X and Y shifts
    :return: The shifted 2D Fourier transformed images
    """
    complex_channels = len(grid_ft.shape) == 4 and grid_ft.shape[-1] == 2
    assert len(grid_ft.shape) == 3 or complex_channels
    assert shift.shape[0] == grid_ft.shape[0]
    s = grid_ft.shape[1]
    symmetrized = s % 2 == 1
    if symmetrized:
        s -= 1

    if y_shift is None:
        assert len(shift.shape) == 2 and shift.shape[1] == 2
        x_shift = shift[..., 0]
        y_shift = shift[..., 1]
    else:
        assert len(shift.shape) == 1 and len(y_shift.shape) == 1 and \
               shift.shape[0] == y_shift.shape[0]
        x_shift = shift
        y_shift = y_shift

    x_shift = x_shift / float(s)
    y_shift = y_shift / float(s)

    if symmetrized:
        ls = torch.linspace(-s // 2, s // 2, s + 1)
    else:
        ls = torch.linspace(-s // 2, s // 2 - 1, s)
    lsx = torch.linspace(0, s // 2, s // 2 + 1)
    y, x = torch.meshgrid(ls, lsx, indexing='ij')
    x = x.to(grid_ft.device)
    y = y.to(grid_ft.device)
    dot_prod = 2 * np.pi * (x[None, :, :] * x_shift[:, None, None] + y[None, :, :] * y_shift[:, None, None])
    a = torch.cos(dot_prod)
    b = torch.sin(dot_prod)

    if complex_channels:
        ar = a * grid_ft[..., 0]
        bi = b * grid_ft[..., 1]
        ab_ri = (a + b) * (grid_ft[..., 0] + grid_ft[..., 1])
        r = ar - bi
        i = ab_ri - ar - bi
        return torch.cat([r.unsqueeze(-1), i.unsqueeze(-1)], -1)
    else:
        ar = a * grid_ft.real
        bi = b * grid_ft.imag
        ab_ri = (a + b) * (grid_ft.real + grid_ft.imag)

        return ar - bi + 1j * (ab_ri - ar - bi)


def grid_spectral_sum(grid, indices):
    if len(grid.shape) == len(indices.shape) and np.all(grid.shape == indices.shape):  # Has no batch dimension
        spectrum = torch.zeros(int(torch.max(indices)) + 1).to(grid.device)
        spectrum.scatter_add_(0, indices.long().flatten(), grid.flatten())
    elif len(grid.shape) == len(indices.shape) + 1 and np.all(grid.shape[1:] == indices.shape):  # Has batch dimension
        spectrum = torch.zeros([grid.shape[0], int(torch.max(indices)) + 1]).to(grid.device)
        indices = indices.long().unsqueeze(0).expand([grid.shape[0]] + list(indices.shape))
        spectrum.scatter_add_(1, indices.flatten(1), grid.flatten(1))
    else:
        raise RuntimeError("Shape of grid must match spectral_indices, except along the batch dimension.")
    return spectrum


def grid_spectral_average(grid, indices):
    indices = indices.long()
    spectrum = grid_spectral_sum(grid, indices)
    norm = grid_spectral_sum(torch.ones_like(indices, dtype=torch.float32), indices)
    if len(spectrum.shape) == 2:  # Batch dimension
        return spectrum / norm[None, :]
    else:
        return spectrum / norm


def spectra_to_grid(spectra, indices):
    if len(spectra.shape) == 1:  # Has no batch dimension
        grid = torch.gather(spectra, 0, indices.flatten().long())
    elif len(spectra.shape) == 2:  # Has batch dimension
        indices = indices.unsqueeze(0).expand([spectra.shape[0]] + list(indices.shape))
        grid = torch.gather(spectra.flatten(1), 1, indices.flatten(1).long())
    else:
        raise RuntimeError("Spectra must be at most two-dimensional (one batch dimension).")
    return grid.view(indices.shape)


def spectral_correlation(grid1, grid2, indices, normalize=False, norm_eps=1e-12):
    if np.any(grid1.shape != grid2.shape):
        print('The grids have to be the same shape')
    correlation = torch.real(grid1 * torch.conj(grid2))

    if normalize:
        correlation = grid_spectral_sum(correlation, indices)
        norm1 = grid_spectral_sum(grid1.abs().square(), indices)
        norm2 = grid_spectral_sum(grid2.abs().square(), indices)
        return correlation / ((norm1 * norm2).sqrt() + norm_eps)
    else:
        return grid_spectral_average(correlation, indices)
    

import torch
from typing import Tuple, Union

# Assume get_freq is defined as follows:
def get_freq(
    shape: Tuple[int, ...],
    pixel_size: Union[float, Tuple[float, ...]] = 1.0,
    rfft: bool = False,
    center: bool = True,
    device: str = "cpu"
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Compute frequency coordinates (in cycles per unit length) for an N-dimensional grid using PyTorch.
    Supports non-cubic (or non-square) shapes. If `rfft` is True, the function computes the frequency
    axis for the last dimension using torch.fft.rfftfreq (assuming Hermitian symmetry on that axis),
    while all other dimensions use torch.fft.fftfreq (with an optional FFT shift if center is True).

    Parameters:
      shape (Tuple[int, ...]): A tuple specifying the size along each dimension in the spatial domain.
      pixel_size (float or Tuple[float, ...]): Physical pixel spacing. If given as a float,
                    that spacing is used for all dimensions; if a tuple, it must match the shape.
      rfft (bool): If True, compute the frequency axis for the last dimension using rfftfreq.
      center (bool): If True, full frequency axes (computed via fftfreq) are shifted so that zero frequency is centered.
                     For the rfft axis, the natural ordering (nonnegative frequencies) is preserved.
      device (str): The PyTorch device for the output tensors.

    Returns:
      For 1D input, returns a single PyTorch tensor representing the frequency axis.
      For an N-D input (N > 1), returns a tuple of PyTorch tensors representing the frequency grids
      for each dimension with "ij" indexing.
    """
    ndim = len(shape)
    
    # Broadcast pixel_size if necessary.
    if not isinstance(pixel_size, (tuple, list)):
        pixel_size = (float(pixel_size),) * ndim
    elif len(pixel_size) != ndim:
        raise ValueError("pixel_size must be a single number or a tuple with length equal to the number of dimensions.")
    
    freq_axes = []
    # When rfft=True, only the last dimension uses rfftfreq.
    for i, (n, d_i) in enumerate(zip(shape, pixel_size)):
        if rfft and (i == ndim - 1):
            f = torch.fft.rfftfreq(n, d=d_i)
        else:
            f = torch.fft.fftfreq(n, d=d_i)
            if center:
                f = torch.fft.fftshift(f)
        freq_axes.append(f.to(device))
    
    # Create the N-dimensional frequency grid with "ij" indexing.
    grids = torch.meshgrid(*freq_axes, indexing="ij")
    
    if ndim == 1:
        return grids[0]
    else:
        return tuple(grids)


def get_spectral_indices(
    shape: Tuple[int, ...],
    center: bool = True,
    maxr: Union[int, None] = None,
    rfft: bool = False,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Computes spectral (radial) indices for an N-dimensional frequency grid.
    Uses get_freq to generate the per-axis frequency grids.
    The radial index (the floored Euclidean distance) is computed at each grid point.
    Only the DC component will be be zero.
    The indices are scaled by the minimum shape dimension.
    

    Parameters:
      shape (Tuple[int, ...]): The shape of the spatial-domain tensor.
      center (bool): Whether full FFT axes are centered.
      maxr (int or None): Optionally clip the index values to this maximum.
      rfft (bool): If True, the last axis is computed with torch.fft.rfftfreq (assuming Hermitian symmetry).
      device (str): The device for torch tensors.
      
    Returns:
      indices (torch.Tensor): An integer tensor where each value is the floored radial frequency index.
    """
    # Obtain the frequency grids (in cycles per unit length) using get_freq.
    grids = get_freq(shape, pixel_size=1, rfft=rfft, center=center, device=device)
    
    # Ensure we have a tuple of tensors (even for the 1D case).
    if not isinstance(grids, (tuple, list)):
        grids = (grids,)
    
    # Scale each frequency grid to convert from cycles per unit length to discrete bin-like values.
    # This scaling factor is the product of the number of points along that axis and the sample spacing.
    min_scale = min(shape)
    scaled_grids = []
    for g in grids:
        scaled_grids.append(g * min_scale)
    
    # Compute the squared Euclidean distance at each grid point.
    r2 = sum(g**2 for g in scaled_grids)
    
    # Take the square root, floor the result, and cast to an integer type.
    indices = torch.floor(torch.sqrt(r2)).long()

    # Only DC should be zero
    mask = (r2 > 0) & (indices == 0)
    indices[mask] = 1
    
    # Optionally clip indices to a maximum value.
    if maxr is not None:
        indices = torch.clamp(indices, max=maxr)
    
    return indices


def get_white_filter(grid_ft, spectrum=None, smoothen=None, centered=True, return_power_spectrum=False, eps=1e-30):
    """
    Computes a whitening filter for a given Fourier-transformed grid.

    This function estimates the power spectrum of the input Fourier-transformed grid, inverts it,
    and constructs a corresponding filter that can be applied to whiten the input grid. Optionally,
    a predefined power spectrum can be provided to modify the whitening process.

    Args:
        grid_ft (torch.Tensor): The Fourier-transformed grid (complex-valued tensor).
        spectrum (torch.Tensor, optional): A predefined power spectrum for scaling (default: None).
        smoothen (float, optional): Apply gaussian smoothening on power spectrum with this sigma (default: None).
        centered (bool, optional): Whether the Fourier space representation is centered (default: True).
        return_power_spectrum (bool, optional): Whether to return the computed power spectrum (default: False).
        eps (float, optional): A small constant added to the power spectrum for numerical stability (default: 1e-30).

    Returns:
        torch.Tensor or tuple:
            - filter (torch.Tensor): The whitening filter in Fourier space.
            - power_spectrum (torch.Tensor, optional): The computed power spectrum of the input grid.
              This is only returned if `return_power_spectrum` is set to True.
    """
    power_grid = torch.view_as_real(grid_ft).square().sum(-1)
    idx = get_spectral_indices(grid_ft.shape, center=centered, device=grid_ft.device)  # TODO Cache this
    power_spectrum = grid_spectral_average(power_grid, idx)

    if smoothen is not None and smoothen > 0:
        kernel = make_gaussian_kernel(smoothen).to(grid_ft.device)
        power_spectrum = F.conv1d(power_spectrum[None, None], kernel[None, None], padding='same')[0, 0]

    power_spectrum_inv = 1 / (power_spectrum.sqrt() + eps)

    if spectrum is not None:
        power_spectrum_inv *= spectrum
    filter = spectra_to_grid(power_spectrum_inv, idx)

    if return_power_spectrum:
        return filter, power_spectrum
    else:
        return filter


def whiten_fourier(grid_ft, spectrum=None, smoothen=None, centered=True, return_power_spectrum=False, eps=1e-30):
    
    """
    Whitens a Fourier-transformed grid by normalizing its power spectrum.

    This function computes the power spectrum of the input Fourier-transformed grid, inverts it,
    and applies the inverse power spectrum to the input grid to whiten it. Optionally, a predefined
    power spectrum can be provided for scaling.

    Args:
        grid_ft (torch.Tensor): The Fourier-transformed grid (complex-valued tensor).
        spectrum (torch.Tensor, optional): A predefined power spectrum for scaling (default: None).
        smoothen (float, optional): Apply gaussian smoothening on power spectrum with this sigma (default: None).
        centered (bool, optional): Whether the Fourier space representation is centered (default: False).

    Returns:
        tuple:
            - whitened_ft (torch.Tensor): The whitened Fourier-transformed grid.
            - power_spectrum (torch.Tensor): The computed power spectrum of the input grid.
    """
    filter = get_white_filter(
        grid_ft=grid_ft, 
        spectrum=spectrum,
        smoothen=smoothen,
        centered=centered, 
        return_power_spectrum=return_power_spectrum, 
        eps=eps
    )
    whitened_ft = grid_ft * filter

    if return_power_spectrum:
        return whitened_ft, filter
    else:
        return whitened_ft

def whiten_real(grid, spectrum=None, smoothen=None, return_power_spectrum=False):
    """
    Whitens a real-valued spatial grid by normalizing its Fourier power spectrum.

    This function first transforms the input grid to Fourier space, applies whitening using `whiten_fourier`,
    and then transforms the result back to the spatial domain.

    Args:
        grid (torch.Tensor): The real-valued input grid.
        spectrum (torch.Tensor, optional): A predefined power spectrum for scaling (default: None).
        smoothen (float, optional): Apply gaussian smoothening on power spectrum with this sigma (default: None).
        return_power_spectrum (bool, optional): Whether to return the computed power spectrum (default: False).

    Returns:
        torch.Tensor or tuple:
            - whitened (torch.Tensor): The whitened spatial domain grid.
            - power_spectrum (torch.Tensor, optional): The computed power spectrum (if `return_power_spectrum` is True).
    """
    grid_ft = dft(grid, center=False, real_in=True)
    whitened_ft, power_spectrum = whiten_fourier(
        grid_ft, spectrum=spectrum, smoothen=smoothen, centered=False, return_power_spectrum=True)
    whitened = idft(whitened_ft, centered=False, real_in=True)

    if return_power_spectrum:
        return whitened, power_spectrum
    else:
        return whitened


def interpolate_power_spectrum(power_spectrum_freq, power_spectrum, target_freq):
    """
    Interpolates the power_spectrum onto new positions in target_freq.
    
    Uses:
      - `np.interp` if inputs are NumPy arrays.
      - `torch.nn.functional.grid_sample` if inputs are PyTorch tensors.

    Parameters:
    - power_spectrum_freq (array-like): 1D array/tensor of original frequencies (must be sorted).
    - power_spectrum (array-like): 1D array/tensor of power spectrum values.
    - target_freq (array-like): 1D array/tensor of target frequencies to interpolate onto.

    Returns:
    - Interpolated values, in the same type as the input (NumPy array or PyTorch tensor).
    """
    if isinstance(power_spectrum_freq, np.ndarray):
        return np.interp(target_freq, power_spectrum_freq, power_spectrum)
    elif isinstance(power_spectrum_freq, torch.Tensor):
        return torch_interp(target_freq, power_spectrum_freq, power_spectrum)
    else:
        raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")


def spectral_resolution(ft_size, voxel_size):
    """
    Get list of inverted resolutions (1/Angstroms) for each spectral index in a Fourier transform.
    """
    res = torch.zeros(ft_size)
    res[1:] = torch.arange(1, ft_size) / (2 * voxel_size * ft_size)
    return res


def spectral_index_from_resolution(resolution: float, image_size: int, pixel_size: float):
    """
    Get spectral index from resolution in Angstroms
    """
    return round(image_size * pixel_size / resolution)


def resolution_from_spectral_index(index: int, image_size: int, pixel_size: float):
    """
    Get the resolution in Angstroms.
    """
    return pixel_size * image_size / float(index)


def resolution_from_fsc(fsc, res, threshold=0.5):
    """
    Get the resolution (res) at the FSC (fsc) threshold.
    """
    assert len(fsc) == len(res)
    if torch.is_tensor(fsc):
        i = torch.argmax(fsc < threshold)
    else:
        i = np.argmax(fsc < threshold)
    if i > 0:
        return res[i - 1]
    else:
        return res[0]


def get_fsc_fourier(grid1_df, grid2_df, centered=True):
    indices = get_spectral_indices(grid1_df.shape, center=centered).to(grid1_df.device)
    fsc = spectral_correlation(grid1_df, grid2_df, indices, normalize=True)
    return fsc[:grid1_df.shape[-1]]


def get_fsc_real(grid1, grid2):
    grid1_df = rft(grid1, dim=3, center=False)
    grid2_df = rft(grid2, dim=3, center=False)
    return get_fsc_fourier(grid1_df, grid2_df, centered=False)


def get_power_fourier(grid_df, centered=True):
    indices = get_spectral_indices(grid_df.shape, center=centered)
    power = grid_df.abs().square()
    return grid_spectral_average(power, indices)


def get_power_real(grid):
    grid_df = rft(grid, dim=3, center=False)
    return get_power_fourier(grid_df, centered=False)


def rescale_fourier(grid, out_sz):
    if out_sz % 2 != 0:
        raise Exception(f"Bad output size: {out_sz}")
    if out_sz == grid.shape[0]:
        return grid

    use_torch = torch.is_tensor(grid)

    if len(grid.shape) == 2:
        if grid.shape[0] != (grid.shape[1] - 1) * 2:
            raise Exception("Input must be cubic")

        if use_torch:
            g = torch.zeros((out_sz, out_sz // 2 + 1), device=grid.device, dtype=grid.dtype)
        else:
            g = torch.zeros((out_sz, out_sz // 2 + 1), dtype=grid.dtype)
        i = np.array(grid.shape) // 2
        o = np.array(g.shape) // 2

        if o[0] < i[0]:
            g = grid[i[0] - o[0]: i[0] + o[0], :g.shape[1]]
        elif o[0] > i[0]:
            g[o[0] - i[0]: o[0] + i[0], :grid.shape[1]] = grid
    elif len(grid.shape) == 3:
        if grid.shape[0] != grid.shape[1] or \
                grid.shape[1] != (grid.shape[2] - 1) * 2:
            raise Exception("Input must be cubic")
        if use_torch:
            g = torch.zeros((out_sz, out_sz, out_sz // 2 + 1), device=grid.device, dtype=grid.dtype)
        else:
            g = torch.zeros((out_sz, out_sz, out_sz // 2 + 1), dtype=grid.dtype)
        i = np.array(grid.shape) // 2
        o = np.array(g.shape) // 2

        if o[0] < i[0]:
            g = grid[i[0] - o[0]: i[0] + o[0], i[1] - o[1]: i[1] + o[1], :g.shape[2]]
        elif o[0] > i[0]:
            g[o[0] - i[0]: o[0] + i[0], o[1] - i[1]: o[1] + i[1], :grid.shape[2]] = grid
    else:
        raise RuntimeError("Only 2D and 3D tensors supported.")

    return g


def rescale_real(grid, out_sz):
    grid_ft = dft(grid, center=True, real_in=True)
    grid_ft = rescale_fourier(grid_ft, out_sz)
    return idft(grid_ft, centered=True, real_in=True)

def size_by_voxel_size(size: int, voxel_size: float, target_voxel_size: float) -> int:
    """Takes grid size, current voxel size and target voxel size. 
    Returns closest grid size and voxel size for optimal Fourier rescaling."""
    in_sz = size
    out_sz = int(round(in_sz * voxel_size / target_voxel_size))
    if out_sz % 2 != 0:
        vs1 = voxel_size * in_sz / (out_sz + 1)
        vs2 = voxel_size * in_sz / (out_sz - 1)
        if np.abs(vs1 - target_voxel_size) < np.abs(vs2 - target_voxel_size):
            out_sz += 1
        else:
            out_sz -= 1

    return out_sz, voxel_size * in_sz / out_sz

def rescale_voxelsize(grid, voxel_size, target_voxel_size, realspace=True):
    bz, _ = size_by_voxel_size(grid.size(0), voxel_size=voxel_size, target_voxel_size=target_voxel_size)
    if realspace:
        return rescale_real(grid, bz)
    else:
        return rescale_fourier(grid, bz)

def spectrum_to_grid_mean(spectrum, dim=2):
    assert spectrum.dim() == 1
    count = torch.arange(1, spectrum.size(0) + 1, device=spectrum.device, dtype=spectrum.dtype).pow(dim)
    return (spectrum * count).sum() / count.sum()


def bfactor_grid(
    b_factor: Tensor,
    shape: Tuple[int, ...],
    pixel_size: float,
    rfft: bool = False,
    center: bool = True
):
    freq = get_freq(
        shape=shape,
        pixel_size=pixel_size,
        rfft=rfft,
        device=b_factor.device,
        center=center
    )

    if len(shape) == 1:
        n4 = freq**4
    elif len(shape) == 2:
        freq_x, freq_y = freq
        xx = freq_x**2
        yy = freq_y**2
        n4 = (xx + yy)**2  # Norms squared^2
        
    return torch.exp(-b_factor/4. * n4[None])