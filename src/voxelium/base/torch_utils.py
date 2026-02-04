#!/usr/bin/env python

"""
Simple Pytorch tools
"""

import argparse
import glob
import importlib.util
import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from voxelium.base.plot import get_default_cmap


def standardize(np_input):
    mean = np.mean(np_input, axis=(1, 2, 3, 4))
    mean = np.resize(mean, (np_input.shape[0], 1, 1, 1, 1))
    std = np.std(np_input, axis=(1, 2, 3, 4)) + 1e-12
    std = np.resize(std, (np_input.shape[0], 1, 1, 1, 1))
    return mean, std


def torch_standardize(torch_input):
    mean = torch.mean(torch_input, dim=(1, 2, 3, 4))
    mean = torch.reshape(mean, (torch_input.shape[0], 1, 1, 1, 1))
    std = torch.std(torch_input, dim=(1, 2, 3, 4)) + 1e-12
    std = torch.reshape(std, (torch_input.shape[0], 1, 1, 1, 1))
    return mean, std


def normalize(np_input):
    norm = np.sqrt(np.sum(np.square(np_input), axis=(1, 2, 3, 4))) + 1e-12
    norm = np.resize(norm, (np_input.shape[0], 1, 1, 1, 1))
    return norm


def torch_normalize(torch_input):
    norm = torch.sqrt(torch.sum((torch_input) ** 2, dim=(1, 2, 3, 4))) + 1e-12
    norm = torch.reshape(norm, (torch_input.shape[0], 1, 1, 1, 1))
    return norm


def torch_interp(x: torch.Tensor, y: torch.Tensor, xp: torch.Tensor):
    ''' 
    x : [..., N]
    y : [..., N]
    xp: [..., P]
    '''
    x_min, min_indices = torch.min(x, dim= -1, keepdim = True)
    x_max, max_indices = torch.max(x, dim= -1, keepdim = True)

    y_min = torch.gather(y, -1, min_indices)
    y_max = torch.gather(y, -1, max_indices)

    xp_min = torch.amin(xp, -1, keepdim= True)
    xp_max = torch.amax(xp, -1, keepdim= True)

    ## Handle the case where out of bound value in support
    x = torch.cat([torch.minimum(x_min, xp_min), x, torch.maximum(x_max, xp_max)], dim = -1)
    y = torch.cat([y_min, y, y_max], dim = -1)

    x_sorted, sorted_idx = torch.sort(x, dim = -1)
    y_sorted = torch.gather(y, -1, sorted_idx)

    right_idx = torch.searchsorted(x_sorted, xp)
    left_idx = right_idx.sub(1).clamp(0, x.shape[-1]-1)

    left_dist = xp - torch.gather(x_sorted, -1, left_idx)
    right_dist = torch.gather(x_sorted, -1, right_idx) - xp

    left_y = torch.gather(y_sorted, -1, left_idx)
    right_y = torch.gather(y_sorted, -1, right_idx)

    yp = left_y + left_dist/(left_dist + right_dist) * (right_y - left_y)

    return yp


def make_imshow_fig(data):
    if len(data.shape) == 3:
        data = data[data.shape[0] // 2]

    if torch.is_tensor(data):
        data = data.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(data)
    plt.axis("off")
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)

    try:
        matplotlib.use(backend)
    except ImportError:
        pass

    return fig


def make_scatter_fig(x, y, c=None):
    if torch.is_tensor(x):
        x = x.detach().data.cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    if c is None:
        c = np.arange(len(x))
    else:
        if torch.is_tensor(c):
            c = c.detach().data.cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    alpha = min(10. / np.sqrt(len(x)), 1.)
    ax.scatter(x, y, edgecolors=None, marker='.', c=c, cmap="summer", alpha=alpha)

    mx = np.mean(x)
    sx = np.std(x) * 3
    my = np.mean(y)
    sy = np.std(y) * 3

    ax.set_xlim([mx - sx, mx + sx])
    ax.set_ylim([my - sy, my + sy])

    plt.subplots_adjust(top=0.99, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    try:
        matplotlib.use(backend)
    except ImportError:
        pass

    return fig


def make_heatmap_fig(x, y, bins=400, sigma=4, cm=None):
    if torch.is_tensor(x):
        x = x.detach().data.cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 7))

    lim = 3

    mx = np.mean(x)
    sx = np.std(x) * lim
    my = np.mean(y)
    sy = np.std(y) * lim

    x_min = mx - sx
    x_max = mx + sx
    y_min = my - sy
    y_max = my + sy

    x = ((x - mx) / (2 * sx) + 0.5) * bins
    y = ((y - my) / (2 * sy) + 0.5) * bins

    mask = (0 <= x) & (x < bins - 0.5) & (0 <= y) & (y < bins - 0.5)
    x = np.round(x[mask]).astype(int)
    y = np.round(y[mask]).astype(int)

    z = np.zeros([bins, bins])
    np.add.at(z, (y, x), 1)

    from scipy.ndimage import gaussian_filter
    z = gaussian_filter(z, sigma)

    if cm is None:
        cm = get_default_cmap()

    ax.imshow(z, cmap=cm, extent=(x_min, x_max, y_max, y_min))

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    plt.subplots_adjust(top=0.99, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    try:
        matplotlib.use(backend)
    except ImportError:
        pass

    return fig


def make_line_fig(x, y, y_log=False):
    if torch.is_tensor(x):
        x = x.detach().data.cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().data.cpu().numpy()

    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y)

    if y_log:
        ax.set_yscale('log')

    plt.subplots_adjust(top=0.99, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    try:
        matplotlib.use(backend)
    except ImportError:
        pass

    return fig


def make_series_line_fig(data, y_log=False):
    backend = matplotlib.rcParams['backend']
    matplotlib.use('pdf')  # To avoid issues with disconnected X-server over ssh

    fig, ax = plt.subplots(figsize=(7, 5))
    do_legend = False
    for d in data:
        y = d['y']
        if torch.is_tensor(y):
            y = y.detach().data.cpu().numpy()

        if 'x' in d:
            x = d['x']
            if torch.is_tensor(x):
                x = x.detach().data.cpu().numpy()
        else:
            x = np.arange(len(y))
        ax.plot(x, y,
                label=d['label'] if 'label' in d else None,
                color=d['color'] if 'color' in d else None,
                linestyle=d['linestyle'] if 'linestyle' in d else None
                )
        do_legend = 'label' in d or do_legend

    if do_legend:
        ax.legend()
    ax.grid()

    if y_log:
        ax.set_yscale('log')

    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.99, left=0.1,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    try:
        matplotlib.use(backend)
    except ImportError:
        pass

    return fig


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def optimizer_set_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    # plt.barh(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.barh(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.yticks(range(0, len(ave_grads), 1), layers)
    plt.xlabel("average gradient")
    plt.ylabel("Layers")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)],
               ['max-gradient', 'mean-gradient'])
    plt.show()


def pca_dim_reduction_(x, n_components: int = 2, subsample: int = None):
    x_ = x
    if subsample is not None and subsample < x.size(0):
        idx = torch.randperm(x.size(0), device=x.device)
        x_ = x[idx[:subsample]]

    # Compute the covariance matrix
    cov_mat = torch.cov(x_.t())

    # Compute the eigenvalues and eigenvectors
    eigen_values, eigen_vectors = torch.linalg.eigh(cov_mat, UPLO='U')

    # Sort the eigenvalues and eigenvectors
    sorted_indices = torch.argsort(eigen_values, descending=True)
    sorted_eigenvectors = eigen_vectors[:, sorted_indices]

    # Select the top k eigenvectors (n_components)
    W = sorted_eigenvectors[:, :n_components]

    # Transform the original dataset
    transformed_data = torch.matmul(x, W)

    return transformed_data


def pca_dim_reduction(x, n_components=2, subsample=None, raise_oom=False):
    try:
        transformed_data = pca_dim_reduction_(x, n_components=n_components, subsample=subsample)
    except RuntimeError as e:
        if 'out of memory' in str(e) and not raise_oom:
            transformed_data = pca_dim_reduction_(x.to("cpu"), n_components=n_components, subsample=subsample)
            transformed_data = transformed_data.to(x.device)

    return transformed_data



def torch_interp1d(x, xp, fp, left=None, right=None):
    """
    A PyTorch implementation of numpy.interp for 1-D tensors.
    Args:
      x:  tensor of query points, shape (N,)
      xp: 1-D tensor of known x-coordinates, must be sorted ascending, shape (M,)
      fp: tensor of values at xp, same shape as xp (or broadcastable), shape (M,)
      left:  value to use for x < xp[0]  (default: fp[0])
      right: value to use for x > xp[-1] (default: fp[-1])
    Returns:
      tensor of interpolated values, shape (N,)
    """
    # ensure inputs are 1-D
    x = x.flatten()
    xp = xp.flatten()
    fp = fp.flatten()
    
    # where would each x land in xp?
    idx = torch.searchsorted(xp, x)  # returns indices in [0..M]
    
    # clamp indices to valid interpolation range [1..M-1]
    idx_lo = idx.clamp(min=1, max=xp.numel()-1)
    
    x0 = xp[idx_lo - 1]
    x1 = xp[idx_lo    ]
    y0 = fp[idx_lo - 1]
    y1 = fp[idx_lo    ]
    
    # compute fraction (x - x0)/(x1 - x0), safely
    denom = x1 - x0
    # avoid division by zero if xp has duplicates
    denom = torch.where(denom == 0, torch.tensor(1., device=denom.device), denom)
    frac = (x - x0) / denom
    
    y = y0 + frac * (y1 - y0)
    
    # handle out-of-bounds
    if left is None:
        left = fp[0]
    if right is None:
        right = fp[-1]
    y = torch.where(x < xp[0], torch.as_tensor(left, device=y.device), y)
    y = torch.where(x > xp[-1], torch.as_tensor(right, device=y.device), y)
    
    return y
