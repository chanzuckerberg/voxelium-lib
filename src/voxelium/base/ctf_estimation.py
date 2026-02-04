#!/usr/bin/env python

"""
Module for fitting contrast transfer function (CTF).
"""

import numpy as np
import torch

import torch
import torch.nn.functional as F

import matplotlib.pylab as plt

from voxelium.base import fast_gaussian_filter

def ellipse_coordinates(a, phi, r):
    """Generate normalized grid coordinates for an ellipse at a given radius r."""
    theta = torch.linspace(-np.pi * 0.55 - phi, np.pi * 0.55 - phi, steps=int(np.pi * r), device=a.device)

    x = r * torch.cos(theta) * a
    y = r * torch.sin(theta) / (a + 1e-12)

    # Apply astigmatic transformation
    x, y = x * torch.cos(phi) - y * torch.sin(phi), \
           x * torch.sin(phi) + y * torch.cos(phi)

    return x, y

def sampling_ellipse_coordinates(a, phi, r, H, W, normalize=False):
    x, y = ellipse_coordinates(a, phi, r)
    y += H // 2

    mask = (1 < x) & (x < W - 1) & (1 < y) & (y < H - 1)
    x = x[mask]
    y = y[mask]

    # Normalize to [-1, 1] for sample_grid
    if normalize:
        x = 2 * x / W - 1
        y = 2 * y / H - 1

    return torch.stack([x, y], dim=-1)

def ellipse_path_average(image, a, phi, min_r=0, max_r=None):
    H, W = image.shape
    image = image.unsqueeze(0).unsqueeze(0)

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, device=image.device)

    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, device=image.device)

    # User-defined range of radii
    max_r = min(W, H) if max_r is None else max_r  # Largest radius based on image size
    num_radii = max_r - min_r  # Number of circles
    radii = torch.linspace(min_r, max_r, num_radii, device=image.device)  # Radii to sample

    average = torch.zeros(num_radii, device=image.device)
    for i, r in enumerate(radii):
        coords = sampling_ellipse_coordinates(a, phi, r, H, W, normalize=True)
        coords = coords.unsqueeze(0).unsqueeze(2)
        
        # Sample pixel intensities using grid_sample (efficient bilinear interpolation)
        sampled_intensities = F.grid_sample(image, coords, align_corners=True, mode='bicubic').squeeze()
        average[i] = sampled_intensities.mean()

    return average

def elipse_fit_loss_function(params, image):
    """Compute loss as intensity variance along ellipses spanning all radii."""
    a, phi = params
    total_variance = 0.0

    H, W = image.shape
    image = image.unsqueeze(0).unsqueeze(0)

    # User-defined range of radii
    r_min = 3  # Smallest radius
    r_max = min(W, H) - 2  # Largest radius based on image size
    num_radii = min(30, r_max - r_min)  # Number of circles
    radii = torch.linspace(r_min, r_max, num_radii, device=image.device)  # Radii to sample

    for r in radii:
        coords = sampling_ellipse_coordinates(a, phi, r, H, W, normalize=True)
        coords = coords.unsqueeze(0).unsqueeze(2)
        
        # Sample pixel intensities using grid_sample (efficient bilinear interpolation)
        sampled_intensities = F.grid_sample(image, coords, align_corners=True, mode='bicubic').squeeze()

        # Compute variance along each ellipse
        total_variance += torch.var(sampled_intensities) / r**2 
    
    # Return average variance over all radii
    return total_variance / len(radii)

def fit_elipses(img, verbose=False):
    # Define global optimization parameters: a (major axis), b (minor axis), phi (angle)
    params = torch.tensor([1.0, 0.0], requires_grad=True, device="cuda")  # Start with circles

    # Move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    params = params.to(device)

    pad = 64
    downsampled_images = [img]
    img_pad = F.pad(img[None, None], (pad, pad, pad, pad), mode='replicate')
    for i in [2, 4, 8, 16, 32, 64]:
        img_ = fast_gaussian_filter(img_pad, kernel_sigma=i-1)
        img_ = img_pad[:, :, pad:-pad, pad:-pad]
        img_ = F.avg_pool2d(img_, kernel_size=i, stride=i)[0, 0]
        downsampled_images.append(img_)

    lr = torch.linspace(0.02, 0.002, len(downsampled_images))
    steps = torch.linspace(30, 20, len(downsampled_images))
    # Gradient descent loop
    for i in range(len(downsampled_images)):
        optimizer = torch.optim.Adam([params], lr=lr[i]) 
        # for j in range(int(steps[i])):
        for j in range(100):
            params_old = params.clone()
            optimizer.zero_grad()
            loss = ellipse_loss_function(params, downsampled_images[-i-1])
            loss.backward()
            optimizer.step()

            a_diff, phi_diff = (params_old - params).detach().cpu().numpy()
            if a_diff < 1e-4 and phi_diff < 1e-3:
                break

        a_opt, phi_opt = params.detach().cpu().numpy()
        if verbose:
            print(f"Optimized Parameters - a: {a_opt}, Angle: {phi_opt * 180 / np.pi}")
    return a_opt, phi_opt

def plot_elipses(img, a, phi, num_radii=10, filter_img=None):
    fig, ax = plt.subplots(figsize=(15, 9))

    if filter_img is not None and filter_img > 0:
        img = fast_gaussian_filter(img, kernel_sigma=filter_img)
        # img = F.avg_pool2d(img[None, None], kernel_size=filter_img, stride=filter_img)[0, 0]
    ax.imshow(img.cpu().detach().numpy(), cmap="grey")

    H, W = img.shape

    r_min = 1  # Smallest radius
    r_max = min(W, H) - 2  # Largest radius based on image size
    radii = torch.linspace(r_min, r_max, num_radii)  # Radii to sample

    for i in range(len(radii)):
        # coords = sampling_coordinates(torch.tensor(1.1, device=device), torch.tensor(45 *np.pi/180, device=device), radii[i], H, W).cpu().detach().numpy()
        coords = sampling_ellipse_coordinates(torch.tensor(a), torch.tensor(phi), radii[i], H, W).cpu().detach().numpy()
        ax.plot(coords[:, 0], coords[:, 1], "-")