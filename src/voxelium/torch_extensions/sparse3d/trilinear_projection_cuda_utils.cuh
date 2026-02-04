#pragma once
#ifndef SPARSE_3D_TRILINEAR_PROJECTION_CUDA_UTILS_H
#define SPARSE_3D_TRILINEAR_PROJECTION_CUDA_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/macros/Macros.h>

#include "base/base_cuda.cuh"
#include "torch_extensions/sparse3d/trilinear_projection_cuda_kernels.h"

template <typename scalar_t>
__device__ inline scalar_t _helper_cube_interpolation_coordinates(
    scalar_t xp, scalar_t yp, scalar_t zp, const int init_offset,
    int xs[2], int ys[2], int zs[2],
    scalar_t &fx, scalar_t &fy, scalar_t &fz
)
{
    scalar_t conj = 1.; // Complex conjugate
    if (xp < 0) // Hermitian half only
    {
        conj = -1.;
        xp = -xp;
        yp = -yp;
        zp = -zp;
    }

    xs[0] = floor(xp);
    fx = xp - xs[0];
    xs[1] = xs[0] + 1;

    ys[0] = floor(yp);
    fy = yp - ys[0];
    ys[0] += init_offset;
    ys[1] = ys[0] + 1;

    zs[0] = floor(zp);
    fz = zp - zs[0];
    zs[0] += init_offset;
    zs[1] = zs[0] + 1;

    return conj;
}

#endif // SPARSE_3D_TRILINEAR_PROJECTION_CUDA_UTILS_H