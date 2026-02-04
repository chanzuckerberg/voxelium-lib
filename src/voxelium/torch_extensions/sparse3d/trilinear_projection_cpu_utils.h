#pragma once
#ifndef SPARSE_3D_TRILINEAR_PROJECTION_CPU_UTILS_H
#define SPARSE_3D_TRILINEAR_PROJECTION_CPU_UTILS_H

#include "torch_extensions/sparse3d/trilinear_projection_cpu_kernels.h"

template <typename scalar_t>
inline void _helper_rotate_coordinates(
    const torch::TensorAccessor<scalar_t, 2> rot_matrix,
    const scalar_t x, const scalar_t y,
    scalar_t &xp, scalar_t &yp, scalar_t &zp
)
{
    xp = rot_matrix[0][0] * x + rot_matrix[1][0] * y;
    yp = rot_matrix[0][1] * x + rot_matrix[1][1] * y;
    zp = rot_matrix[0][2] * x + rot_matrix[1][2] * y;
}

template <typename scalar_t>
inline scalar_t _helper_cube_interpolation_coordinates(
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

#endif // SPARSE_3D_TRILINEAR_PROJECTION_CPU_UTILS_H