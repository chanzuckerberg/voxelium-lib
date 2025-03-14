#!/usr/bin/env python

"""
Setup module for Voxelium
"""

import os
import sys

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

def print_debug_msg():
    print("-------------------------------------- ")
    print("------------- DEBUG MODE ------------- ")
    print("-------------------------------------- ")

nvcc_architectures = [] #["61", "70", "75", "80", "86", "87", "89", "90"]

debug = False
_DEBUG_LEVEL = os.environ.get('VOXELIUM_DEBUG', '0')
if len(os.environ.get('VOXELIUM_DEBUG', '')) > 0:
    debug = True

build_extensions = True
if len(os.environ.get('VOXELIUM_SKIP_EXT', '')) > 0:
    build_extensions = False

sys.path.insert(0, f'{os.path.dirname(__file__)}/voxelium')

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)),  "voxelium")

include_dirs = [project_root]

cxx_extra_compile_args = []
nvcc_extra_compile_args = []

for arch in nvcc_architectures:
    nvcc_extra_compile_args += [f"-gencode=arch=compute_{arch},code=sm_{arch}"]

if debug:
    print_debug_msg()
    cxx_extra_compile_args += ["-g", "-O0", "-DDEBUG=%s" % _DEBUG_LEVEL, "-UNDEBUG"]
    nvcc_extra_compile_args += ["-G", "-lineinfo"]
else:
    cxx_extra_compile_args += ["-DNDEBUG", "-O3"]
nvcc_extra_compile_args += cxx_extra_compile_args


voxelium_sparse3d_ext = CUDAExtension(
    name='voxelium_sparse3d',
    sources=[
        'voxelium/torch_extensions/sparse3d/pybind.cpp',
        'voxelium/torch_extensions/sparse3d/trilinear_projection.cpp',
        'voxelium/torch_extensions/sparse3d/trilinear_projection_cpu_forward_kernel.cpp',
        'voxelium/torch_extensions/sparse3d/trilinear_projection_cpu_backward_kernel.cpp',
        'voxelium/torch_extensions/sparse3d/trilinear_projection_cuda_forward_kernel.cu',
        'voxelium/torch_extensions/sparse3d/trilinear_projection_cuda_backward_kernel.cu',
        'voxelium/torch_extensions/sparse3d/volume_extraction.cpp',
        'voxelium/torch_extensions/sparse3d/volume_extraction_cpu_forward_kernel.cpp',
        'voxelium/torch_extensions/sparse3d/volume_extraction_cpu_backward_kernel.cpp',
        'voxelium/torch_extensions/sparse3d/volume_extraction_cuda_forward_kernel.cu',
        'voxelium/torch_extensions/sparse3d/volume_extraction_cuda_backward_kernel.cu',
    ],
    include_dirs=include_dirs,
    extra_compile_args={'cxx': cxx_extra_compile_args, 'nvcc': nvcc_extra_compile_args},
)

inplace_topk_ext = CUDAExtension(
    name='voxelium_topk',
    sources=[
        'voxelium/torch_extensions/inplace_topk/inplace_topk.cpp',
        'voxelium/torch_extensions/inplace_topk/inplace_topk_cpu_kernels.cpp',
        'voxelium/torch_extensions/inplace_topk/inplace_topk_cuda_kernels.cu',
        'voxelium/torch_extensions/inplace_topk/pybind.cpp',
    ],
    include_dirs=include_dirs,
    extra_compile_args={'cxx': cxx_extra_compile_args, 'nvcc': nvcc_extra_compile_args},
)

if build_extensions:
    ext_modules = [
        voxelium_sparse3d_ext, 
        inplace_topk_ext
    ]
else:
    ext_modules = None

setup(
    name='Voxelium',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages()
)

if debug:
    print_debug_msg()
