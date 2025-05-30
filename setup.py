#!/usr/bin/env python

"""
Setup module for Voxelium
"""

import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, library_paths

def print_debug_msg():
    print("-------------------------------------- ")
    print("------------- DEBUG MODE ------------- ")
    print("-------------------------------------- ")

# GPU architecture
nvcc_archs_env = os.environ.get("NVCC_ARCHS")
nvcc_architectures = nvcc_archs_env.split(',') if nvcc_archs_env else [
    "61", "70", "75", "80", "86", "87", "89", "90"
]

# Compilation flags
debug = bool(os.environ.get("VOXELIUM_DEBUG"))
build_extensions = not os.environ.get("VOXELIUM_SKIP_EXT")

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)), "voxelium")
sys.path.insert(0, project_root)

include_dirs = [project_root]
library_dirs = library_paths(cuda=True)
libraries = ["torch", "torch_cuda", "c10"]
extra_link_args = ["-Wl,-rpath,$ORIGIN"]  # avoid bundling system libs

cxx_extra_compile_args = ["-g", "-O0", "-DDEBUG=1", "-UNDEBUG"] if debug else ["-DNDEBUG", "-O3"]
nvcc_extra_compile_args = [f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in nvcc_architectures]
nvcc_extra_compile_args += ["-allow-unsupported-compiler"] + cxx_extra_compile_args

if debug:
    print_debug_msg()
    nvcc_extra_compile_args += ["-G", "-lineinfo"]

# CUDA Extensions
voxelium_sparse3d_ext = CUDAExtension(
    name='voxelium.torch_extensions.sparse3d._C',
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
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args={'cxx': cxx_extra_compile_args, 'nvcc': nvcc_extra_compile_args},
    extra_link_args=extra_link_args,
)

inplace_topk_ext = CUDAExtension(
    name='voxelium.torch_extensions.inplace_topk._C',
    sources=[
        'voxelium/torch_extensions/inplace_topk/inplace_topk.cpp',
        'voxelium/torch_extensions/inplace_topk/inplace_topk_cpu_kernels.cpp',
        'voxelium/torch_extensions/inplace_topk/inplace_topk_cuda_kernels.cu',
        'voxelium/torch_extensions/inplace_topk/pybind.cpp',
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args={'cxx': cxx_extra_compile_args, 'nvcc': nvcc_extra_compile_args},
    extra_link_args=extra_link_args,
)

ext_modules = [voxelium_sparse3d_ext, inplace_topk_ext] if build_extensions else []

# Python dependencies
requires = [
    "setuptools >= 64", "wheel", "torch==2.2.2", "torchvision==0.17.2", "loguru",
    "matplotlib", "mrcfile", "numpy==1.*", "vtk", "scikit-learn", "scipy",
    "tensorboard", "tqdm", "starfile", "umap-learn", "imageio", "msgpack", "healpy"
]

setup(
    name='Voxelium',
    version='0.0.1a8',
    packages=find_packages(),
    install_requires=requires,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)