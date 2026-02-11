#!/usr/bin/env python
"""
Setup module for Voxelium core library
"""

import os
import sys
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME


def print_debug_msg():
    print("--------------------------------------")
    print("------------- DEBUG MODE -------------")
    print("--------------------------------------")


# -----------------------------------------------------------------------------
# GPU architecture
# -----------------------------------------------------------------------------
nvcc_archs_env = os.environ.get("NVCC_ARCHS")
nvcc_architectures = nvcc_archs_env.split(",") if nvcc_archs_env else [
    "61", "70", "75", "80", "86", "87", "89", "90"
]

# -----------------------------------------------------------------------------
# Build flags
# -----------------------------------------------------------------------------
debug = bool(os.environ.get("VOXELIUM_DEBUG"))
build_extensions = not bool(os.environ.get("VOXELIUM_SKIP_EXT"))

repo_root = os.path.realpath(os.path.dirname(__file__))
src_root = os.path.join(repo_root, "src")
project_root = os.path.join(src_root, "voxelium")

sys.path.insert(0, src_root)  
include_dirs = [project_root]

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

cuda_lib_dirs = []
if CUDA_HOME is not None:
    # common CUDA library locations
    cuda_lib_dirs = [
        os.path.join(CUDA_HOME, "lib64"),
        os.path.join(CUDA_HOME, "lib"),
    ]

library_dirs = [torch_lib_dir] + cuda_lib_dirs

# Torch shared libs (optional; keep if you know you need them)
libraries = ["torch", "torch_cuda", "c10"]

# Avoid bundling system libs into wheels
extra_link_args = ["-Wl,-rpath,$ORIGIN"]

cxx_extra_compile_args = (
    ["-g", "-O0", "-DDEBUG=1", "-UNDEBUG"]
    if debug
    else ["-DNDEBUG", "-O3"]
)

nvcc_extra_compile_args = [
    f"-gencode=arch=compute_{arch},code=sm_{arch}"
    for arch in nvcc_architectures
]
nvcc_extra_compile_args += ["-allow-unsupported-compiler"]
nvcc_extra_compile_args += cxx_extra_compile_args

if debug:
    print_debug_msg()
    nvcc_extra_compile_args += ["-G", "-lineinfo"]


# -----------------------------------------------------------------------------
# CUDA Extensions
# -----------------------------------------------------------------------------
voxelium_sparse3d_ext = CUDAExtension(
    name="voxelium.torch_extensions.sparse3d._C",
    sources=[
        "src/voxelium/torch_extensions/sparse3d/pybind.cpp",
        "src/voxelium/torch_extensions/sparse3d/trilinear_projection.cpp",
        "src/voxelium/torch_extensions/sparse3d/trilinear_projection_cpu_forward_kernel.cpp",
        "src/voxelium/torch_extensions/sparse3d/trilinear_projection_cpu_backward_kernel.cpp",
        "src/voxelium/torch_extensions/sparse3d/trilinear_projection_cuda_forward_kernel.cu",
        "src/voxelium/torch_extensions/sparse3d/trilinear_projection_cuda_backward_kernel.cu",
        "src/voxelium/torch_extensions/sparse3d/volume_extraction.cpp",
        "src/voxelium/torch_extensions/sparse3d/volume_extraction_cpu_forward_kernel.cpp",
        "src/voxelium/torch_extensions/sparse3d/volume_extraction_cpu_backward_kernel.cpp",
        "src/voxelium/torch_extensions/sparse3d/volume_extraction_cuda_forward_kernel.cu",
        "src/voxelium/torch_extensions/sparse3d/volume_extraction_cuda_backward_kernel.cu",
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args={
        "cxx": cxx_extra_compile_args,
        "nvcc": nvcc_extra_compile_args,
    },
    extra_link_args=extra_link_args,
)

inplace_topk_ext = CUDAExtension(
    name="voxelium.torch_extensions.inplace_topk._C",
    sources=[
        "src/voxelium/torch_extensions/inplace_topk/inplace_topk.cpp",
        "src/voxelium/torch_extensions/inplace_topk/inplace_topk_cpu_kernels.cpp",
        "src/voxelium/torch_extensions/inplace_topk/inplace_topk_cuda_kernels.cu",
        "src/voxelium/torch_extensions/inplace_topk/pybind.cpp",
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args={
        "cxx": cxx_extra_compile_args,
        "nvcc": nvcc_extra_compile_args,
    },
    extra_link_args=extra_link_args,
)

ext_modules = (
    [voxelium_sparse3d_ext, inplace_topk_ext]
    if build_extensions
    else []
)

requires = [
    "setuptools>=64",
    "wheel",
    "torch==2.6.0",
    "torchvision==0.21.0",
    "loguru",
    "matplotlib",
    "mrcfile",
    "numpy==1.*",
    "scikit-learn",
    "scipy",
    "tensorboard",
    "alive_progress",
]

setup(
    name="Voxelium",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requires,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)