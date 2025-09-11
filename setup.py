#!/usr/bin/env python
import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
    include_paths,
    library_paths,
)

# --------------------- helpers & env toggles ---------------------
def _bool_env(name: str, default="0") -> bool:
    val = os.environ.get(name, default)
    if val is None:
        return False
    return str(val).strip().lower() not in ("", "0", "false", "no")

DEBUG = _bool_env("VOXELIUM_DEBUG", "0")
SKIP_EXT = _bool_env("VOXELIUM_SKIP_EXT", "0")
# Set this to 0 in CI when you preinstall CPU torch to keep downloads small
LINK_TORCH_CUDA = _bool_env("VOXELIUM_LINK_TORCH_CUDA", "1")

NVCC_ARCHS = os.environ.get("NVCC_ARCHS", "61,70,75,80,86,87,89,90")
NVCC_ARCH_LIST = [a.strip() for a in NVCC_ARCHS.split(",") if a.strip()]

# --------------------- paths ---------------------
PROJECT_ROOT = os.path.realpath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "voxelium")
sys.path.insert(0, SRC_ROOT)

# Respect CUDA_HOME/CUDA_PATH if present (optional – fine if missing with CPU torch)
CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

TORCH_INC_DIRS = list(include_paths())        # torch headers
TORCH_LIB_DIRS = list(library_paths())        # torch libs

CUDA_INC_DIRS = []
CUDA_LIB_DIRS = []
if CUDA_HOME and os.path.isdir(CUDA_HOME):
    inc = os.path.join(CUDA_HOME, "include")
    lib64 = os.path.join(CUDA_HOME, "lib64")
    if os.path.isdir(inc):
        CUDA_INC_DIRS.append(inc)
    if os.path.isdir(lib64):
        CUDA_LIB_DIRS.append(lib64)

INCLUDE_DIRS = [SRC_ROOT] + TORCH_INC_DIRS + CUDA_INC_DIRS
LIBRARY_DIRS = TORCH_LIB_DIRS + CUDA_LIB_DIRS

# --------------------- link & compile flags ---------------------
LIBRARIES = ["torch", "c10"] + (["torch_cuda"] if LINK_TORCH_CUDA else [])
EXTRA_LINK_ARGS = ["-Wl,-rpath,$ORIGIN", "-Wl,--as-needed"]

CXX_FLAGS = ["-g", "-O0", "-DDEBUG=1", "-UNDEBUG"] if DEBUG else ["-DNDEBUG", "-O3"]
NVCC_FLAGS = [f"-gencode=arch=compute_{sm},code=sm_{sm}" for sm in NVCC_ARCH_LIST]
NVCC_FLAGS += ["-allow-unsupported-compiler"] + CXX_FLAGS
if DEBUG:
    NVCC_FLAGS += ["-G", "-lineinfo"]

# --------------------- extensions ---------------------
def make_sparse3d_ext():
    return CUDAExtension(
        name="voxelium.torch_extensions.sparse3d._C",
        sources=[
            "voxelium/torch_extensions/sparse3d/pybind.cpp",
            "voxelium/torch_extensions/sparse3d/trilinear_projection.cpp",
            "voxelium/torch_extensions/sparse3d/trilinear_projection_cpu_forward_kernel.cpp",
            "voxelium/torch_extensions/sparse3d/trilinear_projection_cpu_backward_kernel.cpp",
            "voxelium/torch_extensions/sparse3d/trilinear_projection_cuda_forward_kernel.cu",
            "voxelium/torch_extensions/sparse3d/trilinear_projection_cuda_backward_kernel.cu",
            "voxelium/torch_extensions/sparse3d/volume_extraction.cpp",
            "voxelium/torch_extensions/sparse3d/volume_extraction_cpu_forward_kernel.cpp",
            "voxelium/torch_extensions/sparse3d/volume_extraction_cpu_backward_kernel.cpp",
            "voxelium/torch_extensions/sparse3d/volume_extraction_cuda_forward_kernel.cu",
            "voxelium/torch_extensions/sparse3d/volume_extraction_cuda_backward_kernel.cu",
        ],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES,
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        extra_link_args=EXTRA_LINK_ARGS,
    )

def make_inplace_topk_ext():
    return CUDAExtension(
        name="voxelium.torch_extensions.inplace_topk._C",
        sources=[
            "voxelium/torch_extensions/inplace_topk/inplace_topk.cpp",
            "voxelium/torch_extensions/inplace_topk/inplace_topk_cpu_kernels.cpp",
            "voxelium/torch_extensions/inplace_topk/inplace_topk_cuda_kernels.cu",
            "voxelium/torch_extensions/inplace_topk/pybind.cpp",
        ],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES,
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        extra_link_args=EXTRA_LINK_ARGS,
    )

ext_modules = [] if SKIP_EXT else [make_sparse3d_ext(), make_inplace_topk_ext()]

# --------------------- setup() ---------------------
# Keep runtime deps in pyproject.toml; avoid duplication here.
setup(
    name="Voxelium",
    version="0.0.3",
    packages=find_packages(include=["voxelium", "voxelium.*"]),
    install_requires=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)