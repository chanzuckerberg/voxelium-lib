#!/usr/bin/env python
import os
import sys
from setuptools import setup, find_packages

# We import these inside a try so that if torch is missing, we fail fast with a good message
try:
    from torch.utils.cpp_extension import (
        CUDAExtension,
        BuildExtension,
        include_paths,
        library_paths,
    )
except Exception as e:
    # If we are supposed to build extensions (default) but torch isn't importable,
    # this should surface as a clear error. For pure-Python builds, users can set
    # VOXELIUM_SKIP_EXT=1.
    if not os.environ.get("VOXELIUM_SKIP_EXT"):
        raise RuntimeError(
            "PyTorch is required to build Voxelium CUDA extensions, but was not found. "
            "In CI, either install torch in the build environment (e.g. CIBW_BEFORE_BUILD) "
            "or disable build isolation so your installed torch is visible."
        ) from e
    # Fall back to pure-python if explicitly requested
    CUDAExtension = None
    BuildExtension = None

def print_debug_msg():
    print("-------------------------------------- ")
    print("------------- DEBUG MODE ------------- ")
    print("-------------------------------------- ")

# ===== Build configuration =====
# NVCC arch list (can be overridden via NVCC_ARCHS=…)
nvcc_archs_env = os.environ.get("NVCC_ARCHS")
nvcc_architectures = nvcc_archs_env.split(",") if nvcc_archs_env else [
    "61", "70", "75", "80", "86", "87", "89", "90"
]

# Debug build?
debug = bool(os.environ.get("VOXELIUM_DEBUG"))
# Build native extensions? (set VOXELIUM_SKIP_EXT=1 to skip)
build_extensions = not os.environ.get("VOXELIUM_SKIP_EXT")

# Optional: don’t explicitly link torch_cuda (keeps wheels less coupled in CI)
link_torch_cuda = os.environ.get("VOXELIUM_LINK_TORCH_CUDA", "1") != "0"

# Resolve project root for headers we ship
project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)), "voxelium")
sys.path.insert(0, project_root)

# CUDA toolkit location (only used by NVCC; PyTorch helpers will also consult CUDA_HOME)
CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"

# Torch helper include/lib paths — do NOT pass unsupported kwargs
extra_include_dirs = []
extra_library_dirs = []
if build_extensions and CUDAExtension is not None:
    try:
        extra_include_dirs = list(include_paths())
        extra_library_dirs = list(library_paths())
    except TypeError:
        # Older torch versions might not have these helpers; leave as empty lists
        extra_include_dirs = []
        extra_library_dirs = []

include_dirs = [project_root] + extra_include_dirs
library_dirs = extra_library_dirs

# Link settings
libraries = []
if link_torch_cuda:
    # When linking, prefer minimal explicit list; auditwheel step excludes these anyway.
    libraries = ["torch", "c10", "torch_cuda"]

extra_link_args = ["-Wl,-rpath,$ORIGIN"]  # avoid bundling system libs; keep relative lookup

cxx_extra_compile_args = ["-g", "-O0", "-DDEBUG=1", "-UNDEBUG"] if debug else ["-DNDEBUG", "-O3"]
nvcc_extra_compile_args = [f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in nvcc_architectures]
nvcc_extra_compile_args += ["-allow-unsupported-compiler"] + cxx_extra_compile_args
if debug:
    print_debug_msg()
    nvcc_extra_compile_args += ["-G", "-lineinfo"]

# ===== Define CUDA extensions =====
ext_modules = []
if build_extensions and CUDAExtension is not None:
    voxelium_sparse3d_ext = CUDAExtension(
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
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={"cxx": cxx_extra_compile_args, "nvcc": nvcc_extra_compile_args},
        extra_link_args=extra_link_args,
    )

    inplace_topk_ext = CUDAExtension(
        name="voxelium.torch_extensions.inplace_topk._C",
        sources=[
            "voxelium/torch_extensions/inplace_topk/inplace_topk.cpp",
            "voxelium/torch_extensions/inplace_topk/inplace_topk_cpu_kernels.cpp",
            "voxelium/torch_extensions/inplace_topk/inplace_topk_cuda_kernels.cu",
            "voxelium/torch_extensions/inplace_topk/pybind.cpp",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={"cxx": cxx_extra_compile_args, "nvcc": nvcc_extra_compile_args},
        extra_link_args=extra_link_args,
    )

    ext_modules = [voxelium_sparse3d_ext, inplace_topk_ext]

# Fail fast with a clear message if extensions are expected but torch is missing
if build_extensions and CUDAExtension is None:
    raise RuntimeError(
        "Building Voxelium extensions was requested, but PyTorch’s cpp_extension "
        "couldn’t be imported. Make sure torch is installed in the build env."
    )

# ===== Setup =====
setup(
    name="Voxelium",
    version="0.0.3",
    packages=find_packages(),
    install_requires=[],  # runtime deps are in pyproject.toml
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
)