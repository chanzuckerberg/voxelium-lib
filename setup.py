#!/usr/bin/env python
"""
Setup module for Voxelium
"""
import os
import sys
from setuptools import setup, find_packages

# --- Torch/extension import with friendly guidance ---------------------------
# We import lazily so we can print a clear error if torch isn't visible.
def _import_torch_build_bits():
    try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension, library_paths
        return CUDAExtension, BuildExtension, library_paths
    except Exception as e:
        raise RuntimeError(
            "PyTorch is required to build Voxelium CUDA extensions, but was not found.\n"
            "You are expected to use the pre-installed GPU Torch image OR expose your preinstalled "
            "torch to the build by disabling build isolation.\n\n"
            "Fixes:\n"
            "  * In CI with cibuildwheel: set CIBW_ENVIRONMENT='PIP_NO_BUILD_ISOLATION=1' and use your "
            "    custom manylinux CUDA image that already has torch==2.8.0 installed per interpreter.\n"
            "  * Locally: `pip install torch==2.8.0 torchvision` in the active interpreter BEFORE "
            "    building, or run `pip wheel . --no-build-isolation`.\n\n"
            f"Original import error:\n{e}"
        )

def print_debug_msg():
    print("-------------------------------------- ")
    print("------------- DEBUG MODE ------------- ")
    print("-------------------------------------- ")

# GPU architecture list (override via NVCC_ARCHS="61,70,75,80,86,87,89,90")
nvcc_archs_env = os.environ.get("NVCC_ARCHS")
nvcc_architectures = nvcc_archs_env.split(',') if nvcc_archs_env else [
    "61", "70", "75", "80", "86", "87", "89", "90"
]

# Compilation flags
debug = bool(os.environ.get("VOXELIUM_DEBUG"))
# VOXELIUM_SKIP_EXT=1 allows pure-Python builds (e.g., for doc builds)
build_extensions = not os.environ.get("VOXELIUM_SKIP_EXT")

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)), "voxelium")
sys.path.insert(0, project_root)

# If we are building extensions, we need torch's helpers
if build_extensions:
    CUDAExtension, BuildExtension, library_paths = _import_torch_build_bits()

include_dirs = [project_root]
library_dirs = library_paths(cuda=True) if build_extensions else []
# If someone explicitly disables CUDA linkage, they can set VOXELIUM_LINK_TORCH_CUDA=0.
link_torch_cuda = os.environ.get("VOXELIUM_LINK_TORCH_CUDA", "1") != "0"

libraries = ["torch", "c10"]
if link_torch_cuda:
    libraries.extend(["torch_cuda", "c10_cuda"])

# Avoid bundling system libs; rely on rpaths (auditwheel will keep torch libs excluded in CI)
extra_link_args = ["-Wl,-rpath,$ORIGIN"]

cxx_extra_compile_args = ["-g", "-O0", "-DDEBUG=1", "-UNDEBUG"] if debug else ["-DNDEBUG", "-O3"]

nvcc_extra_compile_args = [f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in nvcc_architectures]
nvcc_extra_compile_args += ["-allow-unsupported-compiler"] + cxx_extra_compile_args

if debug:
    print_debug_msg()
    nvcc_extra_compile_args += ["-G", "-lineinfo"]

ext_modules = []
if build_extensions:
    # CUDA Extensions: sparse3d
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

    # CUDA Extensions: inplace_topk
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

    ext_modules = [voxelium_sparse3d_ext, inplace_topk_ext]

# Python dependencies at runtime
requires = [
    "setuptools >= 64",
    "wheel",
    "torch==2.8.0",
    "torchvision",
    "loguru",
    "matplotlib",
    "mrcfile",
    "numpy==1.*",
    "vtk",
    "scikit-learn",
    "scipy",
    "tensorboard",
    "tqdm",
    "starfile",
    "umap-learn",
    "imageio",
    "msgpack",
    "healpy",
]

setup(
    name='Voxelium',
    version='0.0.3',
    packages=find_packages(),
    install_requires=requires,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension} if build_extensions else {},
)