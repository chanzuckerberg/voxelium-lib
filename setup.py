#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "", "no", "No")


class BuildExtensions(_build_ext):
    """
    Lazy-import torch/cpp_extension so setup can run without torch present.
    All CUDA/PyTorch specifics are resolved only when extensions are built.
    """
    def build_extensions(self):
        # If user wants to skip native extensions entirely
        if _bool_env("VOXELIUM_SKIP_EXT", False):
            self.extensions = []
            return super().build_extensions()

        try:
            # Import here so pyproject's isolated build (which initially lacks torch) can still import setup.py
            import torch
            from torch.utils.cpp_extension import CUDAExtension, include_paths, library_paths
        except Exception as e:
            raise RuntimeError(
                "PyTorch is required to build Voxelium's CUDA extensions. "
                "Install torch (CPU or CUDA) in the build environment before compiling."
            ) from e

        # Resolve CUDA roots/paths
        CUDA_HOME = (
            os.environ.get("CUDA_HOME")
            or os.environ.get("CUDA_PATH")
            or "/usr/local/cuda"
        )

        # Build options (architectures, debug flags)
        nvcc_archs_env = os.environ.get("NVCC_ARCHS")
        nvcc_architectures = (
            nvcc_archs_env.split(",")
            if nvcc_archs_env
            else ["61", "70", "75", "80", "86", "87", "89", "90"]
        )

        debug = _bool_env("VOXELIUM_DEBUG", False)

        cxx_extra_compile_args = (
            ["-g", "-O0", "-DDEBUG=1", "-UNDEBUG"]
            if debug else
            ["-DNDEBUG", "-O3"]
        )
        nvcc_extra_compile_args = [
            f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in nvcc_architectures
        ]
        nvcc_extra_compile_args += ["-allow-unsupported-compiler"] + cxx_extra_compile_args
        if debug:
            nvcc_extra_compile_args += ["-G", "-lineinfo"]

        # Include/library dirs: add project root + torch helpers
        project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)), "voxelium")
        inc_dirs = [project_root] + include_paths(cuda_home=CUDA_HOME)
        lib_dirs = library_paths(cuda_home=CUDA_HOME)

        # Link libraries
        link_torch_cuda = _bool_env("VOXELIUM_LINK_TORCH_CUDA", True)
        # Always need torch & c10; add torch_cuda only when requested
        libraries = ["torch", "c10"] + (["torch_cuda"] if link_torch_cuda else [])

        # Keep rpath local so we don't accidentally vendor system libs
        extra_link_args = ["-Wl,-rpath,$ORIGIN"]

        # Define CUDA extensions
        extensions = [
            CUDAExtension(
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
                include_dirs=inc_dirs,
                library_dirs=lib_dirs,
                libraries=libraries,
                extra_compile_args={"cxx": cxx_extra_compile_args, "nvcc": nvcc_extra_compile_args},
                extra_link_args=extra_link_args,
            ),
            CUDAExtension(
                name="voxelium.torch_extensions.inplace_topk._C",
                sources=[
                    "voxelium/torch_extensions/inplace_topk/inplace_topk.cpp",
                    "voxelium/torch_extensions/inplace_topk/inplace_topk_cpu_kernels.cpp",
                    "voxelium/torch_extensions/inplace_topk/inplace_topk_cuda_kernels.cu",
                    "voxelium/torch_extensions/inplace_topk/pybind.cpp",
                ],
                include_dirs=inc_dirs,
                library_dirs=lib_dirs,
                libraries=libraries,
                extra_compile_args={"cxx": cxx_extra_compile_args, "nvcc": nvcc_extra_compile_args},
                extra_link_args=extra_link_args,
            ),
        ]

        # Replace any placeholder list with the real ones
        self.extensions = extensions
        return super().build_extensions()


setup(
    name="Voxelium",
    version="0.0.3",
    packages=find_packages(),
    # Keep runtime deps in pyproject.toml to avoid duplication/conflicts
    install_requires=[],
    cmdclass={"build_ext": BuildExtensions},
    # ext_modules is intentionally empty here; they’re created lazily in BuildExtensions
    ext_modules=[],
)