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

nvcc_archs_env = os.environ.get("NVCC_ARCHS")
if nvcc_archs_env:
    nvcc_architectures = nvcc_archs_env.split(',')
else:
    nvcc_architectures = ["61", "70", "75", "80", "86", "87", "89", "90"]

debug = bool(os.environ.get("VOXELIUM_DEBUG"))
build_extensions = not os.environ.get("VOXELIUM_SKIP_EXT")

sys.path.insert(0, f'{os.path.dirname(__file__)}/voxelium')

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)),  "voxelium")
libtorch_dir = os.path.abspath("third_party/libtorch")

include_dirs = [
    project_root,
    os.path.join(libtorch_dir, "include"),
    os.path.join(libtorch_dir, "include", "torch", "csrc", "api", "include")
]

library_dirs = [os.path.join(libtorch_dir, "lib")]
libraries = ["torch", "torch_cpu", "c10"]

cxx_extra_compile_args = []
nvcc_extra_compile_args = []

for arch in nvcc_architectures:
    nvcc_extra_compile_args += [f"-gencode=arch=compute_{arch},code=sm_{arch}"]

nvcc_extra_compile_args.append("-allow-unsupported-compiler")

if debug:
    print_debug_msg()
    cxx_extra_compile_args += ["-g", "-O0", f"-DDEBUG={os.environ.get('VOXELIUM_DEBUG', '0')}", "-UNDEBUG"]
    nvcc_extra_compile_args += ["-G", "-lineinfo"]
else:
    cxx_extra_compile_args += ["-DNDEBUG", "-O3"]

nvcc_extra_compile_args += cxx_extra_compile_args
extra_link_args = ["-Wl,-rpath,$ORIGIN/lib"]

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

ext_modules = [voxelium_sparse3d_ext, inplace_topk_ext] if build_extensions else None

requires = [
    "setuptools >= 64", "wheel", "torch==2.2.*", "torchvision", "loguru",
    "matplotlib", "mrcfile", "numpy==1.*", "vtk", "scikit-learn", "scipy",
    "tensorboard", "tqdm", "starfile", "umap-learn", "imageio", "msgpack", "healpy"
]

setup(
    name='Voxelium',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages(),
    install_requires=requires,
    package_data={"voxelium": ["lib/*.so*"]},
    include_package_data=True,
)

if debug:
    print_debug_msg()
