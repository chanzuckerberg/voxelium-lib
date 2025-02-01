#!/usr/bin/env python

"""
Setup module for Voxelium
"""

import os

from distutils.sysconfig import get_python_inc
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

python_root = get_python_inc()
project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)),  "voxelium")

ext_modules = CUDAExtension(
        name='test',
        sources=[],
        include_dirs=[python_root, project_root]
    )

print("Includes Paths:")
paths = ext_modules.include_dirs
for p in paths:
    if os.path.exists(p):
        p = os.path.join(p, "**")
        print(p)