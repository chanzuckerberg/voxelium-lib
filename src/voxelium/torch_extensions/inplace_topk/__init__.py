#!/usr/bin/env python

# Load the compiled C++/CUDA extension first
try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "Could not find compiled extension '_C' in voxelium.torch_extensions.sparse3d. "
        "Make sure you built the extension with setup.py"
    ) from e

from .inplace_topk import *