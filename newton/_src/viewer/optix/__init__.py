"""Lightweight OptiX helper framework for Warp examples.

This package mirrors core ideas from the C# MiniOptix/MiniOptixScene helpers:
- Handle buffers and typed handles
- Mesh/scene helpers
- SBT packing helpers
- Launch-params packing utilities
"""

from .camera import FreeCamera
from .handles import Handle, HandleBuffer
from .hit_kernels import HitKernel, HitKernelManager
from .launch_params import build_launch_params_dtype, pack_launch_params_bytes
from .mesh import MeshWithAccelerationStructure, TriangleMeshGpu
from .mini_renderer import MiniRenderer, build_renderer_params_dtype, pose7_to_mat4
from .sbt_helpers import SbtKernelManager
from .scene_core import SceneCore, SceneState

__all__ = [
    "FreeCamera",
    "Handle",
    "HandleBuffer",
    "HitKernel",
    "HitKernelManager",
    "MeshWithAccelerationStructure",
    "MiniRenderer",
    "SbtKernelManager",
    "SceneCore",
    "SceneState",
    "TriangleMeshGpu",
    "build_launch_params_dtype",
    "build_renderer_params_dtype",
    "pack_launch_params_bytes",
    "pose7_to_mat4",
]
