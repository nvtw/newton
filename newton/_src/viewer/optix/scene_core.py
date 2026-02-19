from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np

import warp as wp

from .handles import Handle, HandleBuffer
from .mesh import MeshWithAccelerationStructure, TriangleMeshGpu, create_cube_mesh, create_plane_mesh


class SceneState(IntFlag):
    OK = 0
    INSTANCES_CHANGED = 1
    MESHES_CHANGED = 2
    MATERIALS_CHANGED = 4
    INSTANCE_TRANSFORMS_CHANGED = 8


@dataclass
class OptixInstance:
    mesh_handle: Handle
    transform: np.ndarray
    sbt_offset: int
    visibility_mask: int = 255
    instance_id: int = -1
    color: int = 0xFFFFFFFF


class InstanceAccelerationStructure:
    """Minimal IAS helper inspired by MiniOptixScene.InstanceAccelerationStructure."""

    def __init__(self, optix, ctx):
        self.optix = optix
        self.ctx = ctx
        self.instances_gpu = None
        self.traversable = 0
        self._temp = None
        self._out = None

    def rebuild_ias(self, instances: list[object], stream=0) -> int:
        arr = np.asarray(instances)
        self.instances_gpu = wp.array(arr.view(np.uint8), dtype=wp.uint8, device="cuda")

        build_input = self.optix.BuildInputInstanceArray()
        build_input.instances = self.instances_gpu.ptr
        build_input.numInstances = len(instances)

        accel_options = self.optix.AccelBuildOptions(
            buildFlags=int(self.optix.BUILD_FLAG_ALLOW_UPDATE),
            operation=self.optix.BUILD_OPERATION_BUILD,
        )
        sizes = self.ctx.accelComputeMemoryUsage([accel_options], [build_input])
        self._temp = wp.empty(sizes.tempSizeInBytes, dtype=wp.uint8, device="cuda")
        self._out = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device="cuda")
        self.traversable = int(
            self.ctx.accelBuild(
                stream,
                [accel_options],
                [build_input],
                self._temp.ptr,
                sizes.tempSizeInBytes,
                self._out.ptr,
                sizes.outputSizeInBytes,
                [],
            )
        )
        return self.traversable


class SceneCore:
    """Python counterpart to SceneCore.cs with pragmatic API."""

    def __init__(self) -> None:
        self.state = SceneState.OK
        self.meshes: HandleBuffer[MeshWithAccelerationStructure] = HandleBuffer()
        self.instances: HandleBuffer[OptixInstance] = HandleBuffer()
        self.default_hit_kernel_handle = Handle.invalid()

    def clear_render_content(self) -> None:
        self.meshes.clear()
        self.instances.clear()
        self.state = SceneState.OK

    def add_mesh(self, mesh: TriangleMeshGpu) -> Handle:
        handle = self.meshes.add(MeshWithAccelerationStructure(mesh))
        self.state |= SceneState.MESHES_CHANGED
        return handle

    def add_cube(self, size: float) -> Handle:
        v, i = create_cube_mesh(size)
        return self.add_mesh(TriangleMeshGpu(v, i))

    def add_plane(self, width: float, height: float) -> Handle:
        v, i = create_plane_mesh(width, height)
        return self.add_mesh(TriangleMeshGpu(v, i))

    def add_instance(
        self,
        transform: np.ndarray,
        mesh_handle: Handle,
        hit_kernel_sbt_offset: int,
        overwrite_instance_id: int = -1,
        visibility_mask: int = 255,
        color: int = 0xFFFFFFFF,
    ) -> Handle:
        handle = self.instances.add_empty()
        instance_id = handle.value if overwrite_instance_id < 0 else overwrite_instance_id
        inst = OptixInstance(
            mesh_handle=mesh_handle,
            transform=np.asarray(transform, dtype=np.float32),
            sbt_offset=int(hit_kernel_sbt_offset),
            visibility_mask=int(visibility_mask),
            instance_id=int(instance_id),
            color=int(color),
        )
        self.instances.set_value(handle, inst)
        self.state |= SceneState.INSTANCES_CHANGED
        return handle

    def set_instance_transform(self, handle: Handle, transform: np.ndarray) -> None:
        inst = self.instances.get_value(handle)
        inst.transform = np.asarray(transform, dtype=np.float32)
        self.instances.set_value(handle, inst)
        self.state |= SceneState.INSTANCE_TRANSFORMS_CHANGED

    def get_instance_count(self) -> int:
        return self.instances.count
