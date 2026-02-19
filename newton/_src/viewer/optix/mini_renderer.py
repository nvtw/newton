from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import warp as wp

from .camera import FreeCamera
from .example_pyoptix_empty_buffer import (
    _compile_cuda_source_to_ptx,
    _create_optix_context,
    _load_native_section,
    _load_optix_device_header_text,
    _require_optix,
)
from .mesh import TriangleMeshGpu
from .sbt_helpers import SbtKernelManager
from .scene_core import SceneCore, SceneState


def _quat_to_mat3(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    zz = qz * z2
    xy = qx * y2
    xz = qx * z2
    yz = qy * z2
    wx = qw * x2
    wy = qw * y2
    wz = qw * z2
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def pose7_to_mat4(pose: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Convert [tx, ty, tz, qx, qy, qz, qw] pose to 4x4 matrix."""

    p = np.asarray(pose, dtype=np.float32).reshape(-1)
    if p.size != 7:
        raise ValueError("Expected pose with 7 values [tx, ty, tz, qx, qy, qz, qw]")
    tx, ty, tz, qx, qy, qz, qw = [float(v) for v in p]
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1.0e-8:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = _quat_to_mat3(qx, qy, qz, qw)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def rotation_y_mat4(angle_radians: float) -> np.ndarray:
    c = float(math.cos(angle_radians))
    s = float(math.sin(angle_radians))
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def pack_rgba8(r: int, g: int, b: int, a: int = 255) -> int:
    rr = max(0, min(255, int(r)))
    gg = max(0, min(255, int(g)))
    bb = max(0, min(255, int(b)))
    aa = max(0, min(255, int(a)))
    return (aa << 24) | (bb << 16) | (gg << 8) | rr


def _mat4_to_optix_transform12(m: np.ndarray) -> list[float]:
    m = np.asarray(m, dtype=np.float32).reshape(4, 4)
    # OptiX expects a 3x4 row-major transform.
    return [
        float(m[0, 0]),
        float(m[0, 1]),
        float(m[0, 2]),
        float(m[0, 3]),
        float(m[1, 0]),
        float(m[1, 1]),
        float(m[1, 2]),
        float(m[1, 3]),
        float(m[2, 0]),
        float(m[2, 1]),
        float(m[2, 2]),
        float(m[2, 3]),
    ]


def build_renderer_params_dtype() -> np.dtype:
    names = [
        "image",
        "width",
        "height",
        "frame_id",
        "trav_handle",
        "cam_px",
        "cam_py",
        "cam_pz",
        "cam_ux",
        "cam_uy",
        "cam_uz",
        "cam_vx",
        "cam_vy",
        "cam_vz",
        "cam_wx",
        "cam_wy",
        "cam_wz",
        "instance_vertex_ptrs",
        "instance_index_ptrs",
        "instance_normal_ptrs",
        "instance_color_ptrs",
        "light_dx",
        "light_dy",
        "light_dz",
    ]
    formats = ["u8", "u4", "u4", "u4", "u8"] + ["f4"] * 12 + ["u8", "u8", "u8", "u8", "f4", "f4", "f4"]
    offsets = [0, 8, 12, 16, 24, *list(range(32, 80, 4)), 80, 88, 96, 104, 112, 116, 120]
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": 128})


def build_optix_instance_dtype() -> np.dtype:
    """ABI-compatible OptiX instance layout (80 bytes)."""

    names = [
        "transform",
        "instanceId",
        "sbtOffset",
        "visibilityMask",
        "flags",
        "traversableHandle",
    ]
    formats = [("f4", (12,)), "u4", "u4", "u4", "u4", "u8"]
    offsets = [0, 48, 52, 56, 60, 64]
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": 80})


@dataclass
class _RendererResources:
    module: object
    pipeline: object
    sbt: object
    sbt_keepalive: dict
    gas_buffers: list[wp.array]
    blas_handles: dict[int, int]
    tlas_handle: int
    tlas_temp: wp.array
    tlas_out: wp.array
    tlas_instances: wp.array
    instance_vertex_ptrs: wp.array
    instance_index_ptrs: wp.array
    instance_normal_ptrs: wp.array
    instance_colors: wp.array


class MiniRenderer:
    """Minimal Python renderer inspired by MiniRenderer.RayTracingRenderer."""

    def __init__(self, device: str = "cuda", width: int = 960, height: int = 540):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.frame_id = 0

        self.optix = _require_optix()
        wp.init()
        with wp.ScopedDevice(self.device):
            if not wp.get_device().is_cuda:
                raise RuntimeError("MiniRenderer requires CUDA")
            self.wp_device = wp.get_device(self.device)
            cu_context = (
                self.wp_device.context.value
                if hasattr(self.wp_device.context, "value")
                else int(self.wp_device.context)
            )
            self.ctx, self.logger = _create_optix_context(self.optix, int(cu_context))

        self.scene = SceneCore()
        self.camera = FreeCamera.create_default()
        self.resources: _RendererResources | None = None

        self.params_dtype = build_renderer_params_dtype()
        self.params_host = np.zeros(1, dtype=self.params_dtype)
        self.params_device = wp.empty(self.params_dtype.itemsize, dtype=wp.uint8, device="cuda")
        self.color = wp.empty(self.width * self.height, dtype=wp.uint32, device="cuda")

    def register_cube(self, size: float = 1.0):
        return self.scene.add_cube(size)

    def register_plane(self, width: float = 10.0, height: float = 10.0):
        return self.scene.add_plane(width, height)

    def register_obj(self, path: str | Path, scale: float = 1.0):
        mesh = TriangleMeshGpu.from_obj(path=path, scale=scale)
        return self.scene.add_mesh(mesh)

    def add_render_instance(self, mesh_handle, pose7, color_rgba8: int | None = None):
        transform = pose7_to_mat4(pose7)
        color = int(color_rgba8) if color_rgba8 is not None else pack_rgba8(255, 255, 255, 255)
        return self.scene.add_instance(transform, mesh_handle, hit_kernel_sbt_offset=0, color=color)

    def set_instance_transform(self, instance_handle, pose7):
        transform = pose7_to_mat4(pose7)
        self.scene.set_instance_transform(instance_handle, transform)

    def set_instance_transform_matrix(self, instance_handle, transform: np.ndarray):
        self.scene.set_instance_transform(instance_handle, np.asarray(transform, dtype=np.float32).reshape(4, 4))

    def get_instance_transform_matrix(self, instance_handle) -> np.ndarray:
        inst = self.scene.instances.get_value(instance_handle)
        return np.asarray(inst.transform, dtype=np.float32).reshape(4, 4).copy()

    def set_camera_pose(self, position, target, up=(0.0, 1.0, 0.0)):
        self.camera.set_pose(position, target, up)

    def _build_ptx(self) -> bytes:
        source = (
            _load_optix_device_header_text()
            + _load_native_section("renderer_launch_params")
            + _load_native_section("renderer_trace_programs")
            + "\n"
        )
        return _compile_cuda_source_to_ptx(source, module_tag="renderer_trace", device=self.device)

    def _build_pipeline(self, ptx: bytes):
        kwargs = {
            "usesMotionBlur": False,
            "traversableGraphFlags": int(self.optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING),
            "numPayloadValues": 1,
            "numAttributeValues": 2,
            "exceptionFlags": int(self.optix.EXCEPTION_FLAG_NONE),
            "pipelineLaunchParamsVariableName": "renderer_params",
        }
        if self.optix.version()[1] >= 2:
            kwargs["usesPrimitiveTypeFlags"] = int(self.optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
        pipeline_options = self.optix.PipelineCompileOptions(**kwargs)
        module_options = self.optix.ModuleCompileOptions(
            maxRegisterCount=self.optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=self.optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=self.optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )
        module, log = self.ctx.moduleCreate(module_options, pipeline_options, ptx)
        if log:
            print(f"Module create log:\n{log}")

        manager = SbtKernelManager(self.optix, self.ctx, module, num_ray_subtypes=1)
        manager.set_raygen_kernel("__raygen__renderer")
        manager.add_miss_kernels(["__miss__renderer", "__miss__shadow"])
        manager.register_hit_shader_type("__closesthit__renderer")

        link_options = self.optix.PipelineLinkOptions()
        link_options.maxTraceDepth = 1
        pipeline = self.ctx.pipelineCreate(pipeline_options, link_options, manager.get_all_program_groups(), "")
        pipeline.setStackSize(2 * 1024, 2 * 1024, 2 * 1024, 2)
        sbt_resources = manager.build_sbt(device="cuda")
        return module, pipeline, sbt_resources.sbt, sbt_resources.keepalive

    def _ensure_blas(self) -> tuple[dict[int, int], list[wp.array]]:
        blas_handles: dict[int, int] = {}
        keepalive: list[wp.array] = []
        for mesh_handle, mesh_wrap in self.scene.meshes.items():
            mesh_wrap.upload_and_rebuild_acceleration_structure(self.optix, self.ctx, stream=0, device=self.device)
            if mesh_wrap.gas_handle == 0:
                raise RuntimeError(f"Failed to build BLAS for mesh handle {mesh_handle.value}")
            blas_handles[mesh_handle.value] = int(mesh_wrap.gas_handle)
            if mesh_wrap.mesh.d_vertices is not None:
                keepalive.append(mesh_wrap.mesh.d_vertices)
            if mesh_wrap.mesh.d_indices is not None:
                keepalive.append(mesh_wrap.mesh.d_indices)
            if mesh_wrap.d_temp is not None:
                keepalive.append(mesh_wrap.d_temp)
            if mesh_wrap.d_gas is not None:
                keepalive.append(mesh_wrap.d_gas)
        return blas_handles, keepalive

    def _build_optix_instances(self, blas_handles: dict[int, int]) -> np.ndarray:
        inst_dtype = build_optix_instance_dtype()
        inst_np = np.zeros(max(self.scene.instances.count, 1), dtype=inst_dtype)
        if self.scene.instances.count == 0:
            # Keep TLAS valid even for empty user instance set.
            first_mesh = next(iter(self.scene.meshes.items()), None)
            if first_mesh is None:
                raise RuntimeError("Scene has no meshes for fallback instance")
            mesh_handle, _ = first_mesh
            inst_np["transform"][0] = np.asarray(
                _mat4_to_optix_transform12(np.eye(4, dtype=np.float32)), dtype=np.float32
            )
            inst_np["instanceId"][0] = np.uint32(0)
            inst_np["sbtOffset"][0] = np.uint32(0)
            inst_np["visibilityMask"][0] = np.uint32(255)
            inst_np["flags"][0] = np.uint32(int(self.optix.INSTANCE_FLAG_NONE))
            inst_np["traversableHandle"][0] = np.uint64(blas_handles[mesh_handle.value])
            return inst_np

        idx = 0
        for _, inst in self.scene.instances.items():
            if inst.mesh_handle.value not in blas_handles:
                raise RuntimeError(f"Missing BLAS for mesh handle {inst.mesh_handle.value}")
            inst_np["transform"][idx] = np.asarray(_mat4_to_optix_transform12(inst.transform), dtype=np.float32)
            # Use dense instance ids to index per-instance shader data arrays.
            inst_np["instanceId"][idx] = np.uint32(idx)
            inst_np["sbtOffset"][idx] = np.uint32(inst.sbt_offset)
            inst_np["visibilityMask"][idx] = np.uint32(inst.visibility_mask)
            inst_np["flags"][idx] = np.uint32(int(self.optix.INSTANCE_FLAG_NONE))
            inst_np["traversableHandle"][idx] = np.uint64(blas_handles[inst.mesh_handle.value])
            idx += 1
        return inst_np

    def _build_instance_shader_data(self) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        count = max(self.scene.instances.count, 1)
        vertex_ptrs = np.zeros(count, dtype=np.uint64)
        index_ptrs = np.zeros(count, dtype=np.uint64)
        normal_ptrs = np.zeros(count, dtype=np.uint64)
        colors = np.zeros(count, dtype=np.uint32)

        if self.scene.instances.count == 0:
            first_mesh = next(iter(self.scene.meshes.items()), None)
            if first_mesh is None:
                raise RuntimeError("Scene has no meshes for fallback shader data")
            _, mesh_wrap = first_mesh
            if (
                mesh_wrap.mesh.d_vertices is None
                or mesh_wrap.mesh.d_indices is None
                or mesh_wrap.mesh.d_normals is None
            ):
                raise RuntimeError("Fallback mesh device buffers not available")
            vertex_ptrs[0] = np.uint64(mesh_wrap.mesh.d_vertices.ptr)
            index_ptrs[0] = np.uint64(mesh_wrap.mesh.d_indices.ptr)
            normal_ptrs[0] = np.uint64(mesh_wrap.mesh.d_normals.ptr)
            colors[0] = np.uint32(0xFFB0B0B0)
        else:
            idx = 0
            for _, inst in self.scene.instances.items():
                mesh_wrap = self.scene.meshes.get_value(inst.mesh_handle)
                if (
                    mesh_wrap.mesh.d_vertices is None
                    or mesh_wrap.mesh.d_indices is None
                    or mesh_wrap.mesh.d_normals is None
                ):
                    raise RuntimeError(f"Mesh device buffers not available for handle {inst.mesh_handle.value}")
                vertex_ptrs[idx] = np.uint64(mesh_wrap.mesh.d_vertices.ptr)
                index_ptrs[idx] = np.uint64(mesh_wrap.mesh.d_indices.ptr)
                normal_ptrs[idx] = np.uint64(mesh_wrap.mesh.d_normals.ptr)
                colors[idx] = np.uint32(int(inst.color))
                idx += 1

        d_vptr = wp.array(vertex_ptrs, dtype=wp.uint64, device=self.device)
        d_iptr = wp.array(index_ptrs, dtype=wp.uint64, device=self.device)
        d_nptr = wp.array(normal_ptrs, dtype=wp.uint64, device=self.device)
        d_col = wp.array(colors, dtype=wp.uint32, device=self.device)
        return d_vptr, d_iptr, d_nptr, d_col

    def _build_or_update_tlas(
        self, instance_np: np.ndarray, previous: _RendererResources | None
    ) -> tuple[int, wp.array, wp.array, wp.array]:
        instance_bytes = np.ascontiguousarray(instance_np).view(np.uint8).reshape(-1)
        d_instances = wp.array(instance_bytes, dtype=wp.uint8, device=self.device)

        build_input = self.optix.BuildInputInstanceArray()
        build_input.instances = int(d_instances.ptr)
        build_input.numInstances = int(instance_np.shape[0])

        build_flags = int(self.optix.BUILD_FLAG_ALLOW_UPDATE)
        do_update = previous is not None and bool(self.scene.state & SceneState.INSTANCE_TRANSFORMS_CHANGED)
        operation = self.optix.BUILD_OPERATION_UPDATE if do_update else self.optix.BUILD_OPERATION_BUILD
        accel_options = self.optix.AccelBuildOptions(buildFlags=build_flags, operation=operation)
        sizes = self.ctx.accelComputeMemoryUsage([accel_options], [build_input])

        if operation == self.optix.BUILD_OPERATION_UPDATE and previous is not None:
            temp_nbytes = getattr(sizes, "tempUpdateSizeInBytes", 0) or sizes.tempSizeInBytes
            d_temp = wp.empty(temp_nbytes, dtype=wp.uint8, device=self.device)
            d_out = previous.tlas_out
            out_nbytes = int(d_out.shape[0])
        else:
            temp_nbytes = int(sizes.tempSizeInBytes)
            out_nbytes = int(sizes.outputSizeInBytes)
            d_temp = wp.empty(temp_nbytes, dtype=wp.uint8, device=self.device)
            d_out = wp.empty(out_nbytes, dtype=wp.uint8, device=self.device)

        handle = int(
            self.ctx.accelBuild(
                0,
                [accel_options],
                [build_input],
                int(d_temp.ptr),
                temp_nbytes,
                int(d_out.ptr),
                out_nbytes,
                [],
            )
        )
        return handle, d_temp, d_out, d_instances

    def rebuild_instance_tree(self):
        with wp.ScopedDevice(self.device):
            if self.scene.state == SceneState.OK and self.resources is not None:
                return

            previous = self.resources
            if previous is None:
                ptx = self._build_ptx()
                module, pipeline, sbt, sbt_keepalive = self._build_pipeline(ptx)
            else:
                module = previous.module
                pipeline = previous.pipeline
                sbt = previous.sbt
                sbt_keepalive = previous.sbt_keepalive

            blas_handles, blas_keepalive = self._ensure_blas()
            inst_np = self._build_optix_instances(blas_handles)
            tlas_handle, tlas_temp, tlas_out, tlas_instances = self._build_or_update_tlas(inst_np, previous=previous)
            d_vptr, d_iptr, d_nptr, d_col = self._build_instance_shader_data()

            self.resources = _RendererResources(
                module=module,
                pipeline=pipeline,
                sbt=sbt,
                sbt_keepalive=sbt_keepalive,
                gas_buffers=blas_keepalive,
                blas_handles=blas_handles,
                tlas_handle=tlas_handle,
                tlas_temp=tlas_temp,
                tlas_out=tlas_out,
                tlas_instances=tlas_instances,
                instance_vertex_ptrs=d_vptr,
                instance_index_ptrs=d_iptr,
                instance_normal_ptrs=d_nptr,
                instance_colors=d_col,
            )
            self._ias_handle = tlas_handle
            self.scene.state = SceneState.OK

    def render_headless(self, frames: int = 1) -> np.ndarray:
        with wp.ScopedDevice(self.device):
            for _ in range(max(int(frames), 1)):
                self.render_frame()
            return self.color.numpy().reshape(self.height, self.width)

    def render_frame(self) -> None:
        with wp.ScopedDevice(self.device):
            self.rebuild_instance_tree()
            assert self.resources is not None
            self.frame_id += 1
            pos, u, v, w = self.camera.get_basis(self.width, self.height)

            self.params_host["image"][0] = np.uint64(self.color.ptr)
            self.params_host["width"][0] = np.uint32(self.width)
            self.params_host["height"][0] = np.uint32(self.height)
            self.params_host["frame_id"][0] = np.uint32(self.frame_id)
            self.params_host["trav_handle"][0] = np.uint64(self._ias_handle)

            self.params_host["cam_px"][0], self.params_host["cam_py"][0], self.params_host["cam_pz"][0] = pos
            self.params_host["cam_ux"][0], self.params_host["cam_uy"][0], self.params_host["cam_uz"][0] = u
            self.params_host["cam_vx"][0], self.params_host["cam_vy"][0], self.params_host["cam_vz"][0] = v
            self.params_host["cam_wx"][0], self.params_host["cam_wy"][0], self.params_host["cam_wz"][0] = w
            self.params_host["instance_vertex_ptrs"][0] = np.uint64(self.resources.instance_vertex_ptrs.ptr)
            self.params_host["instance_index_ptrs"][0] = np.uint64(self.resources.instance_index_ptrs.ptr)
            self.params_host["instance_normal_ptrs"][0] = np.uint64(self.resources.instance_normal_ptrs.ptr)
            self.params_host["instance_color_ptrs"][0] = np.uint64(self.resources.instance_colors.ptr)
            self.params_host["light_dx"][0] = np.float32(0.40)
            self.params_host["light_dy"][0] = np.float32(0.85)
            self.params_host["light_dz"][0] = np.float32(0.30)

            params_bytes = self.params_host.view(np.uint8).reshape(-1)
            wp.copy(self.params_device, wp.array(params_bytes, dtype=wp.uint8, device="cpu", copy=False))

            self.optix.launch(
                self.resources.pipeline,
                0,
                self.params_device.ptr,
                self.params_dtype.itemsize,
                self.resources.sbt,
                self.width,
                self.height,
                1,
            )
            wp.synchronize_device(self.device)
