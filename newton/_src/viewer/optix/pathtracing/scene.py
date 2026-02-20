# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Scene management for path tracing viewer.
Handles mesh loading, BLAS/TLAS construction, and instance management.
"""

import logging

import numpy as np
import warp as wp

from ..color_utils import srgb_to_linear_rgb
from ..transform_utils import mat4_to_optix_transform12
from .materials import MaterialManager

logger = logging.getLogger(__name__)


def _create_vertex_buffers_dtype():
    """Create numpy dtype for VertexBuffers structure (56 bytes)."""
    return np.dtype(
        [
            ("positionAddress", np.uint64),
            ("normalAddress", np.uint64),
            ("colorAddress", np.uint64),
            ("tangentAddress", np.uint64),
            ("texCoord0Address", np.uint64),
            ("texCoord1Address", np.uint64),
            ("prevPositionAddress", np.uint64),
        ]
    )


def _create_render_primitive_dtype():
    """Create numpy dtype for RenderPrimitive structure (72 bytes)."""
    vb_dt = _create_vertex_buffers_dtype()
    return np.dtype(
        [
            ("indexAddress", np.uint64),
            ("materialIdAddress", np.uint64),
            ("vertexBuffer", vb_dt),
        ]
    )


def _create_render_node_dtype():
    """Create numpy dtype for RenderNode structure (136 bytes)."""
    return np.dtype(
        [
            ("objectToWorld", np.float32, (4, 4)),
            ("worldToObject", np.float32, (4, 4)),
            ("materialID", np.int32),
            ("renderPrimID", np.int32),
        ]
    )


def _create_scene_description_dtype():
    """Create numpy dtype for SceneDescription structure (48 bytes)."""
    return np.dtype(
        [
            ("materialAddress", np.uint64),
            ("renderNodeAddress", np.uint64),
            ("renderPrimitiveAddress", np.uint64),
            ("lightAddress", np.uint64),
            ("numLights", np.int32),
            ("_padding", np.int32),
        ]
    )


def _build_optix_instance_dtype() -> np.dtype:
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


class Mesh:
    """Represents a mesh with vertices, indices, and GPU buffers."""

    def __init__(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray = None,
        texcoords: np.ndarray = None,
        texcoords1: np.ndarray = None,
        material_id: int = 0,
    ):
        """
        Create a mesh.

        Args:
            vertices: Nx3 array of vertex positions
            indices: Mx3 array of triangle indices
            normals: Nx3 array of vertex normals (optional, will compute if None)
            texcoords: Nx2 array of texture coordinates (optional)
            material_id: Material index for this mesh
        """
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.indices = np.ascontiguousarray(indices, dtype=np.uint32)

        if normals is None:
            normals = self._compute_normals(self.vertices, self.indices)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)

        if texcoords is None:
            texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
        self.texcoords = np.ascontiguousarray(texcoords, dtype=np.float32)
        if texcoords1 is None:
            texcoords1 = np.zeros((len(vertices), 2), dtype=np.float32)
        self.texcoords1 = np.ascontiguousarray(texcoords1, dtype=np.float32)

        self.material_id = material_id

        # Tangents for normal mapping (computed from UVs; used when normalTexIndex >= 0)
        self.tangents = self._compute_tangents(self.vertices, self.indices, self.normals, self.texcoords)

        # GPU buffers (created on build)
        self.d_vertices = None
        self.d_indices = None
        self.d_normals = None
        self.d_tangents = None
        self.d_texcoords = None
        self.d_texcoords1 = None
        self.d_material_ids = None

    def _compute_tangents(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray,
        texcoords: np.ndarray,
    ) -> np.ndarray:
        """Compute vertex tangents from UVs using vectorized operations."""
        n_verts = len(vertices)
        tangents = np.zeros((n_verts, 4), dtype=np.float32)
        tan1 = np.zeros((n_verts, 3), dtype=np.float32)
        tan2 = np.zeros((n_verts, 3), dtype=np.float32)

        tri = indices.astype(np.int64, copy=False)
        i0 = tri[:, 0]
        i1 = tri[:, 1]
        i2 = tri[:, 2]

        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        uv0 = texcoords[i0]
        uv1 = texcoords[i1]
        uv2 = texcoords[i2]

        e1 = v1 - v0
        e2 = v2 - v0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0

        denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
        valid = np.abs(denom) >= 1.0e-8
        if np.any(valid):
            inv = np.zeros_like(denom, dtype=np.float32)
            inv[valid] = 1.0 / denom[valid]
            inv3 = inv[:, None]
            tri_tangent = (e1 * duv2[:, 1:2] - e2 * duv1[:, 1:2]) * inv3
            tri_bitangent = (e2 * duv1[:, 0:1] - e1 * duv2[:, 0:1]) * inv3
            tri_tangent[~valid] = 0.0
            tri_bitangent[~valid] = 0.0

            np.add.at(tan1, i0, tri_tangent)
            np.add.at(tan1, i1, tri_tangent)
            np.add.at(tan1, i2, tri_tangent)
            np.add.at(tan2, i0, tri_bitangent)
            np.add.at(tan2, i1, tri_bitangent)
            np.add.at(tan2, i2, tri_bitangent)

        n = normals
        tangent = tan1 - n * np.sum(n * tan1, axis=1, keepdims=True)
        t_len = np.linalg.norm(tangent, axis=1, keepdims=True)
        good = t_len[:, 0] > 1.0e-8
        tangent[good] /= t_len[good]

        fallback_x = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n_verts, 1))
        fallback_y = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n_verts, 1))
        use_y = np.abs(n[:, 0]) > 0.9
        tangent[~good] = np.where(use_y[~good, None], fallback_y[~good], fallback_x[~good])

        cross_nt = np.cross(n, tangent)
        handedness = np.where(np.sum(cross_nt * tan2, axis=1) < 0.0, -1.0, 1.0).astype(np.float32)

        tangents[:, :3] = tangent
        tangents[:, 3] = handedness
        return np.ascontiguousarray(tangents, dtype=np.float32)

    def _compute_normals(self, vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Compute vertex normals from face normals."""
        normals = np.zeros_like(vertices)

        for tri in indices:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            face_normal = np.cross(e1, e2)

            normals[tri[0]] += face_normal
            normals[tri[1]] += face_normal
            normals[tri[2]] += face_normal

        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        normals = normals / lengths

        return normals

    def upload_to_gpu(self):
        """Upload mesh data to GPU."""
        self.d_vertices = wp.array(self.vertices.flatten(), dtype=wp.float32, device="cuda")
        self.d_indices = wp.array(self.indices.flatten(), dtype=wp.uint32, device="cuda")
        self.d_normals = wp.array(self.normals.flatten(), dtype=wp.float32, device="cuda")
        self.d_tangents = wp.array(self.tangents.flatten(), dtype=wp.float32, device="cuda")
        self.d_texcoords = wp.array(self.texcoords.flatten(), dtype=wp.float32, device="cuda")
        self.d_texcoords1 = wp.array(self.texcoords1.flatten(), dtype=wp.float32, device="cuda")

        # Per-triangle material IDs
        num_triangles = len(self.indices)
        material_ids = np.full(num_triangles, self.material_id, dtype=np.uint32)
        self.d_material_ids = wp.array(material_ids, dtype=wp.uint32, device="cuda")


class Instance:
    """Represents an instance of a mesh with a transform."""

    def __init__(self, mesh_index: int, transform: np.ndarray = None):
        """
        Create an instance.

        Args:
            mesh_index: Index of the mesh in the scene
            transform: 4x4 transformation matrix (identity if None)
        """
        self.mesh_index = mesh_index
        if transform is None:
            transform = np.eye(4, dtype=np.float32)
        self.transform = np.ascontiguousarray(transform, dtype=np.float32)
        self.prev_transform = self.transform.copy()


class Scene:
    """
    Scene management for path tracing.
    Handles meshes, instances, materials, and acceleration structures.
    """

    def __init__(self, optix_ctx):
        """
        Create a scene.

        Args:
            optix_ctx: OptiX device context
        """
        self._optix = None  # Will be set when building
        self._ctx = optix_ctx
        self.materials = MaterialManager()
        self._meshes = []
        self._instances = []

        # GPU buffers
        self._render_primitives = None
        self._render_nodes = None
        self._scene_desc = None
        self._instance_material_ids = None
        self._compact_materials = None
        self._instance_render_prim_ids = None
        self._texture_descs = None
        self._texture_data = None
        self._gltf_textures = []

        # Acceleration structures
        self._gas_handles = []
        self._gas_buffers = []
        self._ias_handle = None
        self._ias_buffer = None
        self._instance_buffer = None
        self._instance_np_cache = None
        self._instance_np_capacity = 0
        self._tlas_temp_buffer = None
        self._tlas_temp_capacity = 0
        self._tlas_output_capacity = 0

        # Keepalive references
        self._keepalive = {}

    @property
    def mesh_count(self) -> int:
        return len(self._meshes)

    @property
    def instance_count(self) -> int:
        return len(self._instances)

    @property
    def tlas_handle(self) -> int:
        """Get the TLAS traversable handle."""
        return self._ias_handle if self._ias_handle else 0

    @property
    def scene_desc_address(self) -> int:
        """Get the device address of the scene description."""
        return self._scene_desc.ptr if self._scene_desc else 0

    @property
    def instance_material_ids_address(self) -> int:
        """Get device address of per-instance material id buffer."""
        return self._instance_material_ids.ptr if self._instance_material_ids is not None else 0

    @property
    def compact_materials_address(self) -> int:
        """Get device address of compact material table."""
        return self._compact_materials.ptr if self._compact_materials is not None else 0

    @property
    def render_nodes_address(self) -> int:
        """Get device address of render node array."""
        return self._render_nodes.ptr if self._render_nodes is not None else 0

    @property
    def render_primitives_address(self) -> int:
        """Get device address of render primitive array."""
        return self._render_primitives.ptr if self._render_primitives is not None else 0

    @property
    def instance_render_prim_ids_address(self) -> int:
        """Get device address of per-instance render primitive id buffer."""
        return self._instance_render_prim_ids.ptr if self._instance_render_prim_ids is not None else 0

    @property
    def texture_descs_address(self) -> int:
        """Get device address of texture descriptor buffer."""
        return self._texture_descs.ptr if self._texture_descs is not None else 0

    @property
    def texture_data_address(self) -> int:
        """Get device address of texture texel buffer."""
        return self._texture_data.ptr if self._texture_data is not None else 0

    @property
    def texture_count(self) -> int:
        """Get number of loaded glTF textures."""
        return len(self._gltf_textures)

    @property
    def has_meshes(self) -> bool:
        """Return True when at least one mesh is present."""
        return bool(self._meshes)

    def get_instance_material_ids_host(self) -> np.ndarray | None:
        """Return per-instance material IDs as a host NumPy array copy."""
        if self._instance_material_ids is None:
            return None
        return self._instance_material_ids.numpy()

    def set_instance_material_ids_host(self, material_ids: np.ndarray):
        """Upload per-instance material IDs from host memory."""
        self._instance_material_ids = wp.array(np.asarray(material_ids, dtype=np.uint32), dtype=wp.uint32, device="cuda")

    def set_compact_material_bytes(self, compact_bytes: np.ndarray):
        """Upload compact material table bytes to GPU."""
        self._compact_materials = wp.array(np.asarray(compact_bytes, dtype=np.uint8), dtype=wp.uint8, device="cuda")

    def set_gltf_textures(self, textures: list[np.ndarray], srgb_texture_indices: set[int] | None = None):
        """
        Set glTF texture list as RGBA float32 images.

        Args:
            textures: RGBA textures in normalized [0,1] space.
            srgb_texture_indices: Texture indices that represent color data and
                therefore require sRGB->linear decode to match Vulkan hardware
                sampling with ``R8G8B8A8_SRGB``.
        """
        srgb_indices = srgb_texture_indices or set()
        converted = []
        for tex_idx, tex in enumerate(textures):
            tex_f = np.ascontiguousarray(tex, dtype=np.float32)
            if tex_idx in srgb_indices:
                tex_f = tex_f.copy()
                tex_f[..., :3] = srgb_to_linear_rgb(np.clip(tex_f[..., :3], 0.0, 1.0))
            converted.append(tex_f)
        self._gltf_textures = converted

    def add_mesh(self, mesh: Mesh) -> int:
        """Add a mesh to the scene and return its index."""
        self._meshes.append(mesh)
        return len(self._meshes) - 1

    def add_instance(self, mesh_index: int, transform: np.ndarray = None) -> int:
        """Add an instance of a mesh and return its index."""
        instance = Instance(mesh_index, transform)
        self._instances.append(instance)
        return len(self._instances) - 1

    def set_instance_transform(self, instance_index: int, transform: np.ndarray):
        """Update an instance's transform."""
        if 0 <= instance_index < len(self._instances):
            inst = self._instances[instance_index]
            inst.prev_transform = inst.transform.copy()
            inst.transform = np.ascontiguousarray(transform, dtype=np.float32)

    def create_cornell_box(self):
        """Create a Cornell Box scene."""
        self.clear()

        # Materials
        white = self.materials.add_diffuse((0.8, 0.8, 0.8))
        red = self.materials.add_diffuse((0.8, 0.1, 0.1))
        green = self.materials.add_diffuse((0.1, 0.8, 0.1))
        light = self.materials.add_emissive((1.0, 0.95, 0.85), 15.0)

        # Floor
        self.add_box((-2, -2, -2), (2, -1.9, 2), white)
        # Ceiling
        self.add_box((-2, 1.9, -2), (2, 2, 2), white)
        # Back wall
        self.add_box((-2, -2, -2), (2, 2, -1.9), white)
        # Left wall (red)
        self.add_box((-2, -2, -2), (-1.9, 2, 2), red)
        # Right wall (green)
        self.add_box((1.9, -2, -2), (2, 2, 2), green)
        # Short box
        self.add_box((-0.8, -1.9, -0.8), (0.2, -0.5, 0.2), white)
        # Tall box
        self.add_box((0.3, -1.9, -1.2), (1.2, 0.3, -0.3), white)
        # Light
        self.add_box((-0.5, 1.85, -0.5), (0.5, 1.89, 0.5), light)

    def load_from_gltf(self, gltf_path: str) -> bool:
        """Load a glTF/GLB scene into this scene."""
        from .asset_loaders import load_scene_from_gltf  # noqa: PLC0415

        self.clear()
        return bool(load_scene_from_gltf(self, gltf_path))

    def load_from_obj(self, obj_path: str) -> bool:
        """Load an OBJ scene into this scene."""
        from .asset_loaders import load_scene_from_obj  # noqa: PLC0415

        self.clear()
        return bool(load_scene_from_obj(self, obj_path))

    def add_box(self, min_pt: tuple, max_pt: tuple, material_id: int) -> int:
        """Add a box mesh and return the instance index."""
        vertices, indices = self._create_box_geometry(min_pt, max_pt)
        mesh = Mesh(vertices, indices, material_id=material_id)
        mesh_idx = self.add_mesh(mesh)
        return self.add_instance(mesh_idx)

    def add_sphere(
        self,
        center: tuple,
        radius: float,
        segments: int,
        material_id: int,
    ) -> int:
        """
        Add a UV sphere mesh and return the instance index.

        Args:
            center: Sphere center
            radius: Sphere radius
            segments: Horizontal tessellation (minimum 8)
            material_id: Material index
        """
        vertices, indices, normals, texcoords = self._create_sphere_geometry(
            center=center,
            radius=radius,
            segments=max(8, int(segments)),
        )
        mesh = Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            texcoords=texcoords,
            material_id=material_id,
        )
        mesh_idx = self.add_mesh(mesh)
        return self.add_instance(mesh_idx)

    def _create_box_geometry(self, min_pt: tuple, max_pt: tuple) -> tuple:
        """Create box vertices and indices."""
        mn = np.array(min_pt, dtype=np.float32)
        mx = np.array(max_pt, dtype=np.float32)

        # 8 corners
        c = np.array(
            [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]],
                [mn[0], mx[1], mx[2]],
            ],
            dtype=np.float32,
        )

        # 6 faces, 4 vertices each (with normals)
        vertices = []
        indices = []

        def add_quad(v0, v1, v2, v3, normal):
            base = len(vertices)
            vertices.extend([v0, v1, v2, v3])
            indices.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

        # Front (+Z)
        add_quad(c[4], c[5], c[6], c[7], [0, 0, 1])
        # Back (-Z)
        add_quad(c[1], c[0], c[3], c[2], [0, 0, -1])
        # Left (-X)
        add_quad(c[0], c[4], c[7], c[3], [-1, 0, 0])
        # Right (+X)
        add_quad(c[5], c[1], c[2], c[6], [1, 0, 0])
        # Bottom (-Y)
        add_quad(c[0], c[1], c[5], c[4], [0, -1, 0])
        # Top (+Y)
        add_quad(c[7], c[6], c[2], c[3], [0, 1, 0])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def _create_sphere_geometry(
        self,
        center: tuple,
        radius: float,
        segments: int,
    ) -> tuple:
        """Create UV sphere vertices, indices, normals, and UVs."""
        cx, cy, cz = center
        lat_segments = max(4, segments // 2)
        lon_segments = max(8, segments)

        vertices = []
        normals = []
        texcoords = []
        indices = []

        for y in range(lat_segments + 1):
            v = y / lat_segments
            theta = v * np.pi
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for x in range(lon_segments + 1):
                u = x / lon_segments
                phi = u * (2.0 * np.pi)
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                nx = sin_theta * cos_phi
                ny = cos_theta
                nz = sin_theta * sin_phi

                px = cx + radius * nx
                py = cy + radius * ny
                pz = cz + radius * nz

                vertices.append([px, py, pz])
                normals.append([nx, ny, nz])
                texcoords.append([u, 1.0 - v])

        stride = lon_segments + 1
        for y in range(lat_segments):
            for x in range(lon_segments):
                i0 = y * stride + x
                i1 = i0 + 1
                i2 = i0 + stride
                i3 = i2 + 1
                indices.append([i0, i2, i1])
                indices.append([i1, i2, i3])

        return (
            np.array(vertices, dtype=np.float32),
            np.array(indices, dtype=np.uint32),
            np.array(normals, dtype=np.float32),
            np.array(texcoords, dtype=np.float32),
        )

    def clear(self):
        """Clear all meshes and instances."""
        self._meshes.clear()
        self._instances.clear()
        self._gas_handles.clear()
        self._gas_buffers.clear()
        self._ias_handle = None
        self._ias_buffer = None
        self._instance_buffer = None
        self._instance_np_cache = None
        self._instance_np_capacity = 0
        self._tlas_temp_buffer = None
        self._tlas_temp_capacity = 0
        self._tlas_output_capacity = 0
        self._keepalive.clear()
        self.materials.clear()
        self._instance_material_ids = None
        self._compact_materials = None
        self._instance_render_prim_ids = None
        self._texture_descs = None
        self._texture_data = None
        self._gltf_textures = []

    def build(self, optix_module):
        """
        Build the scene: upload meshes, create BLASes, and create TLAS.

        Args:
            optix_module: The imported optix module
        """
        self._optix = optix_module

        if len(self._meshes) == 0:
            logger.info("No meshes to build.")
            return

        logger.info("Building scene with %d meshes and %d instances.", len(self._meshes), len(self._instances))

        # Upload meshes to GPU
        for mesh in self._meshes:
            mesh.upload_to_gpu()

        # Build BLAS for each mesh
        self._build_blas()

        # Build TLAS
        self._build_tlas()

        # Build scene buffers
        self._build_scene_buffers()

        total_verts = sum(len(m.vertices) for m in self._meshes)
        total_tris = sum(len(m.indices) for m in self._meshes)
        logger.info("Scene build complete: %d vertices, %d triangles.", total_verts, total_tris)

    def rebuild_tlas(self):
        """Rebuild only TLAS after instance transform updates."""
        if len(self._meshes) == 0 or len(self._instances) == 0:
            return
        if self._optix is None:
            return
        if len(self._gas_handles) != len(self._meshes):
            # Fallback if a full build has not been completed yet.
            self.build(self._optix)
            return
        self._build_tlas()

    def _build_blas(self):
        """Build bottom-level acceleration structures for all meshes."""
        optix = self._optix

        accel_options = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
            operation=optix.BUILD_OPERATION_BUILD,
        )

        for i, mesh in enumerate(self._meshes):
            tri = optix.BuildInputTriangleArray()
            tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
            tri.numVertices = len(mesh.vertices)
            tri.vertexStrideInBytes = 12
            tri.vertexBuffers = [mesh.d_vertices.ptr]
            tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
            tri.numIndexTriplets = len(mesh.indices)
            tri.indexStrideInBytes = 12
            tri.indexBuffer = mesh.d_indices.ptr
            tri.flags = [optix.GEOMETRY_FLAG_NONE]
            tri.numSbtRecords = 1

            sizes = self._ctx.accelComputeMemoryUsage([accel_options], [tri])
            d_temp = wp.empty(sizes.tempSizeInBytes, dtype=wp.uint8, device="cuda")
            d_gas = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device="cuda")

            handle = self._ctx.accelBuild(
                0,
                [accel_options],
                [tri],
                d_temp.ptr,
                sizes.tempSizeInBytes,
                d_gas.ptr,
                sizes.outputSizeInBytes,
                [],
            )

            self._gas_handles.append(int(handle))
            self._gas_buffers.append(d_gas)
            self._keepalive[f"gas_temp_{i}"] = d_temp

        wp.synchronize_device("cuda")

    def _build_tlas(self):
        """Build top-level acceleration structure."""
        optix = self._optix

        if len(self._instances) == 0:
            return
        inst_dtype = _build_optix_instance_dtype()
        count = len(self._instances)
        if self._instance_np_capacity < count:
            self._instance_np_capacity = max(count, 1, self._instance_np_capacity * 2)
            self._instance_np_cache = np.zeros(self._instance_np_capacity, dtype=inst_dtype)
        inst_np = self._instance_np_cache[:count]

        for i, inst in enumerate(self._instances):
            gas_handle = self._gas_handles[inst.mesh_index]
            inst_np["transform"][i] = mat4_to_optix_transform12(inst.transform)
            inst_np["instanceId"][i] = np.uint32(i)
            inst_np["sbtOffset"][i] = np.uint32(0)
            inst_np["visibilityMask"][i] = np.uint32(0xFF)
            inst_np["flags"][i] = np.uint32(int(optix.INSTANCE_FLAG_NONE))
            inst_np["traversableHandle"][i] = np.uint64(gas_handle)

        inst_bytes = inst_np.view(np.uint8).reshape(-1)
        if self._instance_buffer is None or self._instance_buffer.shape[0] != inst_bytes.shape[0]:
            self._instance_buffer = wp.empty(inst_bytes.shape[0], dtype=wp.uint8, device="cuda")
        self._instance_buffer.assign(inst_bytes)

        accel_options = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_ALLOW_UPDATE),
            operation=optix.BUILD_OPERATION_BUILD,
        )

        ias_input = optix.BuildInputInstanceArray()
        ias_input.instances = int(self._instance_buffer.ptr)
        ias_input.numInstances = count

        sizes = self._ctx.accelComputeMemoryUsage([accel_options], [ias_input])
        required_temp = int(sizes.tempSizeInBytes)
        required_output = int(sizes.outputSizeInBytes)

        # Reuse TLAS scratch/output buffers across rebuilds.
        # Overallocate slightly to reduce realloc churn when scene size
        # changes by a small amount.
        if self._tlas_temp_buffer is None or self._tlas_temp_capacity < required_temp:
            self._tlas_temp_capacity = max(required_temp, int(required_temp * 1.25))
            self._tlas_temp_buffer = wp.empty(self._tlas_temp_capacity, dtype=wp.uint8, device="cuda")
        if self._ias_buffer is None or self._tlas_output_capacity < required_output:
            self._tlas_output_capacity = max(required_output, int(required_output * 1.25))
            self._ias_buffer = wp.empty(self._tlas_output_capacity, dtype=wp.uint8, device="cuda")

        self._ias_handle = self._ctx.accelBuild(
            0,
            [accel_options],
            [ias_input],
            self._tlas_temp_buffer.ptr,
            required_temp,
            self._ias_buffer.ptr,
            required_output,
            [],
        )

        self._keepalive["ias_temp"] = self._tlas_temp_buffer
        wp.synchronize_device("cuda")

    def _build_scene_buffers(self):
        """Build GPU buffers for scene description."""
        # Build RenderPrimitive array
        rp_dtype = _create_render_primitive_dtype()
        render_primitives = np.zeros(len(self._meshes), dtype=rp_dtype)

        for i, mesh in enumerate(self._meshes):
            rp = render_primitives[i]
            rp["indexAddress"] = mesh.d_indices.ptr
            rp["materialIdAddress"] = mesh.d_material_ids.ptr
            rp["vertexBuffer"]["positionAddress"] = mesh.d_vertices.ptr
            rp["vertexBuffer"]["normalAddress"] = mesh.d_normals.ptr
            rp["vertexBuffer"]["tangentAddress"] = mesh.d_tangents.ptr
            rp["vertexBuffer"]["texCoord0Address"] = mesh.d_texcoords.ptr
            rp["vertexBuffer"]["texCoord1Address"] = mesh.d_texcoords1.ptr

        rp_bytes = render_primitives.view(np.uint8).reshape(-1)
        self._render_primitives = wp.array(rp_bytes, dtype=wp.uint8, device="cuda")

        # Build RenderNode array
        rn_dtype = _create_render_node_dtype()
        render_nodes = np.zeros(len(self._instances), dtype=rn_dtype)

        for i, inst in enumerate(self._instances):
            rn = render_nodes[i]
            rn["objectToWorld"] = inst.transform
            rn["worldToObject"] = np.linalg.inv(inst.transform)
            rn["materialID"] = -1  # Use per-triangle materials
            rn["renderPrimID"] = inst.mesh_index

        rn_bytes = render_nodes.view(np.uint8).reshape(-1)
        self._render_nodes = wp.array(rn_bytes, dtype=wp.uint8, device="cuda")

        # Build SceneDescription
        sd_dtype = _create_scene_description_dtype()
        scene_desc = np.zeros(1, dtype=sd_dtype)
        scene_desc[0]["materialAddress"] = self.materials.gpu_address
        scene_desc[0]["renderNodeAddress"] = self._render_nodes.ptr
        scene_desc[0]["renderPrimitiveAddress"] = self._render_primitives.ptr
        scene_desc[0]["lightAddress"] = 0
        scene_desc[0]["numLights"] = 0

        sd_bytes = scene_desc.view(np.uint8).reshape(-1)
        self._scene_desc = wp.array(sd_bytes, dtype=wp.uint8, device="cuda")

        # Build per-instance material ID lookup buffer.
        instance_material_ids = np.zeros(len(self._instances), dtype=np.uint32)
        instance_render_prim_ids = np.zeros(len(self._instances), dtype=np.uint32)
        for i, inst in enumerate(self._instances):
            instance_material_ids[i] = np.uint32(self._meshes[inst.mesh_index].material_id)
            instance_render_prim_ids[i] = np.uint32(inst.mesh_index)
        self._instance_material_ids = wp.array(instance_material_ids, dtype=wp.uint32, device="cuda")
        self._instance_render_prim_ids = wp.array(instance_render_prim_ids, dtype=wp.uint32, device="cuda")

        # Build compact material table for robust device-side lookup.
        compact_dt = np.dtype(
            [
                ("baseColor", np.float32, (3,)),
                ("emissive", np.float32, (3,)),
                ("roughness", np.float32),
                ("metallic", np.float32),
                ("transmission", np.float32),
                ("ior", np.float32),
                ("specularColor", np.float32, (3,)),
                ("specular", np.float32),
                ("clearcoat", np.float32),
                ("clearcoatRoughness", np.float32),
                ("clearcoatNormalTexIndex", np.int32),
                ("clearcoatNormalTexCoord", np.int32),
                ("opacity", np.float32),
                ("baseColorTexIndex", np.int32),
                ("baseColorTexCoord", np.int32),
                ("metallicRoughnessTexIndex", np.int32),
                ("metallicRoughnessTexCoord", np.int32),
                ("normalTexIndex", np.int32),
                ("normalTexCoord", np.int32),
                ("emissiveTexIndex", np.int32),
                ("emissiveTexCoord", np.int32),
                ("normalScale", np.float32),
                ("baseColorUvTransform", np.float32, (6,)),
                ("metallicRoughnessUvTransform", np.float32, (6,)),
                ("normalUvTransform", np.float32, (6,)),
                ("emissiveUvTransform", np.float32, (6,)),
                ("clearcoatNormalUvTransform", np.float32, (6,)),
            ],
            align=True,
        )
        compact = np.zeros(self.materials.count, dtype=compact_dt)
        for i, mat in enumerate(self.materials._materials):
            compact[i]["baseColor"] = mat["pbrBaseColorFactor"][:3]
            compact[i]["emissive"] = mat["emissiveFactor"]
            compact[i]["roughness"] = mat["pbrRoughnessFactor"]
            compact[i]["metallic"] = mat["pbrMetallicFactor"]
            compact[i]["transmission"] = mat["transmissionFactor"]
            compact[i]["ior"] = mat["ior"]
            compact[i]["specularColor"] = mat["specularColorFactor"]
            compact[i]["specular"] = mat["specularFactor"]
            compact[i]["clearcoat"] = mat["clearcoatFactor"]
            compact[i]["clearcoatRoughness"] = mat["clearcoatRoughness"]
            compact[i]["clearcoatNormalTexIndex"] = mat["clearcoatNormalTexture"]["index"]
            compact[i]["clearcoatNormalTexCoord"] = mat["clearcoatNormalTexture"]["texCoord"]
            compact[i]["opacity"] = mat["pbrBaseColorFactor"][3]
            compact[i]["baseColorTexIndex"] = mat["pbrBaseColorTexture"]["index"]
            compact[i]["baseColorTexCoord"] = mat["pbrBaseColorTexture"]["texCoord"]
            compact[i]["metallicRoughnessTexIndex"] = mat["pbrMetallicRoughnessTexture"]["index"]
            compact[i]["metallicRoughnessTexCoord"] = mat["pbrMetallicRoughnessTexture"]["texCoord"]
            compact[i]["normalTexIndex"] = mat["normalTexture"]["index"]
            compact[i]["normalTexCoord"] = mat["normalTexture"]["texCoord"]
            compact[i]["emissiveTexIndex"] = mat["emissiveTexture"]["index"]
            compact[i]["emissiveTexCoord"] = mat["emissiveTexture"]["texCoord"]
            compact[i]["normalScale"] = mat["normalTextureScale"]
            compact[i]["baseColorUvTransform"] = (
                mat["pbrBaseColorTexture"]["uvTransform00"],
                mat["pbrBaseColorTexture"]["uvTransform01"],
                mat["pbrBaseColorTexture"]["uvTransform02"],
                mat["pbrBaseColorTexture"]["uvTransform10"],
                mat["pbrBaseColorTexture"]["uvTransform11"],
                mat["pbrBaseColorTexture"]["uvTransform12"],
            )
            compact[i]["metallicRoughnessUvTransform"] = (
                mat["pbrMetallicRoughnessTexture"]["uvTransform00"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform01"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform02"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform10"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform11"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform12"],
            )
            compact[i]["normalUvTransform"] = (
                mat["normalTexture"]["uvTransform00"],
                mat["normalTexture"]["uvTransform01"],
                mat["normalTexture"]["uvTransform02"],
                mat["normalTexture"]["uvTransform10"],
                mat["normalTexture"]["uvTransform11"],
                mat["normalTexture"]["uvTransform12"],
            )
            compact[i]["emissiveUvTransform"] = (
                mat["emissiveTexture"]["uvTransform00"],
                mat["emissiveTexture"]["uvTransform01"],
                mat["emissiveTexture"]["uvTransform02"],
                mat["emissiveTexture"]["uvTransform10"],
                mat["emissiveTexture"]["uvTransform11"],
                mat["emissiveTexture"]["uvTransform12"],
            )
            compact[i]["clearcoatNormalUvTransform"] = (
                mat["clearcoatNormalTexture"]["uvTransform00"],
                mat["clearcoatNormalTexture"]["uvTransform01"],
                mat["clearcoatNormalTexture"]["uvTransform02"],
                mat["clearcoatNormalTexture"]["uvTransform10"],
                mat["clearcoatNormalTexture"]["uvTransform11"],
                mat["clearcoatNormalTexture"]["uvTransform12"],
            )
        compact_bytes = compact.view(np.uint8).reshape(-1)
        self._compact_materials = wp.array(compact_bytes, dtype=wp.uint8, device="cuda")

        # Build texture descriptor/texel buffers for shader-side glTF texture sampling.
        if self._gltf_textures:
            desc_dtype = np.dtype(
                [
                    ("offset", np.uint32),
                    ("width", np.uint32),
                    ("height", np.uint32),
                    ("_pad", np.uint32),
                ],
                align=True,
            )
            desc = np.zeros(len(self._gltf_textures), dtype=desc_dtype)
            packed_texels = []
            texel_offset = 0
            for i, tex in enumerate(self._gltf_textures):
                h, w = int(tex.shape[0]), int(tex.shape[1])
                desc[i]["offset"] = np.uint32(texel_offset)
                desc[i]["width"] = np.uint32(w)
                desc[i]["height"] = np.uint32(h)
                flat_tex = tex.reshape(-1, 4)
                packed_texels.append(flat_tex)
                texel_offset += flat_tex.shape[0]

            desc_bytes = desc.view(np.uint8).reshape(-1)
            self._texture_descs = wp.array(desc_bytes, dtype=wp.uint8, device="cuda")

            texels = np.concatenate(packed_texels, axis=0).astype(np.float32, copy=False)
            self._texture_data = wp.array(texels.reshape(-1), dtype=wp.float32, device="cuda")
        else:
            self._texture_descs = None
            self._texture_data = None
