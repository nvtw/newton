from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import warp as wp


def compute_vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    for tri in np.asarray(indices, dtype=np.uint32):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        e1 = p1 - p0
        e2 = p2 - p0
        n = np.cross(e1, e2).astype(np.float32)
        normals[i0] += n
        normals[i1] += n
        normals[i2] += n
    lens = np.linalg.norm(normals, axis=1)
    mask = lens > 1.0e-8
    normals[mask] /= lens[mask][:, None]
    normals[~mask] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return normals.astype(np.float32)


def create_cube_mesh(size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    s = 0.5 * float(size)
    vertices = np.array(
        [
            [-s, -s, -s],
            [s, -s, -s],
            [s, s, -s],
            [-s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [s, s, s],
            [-s, s, s],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.uint32,
    )
    return vertices, indices


def create_plane_mesh(width: float = 1.0, height: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    hw = 0.5 * float(width)
    hh = 0.5 * float(height)
    vertices = np.array(
        [[-hw, 0.0, -hh], [hw, 0.0, -hh], [hw, 0.0, hh], [-hw, 0.0, hh]],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return vertices, indices


def load_obj_mesh(path: str | Path, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh from a Wavefront OBJ file.

    Supports vertex (`v`) and face (`f`) records. Polygonal faces are
    triangulated as a fan. Texture/normal indices are ignored.
    """

    obj_path = Path(path)
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    src_vertices: list[list[float]] = []
    out_vertices: list[list[float]] = []
    out_indices: list[list[int]] = []
    remap: dict[int, int] = {}

    def get_out_index(obj_idx: int) -> int:
        if obj_idx in remap:
            return remap[obj_idx]
        v = src_vertices[obj_idx]
        out_idx = len(out_vertices)
        out_vertices.append([float(scale) * float(v[0]), float(scale) * float(v[1]), float(scale) * float(v[2])])
        remap[obj_idx] = out_idx
        return out_idx

    with obj_path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if parts[0] == "v" and len(parts) >= 4:
                src_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                face_idx: list[int] = []
                for tok in parts[1:]:
                    # token shape: v, v/t, v//n, v/t/n
                    v_str = tok.split("/")[0]
                    if not v_str:
                        continue
                    raw = int(v_str)
                    obj_i = (raw - 1) if raw > 0 else (len(src_vertices) + raw)
                    face_idx.append(get_out_index(obj_i))
                for i in range(1, len(face_idx) - 1):
                    out_indices.append([face_idx[0], face_idx[i], face_idx[i + 1]])

    if not out_vertices or not out_indices:
        raise ValueError(f"OBJ contains no triangle geometry: {obj_path}")

    return np.asarray(out_vertices, dtype=np.float32), np.asarray(out_indices, dtype=np.uint32)


@dataclass
class TriangleMeshGpu:
    vertices: np.ndarray
    indices: np.ndarray
    normals: np.ndarray | None = None
    d_vertices: wp.array | None = None
    d_indices: wp.array | None = None
    d_normals: wp.array | None = None

    def upload_gpu_data(self, device: str = "cuda") -> None:
        if self.normals is None:
            self.normals = compute_vertex_normals(self.vertices, self.indices)
        self.d_vertices = wp.array(self.vertices, dtype=wp.float32, device=device)
        self.d_indices = wp.array(self.indices, dtype=wp.uint32, device=device)
        self.d_normals = wp.array(self.normals, dtype=wp.float32, device=device)

    @classmethod
    def cube(cls, size: float = 1.0) -> TriangleMeshGpu:
        v, i = create_cube_mesh(size)
        return cls(vertices=v, indices=i, normals=compute_vertex_normals(v, i))

    @classmethod
    def plane(cls, width: float = 1.0, height: float = 1.0) -> TriangleMeshGpu:
        v, i = create_plane_mesh(width, height)
        n = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (v.shape[0], 1))
        return cls(vertices=v, indices=i, normals=n)

    @classmethod
    def from_obj(cls, path: str | Path, scale: float = 1.0) -> TriangleMeshGpu:
        v, i = load_obj_mesh(path=path, scale=scale)
        return cls(vertices=v, indices=i, normals=compute_vertex_normals(v, i))


@dataclass
class MeshWithAccelerationStructure:
    mesh: TriangleMeshGpu
    gas_handle: int = 0
    d_temp: wp.array | None = None
    d_gas: wp.array | None = None
    _build_flags: int = 0
    update_required: bool = True

    def upload_and_rebuild_acceleration_structure(self, optix, ctx, stream=0, device: str = "cuda") -> None:
        if not self.update_required:
            return
        self.mesh.upload_gpu_data(device=device)

        tri = optix.BuildInputTriangleArray()
        tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
        tri.numVertices = self.mesh.vertices.shape[0]
        tri.vertexStrideInBytes = 12
        tri.vertexBuffers = [self.mesh.d_vertices.ptr]
        tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
        tri.numIndexTriplets = self.mesh.indices.shape[0]
        tri.indexStrideInBytes = 12
        tri.indexBuffer = self.mesh.d_indices.ptr
        tri.flags = [optix.GEOMETRY_FLAG_NONE]
        tri.numSbtRecords = 1

        build_flags = int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS) | int(optix.BUILD_FLAG_ALLOW_UPDATE)
        accel_options = optix.AccelBuildOptions(buildFlags=build_flags, operation=optix.BUILD_OPERATION_BUILD)
        sizes = ctx.accelComputeMemoryUsage([accel_options], [tri])
        self.d_temp = wp.empty(sizes.tempSizeInBytes, dtype=wp.uint8, device=device)
        self.d_gas = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device=device)
        self.gas_handle = int(
            ctx.accelBuild(
                stream,
                [accel_options],
                [tri],
                self.d_temp.ptr,
                sizes.tempSizeInBytes,
                self.d_gas.ptr,
                sizes.outputSizeInBytes,
                [],
            )
        )
        self._build_flags = build_flags
        self.update_required = False
