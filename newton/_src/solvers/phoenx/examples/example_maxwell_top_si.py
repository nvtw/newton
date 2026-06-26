# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf

MAXWELL_TOP_USD_ENV = "NEWTON_MAXWELL_TOP_USD"
MAXWELL_TOP_USD_FILENAME = "MaxwellTopSI2.usda"
DEFAULT_WINDOWS_MAXWELL_TOP_USD_PATH = Path(r"C:\Users\twidmer\Downloads\MaxwellTopSI\MaxwellTopSI2.usda")
FPS = 300
SUBSTEPS = 5
SOLVER_ITERATIONS = 5
CONTACT_GAP = 0.001
CONTACT_MARGIN = 0.0
SOLVER_TYPE = "phoenx"
RIGID_CONTACT_MAX = None
REDUCE_CONTACTS = True
CONTACT_MATCHING = "sticky"
CONTACT_STATS = False
CONTACT_STATS_INTERVAL = 30
CONTACT_STATS_CSV = None
CONTACT_DUMP_FRAMES = ()
CONTACT_DUMP_TOP_N = 12
TRAJECTORY_CSV = None
TRAJECTORY_INTERVAL = 1
SDF_LINEARITY_BAD_RELERR = 0.2


def _linux_download_dir() -> Path:
    user_dirs_path = Path.home() / ".config" / "user-dirs.dirs"
    if user_dirs_path.exists():
        for raw_line in user_dirs_path.read_text().splitlines():
            user_dir_entry = raw_line.strip()
            if not user_dir_entry.startswith("XDG_DOWNLOAD_DIR="):
                continue
            download_dir = user_dir_entry.split("=", 1)[1].strip().strip('"')
            download_dir = download_dir.replace("$HOME", str(Path.home())).replace("${HOME}", str(Path.home()))
            return Path(download_dir).expanduser()

    return Path.home() / "Downloads"


def _find_linux_maxwell_top_usd() -> Path:
    download_dir = _linux_download_dir()
    candidates = (
        download_dir / "MaxwellTopSI" / MAXWELL_TOP_USD_FILENAME,
        download_dir / MAXWELL_TOP_USD_FILENAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if download_dir.exists():
        matches = sorted(download_dir.rglob(MAXWELL_TOP_USD_FILENAME))
        if matches:
            return matches[0]

    return candidates[0]


def _resolve_maxwell_top_usd_path(usd_path: str | None) -> Path:
    if usd_path:
        return Path(usd_path).expanduser()

    env_path = os.environ.get(MAXWELL_TOP_USD_ENV)
    if env_path:
        return Path(env_path).expanduser()

    if sys.platform.startswith("linux"):
        return _find_linux_maxwell_top_usd()

    return DEFAULT_WINDOWS_MAXWELL_TOP_USD_PATH


# World-frame initial angular velocity for the spinning body [deg/s].
# Main2 carries the authored spin in the USD; Spiral is the tip (zero spin).
TOP_BODY_PATH = "/World/Main2/Main2_obj0"
SPIRAL_BODY_PATH = "/World/Spiral/Spiral_obj0"
TRAJECTORY_BODY_PATHS = (TOP_BODY_PATH, SPIRAL_BODY_PATH)
TOP_ANGULAR_VELOCITY_DEG_S = (300.0, 0.0, 10000.0)
TRAJECTORY_FIELDS = [
    "frame",
    "time",
    "body_path",
    "body_id",
    "center_x",
    "center_y",
    "center_z",
    "tip_x",
    "tip_y",
    "tip_z",
    "linear_speed",
    "angular_speed",
    "contact_count",
]
CONTACT_STATS_FIELDS = [
    "frame",
    "contacts",
    "zero_gap",
    "speculative",
    "min_sep",
    "median_sep",
    "max_sep",
    "top_z",
    "top_speed",
    "top_ang_speed",
    "speculative_closing",
    "speculative_closing_frac",
    "jvn_min",
    "jvn_p05",
    "jvn_median",
    "jvn_plus_bias_min",
    "jvn_plus_bias_p05",
    "reducer_candidates",
    "reducer_capacity",
    "reducer_ht_active",
    "reducer_ht_capacity",
    "reducer_ht_insert_failures",
    "edge_contacts",
    "edge_anchor_p50",
    "edge_anchor_p95",
    "edge_anchor_max",
    "edge_anchor_signed_mean",
    "edge_anchor_over_sep_p50",
    "sdf_endpoint_abs_p50",
    "sdf_endpoint_abs_p95",
    "sdf_line_error_p50",
    "sdf_line_error_p95",
    "sdf_min_signed_delta_p05",
    "sdf_nonmonotone_frac",
    "sdf_gen_delta_ratio_p50",
    "sdf_gen_delta_ratio_p95",
    "sdf_gen_delta_relerr_p50",
    "sdf_gen_delta_relerr_p95",
    "sdf_gen_bad_frac",
    "sdf_two_sided_delta_relerr_p50",
    "sdf_two_sided_delta_relerr_p95",
    "sdf_two_sided_bad_frac",
]


@wp.func
def _sample_shape_sdf_world(
    shape: int,
    p_world: wp.vec3,
    shape_data: wp.array[wp.vec4],
    shape_transform: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_data: wp.array[TextureSDFData],
) -> tuple[float, bool]:
    sdf_idx = shape_sdf_index[shape]
    if sdf_idx < 0 or sdf_idx >= texture_sdf_data.shape[0]:
        return float(0.0), False

    texture_sdf = texture_sdf_data[sdf_idx]
    if texture_sdf.coarse_texture.width <= 0:
        return float(0.0), False

    sdf_scale_data = shape_data[shape]
    sdf_scale = wp.vec3(sdf_scale_data[0], sdf_scale_data[1], sdf_scale_data[2])
    if texture_sdf.scale_baked:
        sdf_scale = wp.vec3(1.0, 1.0, 1.0)

    sx = wp.where(wp.abs(sdf_scale[0]) > 1.0e-12, sdf_scale[0], 1.0)
    sy = wp.where(wp.abs(sdf_scale[1]) > 1.0e-12, sdf_scale[1], 1.0)
    sz = wp.where(wp.abs(sdf_scale[2]) > 1.0e-12, sdf_scale[2], 1.0)
    inv_sdf_scale = wp.vec3(1.0 / sx, 1.0 / sy, 1.0 / sz)
    min_sdf_scale = wp.min(wp.min(wp.abs(sx), wp.abs(sy)), wp.abs(sz))
    p_sdf = wp.transform_point(wp.transform_inverse(shape_transform[shape]), p_world)
    p_unscaled = wp.cw_mul(p_sdf, inv_sdf_scale)
    return texture_sample_sdf(texture_sdf, p_unscaled) * min_sdf_scale, True


@wp.kernel(enable_backward=False)
def _measure_sdf_edge_anchor_errors(
    contact_count: wp.array[wp.int32],
    sort_keys: wp.array[wp.int64],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[wp.float32],
    margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_source: wp.array[wp.uint64],
    shape_data: wp.array[wp.vec4],
    shape_transform: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_data: wp.array[TextureSDFData],
    shape_edge_range: wp.array[wp.vec2i],
    mesh_edge_indices: wp.array[wp.vec2i],
    out_errors: wp.array[wp.float32],
    out_signed_errors: wp.array[wp.float32],
    out_separations: wp.array[wp.float32],
    out_sdf_endpoint_abs: wp.array[wp.float32],
    out_sdf_line_error: wp.array[wp.float32],
    out_sdf_min_signed_delta: wp.array[wp.float32],
    out_sdf_gen_delta_ratio: wp.array[wp.float32],
    out_sdf_gen_delta_relerr: wp.array[wp.float32],
    out_sdf_two_sided_delta_relerr: wp.array[wp.float32],
    out_count: wp.array[wp.int32],
):
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    sort_sub_key = int(sort_keys[tid] & wp.int64(0x7FFFFF))
    if (sort_sub_key & 1) != 0:
        return

    mode = (sort_sub_key >> 1) & 1
    edge_idx = sort_sub_key >> 2
    tri_shape = shape0[tid]
    anchor_local = point0[tid]
    if mode != 0:
        tri_shape = shape1[tid]
        anchor_local = point1[tid]

    edge_range = shape_edge_range[tri_shape]
    if edge_idx < 0 or edge_idx >= edge_range[1]:
        return

    mesh_id = shape_source[tri_shape]
    if mesh_id == wp.uint64(0):
        return

    edge = mesh_edge_indices[edge_range[0] + edge_idx]
    mesh = wp.mesh_get(mesh_id)
    scale_data = shape_data[tri_shape]
    scale = wp.vec3(scale_data[0], scale_data[1], scale_data[2])
    xform = shape_transform[tri_shape]
    edge_a = wp.transform_point(xform, wp.cw_mul(mesh.points[edge[0]], scale))
    edge_b = wp.transform_point(xform, wp.cw_mul(mesh.points[edge[1]], scale))

    body = shape_body[tri_shape]
    anchor_world = anchor_local
    if body >= 0:
        anchor_world = wp.transform_point(body_q[body], anchor_local)

    ab = edge_b - edge_a
    denom = wp.dot(ab, ab)
    closest = edge_a
    if denom > 0.0:
        t = wp.clamp(wp.dot(anchor_world - edge_a, ab) / denom, 0.0, 1.0)
        closest = edge_a + t * ab

    p0_world = point0[tid]
    p1_world = point1[tid]
    body0 = shape_body[shape0[tid]]
    body1 = shape_body[shape1[tid]]
    if body0 >= 0:
        p0_world = wp.transform_point(body_q[body0], point0[tid])
    if body1 >= 0:
        p1_world = wp.transform_point(body_q[body1], point1[tid])

    n = normal[tid]
    delta = anchor_world - closest
    signed_sep = wp.dot(p1_world - p0_world, n) - margin0[tid] - margin1[tid]

    tri_anchor_world = p0_world
    sdf_anchor_world = p1_world
    sdf_shape = shape1[tid]
    if mode != 0:
        tri_anchor_world = p1_world
        sdf_anchor_world = p0_world
        sdf_shape = shape0[tid]

    endpoint_abs = float(-1.0)
    line_error = float(-1.0)
    min_signed_delta = float(1.0e20)
    gen_delta_ratio = float(-1.0)
    gen_delta_relerr = float(-1.0)
    two_sided_delta_relerr = float(-1.0)
    segment_len = wp.length(tri_anchor_world - sdf_anchor_world)
    sdf_idx = shape_sdf_index[sdf_shape]
    if sdf_idx >= 0 and sdf_idx < texture_sdf_data.shape[0]:
        texture_sdf = texture_sdf_data[sdf_idx]
        if texture_sdf.coarse_texture.width > 0:
            sdf_scale_data = shape_data[sdf_shape]
            sdf_scale = wp.vec3(sdf_scale_data[0], sdf_scale_data[1], sdf_scale_data[2])
            if texture_sdf.scale_baked:
                sdf_scale = wp.vec3(1.0, 1.0, 1.0)

            sx = wp.where(wp.abs(sdf_scale[0]) > 1.0e-12, sdf_scale[0], 1.0)
            sy = wp.where(wp.abs(sdf_scale[1]) > 1.0e-12, sdf_scale[1], 1.0)
            sz = wp.where(wp.abs(sdf_scale[2]) > 1.0e-12, sdf_scale[2], 1.0)
            inv_sdf_scale = wp.vec3(1.0 / sx, 1.0 / sy, 1.0 / sz)
            min_sdf_scale = wp.min(wp.min(wp.abs(sx), wp.abs(sy)), wp.abs(sz))
            X_sw = wp.transform_inverse(shape_transform[sdf_shape])

            prev = float(0.0)
            sign_sep = float(1.0)
            if signed_sep < 0.0:
                sign_sep = -1.0

            for sample_idx in range(5):
                t = float(sample_idx) * 0.25
                p_world = sdf_anchor_world + (tri_anchor_world - sdf_anchor_world) * t
                p_sdf = wp.transform_point(X_sw, p_world)
                p_unscaled = wp.cw_mul(p_sdf, inv_sdf_scale)
                sdf_val = texture_sample_sdf(texture_sdf, p_unscaled) * min_sdf_scale
                expected = signed_sep * t
                err = wp.abs(sdf_val - expected)
                if sample_idx == 0:
                    endpoint_abs = wp.abs(sdf_val)
                    line_error = err
                else:
                    line_error = wp.max(line_error, err)
                    min_signed_delta = wp.min(min_signed_delta, (sdf_val - prev) * sign_sep)
                prev = sdf_val
    if segment_len > 0.0:
        sdf_at_sdf_anchor, valid_sdf_anchor = _sample_shape_sdf_world(
            sdf_shape,
            sdf_anchor_world,
            shape_data,
            shape_transform,
            shape_sdf_index,
            texture_sdf_data,
        )
        sdf_at_tri_anchor, valid_tri_anchor = _sample_shape_sdf_world(
            sdf_shape,
            tri_anchor_world,
            shape_data,
            shape_transform,
            shape_sdf_index,
            texture_sdf_data,
        )
        if valid_sdf_anchor and valid_tri_anchor:
            gen_delta_ratio = wp.abs(sdf_at_tri_anchor - sdf_at_sdf_anchor) / segment_len
            gen_delta_relerr = wp.abs(gen_delta_ratio - 1.0)

        sdf0_at_p0, valid00 = _sample_shape_sdf_world(
            shape0[tid],
            p0_world,
            shape_data,
            shape_transform,
            shape_sdf_index,
            texture_sdf_data,
        )
        sdf0_at_p1, valid01 = _sample_shape_sdf_world(
            shape0[tid],
            p1_world,
            shape_data,
            shape_transform,
            shape_sdf_index,
            texture_sdf_data,
        )
        sdf1_at_p0, valid10 = _sample_shape_sdf_world(
            shape1[tid],
            p0_world,
            shape_data,
            shape_transform,
            shape_sdf_index,
            texture_sdf_data,
        )
        sdf1_at_p1, valid11 = _sample_shape_sdf_world(
            shape1[tid],
            p1_world,
            shape_data,
            shape_transform,
            shape_sdf_index,
            texture_sdf_data,
        )
        if valid00 and valid01 and valid10 and valid11:
            shape0_delta_ratio = wp.abs(sdf0_at_p1 - sdf0_at_p0) / segment_len
            shape1_delta_ratio = wp.abs(sdf1_at_p1 - sdf1_at_p0) / segment_len
            two_sided_delta_relerr = wp.max(
                wp.abs(shape0_delta_ratio - 1.0),
                wp.abs(shape1_delta_ratio - 1.0),
            )

    out_idx = wp.atomic_add(out_count, 0, 1)
    out_errors[out_idx] = wp.length(delta)
    out_signed_errors[out_idx] = wp.dot(delta, n)
    out_separations[out_idx] = wp.abs(signed_sep)
    out_sdf_endpoint_abs[out_idx] = endpoint_abs
    out_sdf_line_error[out_idx] = line_error
    out_sdf_min_signed_delta[out_idx] = min_signed_delta
    out_sdf_gen_delta_ratio[out_idx] = gen_delta_ratio
    out_sdf_gen_delta_relerr[out_idx] = gen_delta_relerr
    out_sdf_two_sided_delta_relerr[out_idx] = two_sided_delta_relerr


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.solver_type = SOLVER_TYPE
        self.contact_gap = float(CONTACT_GAP)
        self.contact_margin = float(CONTACT_MARGIN)
        self.contact_matching = CONTACT_MATCHING
        self.rigid_contact_max = RIGID_CONTACT_MAX
        self.reduce_contacts = bool(REDUCE_CONTACTS)
        self.sdf_linearity_bad_relerr = float(SDF_LINEARITY_BAD_RELERR)
        self.contact_stats = bool(CONTACT_STATS)
        self.contact_stats_interval = int(CONTACT_STATS_INTERVAL)
        self.contact_stats_csv = CONTACT_STATS_CSV
        self.contact_dump_frames = {int(frame) for frame in CONTACT_DUMP_FRAMES}
        self.contact_dump_top_n = int(CONTACT_DUMP_TOP_N)
        self.trajectory_csv = TRAJECTORY_CSV
        self.trajectory_interval = int(TRAJECTORY_INTERVAL)
        self._contact_stats_csv_header_written = False
        self._trajectory_csv_header_written = False
        self._frame = 0
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SUBSTEPS
        self.sim_time = 0.0
        self._printed_contact_count = False

        usd_path = _resolve_maxwell_top_usd_path(args.usd_path)
        if not usd_path.exists():
            raise FileNotFoundError(
                "example_maxwell_top_si requires the MaxwellTopSI2.usda asset. "
                f"Pass --usd-path, set {MAXWELL_TOP_USD_ENV}, or place it in your Downloads folder."
            )
        sdf_cache_dir = Path(args.sdf_cache_dir).expanduser() if args.sdf_cache_dir else usd_path.parent / ".sdf_cache"

        builder = newton.ModelBuilder()
        builder.sdf_cache_dir = sdf_cache_dir
        result = builder.add_usd(
            str(usd_path),
            schema_resolvers=[
                newton.usd.SchemaResolverNewton(),
                newton.usd.SchemaResolverPhysx(),
            ],
        )
        builder.shape_gap[:] = [self.contact_gap] * len(builder.shape_gap)
        builder.shape_margin[:] = [self.contact_margin] * len(builder.shape_margin)
        builder.shape_collision_filter_pairs.clear()
        print(
            "Loaded MaxwellTopSI2.usda: "
            f"{len(result['path_body_map'])} bodies, "
            f"{len(result['path_joint_map'])} joints, "
            f"{len(result['path_shape_map'])} shapes"
        )
        print(f"SDF texture cache: {sdf_cache_dir}")

        self.model = builder.finalize(skip_validation_joints=True)
        print(
            "Collision setup: "
            f"{self.model.shape_contact_pair_count} contact pairs, "
            f"SDF indices {self.model._shape_sdf_index.numpy().tolist()}, "
            f"gap={self.contact_gap:g} m, margin={self.contact_margin:g} m"
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.top_body_id = result["path_body_map"][TOP_BODY_PATH]
        self._path_body_map = result["path_body_map"]
        self._set_top_angular_velocity(result["path_body_map"])
        self._trajectory_points = self._build_trajectory_points(result["path_body_map"])
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=self.reduce_contacts,
            rigid_contact_max=self.rigid_contact_max,
            contact_matching=self.contact_matching,
        )
        print(
            "Contact pipeline: "
            f"reduce_contacts={self.reduce_contacts}, "
            f"matching={self.contact_matching}, "
            f"sdf_linearity_bad_relerr={self.sdf_linearity_bad_relerr:g}"
        )
        self.contacts = self.collision_pipeline.contacts()
        if self.contact_stats:
            self._edge_anchor_errors = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.float32)
            self._edge_anchor_signed_errors = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.float32)
            self._edge_anchor_separations = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.float32)
            self._sdf_endpoint_abs = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.float32)
            self._sdf_line_error = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.float32)
            self._sdf_min_signed_delta = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.float32)
            self._sdf_gen_delta_ratio = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.float32)
            self._sdf_gen_delta_relerr = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.float32)
            self._sdf_two_sided_delta_relerr = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.float32)
            self._edge_anchor_count = wp.zeros(1, dtype=wp.int32)

        if self.solver_type == "phoenx":
            self.solver = newton.solvers.SolverPhoenX(
                self.model,
                substeps=SUBSTEPS,
                solver_iterations=SOLVER_ITERATIONS,
            )
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=SOLVER_ITERATIONS)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
        print(f"Using {self.solver_type} solver")

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.0, -3.0, 2.0), pitch=-25.0, yaw=135.0)
        self.capture()

    @staticmethod
    def _apply_body_angular_velocity(
        state: newton.State,
        model: newton.Model,
        path_body_map: dict[str, int],
        body_path: str,
        angular_velocity_deg_s: tuple[float, float, float],
    ) -> None:
        """Set world-frame angular velocity [rad/s] on ``state.body_qd``."""
        body_idx = path_body_map[body_path]
        angular_velocity = wp.vec3(
            math.radians(angular_velocity_deg_s[0]),
            math.radians(angular_velocity_deg_s[1]),
            math.radians(angular_velocity_deg_s[2]),
        )
        body_qd_np = state.body_qd.numpy()
        body_qd_np[body_idx, 3:6] = [angular_velocity[0], angular_velocity[1], angular_velocity[2]]
        state.body_qd.assign(body_qd_np)

        model_body_qd_np = model.body_qd.numpy()
        model_body_qd_np[body_idx, 3:6] = body_qd_np[body_idx, 3:6]
        model.body_qd.assign(model_body_qd_np)
        print(f"Set {body_path} angular velocity to {angular_velocity_deg_s} deg/s")

    def _set_top_angular_velocity(self, path_body_map: dict[str, int]) -> None:
        self._apply_body_angular_velocity(
            self.state_0,
            self.model,
            path_body_map,
            TOP_BODY_PATH,
            TOP_ANGULAR_VELOCITY_DEG_S,
        )

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_vec = q[:3]
        w = q[3]
        return v + 2.0 * w * np.cross(q_vec, v) + 2.0 * np.cross(q_vec, np.cross(q_vec, v))

    @staticmethod
    def _body_point_world(body_q: np.ndarray, body_idx: int, point: np.ndarray) -> np.ndarray:
        if body_idx < 0:
            return point
        return body_q[body_idx, :3] + Example._quat_rotate(body_q[body_idx, 3:], point)

    @staticmethod
    def _body_point_velocity(
        body_q: np.ndarray, body_qd: np.ndarray, body_idx: int, point_world: np.ndarray
    ) -> np.ndarray:
        if body_idx < 0:
            return np.zeros(3, dtype=np.float64)
        qd = body_qd[body_idx]
        return qd[:3] + np.cross(qd[3:], point_world - body_q[body_idx, :3])

    @staticmethod
    def _transform_point_np(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        return transform[:3] + Example._quat_rotate(transform[3:], point)

    def _build_trajectory_points(self, path_body_map: dict[str, int]) -> list[tuple[str, int, np.ndarray]]:
        body_q = self.state_0.body_q.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_scale = self.model.shape_scale.numpy()
        shape_transform = self.model.shape_transform.numpy()
        points: list[tuple[str, int, np.ndarray]] = []

        for body_path in TRAJECTORY_BODY_PATHS:
            body_id = path_body_map.get(body_path)
            if body_id is None:
                continue

            best_body_point = None
            best_world_z = float("inf")
            for shape_id, shape_body_id in enumerate(shape_body):
                if int(shape_body_id) != body_id:
                    continue

                shape_source = self.model.shape_source[shape_id]
                vertices = getattr(shape_source, "vertices", None)
                if vertices is None:
                    continue

                scaled_vertices = np.asarray(vertices, dtype=np.float64) * np.asarray(
                    shape_scale[shape_id], dtype=np.float64
                )
                for vertex in scaled_vertices:
                    body_point = self._transform_point_np(shape_transform[shape_id], vertex)
                    world_point = self._body_point_world(body_q, body_id, body_point)
                    if world_point[2] < best_world_z:
                        best_world_z = float(world_point[2])
                        best_body_point = body_point.astype(np.float64)

            if best_body_point is not None:
                points.append((body_path, body_id, best_body_point))
                print(
                    "Tracking trajectory point: "
                    f"{body_path} body={body_id} local={best_body_point.tolist()} initial_z={best_world_z:.6g}"
                )

        return points

    def _write_trajectory_csv(self) -> None:
        if not self.trajectory_csv or self._frame % max(1, self.trajectory_interval) != 0:
            return

        path = Path(self.trajectory_csv)
        write_header = not self._trajectory_csv_header_written and not path.exists()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAJECTORY_FIELDS)
            if write_header:
                writer.writeheader()
            for body_path, body_id, local_point in self._trajectory_points:
                center = body_q[body_id, :3]
                tip = self._body_point_world(body_q, body_id, local_point)
                qd = body_qd[body_id]
                writer.writerow(
                    {
                        "frame": self._frame,
                        "time": self.sim_time,
                        "body_path": body_path,
                        "body_id": body_id,
                        "center_x": float(center[0]),
                        "center_y": float(center[1]),
                        "center_z": float(center[2]),
                        "tip_x": float(tip[0]),
                        "tip_y": float(tip[1]),
                        "tip_z": float(tip[2]),
                        "linear_speed": float(np.linalg.norm(qd[:3])),
                        "angular_speed": float(np.linalg.norm(qd[3:])),
                        "contact_count": contact_count,
                    }
                )
        self._trajectory_csv_header_written = True

    def _edge_anchor_metrics(
        self,
        count: int,
    ) -> dict[str, float | int]:
        if not self.collision_pipeline.deterministic:
            return {"edge_contacts": 0}

        shape_edge_range = getattr(self.model, "shape_edge_range", None)
        mesh_edge_indices = getattr(self.model, "mesh_edge_indices", None)
        contact_sorter = getattr(self.collision_pipeline, "_contact_sorter", None)
        if shape_edge_range is None or mesh_edge_indices is None or contact_sorter is None:
            return {"edge_contacts": 0}

        self._edge_anchor_count.zero_()
        wp.launch(
            _measure_sdf_edge_anchor_errors,
            dim=count,
            inputs=[
                self.contacts.rigid_contact_count,
                contact_sorter.sorted_keys_view,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.contacts.rigid_contact_point0,
                self.contacts.rigid_contact_point1,
                self.contacts.rigid_contact_normal,
                self.contacts.rigid_contact_margin0,
                self.contacts.rigid_contact_margin1,
                self.state_0.body_q,
                self.model.shape_body,
                self.model.shape_source_ptr,
                self.collision_pipeline.geom_data,
                self.collision_pipeline.geom_transform,
                self.model._shape_sdf_index,
                self.model._texture_sdf_data,
                shape_edge_range,
                mesh_edge_indices,
                self._edge_anchor_errors,
                self._edge_anchor_signed_errors,
                self._edge_anchor_separations,
                self._sdf_endpoint_abs,
                self._sdf_line_error,
                self._sdf_min_signed_delta,
                self._sdf_gen_delta_ratio,
                self._sdf_gen_delta_relerr,
                self._sdf_two_sided_delta_relerr,
                self._edge_anchor_count,
            ],
            device=self.device,
        )

        edge_count = int(self._edge_anchor_count.numpy()[0])
        anchor_errors = self._edge_anchor_errors.numpy()[:edge_count].astype(np.float32)
        signed_errors = self._edge_anchor_signed_errors.numpy()[:edge_count].astype(np.float32)
        sep_abs = self._edge_anchor_separations.numpy()[:edge_count].astype(np.float32)
        sdf_endpoint_abs = self._sdf_endpoint_abs.numpy()[:edge_count].astype(np.float32)
        sdf_line_error = self._sdf_line_error.numpy()[:edge_count].astype(np.float32)
        sdf_min_signed_delta = self._sdf_min_signed_delta.numpy()[:edge_count].astype(np.float32)
        sdf_gen_delta_ratio = self._sdf_gen_delta_ratio.numpy()[:edge_count].astype(np.float32)
        sdf_gen_delta_relerr = self._sdf_gen_delta_relerr.numpy()[:edge_count].astype(np.float32)
        sdf_two_sided_delta_relerr = self._sdf_two_sided_delta_relerr.numpy()[:edge_count].astype(np.float32)
        valid_ratio = sep_abs > 1.0e-8
        error_to_sep = anchor_errors[valid_ratio] / sep_abs[valid_ratio]
        valid_sdf = sdf_endpoint_abs >= 0.0
        valid_gen_delta = sdf_gen_delta_relerr >= 0.0
        valid_two_sided_delta = sdf_two_sided_delta_relerr >= 0.0

        return {
            "edge_contacts": edge_count,
            "edge_anchor_p50": float(np.percentile(anchor_errors, 50.0)) if edge_count else float("nan"),
            "edge_anchor_p95": float(np.percentile(anchor_errors, 95.0)) if edge_count else float("nan"),
            "edge_anchor_max": float(np.max(anchor_errors)) if edge_count else float("nan"),
            "edge_anchor_signed_mean": float(np.mean(signed_errors)) if edge_count else float("nan"),
            "edge_anchor_over_sep_p50": float(np.percentile(error_to_sep, 50.0)) if len(error_to_sep) else float("nan"),
            "sdf_endpoint_abs_p50": float(np.percentile(sdf_endpoint_abs[valid_sdf], 50.0))
            if np.any(valid_sdf)
            else float("nan"),
            "sdf_endpoint_abs_p95": float(np.percentile(sdf_endpoint_abs[valid_sdf], 95.0))
            if np.any(valid_sdf)
            else float("nan"),
            "sdf_line_error_p50": float(np.percentile(sdf_line_error[valid_sdf], 50.0))
            if np.any(valid_sdf)
            else float("nan"),
            "sdf_line_error_p95": float(np.percentile(sdf_line_error[valid_sdf], 95.0))
            if np.any(valid_sdf)
            else float("nan"),
            "sdf_min_signed_delta_p05": float(np.percentile(sdf_min_signed_delta[valid_sdf], 5.0))
            if np.any(valid_sdf)
            else float("nan"),
            "sdf_nonmonotone_frac": float(
                np.count_nonzero(sdf_min_signed_delta[valid_sdf] < 0.0) / np.count_nonzero(valid_sdf)
            )
            if np.any(valid_sdf)
            else float("nan"),
            "sdf_gen_delta_ratio_p50": float(np.percentile(sdf_gen_delta_ratio[valid_gen_delta], 50.0))
            if np.any(valid_gen_delta)
            else float("nan"),
            "sdf_gen_delta_ratio_p95": float(np.percentile(sdf_gen_delta_ratio[valid_gen_delta], 95.0))
            if np.any(valid_gen_delta)
            else float("nan"),
            "sdf_gen_delta_relerr_p50": float(np.percentile(sdf_gen_delta_relerr[valid_gen_delta], 50.0))
            if np.any(valid_gen_delta)
            else float("nan"),
            "sdf_gen_delta_relerr_p95": float(np.percentile(sdf_gen_delta_relerr[valid_gen_delta], 95.0))
            if np.any(valid_gen_delta)
            else float("nan"),
            "sdf_gen_bad_frac": float(
                np.count_nonzero(sdf_gen_delta_relerr[valid_gen_delta] > self.sdf_linearity_bad_relerr)
                / np.count_nonzero(valid_gen_delta)
            )
            if np.any(valid_gen_delta)
            else float("nan"),
            "sdf_two_sided_delta_relerr_p50": float(
                np.percentile(sdf_two_sided_delta_relerr[valid_two_sided_delta], 50.0)
            )
            if np.any(valid_two_sided_delta)
            else float("nan"),
            "sdf_two_sided_delta_relerr_p95": float(
                np.percentile(sdf_two_sided_delta_relerr[valid_two_sided_delta], 95.0)
            )
            if np.any(valid_two_sided_delta)
            else float("nan"),
            "sdf_two_sided_bad_frac": float(
                np.count_nonzero(sdf_two_sided_delta_relerr[valid_two_sided_delta] > self.sdf_linearity_bad_relerr)
                / np.count_nonzero(valid_two_sided_delta)
            )
            if np.any(valid_two_sided_delta)
            else float("nan"),
        }

    def _write_contact_stats_csv(self, metrics: dict[str, float | int]) -> None:
        if not self.contact_stats_csv:
            return

        path = Path(self.contact_stats_csv)
        write_header = not self._contact_stats_csv_header_written and not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CONTACT_STATS_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
        self._contact_stats_csv_header_written = True

    def _reducer_metrics(self) -> dict[str, float | int]:
        reducer = getattr(getattr(self.collision_pipeline, "narrow_phase", None), "global_contact_reducer", None)
        if reducer is None:
            return {
                "reducer_candidates": 0,
                "reducer_capacity": 0,
                "reducer_ht_active": 0,
                "reducer_ht_capacity": 0,
                "reducer_ht_insert_failures": 0,
            }

        ht_active_slots = reducer.hashtable.active_slots.numpy()
        return {
            "reducer_candidates": int(reducer.contact_count.numpy()[0]),
            "reducer_capacity": int(reducer.capacity),
            "reducer_ht_active": int(ht_active_slots[reducer.hashtable.capacity]),
            "reducer_ht_capacity": int(reducer.hashtable.capacity),
            "reducer_ht_insert_failures": int(reducer.ht_insert_failures.numpy()[0]),
        }

    def _print_contact_dump(
        self,
        count: int,
        shape0: np.ndarray,
        shape1: np.ndarray,
        separations: np.ndarray,
        jvn: np.ndarray,
        jvn_plus_bias: np.ndarray,
    ) -> None:
        if self._frame not in self.contact_dump_frames:
            return

        sorter = getattr(self.collision_pipeline, "_contact_sorter", None)
        sort_keys = None
        if sorter is not None and getattr(sorter, "sorted_keys_view", None) is not None:
            sort_keys = sorter.sorted_keys_view.numpy()[:count]
        match_index = None
        if self.contacts.rigid_contact_match_index is not None:
            match_index = self.contacts.rigid_contact_match_index.numpy()[:count]

        order = np.argsort(jvn)[: max(1, min(self.contact_dump_top_n, count))]
        print(f"[contact-dump frame={self._frame}] worst_jvn_rows={len(order)}")
        active_order = np.where(separations <= 0.0)[0]
        active_order = active_order[np.argsort(separations[active_order])]
        active_order = active_order[: max(1, min(self.contact_dump_top_n, len(active_order)))]

        def print_row(prefix: str, idx: int) -> None:
            sort_sub_key = -1
            edge_idx = -1
            mode = -1
            if sort_keys is not None:
                sort_sub_key = int(sort_keys[idx] & 0x7FFFFF)
                edge_idx = sort_sub_key >> 2
                mode = (sort_sub_key >> 1) & 1
            matched = -999
            if match_index is not None:
                matched = int(match_index[idx])
            print(
                f"  {prefix} "
                f"i={int(idx)} shapes=({int(shape0[idx])},{int(shape1[idx])}) "
                f"sep={float(separations[idx]):.9g} jvn={float(jvn[idx]):.9g} "
                f"jvn_bias={float(jvn_plus_bias[idx]):.9g} match={matched} "
                f"sort_sub={sort_sub_key} edge={edge_idx} mode={mode}"
            )

        for idx in order:
            print_row("jvn", int(idx))
        print(f"[contact-dump frame={self._frame}] active_rows={len(active_order)}")
        for idx in active_order:
            print_row("act", int(idx))

    def _print_contact_stats(self) -> None:
        if not self.contact_stats or self._frame % max(1, self.contact_stats_interval) != 0:
            return

        reported = int(self.contacts.rigid_contact_count.numpy()[0])
        count = min(reported, int(self.contacts.rigid_contact_max))
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        top_q = body_q[self.top_body_id]
        top_qd = body_qd[self.top_body_id]
        if count == 0:
            reducer_metrics = self._reducer_metrics()
            metrics = {
                "frame": self._frame,
                "contacts": 0,
                "zero_gap": 0,
                "speculative": 0,
                "min_sep": float("nan"),
                "median_sep": float("nan"),
                "max_sep": float("nan"),
                "top_z": float(top_q[2]),
                "top_speed": float(np.linalg.norm(top_qd[:3])),
                "top_ang_speed": float(np.linalg.norm(top_qd[3:])),
                "speculative_closing": 0,
                "speculative_closing_frac": float("nan"),
                "jvn_min": float("nan"),
                "jvn_p05": float("nan"),
                "jvn_median": float("nan"),
                "jvn_plus_bias_min": float("nan"),
                "jvn_plus_bias_p05": float("nan"),
                **reducer_metrics,
                "edge_contacts": 0,
                "edge_anchor_p50": float("nan"),
                "edge_anchor_p95": float("nan"),
                "edge_anchor_max": float("nan"),
                "edge_anchor_signed_mean": float("nan"),
                "edge_anchor_over_sep_p50": float("nan"),
                "sdf_endpoint_abs_p50": float("nan"),
                "sdf_endpoint_abs_p95": float("nan"),
                "sdf_line_error_p50": float("nan"),
                "sdf_line_error_p95": float("nan"),
                "sdf_min_signed_delta_p05": float("nan"),
                "sdf_nonmonotone_frac": float("nan"),
                "sdf_gen_delta_ratio_p50": float("nan"),
                "sdf_gen_delta_ratio_p95": float("nan"),
                "sdf_gen_delta_relerr_p50": float("nan"),
                "sdf_gen_delta_relerr_p95": float("nan"),
                "sdf_gen_bad_frac": float("nan"),
                "sdf_two_sided_delta_relerr_p50": float("nan"),
                "sdf_two_sided_delta_relerr_p95": float("nan"),
                "sdf_two_sided_bad_frac": float("nan"),
            }
            self._write_contact_stats_csv(metrics)
            print(
                f"[contact-stats frame={self._frame}] contacts=0 "
                f"top_z={metrics['top_z']:.6g} top_speed={metrics['top_speed']:.6g} "
                f"top_ang_speed={metrics['top_ang_speed']:.6g} "
                f"reducer_candidates={reducer_metrics['reducer_candidates']} "
                f"reducer_ht={reducer_metrics['reducer_ht_active']}/{reducer_metrics['reducer_ht_capacity']} "
                f"reducer_ht_fail={reducer_metrics['reducer_ht_insert_failures']}"
            )
            return

        shape_body = self.model.shape_body.numpy()
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        point0 = self.contacts.rigid_contact_point0.numpy()[:count]
        point1 = self.contacts.rigid_contact_point1.numpy()[:count]
        normal = self.contacts.rigid_contact_normal.numpy()[:count]
        margin0 = self.contacts.rigid_contact_margin0.numpy()[:count]
        margin1 = self.contacts.rigid_contact_margin1.numpy()[:count]

        separations = np.empty(count, dtype=np.float32)
        jvn = np.empty(count, dtype=np.float32)
        jvn_plus_bias = np.empty(count, dtype=np.float32)
        substep_idt = float(FPS * SUBSTEPS)
        for i in range(count):
            body0 = int(shape_body[int(shape0[i])])
            body1 = int(shape_body[int(shape1[i])])
            p0_world = self._body_point_world(body_q, body0, point0[i])
            p1_world = self._body_point_world(body_q, body1, point1[i])
            v0_world = self._body_point_velocity(body_q, body_qd, body0, p0_world)
            v1_world = self._body_point_velocity(body_q, body_qd, body1, p1_world)
            separations[i] = float(np.dot(p1_world - p0_world, normal[i]) - margin0[i] - margin1[i])
            jvn[i] = float(np.dot(v1_world - v0_world, normal[i]))
            jvn_plus_bias[i] = jvn[i] + max(float(separations[i]), 0.0) * substep_idt

        edge_metrics = self._edge_anchor_metrics(count)
        reducer_metrics = self._reducer_metrics()
        zero_gap_count = int(np.count_nonzero(separations <= 0.0))
        speculative = separations > 0.0
        speculative_closing = int(np.count_nonzero(jvn_plus_bias[speculative] < 0.0))
        speculative_count = int(np.count_nonzero(speculative))
        metrics = {
            "frame": self._frame,
            "contacts": count,
            "zero_gap": zero_gap_count,
            "speculative": speculative_count,
            "min_sep": float(separations.min()),
            "median_sep": float(np.median(separations)),
            "max_sep": float(separations.max()),
            "top_z": float(top_q[2]),
            "top_speed": float(np.linalg.norm(top_qd[:3])),
            "top_ang_speed": float(np.linalg.norm(top_qd[3:])),
            "speculative_closing": speculative_closing,
            "speculative_closing_frac": float(speculative_closing / speculative_count) if speculative_count else 0.0,
            "jvn_min": float(jvn.min()),
            "jvn_p05": float(np.percentile(jvn, 5.0)),
            "jvn_median": float(np.median(jvn)),
            "jvn_plus_bias_min": float(jvn_plus_bias.min()),
            "jvn_plus_bias_p05": float(np.percentile(jvn_plus_bias, 5.0)),
            **reducer_metrics,
            **edge_metrics,
        }
        self._write_contact_stats_csv(metrics)
        print(
            f"[contact-stats frame={self._frame}] contacts={count} "
            f"zero_gap={zero_gap_count} speculative={speculative_count} min_sep={separations.min():.6g} "
            f"median_sep={np.median(separations):.6g} max_sep={separations.max():.6g} "
            f"top_z={metrics['top_z']:.6g} top_ang_speed={metrics['top_ang_speed']:.6g} "
            f"spec_closing={speculative_closing}/{speculative_count} "
            f"jvn_min={metrics['jvn_min']:.6g} jvn_bias_min={metrics['jvn_plus_bias_min']:.6g} "
            f"reducer_candidates={reducer_metrics['reducer_candidates']} "
            f"reducer_ht={reducer_metrics['reducer_ht_active']}/{reducer_metrics['reducer_ht_capacity']} "
            f"reducer_ht_fail={reducer_metrics['reducer_ht_insert_failures']} "
            f"edge_contacts={edge_metrics.get('edge_contacts', 0)} "
            f"edge_anchor_p50={edge_metrics.get('edge_anchor_p50', float('nan')):.6g} "
            f"edge_anchor_over_sep_p50={edge_metrics.get('edge_anchor_over_sep_p50', float('nan')):.6g} "
            f"sdf_endpoint_abs_p50={edge_metrics.get('sdf_endpoint_abs_p50', float('nan')):.6g} "
            f"sdf_line_error_p50={edge_metrics.get('sdf_line_error_p50', float('nan')):.6g} "
            f"sdf_min_signed_delta_p05={edge_metrics.get('sdf_min_signed_delta_p05', float('nan')):.6g} "
            f"sdf_nonmonotone_frac={edge_metrics.get('sdf_nonmonotone_frac', float('nan')):.6g} "
            f"sdf_gen_relerr_p50={edge_metrics.get('sdf_gen_delta_relerr_p50', float('nan')):.6g} "
            f"sdf_gen_bad_frac={edge_metrics.get('sdf_gen_bad_frac', float('nan')):.6g} "
            f"sdf_two_sided_bad_frac={edge_metrics.get('sdf_two_sided_bad_frac', float('nan')):.6g}"
        )
        self._print_contact_dump(count, shape0, shape1, separations, jvn, jvn_plus_bias)

    def capture(self):
        if self.contact_stats:
            self.graph = None
            print("Contact stats enabled: running without CUDA graph capture")
        elif self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.model.collide(
            self.state_0,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self._print_contact_stats()
        if self.solver_type == "phoenx":
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
            self.state_0.assign(self.state_1)
        else:
            for _ in range(SUBSTEPS):
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0.assign(self.state_1)
        self._write_trajectory_csv()

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        if not self._printed_contact_count:
            print(f"Initial rigid contacts: {int(self.contacts.rigid_contact_count.numpy()[0])}")
            self._printed_contact_count = True
        self.sim_time += self.frame_dt
        self._frame += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--usd-path",
            type=str,
            default=None,
            help=(
                "Path to MaxwellTopSI2.usda. Defaults to NEWTON_MAXWELL_TOP_USD, "
                "then searches the Linux Downloads folder, or falls back to "
                f"{DEFAULT_WINDOWS_MAXWELL_TOP_USD_PATH} on other platforms."
            ),
        )
        parser.add_argument(
            "--sdf-cache-dir",
            type=str,
            default=None,
            help="Optional SDF cache directory. Defaults to a .sdf_cache sibling of --usd-path.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
