# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""OptiX reference plate - cube pyramid physics demo with studio lighting."""

from __future__ import annotations

import math
import random

import numpy as np
import warp as wp

import newton
import newton.examples

# Toggle between procedural sky (single sun) and studio env map (multi-light).
USE_STUDIO_ENV_MAP = True

# Print the full camera pose to stdout whenever it changes (useful for finding
# the best initial camera position interactively).
PRINT_CAMERA_POSE = False

# Global multiplier for focused spot light intensities (shadow strength).
SPOT_INTENSITY = 1.0

# Pyramid dimensions.
PYRAMID_LEVELS = 15
PLATE_HALF = 1.30

# Fraction of pyramid cubes that glow (emissive).
EMISSIVE_FRACTION = 0.20
EMISSIVE_INTENSITY = 8


def _generate_studio_env_map(width: int = 512, height: int = 256) -> np.ndarray:
    """Generate a lat-long HDR environment map with multi-light studio setup.

    The map contains a smooth sky gradient (white horizon fading to light
    blue-gray at zenith) plus several broad Gaussian "area lights" at
    different positions and tints to produce soft, multi-directional shadows.

    Returns:
        Float32 array of shape ``(height, width, 3)`` in linear HDR.
    """
    env = np.zeros((height, width, 3), dtype=np.float32)

    v_angles = np.linspace(0, math.pi, height, dtype=np.float32)
    u_angles = np.linspace(0, 2.0 * math.pi, width, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u_angles, v_angles)

    dir_x = np.sin(vv) * np.sin(uu)
    dir_y = np.cos(vv)
    dir_z = np.sin(vv) * np.cos(uu)

    horizon_weight = np.exp(-5.0 * dir_y**2)

    sky_zenith = np.array([0.30, 0.35, 0.55], dtype=np.float32)
    sky_horizon_warm = np.array([1.10, 0.80, 0.45], dtype=np.float32)
    sky_horizon_cool = np.array([0.55, 0.60, 0.80], dtype=np.float32)

    key_az = math.radians(-40.0)
    key_lx = math.sin(key_az)
    key_lz = math.cos(key_az)
    horiz_x = np.sin(uu)
    horiz_z = np.cos(uu)
    az_dot = np.clip(horiz_x * key_lx + horiz_z * key_lz, 0.0, 1.0)
    warm_frac = az_dot[..., None]

    sky_horizon_blend = sky_horizon_cool + (sky_horizon_warm - sky_horizon_cool) * warm_frac
    ground_color = np.array([0.40, 0.38, 0.36], dtype=np.float32)

    above = (dir_y >= 0).astype(np.float32)
    base_sky = (
        above[..., None] * (sky_zenith + (sky_horizon_blend - sky_zenith) * horizon_weight[..., None])
        + (1.0 - above[..., None]) * (ground_color + (sky_horizon_blend - ground_color) * horizon_weight[..., None])
    )
    env += base_sky

    # (elevation_deg, azimuth_deg, angular_sigma_deg, intensity, r, g, b)
    lights = [
        # Focused spots (cast hard shadows).
        (30.0, -40.0, 1.5, 300.0, 1.00, 0.95, 0.85),
        (25.0, 55.0, 1.5, 200.0, 0.85, 0.92, 1.00),
        (10.0, -65.0, 1.2, 150.0, 1.00, 0.88, 0.70),
        (18.0, 165.0, 1.5, 150.0, 0.75, 0.80, 1.00),
        # Broad washes (colored ambient).
        (45.0, -30.0, 35.0, 1.0, 1.00, 0.85, 0.55),
        (40.0, 60.0, 35.0, 0.8, 0.50, 0.65, 1.00),
        (40.0, 170.0, 30.0, 0.7, 0.70, 0.55, 1.00),
        (-20.0, 0.0, 40.0, 0.3, 1.00, 0.80, 0.45),
    ]

    for elev_deg, az_deg, sigma_deg, base_intensity, lr, lg, lb in lights:
        intensity = base_intensity * SPOT_INTENSITY if sigma_deg < 10.0 else base_intensity
        elev = math.radians(elev_deg)
        az = math.radians(az_deg)
        sigma = math.radians(sigma_deg)

        lx = math.cos(elev) * math.sin(az)
        ly = math.sin(elev)
        lz = math.cos(elev) * math.cos(az)

        dot = np.clip(dir_x * lx + dir_y * ly + dir_z * lz, -1.0, 1.0)
        ang_dist = np.arccos(dot)

        weight = np.exp(-0.5 * (ang_dist / sigma) ** 2)
        env[..., 0] += weight * intensity * lr
        env[..., 1] += weight * intensity * lg
        env[..., 2] += weight * intensity * lb

    return env


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        if hasattr(self.viewer, "_cam_speed"):
            self.viewer._cam_speed = 3.0

        # --- OptiX scene setup (materials, env map, static geometry) ---
        use_optix = args.viewer == "optix"
        if use_optix:
            if not hasattr(self.viewer, "_ensure_api"):
                raise RuntimeError("Viewer does not expose OptiX API hooks.")
            self.viewer._ensure_api()
            api = getattr(self.viewer, "_api", None)
            if api is None:
                raise RuntimeError("ViewerOptix PathTracerAPI was not created.")

            if USE_STUDIO_ENV_MAP:
                self._setup_studio_env_map(api)
            else:
                api.set_use_procedural_sky(True)
                api.set_sky_parameters(
                    sun_direction=(-0.3, 0.7, 0.5),
                    multiplier=0.6,
                    haze=0.25,
                    red_blue_shift=0.06,
                    saturation=0.7,
                    horizon_height=0.0,
                    ground_color=(0.72, 0.72, 0.74),
                    horizon_blur=0.8,
                    night_color=(0.0, 0.0, 0.0),
                    sun_disk_intensity=0.35,
                    sun_disk_scale=1.2,
                    sun_glow_intensity=0.5,
                    y_is_up=1,
                )

        # --- Physics model ---
        builder = newton.ModelBuilder()

        # Static presenter plate (two layers, Z-up) - top surface at z = 0.
        plate_half = PLATE_HALF
        plate_bottom_thickness = 0.05
        plate_top_thickness = 0.02
        plate_base_z = -(plate_bottom_thickness + plate_top_thickness)

        # Large visible ground floor whose top face meets the plate bottom.
        ground_thickness = 0.02
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, plate_base_z - ground_thickness * 0.5),
                q=wp.quat_identity(),
            ),
            hx=20.0,
            hy=20.0,
            hz=ground_thickness * 0.5,
        )

        builder.add_shape_box(
            -1,
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, plate_base_z + plate_bottom_thickness * 0.5),
                q=wp.quat_identity(),
            ),
            hx=plate_half,
            hy=plate_half,
            hz=plate_bottom_thickness * 0.5,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, plate_base_z + plate_bottom_thickness + plate_top_thickness * 0.5),
                q=wp.quat_identity(),
            ),
            hx=plate_half,
            hy=plate_half,
            hz=plate_top_thickness * 0.5,
        )

        # --- Pyramid of cubes (Z-up) ---
        # Size cubes so the base row (PYRAMID_LEVELS cubes) fits on the plate
        # with a small margin.
        usable_width = 2.0 * plate_half * 0.92
        cube_spacing = usable_width / PYRAMID_LEVELS
        cube_half = cube_spacing * 0.47

        drop_height = 0.005

        self._pyramid_body_start = builder.body_count
        for level in range(PYRAMID_LEVELS):
            count = PYRAMID_LEVELS - level
            z0 = drop_height + level * (2.0 * cube_half)
            x_start = -0.5 * (count - 1) * cube_spacing
            for i in range(count):
                x = x_start + i * cube_spacing
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x, 0.0, z0 + cube_half), q=wp.quat_identity()),
                )
                builder.add_shape_box(body, hx=cube_half, hy=cube_half, hz=cube_half)
        self._pyramid_body_end = builder.body_count

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        self._use_optix = use_optix
        if use_optix:
            self._apply_optix_materials(api)
            self._setup_camera()
        self._emissive_applied = False

        self.capture()

    # ------------------------------------------------------------------ #
    # OptiX helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _setup_studio_env_map(api):
        """Load a procedurally generated studio env map into the path tracer."""
        from newton._src.viewer.optix.pathtracing.environment_map import EnvironmentMap  # noqa: PLC0415

        env_array = _generate_studio_env_map(width=512, height=256)
        env_map = EnvironmentMap()
        env_map.load_from_array(env_array)
        api._viewer._env_map = env_map

    def _apply_optix_materials(self, api):
        """Override shape colours via the OptiX viewer."""
        if not hasattr(self.viewer, "update_shape_colors"):
            return

        palette = [
            [0.85, 0.12, 0.08],
            [0.92, 0.52, 0.08],
            [0.92, 0.82, 0.15],
            [0.22, 0.72, 0.22],
            [0.12, 0.68, 0.68],
            [0.14, 0.38, 0.88],
        ]

        # Randomly pick ~10% of cube shapes to be emissive.
        cube_start = 3
        cube_indices = list(range(cube_start, self.model.shape_count))
        num_emissive = max(1, int(len(cube_indices) * EMISSIVE_FRACTION))
        rng = random.Random(42)
        self._emissive_shapes = set(rng.sample(cube_indices, num_emissive))
        self._emissive_colors = {}

        shape_colors = {}
        for s in range(self.model.shape_count):
            if s == 0:
                shape_colors[s] = [0.86, 0.86, 0.86]  # ground floor
            elif s == 1:
                shape_colors[s] = [0.36, 0.37, 0.40]  # bottom plate
            elif s == 2:
                shape_colors[s] = [0.12, 0.11, 0.11]  # top plate
            else:
                color = palette[(s - cube_start) % len(palette)]
                shape_colors[s] = color
                if s in self._emissive_shapes:
                    self._emissive_colors[s] = color

        self.viewer.update_shape_colors(shape_colors)

    def _apply_emissive_materials(self):
        """Swap selected cube instances to emissive materials after scene build."""
        scene = getattr(self.viewer, "_scene", None)
        if scene is None or not self._emissive_colors:
            return

        # Build shape_index -> OptiX instance ID mapping from the viewer's
        # batch data structures.
        shape_to_inst = {}
        for batch in self.viewer._shape_instances.values():
            name = batch.name
            inst_ids = self.viewer._instance_name_to_optix_ids.get(name, [])
            for local_idx, s_idx in enumerate(batch.model_shapes):
                if local_idx < len(inst_ids):
                    shape_to_inst[s_idx] = inst_ids[local_idx]

        for s, color in self._emissive_colors.items():
            max_c = max(color[0], color[1], color[2], 1e-6)
            emissive = (
                color[0] / max_c * EMISSIVE_INTENSITY,
                color[1] / max_c * EMISSIVE_INTENSITY,
                color[2] / max_c * EMISSIVE_INTENSITY,
            )
            mat_id = scene.materials.add_gltf_material(
                base_color=(0.0, 0.0, 0.0, 1.0),
                emissive_factor=emissive,
                roughness=1.0,
                metallic=0.0,
            )
            inst_id = shape_to_inst.get(s)
            if inst_id is not None:
                self.viewer._instance_material_map[inst_id] = mat_id

        self.viewer._materials_dirty = True

    def _setup_camera(self):
        """Set initial camera pose via internal viewer state."""
        pos = np.array([0.8579, 1.4811, 3.6018], dtype=np.float32)
        tgt = np.array([0.5751, 1.1556, 2.6995], dtype=np.float32)
        fwd = tgt - pos
        fwd /= np.linalg.norm(fwd)
        self.viewer._cam_pos = pos
        self.viewer._cam_yaw = float(np.degrees(np.arctan2(fwd[0], fwd[2])))
        self.viewer._cam_pitch = float(np.degrees(np.arcsin(np.clip(fwd[1], -1.0, 1.0))))
        self.viewer._cam_fov = 45.0
        self.viewer._cam_user_set = True

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if PRINT_CAMERA_POSE and hasattr(self.viewer, "_camera_changed") and self.viewer._camera_changed():
            p = self.viewer._cam_pos
            yaw = self.viewer._cam_yaw
            pitch = self.viewer._cam_pitch
            fov = self.viewer._cam_fov
            yaw_r = math.radians(yaw)
            pitch_r = math.radians(pitch)
            cos_p = math.cos(pitch_r)
            target = p + np.array([math.sin(yaw_r) * cos_p, math.sin(pitch_r), math.cos(yaw_r) * cos_p])
            print(
                f"api.set_camera_look_at(\n"
                f"    position=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}),\n"
                f"    target=({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}),\n"
                f"    up=(0.0, 1.0, 0.0),\n"
                f"    fov={fov:.1f},\n"
                f")"
            )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

        if not self._emissive_applied and self._use_optix:
            self._apply_emissive_materials()
            self._emissive_applied = True

    def test_final(self):
        if self.sim_time <= 0.0:
            raise ValueError("Example did not advance any frames.")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="optix")
    viewer, args = newton.examples.init(parser=parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
