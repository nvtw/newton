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

"""OptiX reference plate scene with procedural studio environment lighting."""

from __future__ import annotations

import math

import numpy as np

import newton
import newton.examples

# Toggle between procedural sky (single sun) and studio env map (multi-light).
USE_STUDIO_ENV_MAP = True

# Print the full camera pose to stdout whenever it changes (useful for finding
# the best initial camera position interactively).
PRINT_CAMERA_POSE = False

# Global multiplier for focused spot light intensities (shadow strength).
SPOT_INTENSITY = 1.0


def _generate_studio_env_map(width: int = 512, height: int = 256) -> np.ndarray:
    """Generate a lat-long HDR environment map with multi-light studio setup.

    The map contains a smooth sky gradient (white horizon fading to light
    blue-gray at zenith) plus several broad Gaussian "area lights" at
    different positions and tints to produce soft, multi-directional shadows.

    Returns:
        Float32 array of shape ``(height, width, 3)`` in linear HDR.
    """
    env = np.zeros((height, width, 3), dtype=np.float32)

    # Pre-compute per-pixel spherical directions (Y-up lat-long).
    # v in [0, pi]: 0 = north pole (+Y), pi = south pole (-Y).
    # u in [0, 2*pi]: azimuth around Y axis.
    v_angles = np.linspace(0, math.pi, height, dtype=np.float32)
    u_angles = np.linspace(0, 2.0 * math.pi, width, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u_angles, v_angles)

    # Unit direction vectors for every pixel.
    dir_x = np.sin(vv) * np.sin(uu)
    dir_y = np.cos(vv)
    dir_z = np.sin(vv) * np.cos(uu)

    # --- Base sky gradient ---
    # Colorful sky: warm peach/gold near horizon on the key-light side,
    # cool blue at zenith, pale lavender opposite horizon.
    horizon_weight = np.exp(-5.0 * dir_y**2)

    # Blend zenith color from cool blue (overhead) toward warm at horizon.
    sky_zenith = np.array([0.30, 0.35, 0.55], dtype=np.float32)
    sky_horizon_warm = np.array([1.10, 0.80, 0.45], dtype=np.float32)
    sky_horizon_cool = np.array([0.55, 0.60, 0.80], dtype=np.float32)

    # Azimuth-based warm/cool blend: key-light side is warm, opposite is cool.
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

    # --- Studio lights (Gaussian blobs in angular space) ---
    # Each light: (elevation_deg, azimuth_deg, angular_sigma_deg, intensity, r, g, b)
    # Mix of compact directed lights (small sigma, lower elevation) for
    # shadow definition and broader fills for ambient color.
    lights = [
        # === Dominant focused spots (cast hard shadows) ===
        # Warm white key: the main shadow caster
        (30.0, -40.0, 1.5, 300.0, 1.00, 0.95, 0.85),
        # Cold white fill: secondary shadow from opposite side
        (25.0, 55.0, 1.5, 200.0, 0.85, 0.92, 1.00),
        # Warm accent: grazing angle, long shadows
        (10.0, -65.0, 1.2, 150.0, 1.00, 0.88, 0.70),
        # Cool rim: backlight edge definition
        (18.0, 165.0, 1.5, 150.0, 0.75, 0.80, 1.00),
        # === Broad washes (colored ambient, keep sky alive) ===
        # Warm wash from key side
        (45.0, -30.0, 35.0, 1.0, 1.00, 0.85, 0.55),
        # Cool wash from fill side
        (40.0, 60.0, 35.0, 0.8, 0.50, 0.65, 1.00),
        # Back wash: purple tint
        (40.0, 170.0, 30.0, 0.7, 0.70, 0.55, 1.00),
        # Ground bounce
        (-20.0, 0.0, 40.0, 0.3, 1.00, 0.80, 0.45),
    ]

    for elev_deg, az_deg, sigma_deg, intensity, lr, lg, lb in lights:
        # Apply global spot multiplier to focused lights (sigma < 10 deg).
        if sigma_deg < 10.0:
            intensity *= SPOT_INTENSITY
        elev = math.radians(elev_deg)
        az = math.radians(az_deg)
        sigma = math.radians(sigma_deg)

        # Light direction on the unit sphere (Y-up).
        lx = math.cos(elev) * math.sin(az)
        ly = math.sin(elev)
        lz = math.cos(elev) * math.cos(az)

        # Angular distance from each pixel direction to the light direction.
        dot = np.clip(dir_x * lx + dir_y * ly + dir_z * lz, -1.0, 1.0)
        ang_dist = np.arccos(dot)

        # Gaussian falloff.
        weight = np.exp(-0.5 * (ang_dist / sigma) ** 2)
        env[..., 0] += weight * intensity * lr
        env[..., 1] += weight * intensity * lg
        env[..., 2] += weight * intensity * lb

    return env


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame = 0

        if hasattr(self.viewer, "_cam_speed"):
            self.viewer._cam_speed = 3.0

        if args.viewer != "optix":
            raise RuntimeError("This example is intended for the OptiX viewer (`--viewer optix`).")

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

        api.clear_scene()

        bottom_mat = api.create_pbr_material(
            color=(0.36, 0.37, 0.40),
            roughness=0.82,
            metallic=0.02,
        )
        top_mat = api.create_pbr_material(
            color=(0.12, 0.11, 0.11),
            roughness=0.18,
            metallic=0.62,
        )
        ground_mat = api.create_pbr_material(
            color=(0.86, 0.86, 0.86),
            roughness=0.9,
            metallic=0.0,
        )

        # Clean rainbow palette (no pink/rosa): red, orange, yellow, green, teal, blue.
        palette = [
            (0.85, 0.12, 0.08),
            (0.92, 0.52, 0.08),
            (0.92, 0.82, 0.15),
            (0.22, 0.72, 0.22),
            (0.12, 0.68, 0.68),
            (0.14, 0.38, 0.88),
        ]
        pyramid_mats = [api.create_pbr_material(color=c, roughness=0.45, metallic=0.05) for c in palette]

        # Ground plane - much larger than the presenter plate.
        api.add_box(min_pt=(-20.0, -0.09, -20.0), max_pt=(20.0, -0.07, 20.0), material_id=ground_mat)

        # Bottom layer (square presenter plate).
        plate_half = 1.30
        api.add_box(
            min_pt=(-plate_half, -0.07, -plate_half),
            max_pt=(plate_half, -0.02, plate_half),
            material_id=bottom_mat,
        )

        # Top layer: same square footprint, thinner, darker, metallic, top at y=0.
        api.add_box(
            min_pt=(-plate_half, -0.02, -plate_half),
            max_pt=(plate_half, 0.00, plate_half),
            material_id=top_mat,
        )

        # Colorful 3-2-1 box pyramid for quick visual checks.
        cube_half = 0.14
        cube_spacing = 2.05 * cube_half
        base_y = 0.0
        color_idx = 0
        for level, count in enumerate((3, 2, 1)):
            y0 = base_y + level * (2.0 * cube_half)
            y1 = y0 + 2.0 * cube_half
            x_start = -0.5 * (count - 1) * cube_spacing
            for i in range(count):
                x = x_start + i * cube_spacing
                z = 0.0
                mat = pyramid_mats[color_idx % len(pyramid_mats)]
                color_idx += 1
                api.add_box(
                    min_pt=(x - cube_half, y0, z - cube_half),
                    max_pt=(x + cube_half, y1, z + cube_half),
                    material_id=mat,
                )

        api.build_scene()

        # Set camera via the viewer's internal state so it doesn't get
        # overwritten by _sync_camera (api.set_camera_look_at only sets the
        # pathtracing camera which the viewer overwrites each frame).
        pos = np.array([0.8579, 1.4811, 3.6018], dtype=np.float32)
        tgt = np.array([0.5751, 1.1556, 2.6995], dtype=np.float32)
        fwd = tgt - pos
        fwd /= np.linalg.norm(fwd)
        self.viewer._cam_pos = pos
        self.viewer._cam_yaw = float(np.degrees(np.arctan2(fwd[0], fwd[2])))
        self.viewer._cam_pitch = float(np.degrees(np.arcsin(np.clip(fwd[1], -1.0, 1.0))))
        self.viewer._cam_fov = 45.0
        self.viewer._cam_user_set = True

    @staticmethod
    def _setup_studio_env_map(api):
        """Load a procedurally generated studio env map into the path tracer."""
        from newton._src.viewer.optix.pathtracing.environment_map import EnvironmentMap  # noqa: PLC0415

        env_array = _generate_studio_env_map(width=512, height=256)
        env_map = EnvironmentMap()
        env_map.load_from_array(env_array)
        api._viewer._env_map = env_map

    def step(self):
        self.frame += 1

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
        self.viewer.begin_frame(self.frame / 60.0)
        self.viewer.end_frame()

    def test_final(self):
        if self.frame <= 0:
            raise ValueError("Example did not advance any frames.")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="optix")
    viewer, args = newton.examples.init(parser=parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
