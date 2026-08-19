# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the RooftopFlat USD scene with the OptiX viewer.

The complete composed USD hierarchy is loaded directly into OptiX. No Newton
model, collision geometry, or physics state is created.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np

import newton.examples

DEFAULT_USD_PATH = Path("/home/twidmer/Documents/Meshes/Scene.usd")
DEFAULT_CAMERA_PATH = "/Root/Room/CineCameraActor4_2"

LOAD_USD_ENVIRONMENT = False
USD_MAX_TEXTURE_SIZE = 4096
OPTIX_DLSS_QUALITY = "quality"
OPTIX_MAX_BOUNCES = 5
OPTIX_RUSSIAN_ROULETTE_START_BOUNCE = 3
SKY_SPHERE_PATH = "/Environment/sky_NightSky/AxisNorth/SkySphere"


def _create_render_overlay(usd_path: Path) -> tuple[tempfile.TemporaryDirectory, Path]:
    """Create a non-destructive USD overlay tailored for OptiX rendering."""
    from pxr import Usd, UsdGeom

    temporary_directory = tempfile.TemporaryDirectory(prefix="newton_rooftop_flat_")
    overlay_path = Path(temporary_directory.name) / "RooftopFlat_render.usda"
    source_stage = Usd.Stage.Open(str(usd_path))
    if source_stage is None:
        raise ValueError(f"OpenUSD could not open stage: {usd_path}")
    stage = Usd.Stage.CreateNew(str(overlay_path))
    stage.GetRootLayer().subLayerPaths.append(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))

    sky_sphere = stage.GetPrimAtPath(SKY_SPHERE_PATH)
    if sky_sphere:
        UsdGeom.Imageable(sky_sphere).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    stage.GetRootLayer().Save()
    return temporary_directory, overlay_path


def _select_authored_camera(viewer, camera_path: str) -> None:
    """Configure the viewer from an authored USD perspective camera."""
    from pxr import UsdGeom

    usd_scene = viewer.usd_scene
    prim = usd_scene.stage.GetPrimAtPath(camera_path)
    handle = usd_scene.get_transform(camera_path)
    if not prim or handle is None or not prim.IsA(UsdGeom.Camera):
        raise ValueError(f"USD camera does not exist or has no transform: {camera_path}")

    camera = UsdGeom.Camera(prim)
    if str(camera.GetProjectionAttr().Get()) != "perspective":
        raise ValueError(f"USD camera is not perspective: {camera_path}")

    focal_length = float(camera.GetFocalLengthAttr().Get() or 0.0)
    aperture = float(camera.GetVerticalApertureAttr().Get() or 0.0)
    if focal_length <= 0.0 or aperture <= 0.0:
        raise ValueError(f"USD camera has invalid lens settings: {camera_path}")

    world = usd_scene.get_world_transform(handle)
    position = world[:3, 3]
    target = position - world[:3, 2]
    fov = math.degrees(2.0 * math.atan(aperture / (2.0 * focal_length)))
    viewer.set_camera_look_at(
        position,
        target,
        fov=float(np.clip(fov, 5.0, 120.0)),
        renderer_space=True,
    )


class Example:
    """Render the retained RooftopFlat USD hierarchy without physics."""

    def __init__(self, viewer, args):
        if not hasattr(viewer, "load_scene_from_usd") or not hasattr(viewer, "set_camera_look_at"):
            raise RuntimeError("The RooftopFlat USD example requires --viewer optix")

        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0

        usd_path = Path(args.usd_path).expanduser().resolve()
        if not usd_path.is_file():
            raise FileNotFoundError(f"RooftopFlat USD stage not found: {usd_path}")

        self._render_overlay, render_path = _create_render_overlay(usd_path)
        if not viewer.load_scene_from_usd(
            str(render_path),
            max_texture_size=args.usd_max_texture_size,
            load_usd_environment=args.usd_environment,
            usd_environment_scale=args.usd_environment_scale,
            load_usd_lights=True,
        ):
            raise RuntimeError(f"OptiX failed to load USD stage: {usd_path}")

        # Preserve authored bulb and fixture emission in the night scene.
        viewer.emissive_material_intensity = 1.0

        if not args.usd_environment:
            # Keep initialization and the interactive time slider on one sky state.
            viewer.time_of_day = 0.0

        viewer.configure_auto_exposure(
            True,
            target_luminance=0.18,
            min_ev=-6.0,
            max_ev=6.0,
            brighten_speed=0.6,
            darken_speed=1.2,
        )
        viewer.set_ray_budget(
            russian_roulette_start_bounce=OPTIX_RUSSIAN_ROULETTE_START_BOUNCE,
        )
        _select_authored_camera(viewer, args.usd_camera)
        print(
            f"[RooftopFlat USD] loaded {usd_path} "
            f"({viewer.usd_scene.transform_count} retained transforms, camera={args.usd_camera})"
        )

    def step(self) -> None:
        """Advance render time without simulating physics."""
        self.sim_time += self.frame_dt

    def reset_in_place(self) -> None:
        """Reset render time without reloading the retained USD scene."""
        self.sim_time = 0.0

    def render(self) -> None:
        """Render the retained USD hierarchy."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify the retained USD rendering hierarchy is valid."""
        assert self.viewer.usd_scene is not None
        assert self.viewer.usd_scene.transform_count > 0


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--usd-path",
        type=str,
        default=str(DEFAULT_USD_PATH),
        help="Composed RooftopFlat USD stage to render.",
    )
    parser.add_argument(
        "--usd-max-texture-size",
        type=int,
        default=USD_MAX_TEXTURE_SIZE,
        help="Maximum source texture dimension before adaptive atlas fitting; use 0 for no per-texture cap.",
    )
    parser.add_argument(
        "--usd-environment",
        action=argparse.BooleanOptionalAction,
        default=LOAD_USD_ENVIRONMENT,
        help="Load a supported USD DomeLight texture into the OptiX environment.",
    )
    parser.add_argument(
        "--usd-environment-scale",
        type=float,
        default=1.0,
        help="Brightness multiplier for the USD DomeLight environment texture.",
    )
    parser.add_argument(
        "--usd-camera",
        type=str,
        default=DEFAULT_CAMERA_PATH,
        help="Authored perspective camera path.",
    )
    parser.set_defaults(
        viewer="optix",
        optix_dlss_quality=OPTIX_DLSS_QUALITY,
        optix_max_bounces=OPTIX_MAX_BOUNCES,
    )
    viewer, args = newton.examples.init(parser)
    if args.usd_max_texture_size == 0:
        args.usd_max_texture_size = None
    example = Example(viewer, args)
    newton.examples.run(example, args)
