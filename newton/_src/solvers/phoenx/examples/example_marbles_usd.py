# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the Marbles USD stage with the OptiX viewer.

This first integration step loads composed USD geometry, materials, textures,
and the optional dome-light environment directly into ``warp_optix``. USD
physics schemas are intentionally not imported into Newton yet. The retained
USD hierarchy is available through ``viewer.usd_scene`` for future PhoenX
transform coupling.

Run as::

    uv run --extra dev python -m newton._src.solvers.phoenx.examples.example_marbles_usd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import newton.examples

DEFAULT_USD_PATH = Path("/home/twidmer/Documents/Meshes/Marbles/Marbles_Assets_with_physics.usd")


class Example:
    """Display a static composed USD scene through the OptiX renderer."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.time = 0.0

        if not hasattr(viewer, "load_scene_from_usd"):
            raise RuntimeError("The Marbles USD example requires --viewer optix")

        usd_path = Path(args.usd_path).expanduser().resolve()
        if not usd_path.is_file():
            raise FileNotFoundError(f"Marbles USD stage not found: {usd_path}")

        loaded = viewer.load_scene_from_usd(
            str(usd_path),
            max_texture_size=args.usd_max_texture_size,
            load_usd_environment=args.usd_environment,
            usd_environment_scale=args.usd_environment_scale,
        )
        if not loaded:
            raise RuntimeError(f"OptiX failed to load USD stage: {usd_path}")

        # The authored /stage/Overview camera is expressed in centimeters.
        # These values preserve its world-space view after stage-unit conversion.
        viewer.set_camera((0.3828325, 3.1660733, 6.887878), pitch=-39.78, yaw=-71.63)

        usd_scene = viewer.usd_scene
        transform_count = len(usd_scene.transforms) if usd_scene is not None else 0
        print(f"[PhoenX Marbles USD] loaded {usd_path} ({transform_count} retained transforms)")

    def step(self) -> None:
        """Advance display time without simulating the static USD stage."""
        self.time += 1.0 / 60.0

    def render(self) -> None:
        """Render the retained USD scene."""
        self.viewer.begin_frame(self.time)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify that the OptiX renderer retained the loaded USD hierarchy."""
        assert self.viewer.usd_scene is not None
        assert len(self.viewer.usd_scene.transforms) > 0


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--usd-path",
        type=str,
        default=str(DEFAULT_USD_PATH),
        help="Composed USD stage to render.",
    )
    parser.add_argument(
        "--usd-max-texture-size",
        type=int,
        default=1024,
        help="Maximum loaded texture dimension; use 0 for the source resolution.",
    )
    parser.add_argument(
        "--usd-environment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load a supported USD DomeLight texture into the OptiX environment.",
    )
    parser.add_argument(
        "--usd-environment-scale",
        type=float,
        default=1.0,
        help="Brightness multiplier for the USD DomeLight environment texture.",
    )
    parser.set_defaults(viewer="optix")
    viewer, args = newton.examples.init(parser)
    if args.usd_max_texture_size == 0:
        args.usd_max_texture_size = None
    example = Example(viewer, args)
    newton.examples.run(example, args)
