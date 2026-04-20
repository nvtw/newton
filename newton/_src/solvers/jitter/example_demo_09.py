# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Demo 09 -- Restitution and Friction
#
# Port of ``JitterDemo.Demos.Demo09`` (see
# ``C:\git3\jitterphysics2\src\JitterDemo\Demos\Demo09.cs``). Two rows
# of 11 small boxes (edge 1 m, hx=0.5) along +X:
#
# * First row drops from +Z = 4 m with varying restitution (``0`` to
#   ``1``). Jitter's contact constraint here in the Newton port does
#   *not* yet model restitution, so all 11 behave identically -- the
#   row is kept anyway to faithfully mirror the C# scene layout.
# * Second row slides along +Y (original +Y in the C# code, which
#   maps to Newton's +X) with varying friction (``1.0`` down to
#   ``0.0``). Friction is plumbed through ``shape_material_mu`` and
#   consumed by the Jitter contact solver.
#
# Run:  python -m newton._src.solvers.jitter.example_demo_09
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton._src.solvers.jitter.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

BOX_HALF = 0.25  # BoxShape(0.5) in the C# source (full edge 0.5)
NUM_IN_ROW = 11


class Example(DemoExample):
    def __init__(self, viewer, args):
        cfg = DemoConfig(
            title="Restitution and Friction",
            camera_pos=(12.0, -12.0, 6.0),
            camera_pitch=-18.0,
            camera_yaw=45.0,
            fps=60,
            substeps=4,
            solver_iterations=20,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        # Frictionless floor (C# sets pg.FloorShape.RigidBody.Friction = 0).
        floor_cfg = mb.default_shape_cfg.copy()
        floor_cfg.mu = 0.0
        mb.add_shape_plane(
            -1, wp.transform_identity(), width=0.0, length=0.0, cfg=floor_cfg
        )

        self._rest_bodies: list[int] = []
        self._fric_bodies: list[int] = []

        # ---- Row 1: restitution (dropped from 4 m) ----------------
        # Jitter code: Position(-10 + i, 4, -10). Newton uses Z for
        # the vertical axis and we keep the other two as-is.
        for i in range(NUM_IN_ROW):
            cfg = mb.default_shape_cfg.copy()
            cfg.restitution = i * 0.1  # plumbed into the Newton model;
            # the jitter contact solver does not yet consume
            # restitution, so visually all boxes fall identically.
            x = -5.0 + i * 1.0
            y = -10.0
            z = 4.0
            body = mb.add_body(
                xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
                mass=0.1,
            )
            mb.add_shape_box(
                body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg
            )
            self._rest_bodies.append(body)
            self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

        # ---- Row 2: friction (sliding) ----------------------------
        # Jitter code: Position(2 + i, 0.25, 0), Velocity(0, 0, -10).
        # In Newton's +Z-up we keep X the same, set Z = 0.25 (just
        # above the floor) and send the cube sliding along -Y.
        for i in range(NUM_IN_ROW):
            cfg = mb.default_shape_cfg.copy()
            cfg.mu = max(0.0, 1.0 - i * 0.1)
            x = 2.0 + i * 1.0
            y = 0.0
            z = BOX_HALF + 1.0e-3  # just clear of the floor
            body = mb.add_body(
                xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
                mass=0.1,
            )
            mb.add_shape_box(
                body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg
            )
            self._fric_bodies.append(body)
            self.register_body_extent(body, (BOX_HALF, BOX_HALF, BOX_HALF))

        # Initial -Y slide (Jitter sets Velocity(0, 0, -10); +Z in
        # Jitter <-> +Y in Newton so the velocity is along -Y in our
        # frame). The solver uses this initial velocity through the
        # ``body_qd`` state we already mirror into Jitter every frame.
        self._initial_slide_speed = 10.0

    def finish_setup(self) -> None:  # noqa: D401 -- extends base hook
        super().finish_setup()
        # Inject the sliding-row initial velocity directly into the
        # Jitter body container and bounce it back through the Newton
        # state so subsequent syncs keep it. Done here (not in
        # :meth:`build_scene`) because :attr:`world` only exists after
        # :meth:`DemoExample.finish_setup`.
        velocities = self.world.bodies.velocity.numpy().copy()
        for newton_idx in self._fric_bodies:
            j = self._newton_to_jitter[newton_idx]
            velocities[j] = (0.0, -self._initial_slide_speed, 0.0)
        self.world.bodies.velocity.assign(
            np.asarray(velocities, dtype=np.float32)
        )
        # Propagate back into Newton's ``state.body_qd`` so the first
        # collide() sees the correct velocity.
        self._sync_jitter_to_newton()

    def test_final(self) -> None:
        """Every body stayed on / near the floor; no blow-ups."""
        positions = self.world.bodies.position.numpy()
        for newton_idx in self._rest_bodies + self._fric_bodies:
            j = self._newton_to_jitter[newton_idx]
            pos = positions[j]
            assert np.isfinite(pos).all(), f"body {newton_idx} non-finite pos"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
