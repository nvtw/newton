# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for hydroelastic contacts running through the
PhoenX solver.

Hydroelastic's reduction kernel publishes per-contact absolute
stiffness ``k_eff * |agg_force| / total_depth`` into
``Contacts.rigid_contact_stiffness``. With Phase 1 of the soft-
contact wiring, PhoenX's contact normal row automatically picks
that up (:mod:`test_soft_contacts`) and routes through the PhysX-
style PD formulation instead of the legacy mass-normalized
Nyquist-rigid path. This module is the end-to-end check that the
two pieces talk: a real hydroelastic narrow phase + the PhoenX
solver produces a stable stack.

Kept tight and single-purpose: a 3-primitive-cube stack (1 m cubes
with hydroelastic SDFs), 1 second of settle, and a displacement /
rotation assertion on each cube. Mirrors the MuJoCo / XPBD
variants in :mod:`newton.tests.test_hydroelastic` so a regression
in the PhoenX soft-contact wiring diverges from the established
parity.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.solver import SolverPhoenX
from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING
from newton.geometry import HydroelasticSDF

# Same scene parameters as test_hydroelastic.py for parity.
CUBE_HALF = 0.5
NUM_CUBES = 3
SIM_DT = 1.0 / 60.0
SIM_SUBSTEPS = 10
SIM_TIME_S = 1.0

# PhoenX is slightly less stiff than MuJoCo on the Nyquist-rigid
# fallback; the absolute-PD path inherits whatever stiffness the
# hydroelastic reduction publishes, so we use the same displacement
# tolerance as the reference tests.
POSITION_THRESHOLD_FACTOR = 0.20
MAX_ROTATION_DEG = 10.0


def _build_stacked_cubes_scene(device):
    """Stack ``NUM_CUBES`` primitive cubes with hydroelastic enabled
    on a ground plane. Returns everything needed to step.

    Mirrors :func:`test_hydroelastic.build_stacked_cubes_scene` for
    ``ShapeType.PRIMITIVE`` but routes through PhoenX's sticky contact
    matching so the solver sees the same contact persistence used by the
    production path.
    """
    narrow_band = CUBE_HALF * 0.2
    contact_gap = CUBE_HALF * 0.2

    builder = newton.ModelBuilder()
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.5,
        sdf_max_resolution=32,
        is_hydroelastic=True,
        sdf_narrow_band_range=(-narrow_band, narrow_band),
        gap=contact_gap,
    )
    builder.add_ground_plane()

    initial_positions = []
    for i in range(NUM_CUBES):
        z_pos = CUBE_HALF + i * CUBE_HALF * 2.0
        initial_positions.append(wp.vec3(0.0, 0.0, z_pos))
        body = builder.add_body(
            xform=wp.transform(initial_positions[-1], wp.quat_identity()),
            label=f"cube_{i}",
        )
        builder.add_shape_box(body=body, hx=CUBE_HALF, hy=CUBE_HALF, hz=CUBE_HALF)

    model = builder.finalize(device=device)

    collision_pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=512,
        broad_phase="explicit",
        contact_matching=PHOENX_CONTACT_MATCHING,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=True,
            anchor_contact=True,
            buffer_fraction=1.0,
        ),
    )
    contacts = model.contacts(collision_pipeline=collision_pipeline)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # PhoenX construction happens AFTER the collision pipeline is
    # attached so :meth:`SolverPhoenX.__init__` reuses the matching
    # pipeline instead of attaching a fresh non-matching one.
    solver = SolverPhoenX(
        model,
        substeps=1,
        solver_iterations=8,
        velocity_iterations=1,
    )

    return (
        model,
        solver,
        state_0,
        state_1,
        control,
        contacts,
        collision_pipeline,
        initial_positions,
    )


@unittest.skipUnless(wp.is_cuda_available(), "Hydroelastic requires CUDA")
class TestPhoenXHydroelasticStack(unittest.TestCase):
    """Three 1 m primitive cubes stacked with hydroelastic contacts
    must remain stable under PhoenX for 1 s."""

    def test_stacked_primitive_cubes(self) -> None:
        device = wp.get_preferred_device()
        (
            model,
            solver,
            state_0,
            state_1,
            _control,
            contacts,
            _collision_pipeline,
            initial_positions,
        ) = _build_stacked_cubes_scene(device)

        num_frames = int(SIM_TIME_S / SIM_DT)
        for _ in range(num_frames):
            # Reduced hydroelastic contacts are geometry-derived surfaces; refresh
            # them at the same small-step cadence used by the reference tests.
            for _ in range(SIM_SUBSTEPS):
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, None, contacts, SIM_DT / SIM_SUBSTEPS)
                state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        position_threshold = POSITION_THRESHOLD_FACTOR * CUBE_HALF

        for i in range(NUM_CUBES):
            expected_z = initial_positions[i][2]
            actual_pos = body_q[i, :3]
            displacement = float(np.linalg.norm(actual_pos - np.array([0.0, 0.0, expected_z])))
            self.assertLess(
                displacement,
                position_threshold,
                msg=(
                    f"cube {i} displaced {displacement:.4f} m (threshold "
                    f"{position_threshold:.4f}); hydroelastic normal force "
                    "is not balancing the stack weight."
                ),
            )

            initial_quat = np.array([0.0, 0.0, 0.0, 1.0])
            final_quat = body_q[i, 3:]
            dot_product = float(np.clip(abs(np.dot(initial_quat, final_quat)), 0.0, 1.0))
            rotation_angle = 2.0 * np.arccos(dot_product)
            self.assertLess(
                rotation_angle,
                np.radians(MAX_ROTATION_DEG),
                msg=(
                    f"cube {i} rotated {np.degrees(rotation_angle):.2f} deg "
                    f"(threshold {MAX_ROTATION_DEG} deg); likely a normal / "
                    "tangent coupling issue in the PhoenX PD contact path."
                ),
            )


def _small_hydroelastic_cube_config(gap: float = 0.0) -> newton.ModelBuilder.ShapeConfig:
    """Hydroelastic 10 cm cube material; ``gap=0`` keeps contacts to overlapping geometry."""
    return newton.ModelBuilder.ShapeConfig(
        mu=0.5,
        sdf_max_resolution=32,
        is_hydroelastic=True,
        sdf_narrow_band_range=(-0.01, 0.01),
        gap=gap,
        density=1000.0,
    )


def _capture_frame(pipeline, solver, state_0, state_1, contacts, frame_dt: float):
    """Capture one collide + step frame that leaves its result in ``state_0``."""

    def frame():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, frame_dt)
        state_0.assign(state_1)

    frame()  # compile and allocate outside the capture
    with wp.ScopedCapture() as capture:
        frame()
    return capture.graph


@unittest.skipUnless(wp.is_cuda_available(), "Hydroelastic requires CUDA")
class TestPhoenXHydroelasticImpact(unittest.TestCase):
    """Hydroelastic contacts publish a stiffness without damping."""

    HALF = 0.05
    DROP = 0.01

    def test_contact_does_not_return_penetration_as_bounce(self) -> None:
        """A cube placed on a cube rests and a dropped cube rebounds below its drop height.

        Contacts are found once per frame of four substeps, so an impact first
        penetrates. An undamped stiff row returns that penetration as
        separation speed: the dropped cube rebounded 71 mm from 10 mm and the
        placed cube bounced 14 mm.
        """
        half = self.HALF
        builder = newton.ModelBuilder()
        builder.default_shape_cfg = _small_hydroelastic_cube_config()
        cubes = []
        for x, drop in ((0.0, 0.0), (0.5, self.DROP)):
            builder.add_shape_box(
                body=-1, xform=wp.transform((x, 0.0, half), wp.quat_identity()), hx=half, hy=half, hz=half
            )
            body = builder.add_body(xform=wp.transform((x, 0.0, 3.0 * half + drop), wp.quat_identity()))
            builder.add_shape_box(body=body, hx=half, hy=half, hz=half)
            cubes.append(body)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            rigid_contact_max=512,
            broad_phase="explicit",
            contact_matching=PHOENX_CONTACT_MATCHING,
            sdf_hydroelastic_config=HydroelasticSDF.Config(reduce_contacts=True, anchor_contact=True),
        )
        contacts = pipeline.contacts()
        solver = SolverPhoenX(model, collision_pipeline=pipeline, substeps=4, solver_iterations=8)
        state_0, state_1 = model.state(), model.state()
        graph = _capture_frame(pipeline, solver, state_0, state_1, contacts, 1.0 / 120.0)

        heights = []
        for _ in range(120):
            wp.capture_launch(graph)
            heights.append(state_0.body_q.numpy()[cubes, 2] - 3.0 * half)
        heights = np.array(heights)
        self.assertTrue(np.isfinite(heights).all())

        placed_bounce = float(heights[:, 0].max())
        self.assertLess(placed_bounce, 1.0e-3, msg=f"placed cube bounced {placed_bounce * 1e3:.2f} mm")
        dropped = heights[:, 1]
        impact = int(np.argmin(dropped))
        rebound = float(dropped[impact:].max())
        self.assertLess(
            rebound, self.DROP, msg=f"cube dropped {self.DROP * 1e3:.0f} mm rebounded {rebound * 1e3:.2f} mm"
        )
        self.assertLess(abs(float(dropped[-1])), 1.0e-3, msg="dropped cube did not come to rest on the cube")


if __name__ == "__main__":
    wp.init()
    unittest.main()
