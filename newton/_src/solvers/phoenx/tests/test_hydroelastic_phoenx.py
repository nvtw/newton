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
from newton._src.solvers.phoenx.solver_config import (
    PHOENX_CONTACT_MATCHING,
)
from newton._src.solvers.phoenx.solver import SolverPhoenX
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
    ``ShapeType.PRIMITIVE``, but wires PhoenX with sticky contact
    matching (needed for its per-pair warm-start).
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
        rigid_contact_max=100,
        broad_phase="explicit",
        contact_matching=PHOENX_CONTACT_MATCHING,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=True,
            anchor_contact=True,
            buffer_fraction=1.0,
        ),
    )
    model._collision_pipeline = collision_pipeline
    contacts = collision_pipeline.contacts()

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # PhoenX construction happens AFTER the collision pipeline is
    # attached so :meth:`SolverPhoenX.__init__` reuses the matching
    # pipeline instead of attaching a fresh non-matching one.
    solver = SolverPhoenX(
        model,
        substeps=SIM_SUBSTEPS,
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
            collision_pipeline,
            initial_positions,
        ) = _build_stacked_cubes_scene(device)

        num_frames = int(SIM_TIME_S / SIM_DT)
        for _ in range(num_frames):
            collision_pipeline.collide(state_0, contacts)
            # PhoenX's own substep loop runs inside ``step``; we stay
            # with one collide + one step per frame.
            solver.step(state_0, state_1, None, contacts, SIM_DT)
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


if __name__ == "__main__":
    wp.init()
    unittest.main()
