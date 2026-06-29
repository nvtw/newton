# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fast dynamic regression tests for box contacts via the MPR/GJK ``v0`` seed.

These are cheap stand-ins for the slow ``example_brick_stacking`` /
``example_ik_cube_stacking`` runs. They guard the box initial-direction seed in
:func:`~newton._src.geometry.mpr.box_face_seed` against two failure modes a bad
seed introduces:

* **Energy injection** -- a seed that places ``v0`` outside the box (e.g. when a
  partner sits past the box footprint) makes MPR report a bogus deep penetration,
  launching shapes upward. ``test_box_pile_on_thin_table_no_energy_injection``
  drops a pile of boxes onto a thin "table" box and asserts nothing is ever
  flung above its start height.
* **Box-box stacking** -- the essence of ``example_brick_stacking`` /
  ``example_ik_cube_stacking``. ``test_box_pile_on_ground_no_energy_injection``
  piles cubes on a thick ground (no thin table involved), so the regression comes
  purely from box-box contacts, and asserts nothing is flung away.

Both reproduce with the unguarded seed and pass once the seed is only used when
it is a valid interior point of the box.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def _simulate(model, num_steps, sim_substeps=10, fps=100):
    """Run XPBD (matching the example configs) and return per-step body z (centers)."""
    solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_contact_relaxation=0.8, angular_damping=0.0)
    collision_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    sim_dt = (1.0 / fps) / sim_substeps
    z_history = []
    for _ in range(num_steps):
        for _ in range(sim_substeps):
            state_0.clear_forces()
            contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
        z_history.append(state_0.body_q.numpy()[: model.body_count, 2].copy())
    final_q = state_0.body_q.numpy()[: model.body_count]
    return np.array(z_history), final_q


def _add_free_box(builder, pos, half=0.15, q=None):
    q = wp.quat_identity() if q is None else q
    body = builder.add_body(xform=wp.transform(p=pos, q=q))
    builder.add_shape_box(body, hx=half, hy=half, hz=half)
    builder.add_articulation([builder.add_joint_free(body)])
    return body


def test_box_pile_on_thin_table_no_energy_injection(test, device):
    """Boxes dropped onto a thin table settle without ever being launched upward.

    The unguarded seed places ``v0`` outside a box when a partner is past its
    footprint (common while a pile jostles), producing a spurious deep penetration
    that flings shapes far above their start height.
    """
    builder = newton.ModelBuilder()
    # Thin static table (80:1 aspect), top face at z = 0.1.
    builder.add_shape_box(
        body=-1, xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()), hx=4.0, hy=4.0, hz=0.1
    )

    half = 0.15
    rng = np.random.default_rng(7)
    initial_z = []
    # A tight 3x3x3 cluster so boxes pile and repeatedly push partners past each
    # other's footprints -- the configuration that exposes a bad seed.
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                pos = wp.vec3(
                    float(ix - 1) * 0.32 + float(rng.random() - 0.5) * 0.04,
                    float(iy - 1) * 0.32 + float(rng.random() - 0.5) * 0.04,
                    0.1 + half + 0.05 + iz * 0.34,
                )
                _add_free_box(builder, pos, half)
                initial_z.append(pos[2])

    model = builder.finalize(device=device)
    initial_z = np.array(initial_z, dtype=np.float32)

    z_history, final_q = _simulate(model, num_steps=120)

    with test.subTest(device=str(device)):
        # No box may ever rise more than a small margin above where it started.
        threshold = initial_z + 0.5
        max_z = z_history.max(axis=0)
        worst = int(np.argmax(max_z - threshold))
        test.assertTrue(
            np.all(max_z <= threshold),
            f"box {worst} reached z={max_z[worst]:.3f}, started at {initial_z[worst]:.3f} (energy injection)",
        )
        # And everything must come to rest resting on the table, not sunk through it.
        final_z = final_q[: model.body_count, 2]
        test.assertTrue(
            np.all(final_z > 0.1 + half - 0.05),
            f"a box sank through the table (min final z={final_z.min():.3f}, table top + half = {0.1 + half:.3f})",
        )


def test_box_pile_on_ground_no_energy_injection(test, device):
    """Cubes piled on a thick ground settle without being launched (box-box only).

    This isolates the box-box contact path that ``example_brick_stacking`` and
    ``example_ik_cube_stacking`` exercise: no thin table is involved, so any
    upward launch comes purely from a bad box-box ``v0`` seed.
    """
    builder = newton.ModelBuilder()
    # Thick static ground so the regression cannot come from a thin collider.
    builder.add_shape_box(
        body=-1, xform=wp.transform(p=wp.vec3(0.0, 0.0, -1.0), q=wp.quat_identity()), hx=4.0, hy=4.0, hz=1.0
    )

    half = 0.2
    rng = np.random.default_rng(3)
    initial_z = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                pos = wp.vec3(
                    float(ix - 1) * 0.42 + float(rng.random() - 0.5) * 0.05,
                    float(iy - 1) * 0.42 + float(rng.random() - 0.5) * 0.05,
                    half + 0.05 + iz * 0.45,
                )
                _add_free_box(builder, pos, half)
                initial_z.append(pos[2])

    model = builder.finalize(device=device)
    initial_z = np.array(initial_z, dtype=np.float32)

    z_history, final_q = _simulate(model, num_steps=120)

    with test.subTest(device=str(device)):
        threshold = initial_z + 0.5
        max_z = z_history.max(axis=0)
        worst = int(np.argmax(max_z - threshold))
        test.assertTrue(
            np.all(max_z <= threshold),
            f"box {worst} reached z={max_z[worst]:.3f}, started at {initial_z[worst]:.3f} (energy injection)",
        )
        # Boxes must remain a settled pile near the ground, not scattered/flung.
        final_xy = final_q[: model.body_count, :2]
        test.assertTrue(
            np.all(np.linalg.norm(final_xy, axis=1) < 3.0),
            f"a box was flung off the ground (max |xy|={np.linalg.norm(final_xy, axis=1).max():.2f})",
        )
        test.assertTrue(
            np.all(final_q[: model.body_count, 2] > half - 0.05),
            f"a box sank through the ground (min final z={final_q[: model.body_count, 2].min():.3f})",
        )


class TestThinBoxStability(unittest.TestCase):
    pass


add_function_test(
    TestThinBoxStability,
    "test_box_pile_on_thin_table_no_energy_injection",
    test_box_pile_on_thin_table_no_energy_injection,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestThinBoxStability,
    "test_box_pile_on_ground_no_energy_injection",
    test_box_pile_on_ground_no_energy_injection,
    devices=get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
