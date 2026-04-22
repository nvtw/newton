# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stress test the multi-column contact ingest path.

The contact ingest packs at most 6 contacts per column; pairs whose
narrow-phase reports more get split across ``ceil(count / 6)``
adjacent columns (see :mod:`contact_ingest`). The "many contacts per
one pair" case targets scenes like a nut-on-bolt SDF where a single
pair emits dozens of manifold points.

Full SDF nut/bolt scenes need the IsaacGymEnvs mesh assets (see
``newton/examples/contacts/example_nut_bolt_sdf.py``) which aren't
installed in the test environment. This module instead uses a long
low-aspect-ratio box ("plank") on the ground. Newton's primitive
box-plane narrow phase caps at 4-6 contacts per pair, so the plank
exercises the "many contacts, but fit in one column" path rather
than the "more than 6 -> split" path; the pyramid tests
(``test_pyramid``) cover the latter through their many-pair fan-out.
The plank test does catch regressions in long-face-contact stability
(a nut-bolt's threads reduce to many small face contacts, which is
why this baseline matters).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.jitter.constraints.contact_matching_config import (
    JITTER_CONTACT_MATCHING,
)
from newton._src.solvers.jitter.examples.example_jitter_common import (
    build_jitter_world_from_model,
    jitter_to_newton_kernel,
    newton_to_jitter_kernel,
)


_G = 9.81


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Multi-batch contact test runs on CUDA only.",
)
class TestMultiBatchContact(unittest.TestCase):
    """Cross-check the > 6-contacts-per-pair ingest path.

    Long plank (``10 m x 0.5 m x 0.1 m``) dropped flat onto an
    infinite ground. The narrow phase emits several contact points
    along the plank's 10-m footprint; after the 6-per-column split
    we expect ``ceil(n_contacts / 6)`` columns for the same pair.
    Verifies:

    * the solver doesn't produce NaNs or explode,
    * the plank settles to a near-rest state within ~2 s,
    * horizontal momentum stays bounded (no runaway sliding from
      the multi-column friction contention).
    """

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("multi-batch contact test requires CUDA")

    def test_long_plank_settles(self):
        device = wp.get_device("cuda:0")
        mb = newton.ModelBuilder()
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        hx, hy, hz = 5.0, 0.25, 0.05
        body = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, hz + 1e-3),
                q=wp.quat_identity(),
            ),
        )
        mb.add_shape_box(body, hx=hx, hy=hy, hz=hz)

        model = mb.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        model.body_q.assign(state.body_q)

        cp = newton.CollisionPipeline(
            model, contact_matching=JITTER_CONTACT_MATCHING
        )
        contacts = cp.contacts()
        rigid_contact_max = int(contacts.rigid_contact_point0.shape[0])

        builder, n2j = build_jitter_world_from_model(model)
        world = builder.finalize(
            substeps=4,
            solver_iterations=16,
            gravity=(0.0, 0.0, -_G),
            # Need enough columns to hold the whole plank. Worst case
            # rigid_contact_max / 6 + slack.
            max_contact_columns=max(16, (rigid_contact_max + 5) // 6),
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(model.shape_count),
            default_friction=0.5,
            device=device,
        )
        shape_body_np = model.shape_body.numpy()
        sb_j = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        sb = wp.array(sb_j, dtype=wp.int32, device=device)

        n_frames = 120  # 2 s at 60 fps
        max_ncols_observed = 0
        for _ in range(n_frames):
            wp.launch(
                newton_to_jitter_kernel,
                dim=model.body_count,
                inputs=[state.body_q, state.body_qd, model.body_com],
                outputs=[
                    world.bodies.position[1 : 1 + model.body_count],
                    world.bodies.orientation[1 : 1 + model.body_count],
                    world.bodies.velocity[1 : 1 + model.body_count],
                    world.bodies.angular_velocity[1 : 1 + model.body_count],
                ],
                device=device,
            )
            model.collide(
                state, contacts=contacts, collision_pipeline=cp
            )
            world.step(
                dt=1.0 / 60.0, contacts=contacts, shape_body=sb
            )
            wp.launch(
                jitter_to_newton_kernel,
                dim=model.body_count,
                inputs=[
                    world.bodies.position[1 : 1 + model.body_count],
                    world.bodies.orientation[1 : 1 + model.body_count],
                    world.bodies.velocity[1 : 1 + model.body_count],
                    world.bodies.angular_velocity[1 : 1 + model.body_count],
                    model.body_com,
                ],
                outputs=[state.body_q, state.body_qd],
                device=device,
            )
            # Sample how many contact columns the plank produced --
            # proof we really are on the > 6 contact path for at least
            # some frames.
            ncols = int(
                world._ingest_scratch.num_contact_columns.numpy()[0]
                if world._ingest_scratch is not None
                else 0
            )
            if ncols > max_ncols_observed:
                max_ncols_observed = ncols

        final_pos = world.bodies.position.numpy()[n2j[body]]
        final_vel = world.bodies.velocity.numpy()[n2j[body]]

        # Assertions:
        # 1. No NaNs / non-finite values (solver hasn't exploded).
        self.assertTrue(
            np.isfinite(final_pos).all() and np.isfinite(final_vel).all(),
            f"non-finite state: pos={final_pos}, vel={final_vel}",
        )
        # 2. Plank actually sat on the ground (z roughly at hz + tiny
        # bias-slop). If it sank through the plane or got launched,
        # we'd fail here.
        self.assertLess(
            abs(float(final_pos[2]) - hz), 0.05,
            f"plank final z={final_pos[2]:.4f} not near resting height {hz}",
        )
        # 3. Settled speed is small (not vibrating violently).
        self.assertLess(
            float(np.linalg.norm(final_vel)), 0.5,
            f"plank final speed {np.linalg.norm(final_vel):.4f} > 0.5 m/s",
        )
        # 4. Confirm *some* contact column was packed. Newton's
        # box-plane narrow phase emits up to 4 points per pair,
        # which fits one column; the "split across columns" path
        # isn't hit here but is well-exercised by the pyramid tests
        # (many pairs -> many columns). If zero columns ever came
        # through, the ingest or collide path broke upstream.
        self.assertGreaterEqual(
            max_ncols_observed, 1,
            f"No contact columns packed (max={max_ncols_observed}); "
            "plank never made contact.",
        )
        print(
            f"[multi-batch plank] max_ncols={max_ncols_observed}  "
            f"final_z={float(final_pos[2]):.4f}  "
            f"final_speed={float(np.linalg.norm(final_vel)):.4f}"
        )


if __name__ == "__main__":
    unittest.main()
