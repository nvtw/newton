# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Speculative-contact regression test for :class:`PhoenXWorld`.

Drops a sphere onto a ground plane from a height that lets Newton's
:class:`CollisionPipeline` generate speculative contacts (contacts with
a positive gap, emitted by the narrow phase ahead of actual impact so
the solver can resolve the closing motion in one substep). A correct
speculative-contact formulation must leave a separating body in
free-fall -- it should only fire when the body would actually cross the
surface within a single substep and cap closing velocity to
``gap / dt``. A soft-spring formulation -- for example
``bias = gap * bias_rate`` with the Nyquist-capped ``bias_rate ~ 0.6 /
dt`` -- creates a virtual wall at the speculative gap distance and
decelerates the body through honey-like drag while still separated.

The test samples velocity across the window where speculative contacts
exist but the body has not yet penetrated, and asserts that the
deviation from analytic free-fall stays below a tight threshold. A
regression to the honey-spring formulation drops measured velocity
tens of percent below the ``-g * t`` curve and trips the threshold
well before the body reaches the surface.
"""

from __future__ import annotations

import unittest

import warp as wp

from newton._src.solvers.phoenx.examples.example_rabbit_pile import (
    DEFAULT_SHAPE_GAP,
)
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

_G = 9.81


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX speculative-contact test runs on CUDA only.",
)
class TestPhoenXSpeculativeContactNoDrag(unittest.TestCase):
    """A free-falling sphere must stay in free-fall while speculative
    contacts exist but the sphere has not yet touched the ground.

    The sphere drops from ~1.5 m above the plane with the PhoenX-default
    contact gap (5 cm). Newton's narrow phase starts emitting contacts
    once the sphere is within the detection shell; the PGS solver must
    leave the sphere's velocity untouched until the sphere physically
    crosses the surface. We check every frame in which contacts exist
    but the sphere is still separated -- the frame-over-frame velocity
    change must match ``-g * dt`` within a few percent.
    """

    def test_sphere_free_falls_through_speculative_zone(self) -> None:
        # Run at 2400 Hz with 1 substep per frame so the speculative
        # zone (5 cm of detection shell + ~6 m/s impact velocity = one
        # frame is ~2.5 mm of closing motion) spans tens of frames
        # and we can sample deep inside the window where a correct
        # ``bias = gap * inv_h`` formulation applies zero impulse but
        # the honey-spring ``bias = gap * bias_rate`` formulation
        # still leaks a visible per-frame deceleration. Lower
        # frame-rates produce ~1-frame-wide discriminating windows
        # that alias past the sampler.
        fps = 2400
        # ``velocity_iterations = 1`` is PhoenX's default (matches
        # ``PhoenXWorld``'s own default). Exercising the relax pass is
        # essential: Box2D's ``s > 0`` speculative branch must run
        # during the relax sweep too. A regression that zeros the
        # speculative bias (or flips coefficients to soft) during relax
        # applies a large brake impulse to any closing body in the
        # speculative window -- the main honey artefact on the rabbit
        # pile. ``_PhoenXScene`` defaults to ``velocity_iterations = 1``
        # (the new minimum after the soft-PD damping split); the
        # explicit value here documents intent.
        scene = _PhoenXScene(
            fps=fps,
            substeps=1,
            solver_iterations=16,
            velocity_iterations=1,
            friction=0.5,
        )
        scene.add_ground_plane()
        radius = 0.1
        initial_z = 2.0
        body = scene.add_sphere(
            position=(0.0, 0.0, initial_z),
            radius=radius,
            density=1000.0,
        )
        # Ensure the sphere inherits the PhoenX-default contact gap so
        # Newton's narrow phase emits speculative contacts before the
        # sphere actually touches the plane.
        scene.mb.default_shape_cfg.gap = DEFAULT_SHAPE_GAP
        scene.finalize()

        dt = 1.0 / fps
        expected_delta_v = -_G * dt

        speculative_samples: list[tuple[float, float, int, float]] = []
        prev_v_z: float | None = None
        prev_lowest_z: float | None = None

        # Run until the sphere physically penetrates the plane. The
        # *pure speculative* window is the set of frames where contacts
        # already exist but the sphere would still strictly clear the
        # surface even under gravity alone during the upcoming frame --
        # i.e. ``lowest_z > |v_z| * dt + safety``. In that window a
        # well-formed solver applies no impulse (Box2D ``s * inv_h``
        # with rigid coefficients stays zero because ``-v_n < s / dt``);
        # the honey-spring bug decelerates the body anyway and trips
        # the tolerance below.
        safety_margin = _G * dt * dt  # one sub-millimetre per frame at
        # 2400 Hz; gravity over one frame -- keeps the predicate off
        # the edge where the sphere genuinely needs the "cap closing
        # at gap / dt" impulse in the last substep before contact.
        for step in range(6000):
            scene.step()
            t = (step + 1) * dt
            v_z = float(scene.body_velocity(body)[2])
            pos_z = float(scene.body_position(body)[2])
            lowest_z = pos_z - radius
            n = int(scene.contacts.rigid_contact_count.numpy()[0])

            if (
                prev_v_z is not None
                and prev_lowest_z is not None
                and n > 0
                and prev_lowest_z > abs(prev_v_z) * dt + safety_margin
            ):
                # Pure speculative frame: at the start of this step the
                # sphere had a gap larger than one frame of closing
                # motion would consume, so a correct speculative row
                # fires with zero impulse and ``delta_v`` must equal
                # ``g * dt`` exactly.
                speculative_samples.append((t, v_z, n, prev_v_z - v_z))

            prev_v_z = v_z
            prev_lowest_z = lowest_z
            if lowest_z <= 0.0:
                # Sphere reached / crossed the surface -- stop before
                # legitimate contact impulses contaminate the sample.
                break

        self.assertGreater(
            len(speculative_samples),
            0,
            "Sphere never saw a speculative contact; widen the drop distance or shrink ``DEFAULT_SHAPE_GAP``.",
        )

        # ``delta = prev_v - current_v``. Under gravity alone this
        # equals ``-expected_delta_v = g * dt``. A soft-spring speculative
        # bug applies an *upward* braking impulse that makes velocity
        # less negative than free-fall, so ``delta`` becomes *smaller*
        # than ``g * dt`` (or even negative if the brake overshoots
        # gravity). We detect both signs of deviation.
        worst_frame_deviation = 0.0
        for _t, _v_z, _n, delta in speculative_samples:
            deviation = abs(delta - (-expected_delta_v))
            if deviation > worst_frame_deviation:
                worst_frame_deviation = deviation

        # Tolerance: 1 mm/s per frame ~ 2.5 % of ``g * dt``
        # (g * dt ~ 41 mm/s at 240 Hz). The honey-spring bug produced
        # per-frame velocity changes that differ from ``g * dt`` by
        # hundreds of mm/s -- two orders of magnitude above this
        # threshold -- so the test trips as soon as the fix is reverted
        # while staying quiet for correct behaviour (FP noise is
        # several orders below 1 mm/s).
        tolerance = 0.001  # m/s per frame
        self.assertLess(
            worst_frame_deviation,
            tolerance,
            f"speculative contacts perturbed a separated body away "
            f"from free-fall: worst per-frame velocity deviation from "
            f"``g * dt`` = {worst_frame_deviation:.6f} m/s "
            f"(tolerance {tolerance:.6f} m/s). "
            f"Samples (t, v_z, n, delta_v): {speculative_samples[:10]}",
        )


if __name__ == "__main__":
    unittest.main()
