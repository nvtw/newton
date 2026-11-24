# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static friction must hold a jittering resting contact in place."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverPhoenX

_AMPLITUDE = 0.0002  # m
_FREQUENCY = 25.0  # Hz; peak acceleration 0.5 g, so nothing lifts off


@wp.kernel
def _shake_plate(
    clock: wp.array[float], dt: float, plate: int, body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector]
):
    clock[0] = clock[0] + dt
    omega = 2.0 * wp.pi * _FREQUENCY
    z = _AMPLITUDE * wp.sin(omega * clock[0])
    body_q[plate] = wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity())
    body_qd[plate] = wp.spatial_vector(wp.vec3(0.0, 0.0, _AMPLITUDE * omega * wp.cos(omega * clock[0])), wp.vec3(0.0))


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX creep test requires CUDA graph capture")
class TestVibrationCreep(unittest.TestCase):
    def test_bottle_case_holds_on_vibrating_support(self):
        """A carton of loose bottles on a support vibrating at 0.5 g must not walk.

        The heavy bottles keep the light carton jittering, so its corners open
        by microns. Those contacts must keep their friction anchors; when they
        were rebuilt on every opening, the carton crept about 0.25 mm/s.
        """
        device = wp.get_cuda_device()
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            plate = builder.add_body(mass=0.0, is_kinematic=True)
            builder.add_shape_box(plate, hx=0.5, hy=0.5, hz=0.02, cfg=builder.ShapeConfig(density=0.0, gap=0.002))
            size = np.array([0.28575, 0.2159, 0.26035])
            hx, hy, hz, wall = *(0.5 * size), 0.0016
            area = 2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2])
            case_z = 0.02 + hz
            case = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, case_z)))
            board = builder.ShapeConfig(density=0.4 / (2.0 * wall * area), gap=0.002)
            for extent, center in (
                ((hx, hy, wall), (0.0, 0.0, -hz + wall)),
                ((hx, hy, wall), (0.0, 0.0, hz - wall)),
                ((wall, hy, hz), (-hx + wall, 0.0, 0.0)),
                ((wall, hy, hz), (hx - wall, 0.0, 0.0)),
                ((hx, wall, hz), (0.0, -hy + wall, 0.0)),
                ((hx, wall, hz), (0.0, hy - wall, 0.0)),
            ):
                builder.add_shape_box(
                    case, xform=wp.transform(wp.vec3(*center)), hx=extent[0], hy=extent[1], hz=extent[2], cfg=board
                )
            radius, half_height = 0.031, 0.115
            glass = builder.ShapeConfig(density=0.55 / (math.pi * radius**2 * 2.0 * half_height), gap=0.002, mu=0.4)
            for i in range(4):
                for j in range(3):
                    bottle = builder.add_body(
                        xform=wp.transform(
                            wp.vec3(
                                (i - 1.5) * 0.0687, (j - 1.0) * 0.0687, case_z - hz + 2.0 * wall + half_height + 5.0e-4
                            )
                        )
                    )
                    builder.add_shape_cylinder(bottle, radius=radius, half_height=half_height, cfg=glass)
            model = builder.finalize()
            solver = SolverPhoenX(
                model,
                substeps=3,
                solver_iterations=4,
                temporal_iterations=True,
                contact_hertz=80.0,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                velocity_iterations=1,
            )
            state_0, state_1, control = model.state(), model.state(), model.control()
            contacts = solver._collision_pipeline.contacts()
            clock = wp.zeros(1, dtype=float)
            dt = 1.0 / 120.0

            def settle():
                nonlocal state_0, state_1
                for _ in range(2):
                    solver.collide(state_0, contacts, dt=dt)
                    solver.step(state_0, state_1, control, contacts, dt)
                    state_0, state_1 = state_1, state_0

            def shake():
                nonlocal state_0, state_1
                for _ in range(2):
                    wp.launch(_shake_plate, 1, [clock, dt, plate, state_0.body_q, state_0.body_qd])
                    solver.collide(state_0, contacts, dt=dt)
                    solver.step(state_0, state_1, control, contacts, dt)
                    state_0, state_1 = state_1, state_0

            for _ in range(60):
                settle()
            start = state_0.body_q.numpy()[case, :2].copy()
            shake()  # compile and warm every kernel before capture
            with wp.ScopedCapture() as capture:
                shake()
            for _ in range(299):
                wp.capture_launch(capture.graph)
            pose = state_0.body_q.numpy()
        self.assertTrue(np.isfinite(pose).all())
        drift = float(np.linalg.norm(pose[case, :2] - start))
        # Before the fix the carton walked 0.86 mm in these 5 s.
        self.assertLess(drift, 2.5e-4, f"carton crept {drift * 1e3:.3f} mm")


if __name__ == "__main__":
    unittest.main()
