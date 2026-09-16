# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Actual canonical pose kernel versus local translation wrapper, CPU only."""

import argparse

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

from .relative_position_control import install


class Fixture:
    """Minimal world harness, using the unmodified canonical integration method."""

    _integrate_positions = PhoenXWorld._integrate_positions

    def __init__(self, device="cpu"):
        self.device = device
        self.num_bodies = 1
        self._has_maximal_dynamic_bodies = True
        self._reduced_articulation = None
        self.substep_dt = 1 / 3600
        self.bodies = body_container_zeros(1, device=self.device)
        self.bodies.position.assign(np.array([[0.025, -0.02, 0.003]], np.float32))
        self.bodies.orientation.assign(np.array([[0, 0, 0, 1]], np.float32))
        self.bodies.velocity.assign(np.array([[1e-6, 2e-6, -3e-6]], np.float32))
        self.bodies.angular_velocity.assign(np.array([[0.1, 0.2, 0.3]], np.float32))
        self.bodies.motion_type.fill_(2)
        self.bodies.island_root.fill_(-1)
        self.bodies.inverse_inertia.assign(np.diag(np.array([1, 2, 3], np.float32))[None])
        self.bodies.inverse_inertia_world.assign(np.array([[1, 2, 3, 0, 0, 0]], np.float32))

    def step(self):
        for _ in range(30):
            self._integrate_positions()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    wp.init()
    reference = Fixture(args.device)
    reference.step()
    install(Fixture)
    candidate = Fixture(args.device)
    candidate.step()
    for key in ("velocity", "angular_velocity", "orientation", "inverse_inertia_world"):
        np.testing.assert_array_equal(getattr(reference.bodies, key).numpy(), getattr(candidate.bodies, key).numpy())
    assert not np.array_equal(reference.bodies.position.numpy(), candidate.bodies.position.numpy())
    expected = candidate._relative_birth.numpy() + candidate._relative_delta.numpy()
    np.testing.assert_array_equal(candidate.bodies.position.numpy(), expected)
    candidate.bodies.position.fill_(wp.vec3f(0.1))
    candidate._relative_delta.fill_(wp.vec3f(999))
    candidate.step()
    np.testing.assert_array_equal(candidate._relative_birth.numpy(), np.full((1, 3), 0.1, np.float32))
    assert np.max(abs(candidate._relative_delta.numpy())) < 1e-6
    if args.device.startswith("cuda"):
        with wp.ScopedCapture(device=args.device) as capture:
            candidate.step()
        for value in (0.2, -0.3):
            candidate.bodies.position.fill_(wp.vec3f(value))
            candidate._relative_delta.fill_(wp.vec3f(999))
            wp.capture_launch(capture.graph)
            wp.synchronize_device(args.device)
            np.testing.assert_array_equal(candidate._relative_birth.numpy(), np.full((1, 3), value, np.float32))
            assert np.max(abs(candidate._relative_delta.numpy())) < 1e-6
        print("PASS CUDA capture replay: device reset reads each new supplied position; poisoned scratch cleared")
    print(
        "PASS actual canonical pose kernel: q, omega, v, inertia byte-identical; translation differs; reset excludes poisoned history"
    )


if __name__ == "__main__":
    main()
