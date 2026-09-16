# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU kernels: tiny constant motion, reset/overwrite, inactive bodies, FP32 only."""

import json

import numpy as np
import warp as wp

from .relative_position_control import accumulate, publish, reset


def main():
    wp.init()
    device = "cpu"
    initial = np.array([[0.025, -0.02, 0.003]] * 3, dtype=np.float32)
    speeds = np.array([[1e-6, 2e-6, -3e-6]] * 3, dtype=np.float32)
    p = wp.array(initial, dtype=wp.vec3f, device=device)
    v = wp.array(speeds, dtype=wp.vec3f, device=device)
    birth = wp.empty_like(p)
    delta = wp.empty_like(p)
    mt = wp.array([1, 0, 1], dtype=wp.int32, device=device)
    sleep = wp.array([-1, -1, 0], dtype=wp.int32, device=device)
    h = np.float32(1 / 3600)
    naive = initial.copy()
    for outer in range(2):
        # Simulate reset/state replacement between outer graph invocations.
        if outer:
            initial += np.float32(0.1)
            p.assign(initial)
            delta.fill_(wp.vec3f(123))
        wp.launch(reset, 3, inputs=[p, birth, delta], device=device)
        expected_delta = np.zeros(3, np.float32)
        for _ in range(30):
            wp.launch(accumulate, 3, inputs=[v, mt, sleep, delta, h, 1], device=device)
            wp.launch(publish, 3, inputs=[p, birth, delta, mt, sleep, 1], device=device)
            expected_delta += speeds[0] * h
            np.testing.assert_array_equal(p.numpy()[0], initial[0] + expected_delta)
            np.testing.assert_array_equal(p.numpy()[1:], initial[1:])
            np.testing.assert_array_equal(v.numpy(), speeds)
            if not outer:
                naive[0] += speeds[0] * h
        np.testing.assert_array_equal(delta.numpy()[0], expected_delta)
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": device,
                "candidate_dtype": "float32",
                "translation_velocity_and_linear_momentum_unchanged": True,
                "static_sleeping_unchanged": True,
                "reset_overwrite_poison_pass": True,
                "scope": "CPU kernels only; actual world wrapper and CUDA capture gate pending; no angular momentum or creep claim",
            }
        )
    )


if __name__ == "__main__":
    main()
