# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the exact implicit-PD drive in the maximal tree projector.

``PHOENX_MAXIMAL_IMPLICIT_DRIVE`` (commit 954f3e3e) folds the implicit PD drive
into the maximal projector's O(tree) recursion and suppresses the redundant PGS
drive row. The property this pins: with the flag ON, a position-driven
free-root articulation solved in ``maximal_projected`` mode tracks the drive
target as crisply as the reduced-coordinate reference at only 2 solver
iterations, and markedly better than with the flag OFF (soft drive).

The flag is baked at import time: ``constraint_joint`` reads it through
``wp.static`` (so the suppressed PGS drive row is a compile-time constant) and
the projector reads it into a module global. The ON case therefore runs in a
subprocess with the environment variable set before Newton is imported; the OFF
case runs in-process (the default). If the implicit-drive term were removed, the
ON tracking gap would collapse back to the soft OFF behaviour and the tight
bound below would fail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

import numpy as np
import warp as wp

import newton

_FRAMES = 90  # each frame advances two 1/60 s steps
_N_HINGES = 5  # free root + 5 driven hinges = 6-body limb (matches the commit)
_DRIVE_KE = 200.0  # moderate: stable, yet the soft PGS row lags along the chain
_DRIVE_KD = 6.0
_HINGE_FREQ_HZ = 0.9  # gentle continuous motion: stable and mostly-tracking
_HINGE_AMPLITUDE = 0.4  # radians
_RESULT_MARKER = "MAXIMAL_IMPLICIT_DRIVE_RESULT:"


def _make_driven_limb(device):
    """Fixed-base serial chain with a position drive on every hinge.

    The implicit-PD drive folds the drive impedance into the projector's
    articulated-inertia recursion, so a multi-link chain amplifies the advantage
    over the soft PGS drive row (which cannot propagate that impedance to the
    parent within 2 iterations). A fixed base (rather than a free root) keeps the
    reference stable at 2 iterations so the comparison is a clean tracking-lag
    signal, not divergence of a tumbling base.
    """
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    base = builder.add_link(is_kinematic=True, mass=2.0)
    builder.add_shape_box(base, hx=0.25, hy=0.12, hz=0.1)
    base_joint = builder.add_joint_fixed(parent=-1, child=base)
    parent = base
    hinges = []
    axes = (newton.Axis.Y, newton.Axis.Z)
    for i in range(_N_HINGES):
        child = builder.add_link(mass=1.0 / (1.0 + 0.3 * i))
        builder.add_shape_box(child, hx=0.18, hy=0.09, hz=0.07)
        hinge = builder.add_joint_revolute(
            parent=parent,
            child=child,
            axis=axes[i % 2],
            parent_xform=wp.transform(wp.vec3(0.32, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.22, 0.0, 0.0), wp.quat_identity()),
            target_ke=_DRIVE_KE,
            target_kd=_DRIVE_KD,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        hinges.append(hinge)
        parent = child
    builder.add_articulation([base_joint, *hinges])
    return builder.finalize(device=device), hinges


def _target_schedule(n_hinges: int) -> np.ndarray:
    """Deterministic gentle sinusoidal position targets, shape ``[frames, hinges]``.

    The soft-vs-crisp drive difference is a *dynamic-tracking* effect: at a
    settled constant target both drives reach the same steady state and the gap
    washes out, while under violent motion exact-angle matching becomes
    chaotically sensitive. A gentle continuous sinusoid keeps the drive in
    steady motion — the reference and the crisp implicit drive track it closely,
    while the softer OFF drive sustains a visible lag along the whole chain.
    """
    frame_dt = 2.0 / 60.0
    t = np.arange(_FRAMES, dtype=np.float64) * frame_dt
    columns = []
    for i in range(n_hinges):
        sign = 1.0 if i % 2 == 0 else -1.0
        phase = 2.0 * np.pi * _HINGE_FREQ_HZ * t + 0.5 * float(i)
        columns.append(sign * _HINGE_AMPLITUDE * np.sin(phase))
    return np.stack(columns, axis=1).astype(np.float32)


def _measure_tracking(device) -> dict[str, list[list[float]]]:
    """Return the per-frame hinge-angle trajectory for the reduced reference and
    the ``maximal_projected`` solve, both driven by the same moving targets.

    The reduced path folds the PD drive into its own factorization, so it is
    unaffected by ``PHOENX_MAXIMAL_IMPLICIT_DRIVE`` and serves as the ground
    truth. Whether ``maximal_projected`` tracks it crisply depends entirely on
    whether the implicit-drive term is active.
    """
    results: dict[str, list[list[float]]] = {}
    for mode in ("reduced", "maximal_projected"):
        model, hinges = _make_driven_limb(device)
        schedule = _target_schedule(len(hinges))
        state0 = model.state()
        state1 = model.state()
        newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)

        control = model.control()
        qd_start = model.joint_qd_start.numpy()
        q_start = model.joint_q_start.numpy()
        hinge_qd = [int(qd_start[j]) for j in hinges]
        hinge_q = [int(q_start[j]) for j in hinges]
        target_buf = np.zeros(model.joint_dof_count, dtype=np.float32)

        solver = newton.solvers.SolverPhoenX(
            model,
            articulation_mode=mode,
            substeps=1,
            solver_iterations=2,
            velocity_iterations=1,
        )
        dt = 1.0 / 60.0
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state0, state1, control, None, dt)
            solver.step(state1, state0, control, None, dt)

        trajectory: list[list[float]] = []
        for frame_targets in schedule:
            for coord, target in zip(hinge_qd, frame_targets, strict=False):
                target_buf[coord] = target
            control.joint_target_q.assign(target_buf)
            wp.capture_launch(capture.graph)
            final_q = state0.joint_q.numpy()
            trajectory.append([float(final_q[c]) for c in hinge_q])
        results[mode] = trajectory
    return results


def _settled_lag_rms(result: dict[str, list[list[float]]]) -> float:
    """RMS deviation of ``maximal_projected`` from the reduced reference over the
    sustained-motion window (the startup ramp is skipped).

    Under gentle continuous motion the softer OFF drive sustains a lag along the
    chain, so its RMS deviation from reduced is large; the crisp implicit drive
    tracks reduced tightly. The RMS averages out per-frame transient noise into
    a clean tracking-lag measure.
    """
    reduced = np.asarray(result["reduced"], dtype=np.float64)
    maximal = np.asarray(result["maximal_projected"], dtype=np.float64)
    diff = maximal[12:] - reduced[12:]
    return float(np.sqrt(np.mean(diff**2)))


class TestMaximalImplicitDrive(unittest.TestCase):
    def test_implicit_drive_crispens_maximal_projected_tracking(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("maximal implicit-drive test requires CUDA graph capture")
        if os.environ.get("PHOENX_MAXIMAL_IMPLICIT_DRIVE", "0").lower() not in ("0", "", "false", "off"):
            self.skipTest("PHOENX_MAXIMAL_IMPLICIT_DRIVE already set in ambient env; cannot measure OFF baseline")

        # OFF baseline (in-process default): soft PGS drive.
        off = _measure_tracking(device)
        gap_off = _settled_lag_rms(off)

        # ON case in a subprocess so the flag is baked at import time.
        env = dict(os.environ)
        env["PHOENX_MAXIMAL_IMPLICIT_DRIVE"] = "1"
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--measure"],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        marker_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_RESULT_MARKER)]
        if proc.returncode != 0 or not marker_lines:
            self.fail(
                "implicit-drive measurement subprocess failed\n"
                f"returncode={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\n"
                f"stderr tail:\n{proc.stderr[-2000:]}"
            )
        payload = json.loads(marker_lines[-1][len(_RESULT_MARKER) :])
        if payload.get("skipped"):
            self.skipTest(f"subprocess skipped: {payload['skipped']}")
        on = {"reduced": payload["reduced"], "maximal_projected": payload["maximal_projected"]}
        gap_on = _settled_lag_rms(on)

        ratio = gap_off / max(gap_on, 1.0e-9)
        print(f"[implicit-drive] settled-lag RMS: off={gap_off:.6f} on={gap_on:.6f} rad  ratio={ratio:.2f}x")

        # The reduced reference must be identical up to noise: it folds the PD
        # drive into its own factorization and is flag-independent.
        np.testing.assert_allclose(
            np.asarray(on["reduced"], dtype=np.float64),
            np.asarray(off["reduced"], dtype=np.float64),
            atol=2.0e-3,
        )

        # ``gap_off`` is itself the "without the feature" measurement: the soft
        # PGS drive. It lags the reference badly.
        self.assertGreater(gap_off, 4.0e-2)
        # With the implicit drive ON the maximal projector tracks the reduced
        # reference far more crisply (measured ~0.016 vs ~0.096 rad, ~6x).
        self.assertLess(gap_on, 3.0e-2)
        # Removing the implicit-drive term collapses ON onto the OFF behaviour,
        # which fails both the bound above and this ratio.
        self.assertGreater(ratio, 3.0)


def _run_measurement_subprocess() -> int:
    device = wp.get_preferred_device()
    if not device.is_cuda:
        print(_RESULT_MARKER + json.dumps({"skipped": "no CUDA device"}))
        return 0
    result = _measure_tracking(device)
    print(_RESULT_MARKER + json.dumps(result))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--measure":
        sys.exit(_run_measurement_subprocess())
    unittest.main()
