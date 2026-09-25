# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic checks for the experimental PhoenX scene tuner."""

from types import SimpleNamespace

import numpy as np

import newton
from newton.examples.phoenx import tune_phoenx
from newton.examples.phoenx.tune_phoenx import Settings, _joint_errors, candidate_settings


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values


def _one_joint(kind, body_pose, axes, dims):
    identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    model = SimpleNamespace(
        joint_count=1,
        joint_parent=_Array([-1]),
        joint_child=_Array([0]),
        joint_X_p=_Array([identity]),
        joint_X_c=_Array([identity]),
        joint_type=_Array([kind]),
        joint_axis=_Array(axes),
        joint_qd_start=_Array([0]),
        joint_dof_dim=_Array([dims]),
        joint_enabled=_Array([True]),
    )
    return _joint_errors(model, np.asarray([body_pose], dtype=float))


def test_prismatic_motion_is_not_a_joint_error():
    body = [1.0, 0.002, 0.0, 0.0, 0.0, 0.0, 1.0]
    linear, angular, flags = _one_joint(newton.JointType.PRISMATIC, body, [[1.0, 0.0, 0.0]], [1, 0])
    assert np.isclose(linear, 0.002)
    assert angular == 0.0
    assert not flags


def test_revolute_motion_is_not_a_joint_error():
    angle = 0.7
    body = [0.0, 0.0, 0.0, 0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
    linear, angular, flags = _one_joint(newton.JointType.REVOLUTE, body, [[0.0, 0.0, 1.0]], [0, 1])
    assert linear == 0.0
    assert np.isclose(angular, 0.0, atol=1.0e-7)
    assert not flags


def test_fixed_rotation_is_measured():
    angle = 0.3
    body = [0.0, 0.0, 0.0, 0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
    _, angular, _ = _one_joint(newton.JointType.FIXED, body, [], [0, 0])
    assert np.isclose(angular, angle)


def test_single_world_does_not_try_multi_world_schedule():
    base = Settings("auto", 8, 4)
    assert all(row.layout != "multi_world" for row in candidate_settings(base, 1))
    assert any(row.layout == "single_world" for row in candidate_settings(base, 2))
    assert any(row.scheduler == "block_world" for row in candidate_settings(base, 2))


def test_search_modes_and_quality_gate(monkeypatch):
    base = Settings("auto", 8, 4)
    assert len(candidate_settings(base, 2, "fast")) < len(candidate_settings(base, 2, "default"))
    assert len(candidate_settings(base, 2, "default")) < len(candidate_settings(base, 2, "thorough"))

    def fake_trial(_scene, setting, _frames, _stride):
        # Halving substeps is fast but breaks the joint; halving iterations
        # preserves it. The tuner must reject the superficially faster option.
        drift = 0.003 if setting.substeps < base.substeps else 0.001
        return {
            "settings": setting,
            "fps": 100.0 * 32 / (setting.substeps * setting.iterations),
            "metrics": {"joint_m": drift, "joint_rad": 0.0, "penetration_m": 0.0},
            "setup_s": 0.0,
            "trial_s": 0.0,
            "unscored": [],
        }

    monkeypatch.setattr(tune_phoenx, "run_trial", fake_trial)
    scene = SimpleNamespace(
        frame_dt=1 / 60,
        collision_updates=1,
        solver_options={"substeps": 8, "solver_iterations": 4},
        model=SimpleNamespace(world_count=2),
    )
    rows, _, winner = tune_phoenx.tune(scene, mode="default", frames=10, limits={"joint_m": 0.0015})
    assert any(row["fps"] > winner["fps"] and not row["passes"] for row in rows)
    assert winner["settings"].substeps == 8
    assert winner["settings"].iterations == 2
    short_rows, _, _ = tune_phoenx.tune(scene, mode="fast", frames=10, time_budget_s=0.001)
    assert len(short_rows) == 1

    def partly_unsupported(scene, setting, frames, stride):
        if setting.scheduler == "block_world":
            raise ValueError("unsupported combination")
        return fake_trial(scene, setting, frames, stride)

    monkeypatch.setattr(tune_phoenx, "run_trial", partly_unsupported)
    rows, _, winner = tune_phoenx.tune(scene, frames=10)
    assert any("unsupported combination" in row.get("error", "") for row in rows)
    assert winner is not None
