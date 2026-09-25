# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic checks for the experimental PhoenX scene tuner."""

from types import SimpleNamespace

import numpy as np
import warp as wp

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
    assert all(row.layout != "single_world" for row in candidate_settings(base, 8192))
    assert any(row.scheduler == "block_world" for row in candidate_settings(base, 2))
    assert any(row.mass_splitting for row in candidate_settings(base, 1, "fast"))
    assert {row.max_colors for row in candidate_settings(base, 2) if row.mass_splitting} == {6, 12, 16}
    assert all(not row.mass_splitting for row in candidate_settings(base, 2, toggle_mass=False))
    assert {row.threads_per_world for row in candidate_settings(base, 2, "thorough")} == {"auto", 8, 16, 32}
    assert not tune_phoenx._may_toggle_mass({"solver_scheme": "tgs"})
    assert not tune_phoenx._may_toggle_mass({"contact_friction_model": "patch"})
    assert not tune_phoenx._may_toggle_mass(
        {"joint_mode": "maximal_direct", "contact_chunk_size": 64, "enable_body_pair_grouping": False}
    )
    assert tune_phoenx._may_toggle_mass(
        {"joint_mode": "maximal_direct", "contact_chunk_size": 64, "enable_body_pair_grouping": True}
    )
    assert tune_phoenx._may_toggle_mass({"mass_splitting_color_group_size": 2})
    grouped = Settings("single_world", 8, 4, mass_splitting=True, color_group_size=2, splitting_batch_size=2)
    assert any(
        not row.mass_splitting and row.color_group_size == 0
        for row in candidate_settings(grouped, 2, "fast", allow_multi_layout=False)
    )
    thorough = candidate_settings(
        grouped,
        2,
        "thorough",
        toggle_mass=False,
        allow_ungrouped=False,
        allow_stride_two=False,
        allow_multi_layout=False,
    )
    assert {row.color_group_size for row in thorough} == {1, 2, 4}
    assert {row.splitting_batch_size for row in thorough} == {1, 2, 4}
    assert {row.prepare_refresh_stride for row in thorough} == {"auto", 1}
    assert all(row.layout == "single_world" for row in thorough)


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
    scene.solver_options["velocity_iterations"] = 2
    rows, _, _ = tune_phoenx.tune(scene, mode="fast", frames=10)
    assert rows


def test_report_compares_reference_and_recommendation():
    base = Settings("single_world", 8, 4)
    chosen = Settings("single_world", 8, 2)
    rows = [
        {
            "settings": base,
            "fps": 20.0,
            "metrics": {"joint_m": 0.001, "joint_rad": 0.01, "penetration_m": 0.002},
            "passes": True,
        },
        {
            "settings": chosen,
            "fps": 25.0,
            "metrics": {"joint_m": 0.0011, "joint_rad": 0.012, "penetration_m": 0.0025},
            "passes": True,
        },
    ]
    report = tune_phoenx.format_comparison(rows[0], rows[1])
    assert "Reference" in report and "Recommended" in report
    assert "20.0" in report and "25.0" in report
    assert "1.000" in report and "1.100" in report
    assert "2.000" in report and "2.500" in report
    assert "iterations: 4 -> 2" in report


def test_probe_passes_contact_update_dt_for_speculative_contacts():
    class Pipeline:
        def contacts(self):
            return SimpleNamespace(rigid_contact_max=0, rigid_contact_count=_Array([0]))

        def collide(self, _state, _contacts, *, dt):
            assert np.isclose(dt, 1 / 120)

    scene = SimpleNamespace(
        model=SimpleNamespace(joint_count=0, device=wp.get_device("cpu")),
        frame_dt=1 / 60,
        collision_updates=2,
        pipeline_factory=Pipeline,
        extra_metrics=None,
    )
    state = SimpleNamespace(body_q=_Array(np.zeros((0, 7))), body_qd=_Array(np.zeros((0, 6))))
    metrics, flags = tune_phoenx._Probe(scene).measure(state)
    assert metrics == {"joint_m": 0.0, "joint_rad": 0.0, "penetration_m": 0.0}
    assert not flags


def test_small_nonzero_reference_does_not_get_large_slack(monkeypatch):
    def fake_trial(_scene, setting, _frames, _stride):
        return {
            "settings": setting,
            "fps": 20.0 if setting.iterations == 4 else 30.0,
            "metrics": {
                "joint_m": 0.00012 if setting.iterations == 4 else 0.00020,
                "joint_rad": 0.0,
                "penetration_m": 0.0,
            },
            "setup_s": 0.0,
            "trial_s": 0.0,
            "unscored": [],
        }

    monkeypatch.setattr(tune_phoenx, "run_trial", fake_trial)
    scene = SimpleNamespace(
        frame_dt=1 / 60,
        collision_updates=1,
        solver_options={"substeps": 8, "solver_iterations": 4},
        model=SimpleNamespace(world_count=1),
    )
    rows, limits, winner = tune_phoenx.tune(scene, mode="fast", frames=10)
    assert np.isclose(limits["joint_m"], 0.000132)
    assert winner is rows[0]


def test_small_fps_gain_keeps_authored_setting(monkeypatch):
    def fake_trial(_scene, setting, _frames, _stride):
        return {
            "settings": setting,
            "fps": 100.0 if setting.iterations == 4 else 103.0,
            "metrics": {"joint_m": 0.001, "joint_rad": 0.0, "penetration_m": 0.0},
            "setup_s": 0.0,
            "trial_s": 0.0,
            "unscored": [],
        }

    monkeypatch.setattr(tune_phoenx, "run_trial", fake_trial)
    scene = SimpleNamespace(
        frame_dt=1 / 60,
        collision_updates=1,
        solver_options={"substeps": 8, "solver_iterations": 4},
        model=SimpleNamespace(world_count=1),
    )
    rows, _, winner = tune_phoenx.tune(scene, mode="fast", frames=10)
    assert winner is rows[0]
    rows, _, winner = tune_phoenx.tune(scene, mode="fast", frames=10, min_gain=0.02)
    assert winner["settings"].iterations == 2


def test_custom_metric_has_no_assumed_angular_floor(monkeypatch):
    def fake_trial(_scene, setting, _frames, _stride):
        return {
            "settings": setting,
            "fps": 100.0 if setting.iterations == 4 else 150.0,
            "metrics": {
                "joint_m": 0.0,
                "joint_rad": 0.0,
                "penetration_m": 0.0,
                "drive_error": 1.0e-5 if setting.iterations == 4 else 1.0e-3,
            },
            "setup_s": 0.0,
            "trial_s": 0.0,
            "unscored": [],
        }

    monkeypatch.setattr(tune_phoenx, "run_trial", fake_trial)
    scene = SimpleNamespace(
        frame_dt=1 / 60,
        collision_updates=1,
        solver_options={"substeps": 8, "solver_iterations": 4},
        model=SimpleNamespace(world_count=1),
    )
    rows, limits, winner = tune_phoenx.tune(scene, mode="fast", frames=10)
    assert np.isclose(limits["drive_error"], 1.1e-5)
    assert winner is rows[0]
