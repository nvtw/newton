"""Saved-mode continuation of the process-local physical relaxation oracle."""

import argparse
import contextlib
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from local_studies.colibri.reference_high_mass_connected_normals import main as reference_main
from newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio import SolverSetting, _build_scene
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from newton._src.solvers.phoenx.tests.test_high_mass_ratio import GRAVITY, _plane_pair_fz_to_body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--output-prefix", default="/tmp/high_mass_mode_continuation_")
    args = parser.parse_args()
    output_prefix = args.output_prefix
    scene, bottom, top = _build_scene(mass_ratio=400, setting=SolverSetting(8, 8), velocity_iterations=1, sor_boost=1)
    for _ in range(240):
        scene.step()
    world = scene.world
    cc = world._contact_container
    frame = [240]
    last_mode = [None]
    records = []
    phase_records = []
    frame_records = []

    def invariants():
        inv_m = world.bodies.inverse_mass.numpy().astype(float)
        active = inv_m > 0
        positions = world.bodies.position.numpy().astype(float)
        v = world.bodies.velocity.numpy().astype(float)
        w = world.bodies.angular_velocity.numpy().astype(float)
        inertia = np.linalg.inv(
            inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy())[active].astype(float)
        )
        momentum = v[active] / inv_m[active, None]
        angular = np.einsum("bij,bj->bi", inertia, w[active])
        energy = 0.5 * np.sum(momentum * v[active]) + 0.5 * np.sum(angular * w[active])
        return {
            "energy_J": float(energy),
            "P": momentum.sum(0).tolist(),
            "L": (angular + np.cross(positions[active], momentum)).sum(0).tolist(),
        }

    def snapshot(suffix):
        n = int(world._cc_valid_count.numpy()[0])
        np.savez(
            output_prefix + "" + suffix + ".npz",
            positions=world.bodies.position.numpy(),
            velocity=world.bodies.velocity.numpy(),
            angular_velocity=world.bodies.angular_velocity.numpy(),
            inverse_mass=world.bodies.inverse_mass.numpy(),
            inverse_inertia=inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy()),
            impulses=cc.impulses.numpy()[:, :n],
            derived=cc.derived.numpy()[:, :n],
            lambdas=cc.lambdas.numpy()[:, :n],
            headers=world._contact_cols.data.numpy(),
            column_count=world._ingest_scratch.num_contact_columns.numpy(),
        )

    dispatcher = type(world._dispatcher)
    original = dispatcher.relax
    original_solve = dispatcher.solve

    def solve(self, idt):
        before = invariants() if frame[0] >= 244 else None
        original_solve(self, idt)
        if before is not None:
            phase_records.append(
                {
                    "frame": frame[0],
                    "substep": world._current_substep_index,
                    "phase": "biased_solve",
                    "before": before,
                    "after": invariants(),
                }
            )

    def wrapped(self, idt):
        if frame[0] < 244:
            original(self, idt)
            return
        before_position = world.bodies.position.numpy()
        before_invariants = invariants()
        suffix = str(frame[0]) + "_" + str(world._current_substep_index)
        snapshot("live_before_" + suffix)
        output = output_prefix + "live_" + suffix
        with patch.object(
            sys,
            "argv",
            [
                "reference",
                "--snapshot",
                output_prefix + "live_before_" + suffix + ".npz",
                "--relax",
                "--output",
                output,
                *(
                    ["--seed", "/tmp/high_mass_relax_245_1_mode_solution.npz"]
                    if frame[0] == 245 and world._current_substep_index == 1
                    else []
                ),
            ],
        ):
            with open(output + ".log", "w") as stream, contextlib.redirect_stdout(stream):
                reference_main()
        report = json.loads(Path(output + ".json").read_text())["coupled_natural_map"]
        if not report["physical_kkt_accepted"]:
            mode_output = output + "_mode"
            mode_args = ["modes", "--snapshot", output + ".npz", "--output", mode_output]
            if last_mode[0]:
                mode_args += ["--prior-mode", last_mode[0]]
            for offset in (0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi):
                with patch.object(sys, "argv", [*mode_args, "--direction-offset", str(offset)]):
                    with open(mode_output + ".log", "a") as stream, contextlib.redirect_stdout(stream):
                        runpy.run_module("local_studies.colibri.reference_coulomb_modes_general", run_name="__main__")
                mode_report = json.loads(Path(mode_output + ".json").read_text())
                if mode_report["accepted"]:
                    break
            if mode_report["accepted"]:
                last_mode[0] = mode_output + ".json"
                with patch.object(
                    sys,
                    "argv",
                    [
                        "reference",
                        "--snapshot",
                        output_prefix + "live_before_" + suffix + ".npz",
                        "--relax",
                        "--output",
                        output,
                        "--seed",
                        mode_output + ".npz",
                    ],
                ):
                    with open(output + "_seed.log", "w") as stream, contextlib.redirect_stdout(stream):
                        reference_main()
                report = json.loads(Path(output + ".json").read_text())["coupled_natural_map"]
        records.append({"frame": frame[0], "substep": world._current_substep_index, **report})
        Path(output_prefix + "live_summary.json").write_text(json.dumps(records, indent=2))
        if not report["physical_kkt_accepted"]:
            raise RuntimeError("Rejected collective relaxation: " + str(report["max_fixedpoint_velocity_residual_m_s"]))
        if report["physical_contact_work_J"] > 1e-8:
            raise RuntimeError("Rejected energy-increasing physical relaxation")
        result = np.load(output + ".npz")
        velocity = result["updated_velocity"].astype(np.float32)
        world.bodies.velocity.assign(velocity[:, :3].copy())
        world.bodies.angular_velocity.assign(velocity[:, 3:].copy())
        impulses = cc.impulses.numpy()
        impulses[:, result["selected_points"]] = result["coupled_solution"].reshape(-1, 3).T
        cc.impulses.assign(impulses)
        assert np.array_equal(before_position, world.bodies.position.numpy())
        phase_records.append(
            {
                "frame": frame[0],
                "substep": world._current_substep_index,
                "phase": "relax",
                "before": before_invariants,
                "after": invariants(),
                "reaction_audit": report["conservation"],
                "physical_contact_work_J": report["physical_contact_work_J"],
            }
        )
        Path(output_prefix + "live_phases.json").write_text(json.dumps(phase_records, indent=2))
        print(
            "Accepted substep",
            world._current_substep_index,
            "KKT",
            report["max_fixedpoint_velocity_residual_m_s"],
            flush=True,
        )

    with patch.object(dispatcher, "relax", wrapped), patch.object(dispatcher, "solve", solve):
        for index in range(240, 244 + args.frames):
            frame[0] = index
            scene._simulate()
            if index >= 244:
                force_top, _, _ = scene.gather_contact_wrench_on_body(top)
                row = {
                    "frame": index,
                    "bottom_speed_m_s": float(np.linalg.norm(scene.body_velocity(bottom))),
                    "top_speed_m_s": float(np.linalg.norm(scene.body_velocity(top))),
                    "separation_m": float(scene.body_position(top)[2] - scene.body_position(bottom)[2]),
                    "top_load_N": float(force_top[2]),
                    "plane_load_N": float(_plane_pair_fz_to_body(scene, bottom)),
                    "expected_top_load_N": 400 * GRAVITY,
                    "expected_plane_load_N": 401 * GRAVITY,
                }
                frame_records.append(row)
                Path(output_prefix + "live_frames.json").write_text(
                    json.dumps(frame_records, indent=2)
                )
                print("Completed frame", index, row, flush=True)
    snapshot("live_final")
    print("Completed", args.frames, "frames of collective relaxation; biased integration unchanged")


if __name__ == "__main__":
    main()
