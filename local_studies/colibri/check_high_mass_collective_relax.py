"""One-frame process-local collective relaxation diagnostic; never production."""

import argparse
import contextlib
import json
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
    args = parser.parse_args()
    scene, bottom, top = _build_scene(mass_ratio=400, setting=SolverSetting(8, 8), velocity_iterations=1, sor_boost=1)
    for _ in range(240):
        scene.step()
    world = scene.world
    cc = world._contact_container
    frame = [240]
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
            "/tmp/high_mass_relax_" + suffix + ".npz",
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
        output = "/tmp/high_mass_relax_live_" + suffix
        with patch.object(
            sys,
            "argv",
            [
                "reference",
                "--snapshot",
                "/tmp/high_mass_relax_live_before_" + suffix + ".npz",
                "--relax",
                "--output",
                output,
            ],
        ):
            with open(output + ".log", "w") as stream, contextlib.redirect_stdout(stream):
                reference_main()
        report = json.loads(Path(output + ".json").read_text())["coupled_natural_map"]
        records.append({"frame": frame[0], "substep": world._current_substep_index, **report})
        Path("/tmp/high_mass_relax_live_summary.json").write_text(json.dumps(records, indent=2))
        if not report["physical_kkt_accepted"]:
            raise RuntimeError("Rejected collective relaxation: " + str(report["max_fixedpoint_velocity_residual_m_s"]))
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
        Path("/tmp/high_mass_relax_live_phases.json").write_text(json.dumps(phase_records, indent=2))
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
                Path("/tmp/high_mass_relax_live_frames.json").write_text(json.dumps(frame_records, indent=2))
                print("Completed frame", index, row, flush=True)
    snapshot("live_final")
    print("Completed", args.frames, "frames of collective relaxation; biased integration unchanged")


if __name__ == "__main__":
    main()
