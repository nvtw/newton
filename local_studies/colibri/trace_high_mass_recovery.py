"""Observe phase energy and contact-recovery history without changing equations."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio import SolverSetting, _build_scene
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np
from newton._src.solvers.phoenx.constraints import constraint_contact as schema


def main():
    scene, _, _ = _build_scene(mass_ratio=400, setting=SolverSetting(8, 8), velocity_iterations=1, sor_boost=1)
    for _ in range(240):
        scene.step()
    world = scene.world
    cc = world._contact_container
    records = []
    current_frame = [240]
    inv_m = world.bodies.inverse_mass.numpy().astype(np.float64)
    active = inv_m > 0

    def snapshot():
        v = world.bodies.velocity.numpy().astype(np.float64)
        w = world.bodies.angular_velocity.numpy().astype(np.float64)
        inv_i = inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy()).astype(np.float64)
        energy = 0.5 * np.sum(v[active] ** 2 / inv_m[active, None]) + 0.5 * np.einsum(
            "bi,bij,bj", w[active], np.linalg.inv(inv_i[active]), w[active]
        )
        n = int(world._cc_valid_count.numpy()[0])
        derived = cc.derived.numpy()[:, :n].astype(np.float64)
        references = cc.lambdas.numpy()[:, :n].astype(np.float64)
        impulses = cc.impulses.numpy()[:, :n].astype(np.float64)
        headers = world._contact_cols.data.numpy().view(np.int32)
        normal_velocity = []
        for cid in range(int(world._ingest_scratch.num_contact_columns.numpy()[0])):
            b0, b1 = headers[int(schema._OFF_BODY1), cid], headers[int(schema._OFF_BODY2), cid]
            first, count = headers[int(schema._OFF_CONTACT_FIRST), cid], headers[int(schema._OFF_CONTACT_COUNT), cid]
            for k in range(first, first + count):
                rel = v[b1] + np.cross(w[b1], derived[12:15, k]) - v[b0] - np.cross(w[b0], derived[9:12, k])
                normal_velocity.append(float(references[:3, k] @ rel))
        return {
            "energy": float(energy),
            "impulses": impulses,
            "derived": derived,
            "references": references,
            "min_normal_velocity": min(normal_velocity, default=0.0),
            "normal_velocity": np.asarray(normal_velocity),
        }

    dispatcher = type(world._dispatcher)
    original_solve, original_relax = dispatcher.solve, dispatcher.relax

    def observe(original, phase):
        def wrapped(self, idt):
            before = snapshot()
            original(self, idt)
            after = snapshot()
            impulses, derived = after["impulses"], after["derived"]
            normal = impulses[0]
            residual = after["normal_velocity"].copy()
            if phase == "solve":
                residual += derived[3]
                penetrating = derived[3] <= 0.0
                residual[penetrating] += (
                    (0.05829954519867897 / 0.9417003989219666) * normal[penetrating] / derived[0, penetrating]
                )
            else:
                residual[derived[3] > 0.0] = 0.0
            projected_residual = np.where(normal > 1.0e-8, np.abs(residual), np.maximum(-residual, 0.0))
            tangent = np.linalg.norm(impulses[1:3], axis=0)
            records.append(
                {
                    "frame": current_frame[0],
                    "substep": int(world._current_substep_index),
                    "phase": phase,
                    "energy_before_J": before["energy"],
                    "energy_after_J": after["energy"],
                    "energy_delta_J": after["energy"] - before["energy"],
                    "max_normal_fixedpoint_residual_m_s": float(np.max(projected_residual, initial=0.0)),
                    "min_normal_velocity_m_s": after["min_normal_velocity"],
                    "max_normal_bias_m_s": float(np.max(np.abs(derived[3]), initial=0)),
                    "max_tangent_bias_m_s": float(np.max(np.linalg.norm(derived[4:6], axis=0), initial=0)),
                    "normal_bias_impulse_dot_J": float(np.sum(normal * derived[3])),
                    "tangent_bias_impulse_dot_J": float(np.sum(impulses[1:3] * derived[4:6])),
                    "saturated_fraction": float(np.mean((normal > 1e-8) & (tangent >= 0.98 * 0.5 * normal))),
                    "changed_material_anchors": int(
                        np.count_nonzero(np.any(before["references"][6:12] != after["references"][6:12], axis=0))
                    ),
                }
            )

        return wrapped

    with (
        patch.object(dispatcher, "solve", observe(original_solve, "solve")),
        patch.object(dispatcher, "relax", observe(original_relax, "relax")),
    ):
        for frame in range(240, 256):
            current_frame[0] = frame
            scene._simulate()
    Path("/tmp/high_mass_recovery_trace.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            {
                "records": len(records),
                "energy_delta_by_phase": {
                    phase: sum(r["energy_delta_J"] for r in records if r["phase"] == phase)
                    for phase in ("solve", "relax")
                },
                "max_tangent_bias": max(r["max_tangent_bias_m_s"] for r in records),
                "max_normal_bias": max(r["max_normal_bias_m_s"] for r in records),
                "min_normal_velocity": min(r["min_normal_velocity_m_s"] for r in records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
