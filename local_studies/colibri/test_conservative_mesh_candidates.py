# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise dormant mesh witnesses and an internal velocity reversal."""

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from local_studies.colibri.conservative_mesh_candidates import install_conservative_mesh_candidates


def box_separation(q):
    """Compute exact box separating-axis signed overlap, including rotation."""
    rotations = [np.asarray(wp.quat_to_matrix(wp.quat(*row[3:]))).reshape(3, 3) for row in q]
    extents = [np.array([0.05, 0.05, 0.05]), np.array([0.05, 0.15, 0.15])]
    axes = [*rotations[0].T, *rotations[1].T]
    axes.extend(np.cross(a, b) for a in rotations[0].T for b in rotations[1].T)
    gaps = []
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-8:
            continue
        unit_axis = axis / length
        radius = sum(np.abs(rotation.T @ unit_axis) @ extent for rotation, extent in zip(rotations, extents, strict=True))
        gaps.append(abs((q[1, :3] - q[0, :3]) @ unit_axis) - radius)
    return float(max(gaps))


def run_case(conservative, reverse):
    mesh = newton.Mesh.create_box(0.05, compute_normals=False, compute_uvs=False)
    mesh.build_sdf(device="cuda:0", max_resolution=64)
    wall = newton.Mesh.create_box(0.05, 0.15, 0.15, compute_normals=False, compute_uvs=False)
    wall.build_sdf(device="cuda:0", max_resolution=64)
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0001
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.6)
    normal = np.asarray(wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0)))
    for sign in (-1, 1):
        body = builder.add_body(xform=wp.transform(wp.vec3(*(sign * 0.0505 * normal)), rotation))
        builder.add_shape_mesh(
            body,
            mesh=mesh if sign < 0 else wall,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=1000.0 if sign < 0 else 1000.0 / 9.0),
        )
        builder.body_qd[body] = (sign * 0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device="cuda:0")
    physical_gaps = model.shape_gap.numpy().copy()
    original = None
    if conservative:
        original = install_conservative_mesh_candidates()
    try:
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=wp.array([[0, 1]], dtype=wp.vec2i, device=model.device),
            speculative_contact_gap_max=0.005,
            rigid_contact_max=512,
            contact_matching="sticky",
        )
    finally:
        if original is not None:
            original()
    solver = newton.solvers.SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        step_layout="single_world",
        substeps=1,
        solver_iterations=8,
        sor_boost=1.0,
    )
    state = model.state()
    initial_velocity = state.body_qd.numpy()
    initial_velocity[:, :3] = np.array([-0.2, 0.2])[:, None] * normal
    state.body_qd.assign(initial_velocity)
    contacts = pipeline.contacts()
    interval = 1.0 / 120.0
    pipeline.collide(state, contacts, dt=interval)
    count = int(contacts.rigid_contact_count.numpy()[0])
    print(
        "GENERATION",
        conservative,
        reverse,
        count,
        pipeline._shape_search_gap.numpy(),
        pipeline.narrow_phase.global_contact_reducer.contact_count.numpy(),
        pipeline.narrow_phase.global_contact_reducer.ht_insert_failures.numpy(),
        flush=True,
    )
    generated = {
        name: getattr(contacts, name).numpy()[:count].tolist()
        for name in ("rigid_contact_point0", "rigid_contact_point1", "rigid_contact_normal")
    }
    mass = model.body_mass.numpy()
    history = []
    initial_energy = None
    for step in range(8):
        if reverse and step == 2:
            velocity = state.body_qd.numpy()
            velocity[:, :3] = np.array([0.8, -0.8])[:, None] * normal
            state.body_qd.assign(velocity)
            initial_energy = float(0.5 * np.sum(mass * np.sum(velocity[:, :3] ** 2, axis=1)))
        state.clear_forces()
        solver.step(state, state, model.control(), contacts, interval / 8)
        q = state.body_q.numpy()
        qd = state.body_qd.numpy()
        linear_momentum = (mass[:, None] * qd[:, :3]).sum(axis=0)
        angular_momentum = np.cross(q[:, :3], mass[:, None] * qd[:, :3]).sum(axis=0)
        inertia = model.body_inertia.numpy()
        for body in range(2):
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*q[body, 3:]))).reshape(3, 3)
            angular_momentum += rotation @ inertia[body] @ rotation.T @ qd[body, 3:]
        energy = float(0.5 * np.sum(mass * np.sum(qd[:, :3] ** 2, axis=1)))
        for body in range(2):
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*q[body, 3:]))).reshape(3, 3)
            energy += float(0.5 * qd[body, 3:] @ rotation @ inertia[body] @ rotation.T @ qd[body, 3:])
        lambdas = solver.world._contact_container.impulses.numpy()
        history.append(
            {
                "step": step,
                "box_sat_separation": box_separation(q),
                "physical_axis_gap": float((q[1, :3] - q[0, :3]) @ normal - 0.1),
                "max_lambda": float(np.max(np.abs(lambdas))),
                "linear_momentum": linear_momentum.tolist(),
                "angular_momentum": angular_momentum.tolist(),
                "energy": energy,
            }
        )
        np.testing.assert_allclose(linear_momentum, 0.0, atol=1e-6)
        np.testing.assert_allclose(angular_momentum, 0.0, atol=1e-6)
        if not reverse or step < 2:
            np.testing.assert_array_equal(lambdas, np.zeros_like(lambdas))
            np.testing.assert_allclose(qd[:, :3], initial_velocity[:, :3], atol=1e-7)
        elif conservative:
            assert energy <= initial_energy + 1e-6, (energy, initial_energy)
    np.testing.assert_array_equal(model.shape_gap.numpy(), physical_gaps)
    return {
        "conservative": conservative,
        "reverse": reverse,
        "contacts": count,
        "generated": generated,
        "history": history,
    }


def main():
    result = [run_case(conservative, reverse) for conservative, reverse in ((False, True), (True, False), (True, True))]
    Path("/tmp/conservative_mesh_candidates.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result[0]["contacts"] == 0
    assert result[0]["history"][-1]["physical_axis_gap"] < -0.001
    assert result[1]["contacts"] > 0
    assert result[2]["contacts"] > 0
    assert min(row["box_sat_separation"] for row in result[2]["history"]) > -1e-5


if __name__ == "__main__":
    main()
