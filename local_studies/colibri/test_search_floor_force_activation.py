"""CPU-only force-pulse detection and physical activation control."""

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from local_studies.colibri.probe_gear_search_floor import floor_search
from newton._src.sim.collide import compute_shape_velocities


def run_case(floor, pulse):
    """Generate once at rest, then apply a paired central force pulse."""
    device = "cpu"
    axis = np.array([0.8, 0.6, 0.0])
    origin = np.array([-0.2, 0.3, 0.4])
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0001
    for sign in (-1, 1):
        body = builder.add_body(xform=wp.transform(wp.vec3(*(origin + sign * 0.0505 * axis)), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5))
    model = builder.finalize(device=device)
    physical = model.shape_gap.numpy().copy()
    margin = model.shape_margin.numpy().copy()
    pipeline = newton.CollisionPipeline(
        model,
        speculative_contact_gap_max=0.005,
        speculative_contact_velocity_filter=False,
        rigid_contact_max=32,
        contact_matching="sticky",
    )
    solver = newton.solvers.SolverPhoenX(
        model, collision_pipeline=pipeline, step_layout="single_world", substeps=1, solver_iterations=8, sor_boost=1.0
    )
    state = model.state()
    contacts = pipeline.contacts()
    old = wp.launch

    def launch(*args, **kwargs):
        result = old(*args, **kwargs)
        kernel = kwargs.get("kernel", args[0] if args else None)
        if floor and kernel is compute_shape_velocities:
            old(
                floor_search,
                model.shape_count,
                inputs=[model.shape_body, model.shape_gap, pipeline._shape_search_gap, floor],
                device=device,
            )
        return result

    wp.launch = launch
    interval = 1.0 / 120
    try:
        pipeline.collide(state, contacts, dt=interval)
    finally:
        wp.launch = old
    count = int(contacts.rigid_contact_count.numpy()[0])
    expected_count = 1 if floor else 0
    assert count == expected_count, (floor, count)
    np.testing.assert_array_equal(state.body_qd.numpy(), np.zeros((2, 6)))
    mass = model.body_mass.numpy().astype(float)
    dt = interval / 8
    rows = []
    work = 0.0
    initial_geometry = state.body_q.numpy().copy()
    for step in range(8):
        before = state.body_qd.numpy().astype(float)
        force = np.zeros((2, 6), dtype=np.float32)
        if pulse and step == 2:
            force[:, :3] = np.array([1.0, -1.0])[:, None] * axis * mass[:, None] * (0.8 / dt)
        state.clear_forces()
        state.body_f.assign(force)
        kick = dt * force[:, :3].astype(float)
        free = before[:, :3] + kick / mass[:, None]
        step_work = float(np.sum(kick * (before[:, :3] + free) * 0.5))
        work += step_work
        solver.step(state, state, model.control(), contacts, dt)
        q, velocity = state.body_q.numpy().astype(float), state.body_qd.numpy().astype(float)
        momentum = mass[:, None] * velocity[:, :3]
        angular = np.cross(q[:, :3], momentum).sum(axis=0)
        inertia = model.body_inertia.numpy().astype(float)
        energy = 0.5 * np.sum(mass[:, None] * velocity[:, :3] ** 2)
        for body in range(2):
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*q[body, 3:])), dtype=float).reshape(3, 3)
            world_inertia = rotation @ inertia[body] @ rotation.T
            angular += world_inertia @ velocity[body, 3:]
            energy += 0.5 * velocity[body, 3:] @ world_inertia @ velocity[body, 3:]
        p = momentum.sum(axis=0)
        impulse = solver.world._contact_container.impulses.numpy()
        maximum = float(np.max(np.abs(impulse))) if impulse.size else 0.0
        residual_impulse = mass[:, None] * (velocity[:, :3] - before[:, :3]) - kick
        if count:
            basis = solver.world._contact_container.lambdas.numpy()
            n, t = basis[:3, 0].astype(float), basis[3:6, 0].astype(float)
            contact_impulse = impulse[0, 0] * n + impulse[1, 0] * t + impulse[2, 0] * np.cross(n, t)
            np.testing.assert_allclose(residual_impulse[0], -contact_impulse, atol=2e-7)
            np.testing.assert_allclose(residual_impulse[1], contact_impulse, atol=2e-7)
        free_energy = 0.5 * np.sum(mass[:, None] * free**2)
        contact_work = float(np.sum(residual_impulse * (free + velocity[:, :3]) * 0.5))
        for body in range(2):
            # Spherical inertia is isotropic, so rotation does not change it.
            free_energy += 0.5 * before[body, 3:] @ inertia[body] @ before[body, 3:]
            torque_impulse = inertia[body] @ (velocity[body, 3:] - before[body, 3:])
            contact_work += float(0.5 * (velocity[body, 3:] + before[body, 3:]) @ torque_impulse)
        assert abs(float(energy) - free_energy - contact_work) < 2e-7
        assert contact_work <= 2e-7, (step, contact_work)
        np.testing.assert_allclose(p, 0.0, atol=2e-7)
        np.testing.assert_allclose(angular, 0.0, atol=2e-7)
        np.testing.assert_allclose(residual_impulse.sum(axis=0), 0.0, atol=2e-7)
        assert energy <= work + 2e-7, (step, energy, work)
        if not pulse or step < 2:
            assert maximum == 0
            np.testing.assert_array_equal(state.body_q.numpy(), initial_geometry)
            np.testing.assert_array_equal(state.body_qd.numpy(), np.zeros((2, 6)))
        separation = float(np.linalg.norm(q[1, :3] - q[0, :3]) - 0.1)
        rows.append(
            {
                "step": step,
                "separation": separation,
                "max_impulse": maximum,
                "linear_momentum": p.tolist(),
                "angular_momentum": angular.tolist(),
                "contact_midpoint_work": contact_work,
                "force_kick_work": step_work,
                "cumulative_force_kick_work": work,
                "kinetic_energy": float(energy),
            }
        )
    np.testing.assert_array_equal(model.shape_gap.numpy(), physical)
    np.testing.assert_array_equal(model.shape_margin.numpy(), margin)
    return {
        "floor": floor,
        "pulse": pulse,
        "contacts_generated_at_rest": count,
        "physical_gap": physical.tolist(),
        "history": rows,
    }


def main():
    wp.set_device("cpu")
    controls = [run_case(0.0, True), run_case(0.002, False), run_case(0.002, True)]
    assert controls[0]["history"][-1]["separation"] < -0.005
    assert min(row["separation"] for row in controls[2]["history"]) > -1e-5
    assert max(row["max_impulse"] for row in controls[2]["history"]) > 0
    result = {
        "status": "PASS",
        "device": "cpu",
        "cases": controls,
        "scope": "Detection at rest separately checked; actual solver force pulse, no velocity reassignment or fresh collision during8steps.",
    }
    Path("/tmp/search_floor_force_activation_cpu.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
