"""Compare the native two-body rotary drive with an analytical damped oscillator."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton


def analytical(time, inertia, stiffness, damping, target):
    """Exact overdamped response from zero angle and relative angular speed."""
    roots = np.roots([inertia, damping, stiffness])
    assert np.all(np.isreal(roots)) and abs(roots[0] - roots[1]) > 1e-8
    r0, r1 = roots.real
    a, b = np.linalg.solve([[1.0, 1.0], [r0, r1]], [-target, 0.0])
    angle = target + a * np.exp(r0 * time) + b * np.exp(r1 * time)
    speed = a * r0 * np.exp(r0 * time) + b * r1 * np.exp(r1 * time)
    return angle, speed


def run(substeps, velocity_iterations, frames, device):
    inertias = np.array([0.02, 0.01])
    stiffness = damping = 0.5729577951308232
    target = 0.3490658503988659
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    bodies = [
        builder.add_link(mass=1.0, inertia=wp.mat33(float(i), 0.0, 0.0, 0.0, float(i), 0.0, 0.0, 0.0, float(i)))
        for i in inertias
    ]
    root = builder.add_joint_free(bodies[0])
    hinge = builder.add_joint_revolute(
        bodies[0],
        bodies[1],
        axis=(0.0, 0.0, 1.0),
        target_ke=stiffness,
        target_kd=damping,
        target_pos=target,
    )
    builder.add_articulation([root, hinge])
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    solver = newton.solvers.SolverPhoenX(
        model,
        articulation_mode="maximal",
        joint_solver="block_pgs",
        step_layout="single_world",
        substeps=substeps,
        solver_iterations=1,
        velocity_iterations=velocity_iterations,
        velocity_relaxation="final_substep",
        prepare_refresh_stride=1,
        sor_boost=1.0,
        mass_splitting=False,
    )
    graph = None
    if model.device.is_cuda:
        with wp.ScopedCapture(device=model.device) as capture:
            state.clear_forces()
            solver.step(state, state, control, None, 1.0 / 120.0)
        graph = capture.graph
    records = []
    for frame in range(frames):
        if graph is not None:
            wp.capture_launch(graph)
        else:
            state.clear_forces()
            solver.step(state, state, control, None, 1.0 / 120.0)
        q = state.body_q.numpy().copy()
        qd = state.body_qd.numpy().copy()
        theta = 2.0 * np.arctan2(q[:, 5], q[:, 6])
        # Newton spatial vectors store linear velocity first, angular last.
        omega = qd[:, 5]
        time = (frame + 1) / 120.0
        exact_q, exact_v = analytical(time, 1.0 / np.sum(1.0 / inertias), stiffness, damping, target)
        records.append(
            [
                time,
                float(theta[1] - theta[0]),
                float(omega[1] - omega[0]),
                float(exact_q),
                float(exact_v),
                float(inertias @ omega),
            ]
        )
    values = np.array(records)
    discrete = []
    angle = speed = 0.0
    inertia = 1.0 / np.sum(1.0 / inertias)
    dt = 1.0 / (120.0 * substeps)
    for _frame in range(frames):
        for _substep in range(substeps):
            speed = (speed + dt * stiffness / inertia * (target - angle)) / (
                1.0 + dt * damping / inertia + dt * dt * stiffness / inertia
            )
            angle += dt * speed
        discrete.append([angle, speed])
    return {
        "substeps": substeps,
        "velocity_iterations": velocity_iterations,
        "max_angle_error_rad": float(np.max(np.abs(values[:, 1] - values[:, 3]))),
        "max_relative_speed_error_rad_s": float(np.max(np.abs(values[:, 2] - values[:, 4]))),
        "max_axial_angular_momentum_Nms": float(np.max(np.abs(values[:, 5]))),
        "max_discrete_angle_error_rad": float(np.max(np.abs(values[:, 1] - np.asarray(discrete)[:, 0]))),
        "max_discrete_speed_error_rad_s": float(np.max(np.abs(values[:, 2] - np.asarray(discrete)[:, 1]))),
        "final": values[-1].tolist(),
        "trajectory": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("/tmp/colibri_analytical_free_hinge.json"))
    args = parser.parse_args()
    results = [run(substeps, relax, args.frames, args.device) for substeps, relax in ((1, 0), (30, 0), (30, 1))]
    args.output.write_text(
        json.dumps(
            {
                "cases": results,
                "reference": "I_eff theta_ddot + kd theta_dot + ke(theta-target)=0; zero external torque and zero initial momentum",
            },
            indent=2,
        )
    )
    for result in results:
        print({k: v for k, v in result.items() if k != "trajectory"}, flush=True)

    if wp.get_device(args.device).is_cuda and args.frames == 120:
        for result in results:
            assert result["max_discrete_angle_error_rad"] < 3.0e-6
            assert result["max_discrete_speed_error_rad_s"] < 3.0e-6
            assert result["max_axial_angular_momentum_Nms"] < 1.0e-7
        assert results[1]["max_angle_error_rad"] < 0.1 * results[0]["max_angle_error_rad"]
        np.testing.assert_array_equal(results[1]["trajectory"], results[2]["trajectory"])


if __name__ == "__main__":
    main()
