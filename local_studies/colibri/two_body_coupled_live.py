"""Short live CPU-outer/GPU-response two-body reference with native integration."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator
from local_studies.colibri.coupled_support_online import assemble_snapshot, solve_online
from local_studies.colibri.single_block_support_reference import coupled_response
from newton._src.solvers.phoenx.dispatch.single_world_mass_splitting import SingleWorldMassSplittingDispatcher
from newton._src.solvers.phoenx.tests.test_color_group_conservation import _physical_totals
from newton.examples.kamino.example_kamino_colibri import build_scene
from newton.examples.phoenx.example_phoenx_colibri import CONTACT_OFFSETS


def snapshot(w):
    cc = w._contact_container
    b = w.constraints.bilateral
    copy = w._copy_state
    fields = dict(
        position=w.bodies.position,
        orientation=w.bodies.orientation,
        velocity=w.bodies.velocity,
        angular_velocity=w.bodies.angular_velocity,
        inverse_mass=w.bodies.inverse_mass,
        inverse_inertia=w.bodies.inverse_inertia_world,
        headers=w._contact_cols.data,
        column_count=w._ingest_scratch.num_contact_columns,
        lambdas=cc.lambdas,
        derived=cc.derived,
        impulses=cc.impulses,
        joint_data=w.constraints.data,
        copy_velocity=copy.velocity,
        copy_angular_velocity=copy.angular_velocity,
        copy_section_end=copy.section_end,
    )
    for name in (
        "row_count",
        "row_indices",
        "structural_index",
        "row_local",
        "row_dynamic",
        "wrench0",
        "wrench1",
        "bias",
        "reference",
        "dynamic_mass",
        "accumulated",
    ):
        fields["joint_" + name] = getattr(b, name)
    return {k: v.numpy() for k, v in fields.items()}


def gpu_response(a):
    W, B, C = a["W"], a["B"], a["C"]
    rhs = np.column_stack([a["targets"] - B @ a["free"], B @ W @ C.T])
    rows = len(a["old"])

    def arr(x):
        return wp.array(np.asarray(x, dtype=np.float64), dtype=wp.float64, device="cuda:0")

    K = arr(a["K"])
    y = wp.zeros(rhs.shape, dtype=wp.float64, device="cuda:0")
    G = wp.zeros((12, rows), dtype=wp.float64, device="cuda:0")
    v = wp.zeros(12, dtype=wp.float64, device="cuda:0")
    lam = wp.zeros(rows + 1, dtype=wp.float64, device="cuda:0")
    wp.launch(
        coupled_response,
        dim=128,
        block_dim=128,
        inputs=[
            K,
            arr(rhs),
            arr(W @ C.T),
            arr(W @ B.T),
            arr(a["free"]),
            arr(C),
            arr(np.zeros(rows)),
            -1,
            rows,
            wp.zeros((6, 6), dtype=wp.float64, device="cuda:0"),
            y,
            G,
            v,
            lam,
            wp.zeros(12, dtype=wp.float64, device="cuda:0"),
        ],
        device="cuda:0",
    )
    return G.numpy(), v.numpy(), y.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--output", default="/tmp/two_body_coupled_live.json")
    parser.add_argument("--contact-solver", choices=("online", "condensed32"), default="online")
    parser.add_argument("--reference", default="/tmp/colibri_base_frame_totalnormal330_reference.npz")
    args = parser.parse_args()
    builder = build_scene(
        body_count=2,
        fix_base=False,
        contact_gap=0.001,
        source_contact_offsets=True,
        mesh_cylinders=True,
        sdf_resolution=0,
    )
    flower = builder.body_label.index("Flower")
    for i, label in enumerate(builder.shape_label):
        if label in CONTACT_OFFSETS:
            builder.shape_gap[i] = CONTACT_OFFSETS[label]
        if builder.shape_body[i] == flower:
            builder.shape_flags[i] = 0
    model = builder.finalize(skip_validation_joints=True)
    newton.eval_ik(model, model, model.joint_q, model.joint_qd)
    state = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching="sticky",
        rigid_contact_max=8192,
        speculative_contact_gap_max=0.005,
        speculative_contact_velocity_filter=False,
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        articulation_mode="maximal",
        joint_solver="block_pgs",
        step_layout="single_world",
        parallel_contact_prepare=True,
        contact_chunk_size=6,
        mass_splitting=True,
        mass_splitting_color_group_size=4,
        mass_splitting_batch_size=2,
        max_colored_partitions=8,
        substeps=30,
        solver_iterations=1,
        velocity_relaxation="final_substep",
        prepare_refresh_stride=1,
        sor_boost=1.0,
    )
    world = solver.world
    assert not world._colored_contact_rows

    def interval():
        pipeline.collide(state, contacts, dt=1 / 120)
        state.clear_forces()
        solver.step(state, state, control, contacts, 1 / 120)

    with wp.ScopedCapture() as capture:
        interval()
        interval()
    for _ in range(330):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    ref = np.load(args.reference)
    reference_state = {key: ref[key + "_history"][329] if key + "_history" in ref else ref[key] for key in ("q", "qd")}
    gate = {
        key: actual.tobytes() == reference_state[key].tobytes()
        for key, actual in (("q", state.body_q.numpy()), ("qd", state.body_qd.numpy()))
    }
    assert all(gate.values()), gate
    history = [state.body_q.numpy()]
    velocities = [state.body_qd.numpy()]
    records = []
    integration_records = []
    original_solve = SingleWorldMassSplittingDispatcher.solve
    original_relax = SingleWorldMassSplittingDispatcher.relax
    start = time.perf_counter()

    def correct(phase):
        begun = time.perf_counter()
        physical_before, energy_before = _physical_totals(world)
        d = snapshot(world)
        a = assemble_snapshot(d, phase, world.substep_dt, world.num_joints)
        assert a["bodies"] == [1, 2] and a["B"].shape == (6, 12)
        outer_start = time.perf_counter()
        if args.contact_solver == "condensed32":
            from local_studies.colibri.two_body_condensed_gpu import solve as solve_condensed

            lam, iterated_velocity, iterated_joint, response = solve_condensed(a, 32)
            G, vbar, Y = response["response"], response["baseline"], response["joint_solved"]
        else:
            G, vbar, Y = gpu_response(a)
        A = a["C"] @ G
        rhs = a["rhs"] + a["C"] @ (vbar - a["vbar"])
        if args.contact_solver == "condensed32":
            error = 0.0
            if len(lam):
                evaluate, *_ = natural_map_evaluator(A, rhs, a["gamma"], a["mu"])
                error = float(np.max(np.abs(evaluate(lam)[0])))
            np.testing.assert_allclose(iterated_velocity, vbar + G @ lam, atol=1e-10, rtol=0)
            np.testing.assert_allclose(iterated_joint, Y[:, 0] - Y[:, 1:] @ lam, atol=1e-10, rtol=0)
            result = dict(
                accepted=error < 1e-8,
                final_physical_error=error,
                stages=response.get("stages", [dict(contact_sweeps=32, evaluations=0)]),
                scope="Bounded condensed solve; exact work and fallback stages recorded",
            )
        else:
            lam, result = solve_online(A, rhs, a["gamma"], a["mu"], a["old"], physical_jacobian=a["C"])
        outer_seconds = time.perf_counter() - outer_start
        if not result["accepted"]:
            np.savez(Path(args.output).with_suffix(".rejected.npz"), **a, solution=lam)
            raise AssertionError("Unaccepted outer " + json.dumps(result))
        joint = Y[:, 0] - Y[:, 1:] @ lam
        velocity = vbar + G @ lam
        delta = a["C"].T @ (lam - a["old"]) + a["B"].T @ (joint - a["old_joint"])
        physical = a["velocity"] + a["W"] @ delta
        np.testing.assert_allclose(velocity, physical, atol=1e-10, rtol=0)
        equation = a["B"] @ velocity + a["diagonal"] * joint - a["targets"]
        assert np.max(np.abs(equation)) < 1e-8
        moment = np.zeros(6)
        for i, body in enumerate(a["bodies"]):
            p = delta[6 * i : 6 * i + 6]
            moment += np.r_[p[:3], p[3:] + np.cross(d["position"][body], p[:3])]
        ground = sum(
            (g.T @ (lam - a["old"]).reshape(-1, 3)[i] for i, g in enumerate(a["ground_wrenches"])), np.zeros(6)
        )
        moment += np.r_[ground[:3], ground[3:] + np.cross(d["position"][0], ground[:3])]
        assert np.max(np.abs(moment)) < 1e-8
        M = np.linalg.inv(a["W"])
        mid = (velocity + a["velocity"]) * 0.5
        energy = 0.5 * (velocity @ M @ velocity - a["velocity"] @ M @ a["velocity"])
        work = mid @ delta
        assert abs(energy - work) < 1e-12
        # Commit total impulses once, then synchronize all copies before any
        # later native phase. Native pose integration remains unchanged.
        v = d["velocity"].copy()
        omega = d["angular_velocity"].copy()
        for i, body in enumerate(a["bodies"]):
            v[body] = velocity[6 * i : 6 * i + 3]
            omega[body] = velocity[6 * i + 3 : 6 * i + 6]
        impulses = d["impulses"].copy()
        impulses[:, a["points"]] = lam.reshape(-1, 3).T
        accumulated = d["joint_accumulated"].copy()
        offset = 0
        for j in a["joints"]:
            rows = d["joint_row_indices"][j, : d["joint_row_count"][j]]
            accumulated[rows] = joint[offset : offset + len(rows)]
            offset += len(rows)
        world.bodies.velocity.assign(v)
        world.bodies.angular_velocity.assign(omega)
        world._contact_container.impulses.assign(impulses)
        from newton._src.solvers.phoenx.constraints import contact_projection

        if hasattr(contact_projection, "contact_project_friction_metric_with_break"):
            from local_studies.colibri.coupled_friction_history import final_break_flags

            contact_data = d["lambdas"].copy()
            contact_data[12, a["points"]] = final_break_flags(a, lam, velocity)
            world._contact_container.lambdas.assign(contact_data)
        world.constraints.bilateral.accumulated.assign(accumulated)
        world._mass_splitting_broadcast()
        actual_velocity = np.concatenate(
            [np.r_[world.bodies.velocity.numpy()[b], world.bodies.angular_velocity.numpy()[b]] for b in a["bodies"]]
        ).astype(float)
        actual_lam = world._contact_container.impulses.numpy()[:, a["points"]].T.astype(float).ravel()
        actual_joint = np.concatenate(
            [
                world.constraints.bilateral.accumulated.numpy()[d["joint_row_indices"][j, : d["joint_row_count"][j]]]
                for j in a["joints"]
            ]
        ).astype(float)
        actual_delta = a["C"].T @ (actual_lam - a["old"]) + a["B"].T @ (actual_joint - a["old_joint"])
        physical_after, energy_after = _physical_totals(world)
        ground_actual = sum(
            (g.T @ (actual_lam - a["old"]).reshape(-1, 3)[i] for i, g in enumerate(a["ground_wrenches"])),
            np.zeros(6),
        )
        reaction_actual = np.r_[ground_actual[:3], ground_actual[3:] + np.cross(d["position"][0], ground_actual[:3])]
        actual_momentum = physical_after - physical_before + reaction_actual
        actual_work = actual_delta @ ((actual_velocity + a["velocity"]) * 0.5)
        actual_work_error = abs(energy_after - energy_before - actual_work)
        response_rounding = actual_velocity - a["velocity"] - a["W"] @ actual_delta
        actual_joint_error = np.max(np.abs(a["B"] @ actual_velocity + a["diagonal"] * actual_joint - a["targets"]))
        actual_kkt = 0.0
        if len(actual_lam):
            eval_actual, *_ = natural_map_evaluator(
                A, rhs + a["C"] @ (actual_velocity - vbar - G @ actual_lam), a["gamma"], a["mu"]
            )
            actual_kkt = float(np.max(np.abs(eval_actual(actual_lam)[0])))
        if (
            np.max(np.abs(actual_momentum)) >= 1e-8
            or actual_work_error >= 1e-10
            or actual_kkt >= 1e-6
            or actual_joint_error >= 1e-6
        ):
            np.savez(
                Path(args.output).with_suffix(".storage_rejected.npz"),
                **a,
                solution=lam,
                actual_velocity=actual_velocity,
                actual_lam=actual_lam,
                actual_joint=actual_joint,
            )
            raise AssertionError("FP32 physical storage gate failed")
        records.append(
            dict(
                phase=phase,
                actual_momentum_error=float(np.max(np.abs(actual_momentum))),
                actual_work_error=float(actual_work_error),
                actual_contact_residual=float(actual_kkt),
                actual_joint_residual=float(actual_joint_error),
                storage_response_error=float(np.max(np.abs(response_rounding))),
                outer_seconds=outer_seconds,
                solver_stages=result["stages"],
                outer_evaluations=sum(stage.get("evaluations", 0) for stage in result["stages"]),
                normal_seed_iterations=sum(stage.get("normal_iterations", 0) for stage in result["stages"]),
                points=len(a["points"]),
                residual=result["final_physical_error"],
                joint_error=float(np.max(np.abs(equation))),
                momentum_error=float(np.max(np.abs(moment))),
                work_error=float(abs(energy - work)),
                energy_change=float(energy),
                max_joint_target=float(np.max(np.abs(a["targets"][:5]))),
                base_velocity=velocity[:6].tolist(),
                seconds=time.perf_counter() - begun,
            )
        )

    def solve(self, idt):
        original_solve(self, idt)
        if self._world is world:
            correct("biased")

    def relax(self, idt):
        original_relax(self, idt)
        if self._world is world and world._active_velocity_iterations > 0:
            correct("relax")

    original_integrate = world._integrate_positions

    def integrate():
        before, energy_before = _physical_totals(world)
        original_integrate()
        after, energy_after = _physical_totals(world)
        error = float(np.max(np.abs(after - before)))
        integration_records.append(dict(momentum_error=error, torque_free_energy_change=energy_after - energy_before))
        assert error < 1e-8, "Native position integration momentum changed beyond FP32 allowance"

    world._integrate_positions = integrate

    SingleWorldMassSplittingDispatcher.solve = solve
    SingleWorldMassSplittingDispatcher.relax = relax
    error = None
    try:
        for _ in range(2 * args.frames):
            interval()
            history.append(state.body_q.numpy())
            velocities.append(state.body_qd.numpy())
            print("COUPLED_INTERVAL", len(history) - 1, "phases", len(records), flush=True)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        world._integrate_positions = original_integrate
        SingleWorldMassSplittingDispatcher.solve = original_solve
        SingleWorldMassSplittingDispatcher.relax = original_relax
        report = dict(
            reference_only=True,
            frames_requested=args.frames,
            intervals_completed=len(history) - 1,
            baseline_q_qd_byte_gate=gate,
            error=error,
            seconds=time.perf_counter() - start,
            phases=records,
            integration=integration_records,
        )
        Path(args.output).write_text(json.dumps(report, indent=2))
        np.savez(
            Path(args.output).with_suffix(".npz"),
            q_history=history,
            qd_history=velocities,
            failed_q=state.body_q.numpy(),
            failed_qd=state.body_qd.numpy(),
        )
        print(
            "COUPLED_FINAL",
            json.dumps({k: v for k, v in report.items() if k not in ("phases", "integration")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
