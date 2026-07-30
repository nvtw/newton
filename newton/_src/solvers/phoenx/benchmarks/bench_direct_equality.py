# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare direct and PGS joint equalities on ill-conditioned chains.

The benchmark reports graph-replay throughput and joint-anchor error at the
same PhoenX work point. For the direct solve it also reports the condition
number of the assembled mechanism matrix. Compilation and host diagnostics
are excluded from the timing window.

Examples::

    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_direct_equality

    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_direct_equality \
        --lengths 12 26 --mass-ratios 1 10000 --output /tmp/direct-equality.json

    uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_direct_equality \
        --driven --lengths 12 26 --mass-ratios 1 10000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

import newton


@dataclass(frozen=True)
class BenchResult:
    """One direct or PGS measurement."""

    equality_solver: str
    driven: bool
    length: int
    mass_ratio: float
    equation_count: int
    matrix_condition: float | None
    equilibrated_condition: float | None
    anchor_error_m: float
    fps: float
    finite: bool


@dataclass(frozen=True)
class HeterogeneousResult:
    """One heterogeneous direct-solver scheduling measurement."""

    mechanism_lengths: tuple[int, ...]
    driven: bool
    mass_ratio: float
    equation_count: int
    matrix_entry_tasks: int
    compact_panel_slots: int
    rectangular_entry_slots: int
    fps: float
    finite: bool


def _add_link(builder: newton.ModelBuilder, position: wp.vec3, mass: float) -> int:
    inertia = max(mass / 12.0, 1.0e-12)
    return builder.add_link(
        xform=wp.transform(position, wp.quat_identity()),
        mass=mass,
        inertia=((inertia, 0.0, 0.0), (0.0, inertia, 0.0), (0.0, 0.0, inertia)),
    )


def _build_chain(length: int, mass_ratio: float, *, driven: bool = False) -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    parent = -1
    joints = []
    light_mass = 1.0 / mass_ratio
    for index in range(length):
        mass = 1.0 if index % 2 == 0 else light_mass
        child = _add_link(builder, wp.vec3(0.0, 0.0, -float(index) - 0.5), mass)
        parent_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.0) if parent < 0 else wp.vec3(0.0, 0.0, -0.5),
            wp.quat_identity(),
        )
        drive_kwargs = {}
        if driven:
            drive_kwargs = {
                "target_ke": 100.0,
                "target_kd": 10.0,
                "actuator_mode": newton.JointTargetMode.POSITION,
                "effort_limit": 1000.0,
            }
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
                limit_lower=-np.inf,
                limit_upper=np.inf,
                **drive_kwargs,
            )
        )
        parent = child
    builder.add_articulation(joints)
    model = builder.finalize()
    model.set_gravity((9.81, 0.0, 0.0))
    return model


def _build_heterogeneous_mechanisms(
    lengths: tuple[int, ...],
    mass_ratio: float,
    *,
    driven: bool = False,
) -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    light_mass = 1.0 / mass_ratio
    for mechanism, length in enumerate(lengths):
        parent = -1
        joints = []
        offset_x = float(mechanism) * 2.0
        for index in range(length):
            mass = 1.0 if index % 2 == 0 else light_mass
            child = _add_link(
                builder,
                wp.vec3(offset_x, 0.0, -float(index) - 0.5),
                mass,
            )
            parent_xform = wp.transform(
                wp.vec3(offset_x, 0.0, 0.0) if parent < 0 else wp.vec3(0.0, 0.0, -0.5),
                wp.quat_identity(),
            )
            drive_kwargs = {}
            if driven:
                drive_kwargs = {
                    "target_ke": 100.0,
                    "target_kd": 10.0,
                    "actuator_mode": newton.JointTargetMode.POSITION,
                    "effort_limit": 1000.0,
                }
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=child,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=parent_xform,
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
                    limit_lower=-np.inf,
                    limit_upper=np.inf,
                    **drive_kwargs,
                )
            )
            parent = child
        builder.add_articulation(joints)
    model = builder.finalize()
    model.set_gravity((9.81, 0.0, 0.0))
    return model


def _parse_mechanism_group(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*)x([2-9][0-9]*)", value)
    if match is None:
        raise argparse.ArgumentTypeError("mechanism groups must have the form COUNTxLENGTH, for example 32x2")
    return int(match.group(1)), int(match.group(2))


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    xyz = transform[3:6]
    w = transform[6]
    return transform[:3] + point + 2.0 * np.cross(xyz, np.cross(xyz, point) + w * point)


def _maximum_anchor_error(model: newton.Model, state: newton.State) -> float:
    body_q = state.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()

    maximum = 0.0
    for joint in range(int(model.joint_count)):
        parent = int(joint_parent[joint])
        child = int(joint_child[joint])
        point_parent = joint_x_p[joint, :3]
        if parent >= 0:
            point_parent = _transform_point(body_q[parent], point_parent)
        point_child = _transform_point(body_q[child], joint_x_c[joint, :3])
        maximum = max(maximum, float(np.linalg.norm(point_child - point_parent)))
    return maximum


def _matrix_conditions(solver: newton.solvers.SolverPhoenX) -> tuple[float, float]:
    direct = solver._direct_equality_system
    dimension = direct.topology.dimensions[0]
    symbolic = direct.solver.symbolic
    matrix_values = direct.matrix.numpy()
    equilibrated = np.zeros((dimension, dimension), dtype=np.float32)
    rows = symbolic.matrix_row
    columns = symbolic.matrix_column
    values = matrix_values[symbolic.matrix_storage]
    equilibrated[rows, columns] = values
    equilibrated[columns, rows] = values
    row_scale = direct.row_scale.numpy()[:dimension].astype(np.float64)
    unscaled = equilibrated.astype(np.float64) / np.outer(row_scale, row_scale)
    try:
        return float(np.linalg.cond(unscaled)), float(np.linalg.cond(equilibrated.astype(np.float64)))
    except np.linalg.LinAlgError:
        return float("inf"), float("inf")


def _run_one(
    *,
    equality_solver: str,
    driven: bool,
    length: int,
    mass_ratio: float,
    quality_frames: int,
    measure_frames: int,
    substeps: int,
    solver_iterations: int,
) -> BenchResult:
    model = _build_chain(length, mass_ratio, driven=driven)
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=0,
        articulation_mode="maximal",
        joint_equality_solver=equality_solver,
        step_layout="single_world",
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    with wp.ScopedCapture(model.device) as capture:
        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
        solver.step(state_1, state_0, control, None, 1.0 / 60.0)
    graph = capture.graph

    quality_replays = (quality_frames + 1) // 2
    for _ in range(quality_replays):
        wp.capture_launch(graph)
    anchor_error = _maximum_anchor_error(model, state_0)
    if equality_solver == "direct":
        matrix_condition, equilibrated_condition = _matrix_conditions(solver)
    else:
        matrix_condition, equilibrated_condition = None, None

    measure_replays = (measure_frames + 1) // 2
    wp.synchronize_device(model.device)
    start = time.perf_counter()
    for _ in range(measure_replays):
        wp.capture_launch(graph)
    wp.synchronize_device(model.device)
    elapsed = time.perf_counter() - start

    body_q = state_0.body_q.numpy()
    body_qd = state_0.body_qd.numpy()
    finite = bool(np.isfinite(body_q).all() and np.isfinite(body_qd).all())
    direct = solver._direct_equality_system
    equation_count = (
        sum(direct.topology.dimensions) if direct is not None and direct.enabled else (6 if driven else 5) * length
    )
    return BenchResult(
        equality_solver=equality_solver,
        driven=driven,
        length=length,
        mass_ratio=mass_ratio,
        equation_count=equation_count,
        matrix_condition=matrix_condition,
        equilibrated_condition=equilibrated_condition,
        anchor_error_m=anchor_error,
        fps=float(2 * measure_replays / elapsed),
        finite=finite,
    )


def _print_pair(pgs: BenchResult, direct: BenchResult) -> None:
    condition = direct.matrix_condition
    assert condition is not None
    equilibrated_condition = direct.equilibrated_condition
    assert equilibrated_condition is not None
    error_gain = pgs.anchor_error_m / max(direct.anchor_error_m, np.finfo(np.float64).tiny)
    fps_ratio = direct.fps / pgs.fps
    print(
        f"links={direct.length:2d}  ratio={direct.mass_ratio:9.1g}:1  "
        f"driven={direct.driven!s:5s}  equations={direct.equation_count:3d}  "
        f"cond={condition:10.3e}->{equilibrated_condition:10.3e}"
    )
    print(f"  PGS     fps={pgs.fps:9.1f}  anchor={pgs.anchor_error_m:10.3e} m  finite={pgs.finite}")
    print(
        f"  direct  fps={direct.fps:9.1f}  anchor={direct.anchor_error_m:10.3e} m  "
        f"finite={direct.finite}  speed={fps_ratio:6.1%}  error_gain={error_gain:9.1f}x"
    )


def _run_heterogeneous(
    *,
    lengths: tuple[int, ...],
    mass_ratio: float,
    measure_frames: int,
    driven: bool,
    substeps: int,
    solver_iterations: int,
) -> HeterogeneousResult:
    model = _build_heterogeneous_mechanisms(lengths, mass_ratio, driven=driven)
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=substeps,
        solver_iterations=solver_iterations,
        velocity_iterations=0,
        articulation_mode="maximal",
        joint_equality_solver="direct",
        step_layout="single_world",
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    with wp.ScopedCapture(model.device) as capture:
        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
        solver.step(state_1, state_0, control, None, 1.0 / 60.0)
    graph = capture.graph

    for _ in range(10):
        wp.capture_launch(graph)
    measure_replays = (measure_frames + 1) // 2
    wp.synchronize_device(model.device)
    start = time.perf_counter()
    for _ in range(measure_replays):
        wp.capture_launch(graph)
    wp.synchronize_device(model.device)
    elapsed = time.perf_counter() - start

    direct = solver._direct_equality_system
    finite = bool(np.isfinite(state_0.body_q.numpy()).all() and np.isfinite(state_0.body_qd.numpy()).all())
    return HeterogeneousResult(
        mechanism_lengths=lengths,
        driven=driven,
        mass_ratio=mass_ratio,
        equation_count=sum(direct.topology.dimensions),
        matrix_entry_tasks=direct.matrix_storage.size,
        compact_panel_slots=direct.matrix.size,
        rectangular_entry_slots=len(lengths) * direct.max_dimension**2,
        fps=float(2 * measure_replays / elapsed),
        finite=finite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=(12,))
    parser.add_argument("--mass-ratios", nargs="+", type=float, default=(1.0, 1.0e2, 1.0e4, 1.0e6))
    parser.add_argument(
        "--heterogeneous",
        nargs="+",
        type=_parse_mechanism_group,
        help="also benchmark grouped mechanisms as COUNTxLENGTH entries, for example 32x2 8x8 2x26",
    )
    parser.add_argument("--quality-frames", type=int, default=20)
    parser.add_argument("--measure-frames", type=int, default=400)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--solver-iterations", type=int, default=2)
    parser.add_argument(
        "--driven",
        action="store_true",
        help="benchmark finite-effort implicit PD drives in addition to structural joint rows",
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if not wp.is_cuda_available():
        raise RuntimeError("the direct-equality benchmark requires CUDA")
    if any(length < 2 for length in args.lengths):
        parser.error("chain lengths must be at least two")
    if any(ratio < 1.0 for ratio in args.mass_ratios):
        parser.error("mass ratios must be at least one")
    if min(args.quality_frames, args.measure_frames, args.substeps, args.solver_iterations) < 1:
        parser.error("frame and solver work counts must be positive")

    results: list[BenchResult] = []
    for length in args.lengths:
        for mass_ratio in args.mass_ratios:
            pgs = _run_one(
                equality_solver="pgs",
                driven=args.driven,
                length=length,
                mass_ratio=mass_ratio,
                quality_frames=args.quality_frames,
                measure_frames=args.measure_frames,
                substeps=args.substeps,
                solver_iterations=args.solver_iterations,
            )
            direct = _run_one(
                equality_solver="direct",
                driven=args.driven,
                length=length,
                mass_ratio=mass_ratio,
                quality_frames=args.quality_frames,
                measure_frames=args.measure_frames,
                substeps=args.substeps,
                solver_iterations=args.solver_iterations,
            )
            results.extend((pgs, direct))
            _print_pair(pgs, direct)

    heterogeneous = None
    if args.heterogeneous:
        mechanism_lengths = tuple(length for count, length in args.heterogeneous for _ in range(count))
        heterogeneous = _run_heterogeneous(
            lengths=mechanism_lengths,
            mass_ratio=max(args.mass_ratios),
            measure_frames=args.measure_frames,
            driven=args.driven,
            substeps=args.substeps,
            solver_iterations=args.solver_iterations,
        )
        print(
            f"heterogeneous mechanisms={len(mechanism_lengths)}  driven={heterogeneous.driven}  "
            f"equations={heterogeneous.equation_count}  "
            f"matrix_tasks={heterogeneous.matrix_entry_tasks}/{heterogeneous.rectangular_entry_slots}  "
            f"panel_slots={heterogeneous.compact_panel_slots}  "
            f"fps={heterogeneous.fps:.1f}  finite={heterogeneous.finite}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"comparisons": [asdict(result) for result in results]}
        if heterogeneous is not None:
            payload["heterogeneous"] = asdict(heterogeneous)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
