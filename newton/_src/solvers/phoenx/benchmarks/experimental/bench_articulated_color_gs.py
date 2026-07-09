# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare articulation-aware contact GS with looser operator splits.

The benchmark constructs randomized floating revolute trees, forms their
exact dense constrained mobility, and applies small three-row bilateral
contact blocks. It compares three mathematically distinct schedules:

* articulation-aware GS, where only one contact block per articulation is
  active in a color and every impulse uses the exact tree response;
* body-colored Jacobi, where distinct links are incorrectly assumed to be
  independent after the tree response couples them; and
* free-body PGS followed by one mass-metric joint projection per sweep.

The first schedule is the reference for the experimental PhoenX
articulation-contact path. This benchmark is deliberately topology- and
mass-randomized; it is not a robot-specific performance claim.

Example:
    ``uv run --extra dev -m
    newton._src.solvers.phoenx.benchmarks.experimental.bench_articulated_color_gs``
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = value
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _make_case(
    seed: int,
    *,
    body_count: int,
    contact_blocks: int,
    mass_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    parent = np.full(body_count, -1, dtype=np.int32)
    for body in range(1, body_count):
        parent[body] = rng.integers(0, body)

    transform = np.repeat(np.eye(6)[None, :, :], body_count, axis=0)
    motion = np.zeros((body_count, 6))
    for body in range(1, body_count):
        lever = rng.normal(0.0, 0.25, 3)
        transform[body, :3, 3:] = _skew(lever)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        motion[body, :3] = np.cross(lever, axis)
        motion[body, 3:] = axis

    generalized_count = 6 + body_count - 1
    kinematics = np.zeros((body_count, 6, generalized_count))
    kinematics[0, :, :6] = np.eye(6)
    for body in range(1, body_count):
        kinematics[body] = transform[body] @ kinematics[parent[body]]
        kinematics[body, :, 6 + body - 1] = motion[body]
    kinematics = kinematics.reshape(6 * body_count, generalized_count)

    mass = np.zeros((6 * body_count, 6 * body_count))
    mass_exponents = np.linspace(0.0, np.log10(mass_ratio), body_count)
    rng.shuffle(mass_exponents)
    for body, exponent in enumerate(mass_exponents):
        body_mass = 10.0**exponent
        basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        inertia = basis @ np.diag(body_mass * 10.0 ** rng.uniform(-0.4, 0.4, 3)) @ basis.T
        offset = 6 * body
        mass[offset : offset + 3, offset : offset + 3] = body_mass * np.eye(3)
        mass[offset + 3 : offset + 6, offset + 3 : offset + 6] = inertia

    generalized_mass = kinematics.T @ mass @ kinematics
    mobility = kinematics @ np.linalg.solve(generalized_mass, kinematics.T)
    inverse_mass = np.linalg.inv(mass)

    body_choices = rng.choice(body_count, size=contact_blocks, replace=False)
    jacobian = np.zeros((3 * contact_blocks, 6 * body_count))
    for block, body in enumerate(body_choices):
        directions, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        point = rng.normal(0.0, 0.2, 3)
        for row in range(3):
            direction = directions[:, row]
            jacobian[3 * block + row, 6 * body : 6 * body + 3] = direction
            jacobian[3 * block + row, 6 * body + 3 : 6 * body + 6] = np.cross(point, direction)

    initial_velocity = kinematics @ rng.normal(size=generalized_count)
    return mass, inverse_mass, mobility, jacobian, initial_velocity


def _relative_residual(jacobian: np.ndarray, velocity: np.ndarray, initial_norm: float) -> float:
    return float(np.linalg.norm(jacobian @ velocity) / initial_norm)


def _run_case(
    mass: np.ndarray,
    inverse_mass: np.ndarray,
    mobility: np.ndarray,
    jacobian: np.ndarray,
    initial_velocity: np.ndarray,
    *,
    sweeps: int,
) -> dict[str, float]:
    groups = tuple(slice(row, row + 3) for row in range(0, len(jacobian), 3))
    initial_norm = float(np.linalg.norm(jacobian @ initial_velocity))

    velocity = initial_velocity.copy()
    for _ in range(sweeps):
        for group in groups:
            block_jacobian = jacobian[group]
            response = block_jacobian @ mobility @ block_jacobian.T
            delta = np.linalg.solve(response, -(block_jacobian @ velocity))
            velocity += mobility @ block_jacobian.T @ delta
    articulation_gs = _relative_residual(jacobian, velocity, initial_norm)

    velocity = initial_velocity.copy()
    for _ in range(sweeps):
        impulse = np.zeros_like(velocity)
        for group in groups:
            block_jacobian = jacobian[group]
            response = block_jacobian @ mobility @ block_jacobian.T
            delta = np.linalg.solve(response, -(block_jacobian @ velocity))
            impulse += block_jacobian.T @ delta
        velocity += mobility @ impulse
    body_color_jacobi = _relative_residual(jacobian, velocity, initial_norm)

    projection = mobility @ mass
    velocity = initial_velocity.copy()
    for _ in range(sweeps):
        for group in groups:
            block_jacobian = jacobian[group]
            response = block_jacobian @ inverse_mass @ block_jacobian.T
            delta = np.linalg.solve(response, -(block_jacobian @ velocity))
            velocity += inverse_mass @ block_jacobian.T @ delta
        velocity = projection @ velocity
    free_pgs_then_project = _relative_residual(jacobian, velocity, initial_norm)

    return {
        "articulation_gs": articulation_gs,
        "body_color_jacobi": body_color_jacobi,
        "free_pgs_then_project": free_pgs_then_project,
    }


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.body_count < 2:
        raise ValueError("body-count must be at least two")
    if args.contact_blocks < 1 or args.contact_blocks > args.body_count:
        raise ValueError("contact-blocks must be in [1, body-count]")
    if args.cases < 1 or args.sweeps < 1:
        raise ValueError("cases and sweeps must be positive")
    if any(ratio < 1.0 for ratio in args.mass_ratios):
        raise ValueError("mass ratios must be at least one")

    results = []
    for mass_ratio in args.mass_ratios:
        samples: dict[str, list[float]] = {
            "articulation_gs": [],
            "body_color_jacobi": [],
            "free_pgs_then_project": [],
        }
        for case in range(args.cases):
            values = _run_case(
                *_make_case(
                    args.seed + case,
                    body_count=args.body_count,
                    contact_blocks=args.contact_blocks,
                    mass_ratio=mass_ratio,
                ),
                sweeps=args.sweeps,
            )
            for key, value in values.items():
                samples[key].append(value)
        results.append(
            {
                "mass_ratio": mass_ratio,
                "relative_contact_residual": {
                    key: {
                        "median": float(np.median(values)),
                        "p90": float(np.quantile(values, 0.9)),
                        "maximum": float(np.max(values)),
                    }
                    for key, values in samples.items()
                },
            }
        )
    return {
        "schema": "phoenx_articulated_color_gs_v1",
        "cases_per_ratio": args.cases,
        "body_count": args.body_count,
        "contact_blocks": args.contact_blocks,
        "sweeps": args.sweeps,
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass-ratios", nargs="+", type=float, default=(100.0, 10_000.0, 1_000_000.0))
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--body-count", type=int, default=16)
    parser.add_argument("--contact-blocks", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(json.dumps(benchmark(args), indent=args.json_indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
