"""Compare grouping on one unchanged final public state, restoring device arrays."""

import json
import runpy
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from local_studies.colibri.support_star_groups import attach


def collect(root):
    arrays = []
    seen = set()

    def visit(value, depth):
        if value is None or id(value) in seen or depth > 8:
            return
        if isinstance(value, wp.array):
            seen.add(id(value))
            if value.size:
                arrays.append((value, wp.clone(value)))
            return
        if isinstance(value, str | int | float | bool | type | np.ndarray) or callable(value):
            return
        seen.add(id(value))
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, tuple | list):
            items = enumerate(value)
        else:
            items = vars(value).items() if hasattr(value, "__dict__") else ()
        for name, child in items:
            if str(name) not in ("model", "device", "_device", "module", "_cls", "cls", "func", "_struct"):
                visit(child, depth + 1)

    visit(root, 0)
    return arrays


def residual(world):
    cc = world._contact_container
    header = world._contact_cols.data.numpy().view(np.int32)
    normal = cc.lambdas.numpy()
    derived = cc.derived.numpy()
    impulse = cc.impulses.numpy()
    velocity = world.bodies.velocity.numpy()
    spin = world.bodies.angular_velocity.numpy()
    rows = []
    for column in range(int(world._ingest_scratch.num_contact_columns.numpy()[0])):
        if set(header[1:3, column]) != {0, 1}:
            continue
        side = 0 if header[1, column] == 1 else 1
        for k in range(header[5, column], header[5, column] + header[6, column]):
            n = normal[:3, k].astype(float)
            r = derived[9 + side * 3 : 12 + side * 3, k].astype(float)
            v = velocity[1].astype(float) + np.cross(spin[1], r)
            vn = float(np.dot(n, v))
            rows.append(
                {
                    "point": int(k),
                    "normal_velocity": vn,
                    "slip": float(np.linalg.norm(v - vn * n)),
                    "normal_impulse": float(impulse[0, k]),
                    "friction_ratio": float(np.linalg.norm(impulse[1:, k]) / max(impulse[0, k], 1e-30)),
                }
            )
    return {
        "rows": rows,
        "base_copies": int(world._copy_state.count_per_node.numpy()[1]),
        "base_velocity": velocity[1].tolist(),
        "base_spin": spin[1].tolist(),
    }


def compare(example):
    world = example.solver.world
    original_rebuild = world._rebuild_mass_splitting_graph
    original_sweep = world._color_group_sweep
    saved = collect(world)
    attach(world)
    star_rebuild = world._rebuild_mass_splitting_graph
    star_sweep = world._color_group_sweep
    results = {"input": residual(world), "saved_arrays": len(saved)}
    states = {}
    try:
        for label, rebuild, sweep in (
            ("reference", original_rebuild, original_sweep),
            ("reference_repeat", original_rebuild, original_sweep),
            ("support_star", star_rebuild, star_sweep),
            ("reference_after_candidate", original_rebuild, original_sweep),
        ):
            for destination, source in saved:
                wp.copy(destination, source)
            world._rebuild_mass_splitting_graph = rebuild
            world._color_group_sweep = sweep
            rebuild()
            world._dispatcher.solve(1.0 / world.substep_dt)
            wp.synchronize_device(world.device)
            results[label] = residual(world)
            states[label] = tuple(
                array.numpy().copy()
                for array in (
                    world.bodies.velocity,
                    world.bodies.angular_velocity,
                    world._contact_container.impulses,
                    world._contact_container.lambdas,
                    world._contact_container.derived,
                    world.constraints.bilateral.accumulated,
                )
            )
        results["reference_repeat_byte_exact"] = all(
            np.array_equal(a.view(np.uint8), b.view(np.uint8))
            for key in ("reference_repeat", "reference_after_candidate")
            for a, b in zip(states["reference"], states[key], strict=True)
        )
        if not results["reference_repeat_byte_exact"]:
            raise AssertionError("Frozen replay failed to restore all physical inputs")
        Path("/tmp/colibri_support_star_frozen330.json").write_text(json.dumps(results, indent=2))
        print("FROZEN_SUPPORT_STAR", results["saved_arrays"], results["reference_repeat_byte_exact"], flush=True)
    finally:
        for destination, source in saved:
            wp.copy(destination, source)
        world._rebuild_mass_splitting_graph = original_rebuild
        world._color_group_sweep = original_sweep


original_run = newton.examples.run


def run(example, args):
    original_run(example, args)
    compare(example)


newton.examples.run = run
if __name__ == "__main__":
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
