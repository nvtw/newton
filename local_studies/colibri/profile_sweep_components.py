# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Frozen sweep attribution. Ablated outputs are never physical trajectories."""

import ast
import ctypes
import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from local_studies.colibri.freeze_support_star import collect
from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx.constraints import bilateral_joint
from newton._src.solvers.phoenx.dispatch import color_groups
from newton.examples.phoenx.example_phoenx_colibri import Example


class Gate(ast.NodeTransformer):
    def __init__(self, mode):
        self.mode = mode

    def visit_Expr(self, node):
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "dispatch"
        ):
            test = "cid < num_joints" if self.mode == "joints" else "cid >= num_joints"
            return ast.If(test=ast.parse(test, mode="eval").body, body=[node], orelse=[])
        return self.generic_visit(node)


def ablated(mode, phase, soft):
    tree = Gate(mode).visit(ast.parse(inspect.getsource(color_groups.get_sweep_kernel)))
    ast.fix_missing_locations(tree)
    factory = _compile_function(ast.unparse(tree), color_groups, "get_sweep_kernel")
    return factory(phase, soft)


def bilateral_prefix(stage, soft):
    """Keep an observable exact arithmetic prefix; never continue its output."""
    source = inspect.getsource(bilateral_joint.iterate_bilateral_joint_block.func)
    source = source.replace("    use_bias: wp.bool,", "    use_bias: wp.bool,\n    sink: wp.array2d[wp.float64],")
    if stage == "rhs":
        source = source[: source.index("    lower = data.lower[cid]")]
        source += "    for i in range(count):\n        sink[cid, i] = rhs[i]\n"
    elif stage == "solve":
        source = source[: source.index("    impulse0 = wp.spatial_vector()")]
        source += "    for i in range(count):\n        sink[cid, i] = solution[i]\n"
    name = "iterate_bilateral_prefix_" + stage
    source = source.replace("def iterate_bilateral_joint_block(", "def " + name + "(")
    callback = _compile_function(source, bilateral_joint, name)
    tree = ast.parse(inspect.getsource(color_groups.get_sweep_kernel))

    class Replace(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == "sweep":
                node.args.args.append(
                    ast.arg(arg="sink", annotation=ast.parse("wp.array2d[wp.float64]", mode="eval").body)
                )
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "dispatch"
            ):
                return ast.parse(
                    "if cid < num_joints:\n    stage_callback(constraints, cid, bodies, particles, copies, num_bodies, slab, True, sink)"
                ).body[0]
            return self.generic_visit(node)

    tree = Replace().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = types.SimpleNamespace(**vars(color_groups))
    namespace.stage_callback = callback
    return _compile_function(ast.unparse(tree), namespace, "get_sweep_kernel")("iterate", soft)


class Frozen(Exception):
    pass


def main():
    profile = "--profile" in sys.argv
    prefixes = "--bilateral-prefixes" in sys.argv
    sys.argv = [sys.argv[0], "--viewer", "null", "--num-frames", "330"]
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    for _ in range(330):
        example.step()
    wp.synchronize_device(example.model.device)
    world = example.solver.world
    soft = bool(world._dispatch_specialization_flags()["has_soft_contact_pd"])
    wanted = color_groups.get_sweep_kernel("iterate", soft)
    original_launch = wp.launch
    captured = {}

    def intercept(kernel, *positional, **keywords):
        if kernel is wanted:
            captured.update(kernel=kernel, positional=positional, keywords=keywords)
            raise Frozen
        return original_launch(kernel, *positional, **keywords)

    wp.launch = intercept
    try:
        example.simulate()
    except Frozen:
        pass
    finally:
        wp.launch = original_launch
    assert captured, "Did not capture the first physical iterate"
    saved = collect(world)
    variants = {
        "full": wanted,
        "joints": ablated("joints", "iterate", soft),
        "contacts": ablated("contacts", "iterate", soft),
    }
    if prefixes:
        variants = {label: bilateral_prefix(label, soft) for label in ("rhs", "solve", "full")}
        sink = wp.zeros((world.num_joints, 6), dtype=wp.float64, device=world.device)
        saved.append((sink, wp.clone(sink)))
        captured["keywords"] = dict(captured["keywords"])
        if "inputs" in captured["keywords"]:
            captured["keywords"]["inputs"] = [*captured["keywords"]["inputs"], sink]
        else:
            positional = list(captured["positional"])
            positional[1] = [*positional[1], sink]
            captured["positional"] = tuple(positional)
    report = {
        "scope": "Frozen first iterate after 330 frames; ablations are cost attribution only",
        "restored_arrays": len(saved),
        "variants": {},
    }
    ids = world._color_group_data["ids"].numpy()
    starts = world._color_group_data["starts"].numpy()
    count = int(world._color_group_data["num_colors"].numpy()[0])
    ids = ids[: starts[count]]
    headers = world._contact_cols.data.numpy().view(np.int32)
    rows = ids[ids >= world.num_joints] - world.num_joints
    report["workload"] = (
        {
            "colors": count,
            "groups": (count + args.mass_splitting_color_group_size - 1) // args.mass_splitting_color_group_size,
        }
        if hasattr(args, "mass_splitting_color_group_size")
        else {"colors": count, "group_width": world.mass_splitting_color_group_size}
    )
    report["workload"].update(
        joint_rows=int(np.sum(ids < world.num_joints)),
        contact_columns=len(rows),
        contact_points=int(headers[6, rows].sum()),
        max_points_per_column=int(headers[6, rows].max(initial=0)),
    )
    driver = ctypes.CDLL("libcuda.so.1")
    if profile and driver.cuProfilerStart():
        raise RuntimeError("cuProfilerStart failed")
    for label, kernel in variants.items():
        for destination, source in saved:
            wp.copy(destination, source)
        original_launch(kernel, *captured["positional"], **captured["keywords"])
        wp.synchronize_device(world.device)
        first = [
            a.numpy().copy()
            for a in (
                world.bodies.velocity,
                world.bodies.angular_velocity,
                world._copy_state.velocity,
                world._copy_state.angular_velocity,
                world._contact_container.impulses,
                world.constraints.bilateral.accumulated,
            )
        ]
        sink_first = sink.numpy().copy() if prefixes else None
        start = wp.Event(world.device, enable_timing=True)
        end = wp.Event(world.device, enable_timing=True)
        with wp.ScopedCapture(device=world.device) as capture:
            for destination, source in saved:
                wp.copy(destination, source)
            wp.record_event(start, external=True)
            original_launch(kernel, *captured["positional"], **captured["keywords"])
            wp.record_event(end, external=True)
        samples = []
        for _ in range(100):
            wp.capture_launch(capture.graph)
            samples.append(wp.get_event_elapsed_time(start, end))
        again = [
            a.numpy()
            for a in (
                world.bodies.velocity,
                world.bodies.angular_velocity,
                world._copy_state.velocity,
                world._copy_state.angular_velocity,
                world._contact_container.impulses,
                world.constraints.bilateral.accumulated,
            )
        ]
        assert all(a.tobytes() == b.tobytes() for a, b in zip(first, again, strict=True))
        if prefixes:
            assert sink_first.tobytes() == sink.numpy().tobytes()
        report["variants"][label] = {
            "mean_ms": float(np.mean(samples[10:])),
            "median_ms": float(np.median(samples[10:])),
            "p95_ms": float(np.percentile(samples[10:], 95)),
            "identical_input_repeat_exact": True,
            "kernel": kernel.key,
            "sink_norm": float(np.linalg.norm(sink.numpy())) if prefixes else None,
        }
    for destination, source in saved:
        wp.copy(destination, source)
    if profile and driver.cuProfilerStop():
        raise RuntimeError("cuProfilerStop failed")
    output = "/tmp/colibri_bilateral_prefixes.json" if prefixes else "/tmp/colibri_sweep_components.json"
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
