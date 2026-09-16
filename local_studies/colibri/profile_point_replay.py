"""Frozen exact-value substitution diagnostics; never use ablations for simulation.

Install via COLIBRI_STAGE_RUNNER after the corrected native/no-skip wrapper.
The physical run uses the unmodified serial cached candidate. At frame 331,
clone actual first biased and relaxation launch inputs, then independently replay.
"""

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from local_studies.colibri import cached_component_live


def clone_inputs(inputs):
    """Clone all struct arrays with shared-pointer aliases retained."""
    arrays = {}

    def clone(value):
        if isinstance(value, wp.array):
            key = (value.ptr, value.shape, str(value.dtype))
            if key not in arrays:
                arrays[key] = (value, wp.clone(value))
            return arrays[key][1]
        if hasattr(value, "_cls"):
            result = value._cls()
            for name in value._cls.vars:
                setattr(result, name, clone(getattr(value, name)))
            return result
        return value

    result = [clone(value) for value in inputs]
    return result, [pair[1] for pair in arrays.values()]


def make_variants(kernel):
    """Generate exact-value substitutions from the executed kernel source."""
    source_path = Path(inspect.getsourcefile(kernel.func))
    source = source_path.read_text()
    module_ast = ast.parse(source)
    function = next(n for n in module_ast.body if isinstance(n, ast.FunctionDef) and n.name == kernel.key)
    function_text = "\n".join(source.splitlines()[function.decorator_list[0].lineno - 1 : function.end_lineno])
    assert "apply_cached(" in function_text and "_sync_tree()" not in function_text
    assert "row_velocity_local(" not in function_text, "Attribute the serial-global baseline first"
    point_tail = """
            trace[20, contact] = cc_get_normal_lambda(contacts, contact)
            for target in range(2):
                body = tree.body_slot[articulation, target]
                v = bodies.velocity[body]
                w = bodies.angular_velocity[body]
                for component in range(3):
                    trace[6 + target * 6 + component, contact] = v[component]
                    trace[9 + target * 6 + component, contact] = w[component]
                row = tree.dynamic_row[articulation, target]
                if row >= 0:
                    trace[18 + target, contact] = dynamic_accumulated_impulse[row]
"""
    restore = """
                for target in range(2):
                    body = tree.body_slot[articulation, target]
                    bodies.velocity[body] = wp.vec3f(trace[6 + target * 6, contact], trace[7 + target * 6, contact], trace[8 + target * 6, contact])
                    bodies.angular_velocity[body] = wp.vec3f(trace[9 + target * 6, contact], trace[10 + target * 6, contact], trace[11 + target * 6, contact])
                    row = tree.dynamic_row[articulation, target]
                    if row >= 0:
                        dynamic_accumulated_impulse[row] = trace[18 + target, contact]
"""
    variants = {}
    for mode in ("record", "jv", "metric", "apply", "all", "full_keep_jv", "metric_keep_jv"):
        text = function_text.replace("def " + kernel.key, "def replay_" + mode)
        text = text.replace("    use_bias: wp.bool,", "    use_bias: wp.bool,\n    trace: wp.array2d[wp.float32],")
        tree = ast.parse(text)
        edits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("normal_velocity", "tangent_velocity0", "tangent_velocity1"):
                    slot = ("normal_velocity", "tangent_velocity0", "tangent_velocity1").index(name)
                    if mode in ("jv", "all"):
                        edits.append((node.lineno, node.end_lineno, f"                {name} = trace[{slot}, contact]"))
                    elif mode in ("record", "full_keep_jv", "metric_keep_jv"):
                        edits.append(
                            (node.end_lineno + 1, node.end_lineno, f"                trace[{slot}, contact] = {name}")
                        )
                if name == "tangents":
                    if mode in ("metric", "all", "metric_keep_jv"):
                        edits.append(
                            (
                                node.lineno,
                                node.end_lineno,
                                "                    tangents = wp.vec3f(trace[3, contact], trace[4, contact], trace[5, contact])",
                            )
                        )
                    elif mode == "record":
                        edits.append(
                            (
                                node.end_lineno + 1,
                                node.end_lineno,
                                "                    for component in range(3):\n                        trace[3 + component, contact] = tangents[component]",
                            )
                        )
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "apply_cached"
                and mode in ("apply", "all")
            ):
                edits.append((node.lineno, node.end_lineno, restore.strip("\n")))
        lines = text.splitlines()
        for begin, end, replacement in sorted(edits, reverse=True):
            lines[begin - 1 : end] = replacement.splitlines()
        text = "\n".join(lines)
        if mode == "record":
            text += point_tail
        variants[mode] = text
    directory = Path(tempfile.mkdtemp(prefix="colibri_point_replay_"))
    path = directory / "kernels.py"
    path.write_text(source + "\n\n" + "\n\n".join(variants.values()))
    name = kernel.func.__module__.rsplit(".", 1)[0] + ".point_replay_diagnostic"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, {
        "executed_source": str(source_path),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "diagnostic_source": str(path),
    }


def load_snapshot(kernel, prefix, phase, device="cuda:0"):
    """Restore typed arrays and aliases from a complete saved launch manifest."""
    suffix = "biased" if phase else "relax"
    base = Path(str(prefix) + ".point_inputs_" + suffix)
    archive = np.load(str(base) + ".npz")
    metadata = json.loads(Path(str(base) + ".json").read_text())
    restored = {}

    def restore(dtype, key):
        if key not in metadata:
            assert hasattr(dtype, "vars"), (key, dtype)
            value = dtype()
            for name, field in dtype.vars.items():
                setattr(value, name, restore(field.type, key + "__" + name))
            return value
        entry = metadata[key]
        if entry["kind"] == "scalar_or_null":
            return None if entry["value"] is None else dtype(entry["value"])
        if entry["alias_of"] is not None:
            return restored[entry["alias_of"]]
        value = wp.array(archive[key], dtype=dtype.dtype, device=device)
        assert list(value.shape) == entry["shape"], (key, value.shape, entry["shape"])
        restored[key] = value
        return value

    inputs = [restore(arg.type, "input" + str(i)) for i, arg in enumerate(kernel.adj.args)]
    assert bool(inputs[-1]) == phase
    return kernel, {"dim": inputs[0].body_count.shape[0], "device": device}, inputs


def benchmark(snapshot, trace_path=None):
    """Reset every array before each replay; time only one completed kernel."""
    kernel, launch_args, inputs = snapshot
    working, arrays = clone_inputs(inputs)
    pristine = [wp.clone(a) for a in arrays]
    module, provenance = make_variants(kernel)
    trace = wp.full((21, working[11].shape[1]), float("nan"), dtype=wp.float32, device=launch_args["device"])
    launch = wp.launch
    from newton._src.solvers.phoenx.articulations import maximal_contact_gs as native

    def reset():
        for dst, src in zip(arrays, pristine, strict=True):
            wp.copy(dst, src)

    def execute(mode):
        if mode == "native":
            native_args = dict(launch_args)
            native_args["dim"] = working[0].body_count.shape[0] * 64
            native_args["block_dim"] = 64
            launch(native.iterate_maximal_contact_runs_kernel, inputs=[*working[:2], *working[3:]], **native_args)
            return
        target = kernel if mode == "full" else getattr(module, "replay_" + mode)
        launch(target, inputs=working if mode == "full" else [*working, trace], **launch_args)

    reset()
    execute("full")
    wp.synchronize()
    expected = [a.numpy().copy() for a in arrays]
    physical = [working[3].velocity, working[3].angular_velocity, working[4], working[6].lambdas, working[6].impulses]
    expected_physical = [a.numpy().copy() for a in physical]
    reset()
    execute("record")
    wp.synchronize()
    assert all(a.numpy().tobytes() == b.tobytes() for a, b in zip(arrays, expected, strict=True)), (
        "Recording altered full output"
    )
    results = {}
    for mode in ("full", "native", "jv", "metric", "apply", "all", "full_keep_jv", "metric_keep_jv"):
        reset()
        execute(mode)
        wp.synchronize()
        mismatches = [
            i for i, (a, b) in enumerate(zip(arrays, expected, strict=True)) if a.numpy().tobytes() != b.tobytes()
        ]
        if mode == "native":
            mismatches = [
                i
                for i, (a, b) in enumerate(zip(physical, expected_physical, strict=True))
                if a.numpy().tobytes() != b.tobytes()
            ]
        if mismatches:
            raise AssertionError((mode, "output-byte mismatch", mismatches))
        print("REPLAY_BYTE_GATE", bool(inputs[-1]), mode, "PASS", flush=True)
        # Reset copies are outside event interval and are synchronized before it.
        with wp.ScopedCapture(device=launch_args["device"]) as capture:
            execute(mode)
        start = wp.Event(enable_timing=True)
        end = wp.Event(enable_timing=True)
        values = []
        for repeat in range(45):
            reset()
            wp.synchronize()
            wp.record_event(start)
            wp.capture_launch(capture.graph)
            wp.record_event(end)
            wp.synchronize()
            if repeat >= 5:
                values.append(wp.get_event_elapsed_time(start, end))
        print("REPLAY_TIME", bool(inputs[-1]), mode, float(np.median(values)), flush=True)
        results[mode] = {
            "median_ms": float(np.median(values)),
            "min_ms": min(values),
            "max_ms": max(values),
            "samples_ms": values,
            "all_arrays_byte_exact": mode != "native",
            "physical_arrays_byte_exact": True,
        }
    trace_host = trace.numpy()
    if trace_path is not None:
        np.savez_compressed(trace_path, trace=trace_host)
    visited = np.isfinite(trace_host[0])
    metric = np.isfinite(trace_host[3])
    return {
        "visited_points": int(np.count_nonzero(visited)),
        "metric_calls": int(np.count_nonzero(metric)),
        "metric_broken_results": int(np.count_nonzero(metric & (trace_host[5] > 0))),
        "loaded_points_after": int(np.count_nonzero(visited & (trace_host[20] > 0))),
        "provenance": provenance,
        "biased": bool(inputs[-1]),
        "results": results,
        "scope": "One-kernel CUDA graph replay (no Python struct packing inside event interval). Frozen actual input, independent exact-value substitutions; differences are counterfactual costs, not additive exclusive timings. Reset/host/collision/factor/cache setup excluded. No ablated output enters simulation.",
    }


def main():
    """Warm normally, collect two actual launches eagerly, then benchmark clones."""
    assert os.environ.get("COLIBRI_CACHE_REGISTER") != "1"
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    frozen = os.environ.get("COLIBRI_REPLAY_FROZEN_PREFIX")
    if frozen:
        kernels = cached_component_live.load_kernels()
        report = [
            benchmark(
                load_snapshot(kernels.iterate_maximal_contact_runs_kernel, frozen, phase),
                output.with_suffix(".point_trace_" + ("biased" if phase else "relax") + ".npz"),
            )
            for phase in (True, False)
        ]
        output.with_suffix(".point_replay.json").write_text(json.dumps(report, indent=2))
        print("POINT_REPLAY", json.dumps(report), flush=True)
        return
    cached_component_live.install()
    original_run = newton.examples.run
    snapshots = {}

    def run(example, args):
        original_step = example.step
        frame = 0

        def step():
            nonlocal frame
            frame += 1
            if frame != 331:
                return original_step()
            original_launch = wp.launch
            graph = example.graph

            def capture(kernel, *positional, **kwargs):
                if (
                    kernel.key == "iterate_maximal_contact_runs_kernel"
                    and "cached_contact_gs" in kernel.func.__module__
                ):
                    phase = bool(kwargs["inputs"][-1])
                    if phase not in snapshots:
                        cloned, _ = clone_inputs(kwargs["inputs"])
                        snapshots[phase] = (
                            kernel,
                            {k: kwargs[k] for k in ("dim", "device", "block_dim") if k in kwargs},
                            cloned,
                        )
                return original_launch(kernel, *positional, **kwargs)

            example.graph = None
            wp.launch = capture
            try:
                return original_step()
            finally:
                wp.launch = original_launch
                example.graph = graph

        example.step = step
        return original_run(example, args)

    newton.examples.run = run
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        newton.examples.run = original_run
    assert set(snapshots) == {False, True}, snapshots.keys()
    for phase, snapshot in snapshots.items():
        saved = {}
        metadata = {}
        pointers = {}

        def save(value, prefix, storage=saved, meta=metadata, aliases=pointers):
            if isinstance(value, wp.array):
                storage[prefix] = value.numpy().copy()
                key = (value.ptr, value.shape, str(value.dtype))
                meta[prefix] = {
                    "kind": "array",
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                    "alias_of": aliases.get(key),
                }
                aliases.setdefault(key, prefix)
            elif hasattr(value, "_cls"):
                for name in value._cls.vars:
                    save(getattr(value, name), prefix + "__" + name, storage, meta, aliases)

            else:
                meta[prefix] = {"kind": "scalar_or_null", "value": getattr(value, "value", value)}

        for index, value in enumerate(snapshot[2]):
            save(value, "input" + str(index))
        output.with_suffix(".point_inputs_" + ("biased" if phase else "relax") + ".json").write_text(
            json.dumps(metadata, indent=2)
        )
        np.savez_compressed(output.with_suffix(".point_inputs_" + ("biased" if phase else "relax") + ".npz"), **saved)
    report = [
        benchmark(snapshots[phase], output.with_suffix(".point_trace_" + ("biased" if phase else "relax") + ".npz"))
        for phase in (True, False)
    ]
    output.with_suffix(".point_replay.json").write_text(json.dumps(report, indent=2))
    print("POINT_REPLAY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
