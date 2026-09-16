"""Exact per-contact parallelism for native maximal contact mobility refresh.

Local diagnostic only. Each lane owns disjoint contact output slots; all native
factor/geometry arithmetic remains in the original Warp helper.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations import maximal_contact_gs as gs
from newton._src.solvers.phoenx.constraints import constraint_contact as cc


def extract(source, name):
    """Extract a complete decorated function for strict source equivalence."""
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(source.splitlines()[node.decorator_list[0].lineno - 1 : node.end_lineno])


def make_kernel(*, negative_control=False):
    """Change ownership alone while retaining the actually imported native helper."""
    source = Path(gs.__file__).read_text()
    name = "refresh_maximal_contact_mobility_kernel"
    original = extract(source, name)
    helper = extract(source, "_write_exact_contact_mobility")
    staged = Path(
        "/tmp/colibri_merged_friction_proposal/newton/_src/solvers/phoenx/articulations/maximal_contact_gs.py"
    )
    assert original == extract(staged.read_text(), name), "Saved corrected owned refresh differs"
    assert helper == extract(staged.read_text(), "_write_exact_contact_mobility"), "Saved corrected owned math differs"
    gate = "    if lane != wp.int32(0):\n        return\n"
    loop = "        for offset in range(count):"
    assert original.count(gate) == original.count(loop) == 1
    transformed = original.replace("def " + name, "def parallel_refresh")
    if not negative_control:
        transformed = transformed.replace(gate, "")
    transformed = transformed.replace(loop, "        for offset in range(lane, count, wp.int32(_TREE_WIDTH)):")
    imports = """import warp as wp
from newton._src.solvers.phoenx.articulations.maximal_contact_gs import (
    MaximalTreeProjectorData, MaximalContactResponseData, BodyContainer,
    ContactColumnContainer, ContactContainer, _TREE_WIDTH,
    _write_exact_contact_mobility, contact_get_body1, contact_get_body2,
    contact_get_contact_first, contact_get_contact_count,
)
"""
    directory = Path(tempfile.mkdtemp(prefix="colibri_parallel_mobility_"))
    path = directory / "kernels.py"
    path.write_text(imports + original.replace("def " + name, "def serial_refresh") + "\n\n" + transformed + "\n")
    module_name = "local_studies.colibri.parallel_mobility_generated"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return (
        module.parallel_refresh,
        module.serial_refresh,
        {
            "original_path": gs.__file__,
            "original_kernel_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "original_helper_sha256": hashlib.sha256(helper.encode()).hexdigest(),
            "corrected_owned_source_equal": True,
            "candidate_path": str(path),
            "negative_control": negative_control,
        },
    )


def load_inputs(prefix, phase, device):
    """Restore the actual saved launch's exact native refresh inputs and aliases."""
    base = str(prefix) + ".point_inputs_" + ("biased" if phase else "relax")
    archive = np.load(base + ".npz")
    metadata = json.loads(Path(base + ".json").read_text())
    restored = {}

    def restore(dtype, key):
        if key not in metadata:
            value = dtype()
            for name, field in dtype.vars.items():
                setattr(value, name, restore(field.type, key + "__" + name))
            return value
        entry = metadata[key]
        if entry["kind"] == "scalar_or_null":
            return None if entry["value"] is None else dtype(entry["value"])
        alias = entry["alias_of"]
        if alias is not None and alias in restored:
            return restored[alias]
        value = wp.array(archive[key], dtype=dtype.dtype, device=device)
        assert list(value.shape) == entry["shape"]
        restored[key] = value
        return value

    # Exact source launch has an extra response-cache argument, excluded here.
    indices = (0, 1, 3, 5, 6, 9, 10, 11)
    return [
        restore(arg.type, "input" + str(i))
        for arg, i in zip(gs.refresh_maximal_contact_mobility_kernel.adj.args, indices, strict=True)
    ]


def arrays_of(inputs):
    """Collect all input arrays for the read-only factor/geometry gate."""
    arrays = {}

    def walk(value):
        if isinstance(value, wp.array):
            arrays[(value.ptr, value.shape)] = value
        elif hasattr(value, "_cls"):
            for name in value._cls.vars:
                walk(getattr(value, name))

    for value in inputs:
        walk(value)
    return list(arrays.values())


def modify_case(inputs, count, internal):
    """Build a >64-point single column using real geometry and exact factor data."""
    columns, contacts, scheduled, ends = inputs[3:7]
    column = int(scheduled.numpy()[0])
    data = columns.data.numpy().copy()
    ints = data.view(np.int32)
    source = int(ints[cc._OFF_CONTACT_FIRST, column])
    first = 1000
    ints[cc._OFF_CONTACT_FIRST, column] = first
    ints[cc._OFF_CONTACT_COUNT, column] = count
    if internal:
        slots = inputs[0].body_slot.numpy()[0]
        ints[cc._OFF_BODY1, column] = slots[0]
        ints[cc._OFF_BODY2, column] = slots[1]
    columns.data.assign(data)
    ends.assign(np.ones(ends.shape, dtype=np.int32))
    for name in contacts._cls.vars:
        value = getattr(contacts, name)
        if isinstance(value, wp.array):
            data = value.numpy().copy()
            data[:, first : first + count] = data[:, source : source + 1]
            value.assign(data)
    return list(range(first, first + count))


def measure(kernel, inputs):
    """Time graph-only single launches with output reset outside the interval."""
    device = inputs[-1].device
    launch = {"dim": inputs[0].body_count.shape[0] * gs._TREE_WIDTH, "block_dim": gs._TREE_WIDTH, "device": device}
    with wp.ScopedCapture(device=device) as capture:
        wp.launch(kernel, inputs=inputs, **launch)
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    samples = []
    for repeat in range(45):
        inputs[-1].fill_(-9876.5)
        wp.synchronize_device(device)
        wp.record_event(start)
        wp.capture_launch(capture.graph)
        wp.record_event(end)
        wp.synchronize_device(device)
        if repeat >= 5:
            samples.append(wp.get_event_elapsed_time(start, end))
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def main():
    """Gate real snapshots and true per-column overflow, then time actual workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="/tmp/colibri_point_sink_replay331")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    candidate, serial_clone, provenance = make_kernel(negative_control=args.negative_control)
    reference = gs.refresh_maximal_contact_mobility_kernel if wp.get_device(args.device).is_cuda else serial_clone
    results = []
    cases = [("actual_biased", True, None, False), ("actual_relax", False, None, False)]
    cases += [(f"external_{count}", True, count, False) for count in (0, 1, 63, 64, 65, 129)]
    cases += [(f"internal_{count}", True, count, True) for count in (1, 65, 129)]
    for label, phase, count, internal in cases:
        inputs = load_inputs(args.prefix, phase, args.device)
        touched = None if count is None else modify_case(inputs, count, internal)
        output = inputs[-1]
        readonly = [a for a in arrays_of(inputs) if a.ptr != output.ptr]
        before = [a.numpy().tobytes() for a in readonly]
        launch = {
            "dim": inputs[0].body_count.shape[0] * gs._TREE_WIDTH,
            "block_dim": gs._TREE_WIDTH,
            "device": args.device,
        }
        output.fill_(-9876.5)
        wp.launch(reference, inputs=inputs, **launch)
        expected = output.numpy().copy()
        output.fill_(-9876.5)
        wp.launch(candidate, inputs=inputs, **launch)
        actual = output.numpy().copy()
        assert actual.tobytes() == expected.tobytes(), (label, float(np.max(np.abs(actual - expected))))
        assert before == [a.numpy().tobytes() for a in readonly], label + " mutated read-only input"
        if touched is not None:
            keep = np.ones(actual.shape[1], dtype=bool)
            keep[touched] = False
            assert np.all(actual[:, keep] == np.float32(-9876.5))
            if count:
                assert np.any(actual[:, touched[-1]] != np.float32(-9876.5)), "Last point omitted"
        result = {"case": label, "byte_equal": True, "readonly_unchanged": True}
        if count is None and wp.get_device(args.device).is_cuda:
            result["native"] = measure(gs.refresh_maximal_contact_mobility_kernel, inputs)
            result["parallel"] = measure(candidate, inputs)
        results.append(result)
        print("PARALLEL_REFRESH", json.dumps(result), flush=True)
    Path(args.output).write_text(
        json.dumps({"provenance": provenance, "device": args.device, "cases": results}, indent=2)
    )


if __name__ == "__main__":
    main()
