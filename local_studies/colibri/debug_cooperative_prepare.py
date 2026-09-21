"""Capture first exactness failure and distinguish matrix from LDL rounding."""

import inspect
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri import cooperative_bilateral_prepare as cooperative
from local_studies.colibri.check_cooperative_bilateral_prepare import FIELDS, snapshot_fixture
from local_studies.colibri.tail_first_dispatch import _compile_function
from newton._src.solvers.phoenx.tests import test_bilateral_preparation as reference


def kernels():
    source = inspect.getsource(reference._prepare_dynamic_reference.func)
    source = source.replace("_prepare_dynamic_reference(", "capture_scalar_prepare(").replace(
        "copy_state: CopyStateContainer):", "copy_state: CopyStateContainer, debug: wp.array2d[wp.float64]):"
    )
    source = source.replace(
        "    lower = Mat66d()",
        "    for di in range(6):\n        for dj in range(6):\n            debug[cid,di*6+dj]=matrix[di,dj]\n    lower = Mat66d()",
    )
    source = source.replace(
        "            pivot -= lower[i, j] * lower[i, j] * diagonal[j]",
        "            debug[cid,42+i*6+j]=pivot\n            pivot -= lower[i, j] * lower[i, j] * diagonal[j]\n            debug[cid,78+i*6+j]=pivot",
    )
    source = source.replace("        diagonal[i] = pivot", "        debug[cid,36+i]=pivot\n        diagonal[i] = pivot")
    scalar = _compile_function(source, reference, "capture_scalar_prepare")
    source = inspect.getsource(cooperative.cooperative_prepare.func)
    source = source.replace("cooperative_prepare(", "capture_cooperative_prepare(").replace(
        "copy_state: CopyStateContainer):", "copy_state: CopyStateContainer, debug: wp.array2d[wp.float64]):"
    )
    source = source.replace(
        "    lower_row = Vec6d()",
        "    if lane<6:\n        for dj in range(6):\n            debug[cid,lane*6+dj]=matrix_row[dj]\n    lower_row = Vec6d()",
    )
    source = source.replace(
        "                    pivot_local -= lower_row[j] * lower_row[j] * diagonal[j]",
        "                    debug[cid,42+i*6+j]=pivot_local\n                    pivot_local -= lower_row[j] * lower_row[j] * diagonal[j]\n                    debug[cid,78+i*6+j]=pivot_local",
    )
    source = source.replace(
        "            diagonal[i] = pivot",
        "            if lane==0:\n                debug[cid,36+i]=pivot\n            diagonal[i] = pivot",
    )
    candidate = _compile_function(source, cooperative, "capture_cooperative_prepare")
    return scalar, candidate


def main():
    original = wp.launch

    def redirect(kernel, *positional, **kwargs):
        if kernel is reference.prepare_bilateral_joint_blocks:
            dim = kwargs.get("dim", positional[0] if positional else None)
            inputs = kwargs.get("inputs", positional[1] if len(positional) > 1 else None)
            return cooperative.launch(*inputs, dim, kwargs.get("device", "cuda:0"), 32)
        return original(kernel, *positional, **kwargs)

    wp.launch = redirect
    failure = None
    try:
        reference.TestBilateralPreparation().test_fixed_bounds_match_active_rows_and_mass_scales()
    except AssertionError as error:
        traceback = error.__traceback__
        while traceback:
            if traceback.tb_frame.f_code.co_name == "test_fixed_bounds_match_active_rows_and_mass_scales":
                failure = traceback.tb_frame.f_locals.copy()
                break
            traceback = traceback.tb_next
        if failure is None:
            raise
    finally:
        wp.launch = original
    assert failure is not None, "Prior failing fixture no longer reproduces"
    world = failure["world"]
    data = world.constraints.bilateral
    arrays = {
        "joint_data": world.constraints.data.numpy(),
        "inverse_mass": world.bodies.inverse_mass.numpy(),
        "inverse_inertia": world.bodies.inverse_inertia_world.numpy(),
        "copy_count": world._copy_state.count_per_node.numpy(),
    }
    for name in (
        "row_count",
        "row_indices",
        "structural_index",
        "row_local",
        "row_dynamic",
        "wrench0",
        "wrench1",
        "dynamic_mass",
    ):
        arrays["joint_" + name] = getattr(data, name).numpy()
    path = "/tmp/cooperative_prepare_first_failure.npz"
    np.savez(path, **arrays)
    constraints, bodies, copies, count = snapshot_fixture(path)
    outputs = []
    debugs = []
    for index, kernel in enumerate(kernels()):
        for name in FIELDS:
            getattr(constraints.bilateral, name).zero_()
        debug = wp.zeros((count, 114), dtype=wp.float64, device="cuda:0")
        wp.launch(
            kernel, count * (8 if index else 1), [constraints, bodies, copies, debug], device="cuda:0", block_dim=32
        )
        debugs.append(debug.numpy())
        outputs.append({name: getattr(constraints.bilateral, name).numpy() for name in FIELDS})
    active = int(arrays["joint_row_count"][0])
    entries = (
        [i * 6 + j for i in range(active) for j in range(i + 1)]
        + list(range(36, 36 + active))
        + [offset + i * 6 + j for offset in (42, 78) for i in range(active) for j in range(i)]
    )
    differences = [
        {"index": int(i), "scalar": float(debugs[0][0, i]), "cooperative": float(debugs[1][0, i])}
        for i in entries
        if debugs[0][0, i].tobytes() != debugs[1][0, i].tobytes()
    ]
    report = {
        "case": failure["case"],
        "differences": differences,
        "instrumented_outputs_equal": {
            name: outputs[0][name].tobytes() == outputs[1][name].tobytes() for name in FIELDS
        },
        "instrumentation_preserves_original_outputs": [
            {name: outputs[i][name].tobytes() == failure["outputs"][i][name].tobytes() for name in FIELDS}
            for i in (0, 1)
        ],
    }
    Path("/tmp/cooperative_prepare_first_difference.json").write_text(json.dumps(report, indent=2))
    np.savez("/tmp/cooperative_prepare_first_difference.npz", scalar=debugs[0], cooperative=debugs[1])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
