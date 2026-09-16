"""Collect every frozen factor difference and independent Gram backward error."""

import argparse
import inspect
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import warp as wp

from local_studies.colibri.cooperative_bilateral_prepare import launch
from newton._src.solvers.phoenx.tests import test_bilateral_preparation as existing


def measure(local):
    data = local["data"]
    case = local["case"]
    outputs = local["outputs"]
    count = case["count"]
    cid = 0
    indices = data.row_indices.numpy()
    row_local = data.row_local.numpy()
    structural = int(data.structural_index.numpy()[cid])
    w0 = data.wrench0.numpy()
    w1 = data.wrench1.numpy()
    dynamic = data.row_dynamic.numpy()
    mass = data.dynamic_mass.numpy()
    mp.mp.dps = 60
    gram = mp.zeros(count, count)
    for i in range(count):
        row = int(indices[cid, i])
        l = int(row_local[row])
        for j in range(i + 1):
            value = mp.mpf(0)
            for wrench, response in ((w0, outputs[0]["response0"]), (w1, outputs[0]["response1"])):
                for c in range(6):
                    value += mp.mpf(float(wrench[structural, l, c])) * mp.mpf(float(response[cid, j, c]))
            if i == j and dynamic[row]:
                value += 1 / mp.mpf(float(mass[row]))
            gram[i, j] = value
            gram[j, i] = value
    differences = {}
    for name in outputs[0]:
        a, b = outputs[0][name], outputs[1][name]
        differences[name] = {
            "byte_equal": a.tobytes() == b.tobytes(),
            "max_absolute": float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0)),
            "differing_values": int(np.count_nonzero(a != b)),
        }
    errors = []
    for output in outputs:
        if count:
            L = mp.matrix(output["lower"][cid, :count, :count].tolist())
            D = mp.diag(output["diagonal"][cid, :count].tolist())
            delta = L * D * L.T - gram
            absolute = float(max(abs(x) for x in delta))
            scale = float(max(abs(x) for x in gram))
            relative = absolute / scale if scale else absolute
            condition = float(np.linalg.cond(np.array(gram.tolist(), dtype=float)))
        else:
            absolute = relative = 0.0
            condition = None
        errors.append(
            {
                "absolute": absolute,
                "relative": relative,
                "gram_condition_fp64": condition,
                "valid": int(output["valid"][cid]),
            }
        )
    return {
        "case": case,
        "differences": differences,
        "scalar_backward_error": errors[0],
        "cooperative_backward_error": errors[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-dim", type=int, default=32)
    parser.add_argument(
        "--original-candidate", action="store_true", help="Measure the preserved pre-intrinsic candidate"
    )
    parser.add_argument("--output", default="/tmp/cooperative_prepare_all_cases.json")
    args = parser.parse_args()
    if args.original_candidate:
        from local_studies.colibri import cooperative_bilateral_prepare as candidate_module
        from local_studies.colibri.tail_first_dispatch import _compile_function

        source = Path("/tmp/cooperative_bilateral_prepare_before_explicit.py").read_text()
        candidate_module.cooperative_prepare = _compile_function(source, candidate_module, "cooperative_prepare")
    original_launch = wp.launch
    original_assert = np.testing.assert_array_equal
    reports = []

    def redirect(kernel, *positional, **kwargs):
        if kernel is existing.prepare_bilateral_joint_blocks:
            dim = kwargs.get("dim", positional[0] if positional else None)
            inputs = kwargs.get("inputs", positional[1] if len(positional) > 1 else None)
            return launch(*inputs, dim, kwargs.get("device", "cuda:0"), args.block_dim)
        return original_launch(kernel, *positional, **kwargs)

    def collect(*positional, **kwargs):
        frame = inspect.currentframe().f_back
        if frame.f_code.co_name == "test_fixed_bounds_match_active_rows_and_mass_scales":
            if frame.f_locals["field"] == "valid":
                reports.append(measure(frame.f_locals))
            return
        return original_assert(*positional, **kwargs)

    wp.launch = redirect
    np.testing.assert_array_equal = collect
    try:
        existing.TestBilateralPreparation().test_fixed_bounds_match_active_rows_and_mass_scales()
    finally:
        wp.launch = original_launch
        np.testing.assert_array_equal = original_assert
    assert len(reports) == 84
    result = {
        "diagnostic_only": True,
        "exact_optimization_gate_waived": False,
        "cases": reports,
        "exact_cases": sum(all(v["byte_equal"] for v in row["differences"].values()) for row in reports),
        "validity_differences": sum(not row["differences"]["valid"]["byte_equal"] for row in reports),
        "maximum_scalar_backward_error": max(row["scalar_backward_error"]["relative"] for row in reports),
        "maximum_cooperative_backward_error": max(row["cooperative_backward_error"]["relative"] for row in reports),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
