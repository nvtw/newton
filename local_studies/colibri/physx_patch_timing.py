"""Measure patch observer overhead with a mandatory saved trajectory byte gate.

Use as COLIBRI_STAGE_RUNNER under native_conditioned_two_body. This first
control removes observer launches only; it retains solver ledger writes to
avoid changing the physical kernel compilation. It is not production code.
"""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri import physx_patch_reference_live as live


def main():
    """Run unchanged hybrid physics while excluding observation-only launches."""
    reference = Path(os.environ["COLIBRI_PATCH_TIMING_REFERENCE"])
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    assert output.with_suffix(".npz") != reference
    source_hash = hashlib.sha256(Path(live.__file__).read_bytes()).hexdigest()
    observed = {
        live.kernels.record_before,
        live.kernels.record_joint,
        live.kernels.record_endpoints,
        live.kernels.record_after,
        live.kernels.advance_record,
    }
    original_launch = wp.launch
    skipped = dict.fromkeys((k.key for k in observed), 0)

    def launch(kernel, *args, **kwargs):
        if kernel in observed:
            skipped[kernel.key] += 1
            return None
        return original_launch(kernel, *args, **kwargs)

    states, generated = live.install()
    wp.launch = launch
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        wp.launch = original_launch
    assert hashlib.sha256(Path(live.__file__).read_bytes()).hexdigest() == source_hash
    for data in states.values():
        assert not np.any(data["state"].error.numpy()), "Unsupported patch geometry/ownership"
    assert states and all(skipped.values()), "Observer interception did not execute"
    with np.load(reference) as old, np.load(output.with_suffix(".npz")) as new:
        keys = ("q_history", "qd_history", "history_times")
        for key in keys:
            a, b = old[key], new[key]
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.tobytes() == b.tobytes(), f"Trajectory byte mismatch: {key}"
    output.with_suffix(".observer_timing.json").write_text(
        json.dumps(
            {
                "byte_gate": list(keys),
                "reference": str(reference),
                "hybrid_source_sha256": source_hash,
                "generated_source": str(generated),
                "observer_launches_skipped_during_capture_or_eager": skipped,
                "solver_ledger_writes_retained": True,
                "scope": "Diagnostic overhead only; hybrid physical limitations unchanged",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
