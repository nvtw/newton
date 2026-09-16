"""Frozen captured GPU timing; reset input impulses before every replay."""

import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.coupled_support_online import assemble_snapshot
from local_studies.colibri.two_body_condensed_gpu import solve


def main(output="/tmp/colibri_condensed_gpu_timing.json"):
    """Time only prepared device inputs and three native response/PGS kernels."""
    wp.init()
    z = np.load("/tmp/colibri_base_frame_totalnormal_phases330.npz")
    reports = []
    for phase in ("biased", "relax"):
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith(phase + "_solved.")}
        a = assemble_snapshot(d, phase, float(z["dt"][0]), int(z["num_joints"][0]))
        for sweeps in (16, 32):
            calls = []
            launch = wp.launch

            def record(*args, calls=calls, launch=launch, **kwargs):
                calls.append((args, kwargs))
                return launch(*args, **kwargs)

            wp.launch = record
            try:
                reference = solve(a, sweeps)
            finally:
                wp.launch = launch
            assert len(calls) == 3
            impulse = calls[-1][1]["inputs"][7]
            old = wp.array(a["old"], dtype=wp.float64, device="cuda:0")
            with wp.ScopedCapture(device="cuda:0") as capture:
                wp.copy(impulse, old)
                for args, kwargs in calls:
                    launch(*args, **kwargs)
            for _ in range(5):
                wp.capture_launch(capture.graph)
            wp.synchronize_device("cuda:0")
            begin = time.perf_counter()
            for _ in range(100):
                wp.capture_launch(capture.graph)
            wp.synchronize_device("cuda:0")
            ms = (time.perf_counter() - begin) * 10
            np.testing.assert_array_equal(impulse.numpy(), reference[0])
            reports.append(
                {
                    "phase": phase,
                    "sweeps": sweeps,
                    "milliseconds": ms,
                    "replays": 100,
                    "input_reset": True,
                    "scope": "Synchronized captured reset+factor+local blocks+PGS; excludes host assembly, collision, upload and body writeback",
                }
            )
    Path(output).write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
