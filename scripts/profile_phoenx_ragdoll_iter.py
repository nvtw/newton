# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warm the full ragdoll scene, then expose one frame to Nsight Compute."""

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp
from cuda.bindings import runtime

from newton.examples.phoenx.example_phoenx_ragdoll_pile import Example
from newton.viewer import ViewerNull


def main():
    while True:
        utilization = int(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True
            ).splitlines()[0]
        )
        if utilization < 10:
            break
        print(f"GPU busy ({utilization}%); waiting for utilization below 10%.", flush=True)
        time.sleep(10)
    print(f"GPU idle gate: {utilization}%. Warp {wp.__version__}. Building 32,000 ragdolls.", flush=True)
    viewer = ViewerNull()
    try:
        example = Example(
            viewer, SimpleNamespace(grid=(40, 40, 20), seed=42, substeps=8, iterations=4, joint_solver="exact")
        )
        wp.synchronize()
        started = time.perf_counter()
        for frame in range(300):
            example.step()
            if (frame + 1) % 60 == 0:
                wp.synchronize()
                print(f"Warmup {frame + 1}/300 · elapsed {time.perf_counter() - started:.1f} s", flush=True)
        print("Profiling two constraint-iterate launches in frame 301.", flush=True)
        (error,) = runtime.cudaProfilerStart()
        if error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaProfilerStart failed: {error}")
        try:
            example.step()
            wp.synchronize()
        finally:
            runtime.cudaProfilerStop()
        assert np.isfinite(example.state.body_q.numpy()).all()
        assert np.isfinite(example.state.body_qd.numpy()).all()
        example.test_post_step()
        print("Profile frame complete. Rendering is excluded from these kernel measurements.", flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
