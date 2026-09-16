"""Actual Warp FP32 proposal checked with independent physical equations."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.check_full_wrench_fp32 import audit
from local_studies.colibri.full_wrench_candidate import propose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    x = np.load("/tmp/colibri_coupled_normal_stick_reference.npz")
    ids = x["eligible"]
    n = len(ids)
    device = args.device
    inputs = [
        wp.array(x["mobility"][None].astype(np.float32), dtype=wp.spatial_matrixf, device=device),
        wp.array(x["velocity"][None, :6].astype(np.float32), dtype=wp.spatial_vectorf, device=device),
        wp.array(x["point_map"].T.reshape(n, 3, 6).astype(np.float32), dtype=wp.spatial_vectorf, device=device),
        wp.array(x["impulses"][ids].T.astype(np.float32), dtype=float, device=device),
        wp.array(x["coefficients"][ids].astype(np.float32), dtype=float, device=device),
        wp.ones(n, dtype=int, device=device),
        wp.array(np.array([0, n], np.int32), dtype=int, device=device),
    ]
    candidate = wp.zeros((3, n), dtype=float, device=device)
    status = wp.zeros(1, dtype=int, device=device)
    requested = wp.zeros(1, dtype=wp.spatial_vectorf, device=device)
    inputs.extend([candidate, status, requested])
    wp.launch(propose, dim=1, inputs=inputs, device=device)
    assert status.numpy()[0] == 1
    proposed = candidate.numpy().T.copy()
    report = audit(x, proposed)
    assert report["contact_and_homogeneous_joint_certificate"], report
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(propose, dim=1, inputs=inputs, device=device)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(candidate.numpy().T, proposed)
    before = inputs[3].numpy().copy()
    inputs[4].zero_()
    wp.launch(propose, dim=1, inputs=inputs, device=device)
    assert status.numpy()[0] < 0
    np.testing.assert_array_equal(inputs[3].numpy(), before)
    report.update(device=device, proposal_status=1, rejected_mu0_preserves_input=True)
    Path("/tmp/colibri_full_wrench_warp_" + device.replace(":", "_") + ".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
