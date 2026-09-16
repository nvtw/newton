"""Execute target-only projection kernels on a saved actual contact container."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.shared_recovery_live import invalidate, project_targets, restore_targets, save_targets
from local_studies.colibri.shared_recovery_reference import project
from newton._src.solvers.phoenx.constraints.constraint_contact import contact_column_container_zeros
from newton._src.solvers.phoenx.constraints.contact_container import contact_container_zeros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--snapshot", type=Path, default=Path("/tmp/colibri_native_direct_owned_fixed3600.native_state.npz")
    )
    args = parser.parse_args()
    x = np.load(args.snapshot)
    device = args.device
    cc = contact_container_zeros(x["contact_lambdas"].shape[1], device=device)
    for name, key in [("lambdas", "contact_lambdas"), ("derived", "contact_derived"), ("impulses", "contact_impulses")]:
        setattr(cc, name, wp.array(x[key], dtype=float, device=device))
    cols = contact_column_container_zeros(x["contact_columns_data"].shape[1], device=device)
    cols.data = wp.array(x["contact_columns_data"], dtype=float, device=device)
    cap, ncols = cc.lambdas.shape[1], cols.data.shape[1]
    valid = wp.zeros(1, dtype=int, device=device)
    weights = wp.array(np.maximum(x["contact_impulses"][0], 0), dtype=float, device=device)
    saved = wp.zeros((2, cap), dtype=float, device=device)
    diagnostics = wp.zeros((5, ncols), dtype=float, device=device)
    before = cc.derived.numpy().copy()
    wp.launch(save_targets, dim=cap, inputs=[cc, saved], device=device)
    wp.launch(project_targets, dim=ncols, inputs=[cc, cols, weights, valid, saved, diagnostics], device=device)
    np.testing.assert_array_equal(cc.derived.numpy(), before)
    valid.fill_(1)
    wp.launch(project_targets, dim=ncols, inputs=[cc, cols, weights, valid, saved, diagnostics], device=device)
    after = cc.derived.numpy().copy()
    mask = np.ones(before.shape[0], dtype=bool)
    mask[4:6] = False
    np.testing.assert_array_equal(after[mask], before[mask])
    np.testing.assert_array_equal(cc.impulses.numpy(), x["contact_impulses"])
    np.testing.assert_array_equal(cc.lambdas.numpy(), x["contact_lambdas"])
    header = x["contact_columns_data"].view(np.int32)
    max_error = 0.0
    active = 0
    for col in range(ncols):
        first, count = header[5:7, col]
        if count <= 0 or first < 0 or first + count > cap:
            continue
        ids = np.arange(first, first + count)
        n = x["contact_lambdas"][:3, ids].T.astype(float)
        t = x["contact_lambdas"][3:6, ids].T.astype(float)
        tangents = np.stack([t, np.cross(n, t)], axis=1)
        radius = x["contact_columns_data"][3, col] * np.maximum(x["contact_impulses"][0, ids], 0)
        expected, report = project(
            before[9:12, ids].T.astype(float), tangents, n[0], radius, before[4:6, ids].T.astype(float)
        )
        if report["status"] == "PROJECTED_NUMERICAL_RECOVERY_ONLY":
            active += 1
            max_error = max(max_error, float(np.max(abs(after[4:6, ids].T - expected))))
            np.testing.assert_allclose(after[4:6, ids].T, expected, rtol=0, atol=1e-10)
        else:
            np.testing.assert_array_equal(after[4:6, ids], before[4:6, ids])
    assert active == 1
    wp.launch(restore_targets, dim=cap, inputs=[cc, saved], device=device)
    np.testing.assert_array_equal(cc.derived.numpy(), before)
    wp.launch(invalidate, dim=1, inputs=[valid], device=device)
    assert valid.numpy()[0] == 0
    # A point-like patch has no spin recovery coordinate: no regularization or truncation.
    collapsed = before.copy()
    collapsed[9:12] = 0
    cc.derived.assign(collapsed)
    valid.fill_(1)
    wp.launch(save_targets, dim=cap, inputs=[cc, saved], device=device)
    wp.launch(project_targets, dim=ncols, inputs=[cc, cols, weights, valid, saved, diagnostics], device=device)
    np.testing.assert_array_equal(cc.derived.numpy(), collapsed)
    if wp.get_device(device).is_cuda:
        cc.derived.assign(before)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(save_targets, dim=cap, inputs=[cc, saved], device=device)
            wp.launch(project_targets, dim=ncols, inputs=[cc, cols, weights, valid, saved, diagnostics], device=device)
            wp.launch(restore_targets, dim=cap, inputs=[cc, saved], device=device)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(cc.derived.numpy(), before)
    result = {
        "device": device,
        "passed": True,
        "active_patch_count": active,
        "max_FP32_vs_FP64_target_error_m_s": max_error,
        "gates": [
            "first-phase unchanged",
            "all force/history/normal/other rows byte-exact",
            "mu0 unchanged",
            "restore exact",
            "ingest invalidation",
            "singular field unchanged",
        ],
    }
    print(json.dumps(result, indent=2))
    Path("/tmp/colibri_shared_recovery_" + device.replace(":", "_") + ".json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
