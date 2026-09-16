# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Read-only classification of actual material-reference transfer at ingest."""

import json
from pathlib import Path

import warp as wp


@wp.kernel
def classify(
    count: wp.array[int],
    matched: wp.array[int],
    reuse: wp.array[int],
    previous_cid: wp.array[int],
    previous: wp.array2d[float],
    current: wp.array2d[float],
    stats: wp.array[int],
):
    k = wp.tid()
    if k < count[0]:
        old = matched[k]
        if reuse[0] != 0:
            old = k
        reason = int(0)
        if old >= 0:
            reason = 1
            if previous_cid[old] >= 0:
                reason = 2
                if previous[12, old] == 0.0:
                    reason = 3
                    ndot = float(0.0)
                    for j in range(3):
                        ndot += previous[j, old] * current[j, k]
                    if ndot >= 0.95:
                        reason = 4
                        for j in range(13, 27):
                            if previous[j, old] != current[j, k]:
                                reason = 5
        wp.atomic_add(stats, reason, 1)


def install(solver, output):
    """Observe post-ingest buffers before subsequent material-history preparation."""
    world = solver.world
    cc = world._contact_container
    assert cc.lambdas.shape[0] == 27
    stats = wp.zeros(6, dtype=int, device=cc.lambdas.device)
    original = world._ingest_and_warmstart_contacts

    def ingest(*args, **kwargs):
        result = original(*args, **kwargs)
        matched = (
            world._ingest_scratch.sorted_match_index
            if world._enable_body_pair_grouping
            else world._contact_views.rigid_contact_match_index
        )
        wp.launch(
            classify,
            cc.lambdas.shape[1],
            inputs=[
                world._cc_valid_count,
                matched,
                world._reuse_contact_indices,
                world._cid_of_contact_prev,
                cc.prev_lambdas,
                cc.lambdas,
                stats,
            ],
            device=stats.device,
        )
        return result

    world._ingest_and_warmstart_contacts = ingest

    def finish():
        values = stats.numpy()
        Path(output).with_suffix(".history_transfer.json").write_text(
            json.dumps(
                {
                    "fields": [
                        "no_match",
                        "invalid_previous_column",
                        "previous_broken",
                        "normal_rejected",
                        "reference_retained",
                        "eligible_reference_mismatch",
                    ],
                    "counts": values.tolist(),
                    "scope": "Actual post-ingest 14-float birth-reference equality; read-only kernel; subsequent prepare resets not included",
                },
                indent=2,
            )
        )
        assert values[5] == 0, "Eligible matched reference not retained at ingest"

    return finish
