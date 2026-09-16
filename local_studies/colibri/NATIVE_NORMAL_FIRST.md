# All normals before original point friction

`native_normal_first.py` is an isolated native-contact callback replacement.
Within each articulation it visits all original normal rows, then all original
point-friction rows. It retains each contact's actual load, Coulomb metric,
friction break flag, history, targets, lever arms and paired impulse application.
It does not create pooled anchors or redistribute normal loads. The dispatcher
still removes/restores joint bias exactly once around the same callback.

Run with the corrected no-skip fixture:

```
COLIBRI_DIFFERENTIAL_REFERENCE=1 COLIBRI_DIFFERENTIAL_NORMALIZED=1 \
COLIBRI_STAGE_RUNNER=local_studies.colibri.native_normal_first \
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
-m local_studies.colibri.direct_relax_no_skip --native-path direct \
--body-count 2 --frames 600 --substeps 30 --save-history \
--output /tmp/colibri_normal_first600.json
```

Three CPU tests pass: single-point equivalence, a coupled two-point ordering
control with original capacities and impulse/velocity ledger, and preservation
of the exact point-friction source/projection and complete loops. These tests
isolate the schedule and do not certify all live CUDA force behavior. The
actual GPU callback compiled and completed 2-frame and 600-frame runs.

## Result

Same 30 substeps per 120 Hz collision update, SOR 1 and source physics:

| Metric | Original no-skip native | Normal-first |
|---|---:|---:|
| Base creep, 5–10 seconds | 2.25892 um/s | 2.07196 um/s |
| Synchronized step FPS | 20.4885 | 19.1011 |
| Final hinge | near 18.2073 degrees | 18.2073313 degrees |
| Final implicit drive residual | 1.37e-9 | -2.94e-9 |

All finite/source/geometry checks pass. The small creep decrease does not match
the earlier pooled-anchor hybrid's 0.0165 um/s, and the schedule is slower.
Ordering alone therefore does not explain the hybrid's large improvement.
This is an unpromoted diagnostic, not a stable high-performance solution.
No long-run extension or production change was made.

Artifacts: `/tmp/colibri_normal_first600.{json,npz,motion.json,normal_first.json}`.
