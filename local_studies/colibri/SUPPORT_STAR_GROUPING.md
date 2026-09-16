# Local support-star coupling study

Canonical solver defaults are unchanged. This is a convergence experiment, not an exact trajectory optimization.

## Ownership

Recognize dynamic bodies with any static-contact candidate (without an impulse threshold). Merge supported bodies linked by bilateral constraints using deterministic minimum-body union. A star owns every static-contact chunk of its bodies and every incident bilateral row. Unsupported neighboring bodies are not recursively absorbed. Components exceeding64rows retain the original schedule intact. Remaining rows retain their original consecutive-color partition.

Within each partition, variable CSR preserves original color order and original row order. Each stage has disjoint endpoints; different partitions use independently counted mass copies. Existing count-scaled response, common-point impulses and averaging remain unchanged.

## Validation

- CPU and CUDA graph tests cover overlapping stars, support withdrawal and component capacity fallback; every row is visited once, no same-stage endpoint conflict, original order preserved.
- Two unequal-mass supported bodies joined by a hinge exercise the actual grouped biased/relax callbacks. Independent six-component momentum change equals the measured stationary-support wrench, including warm-start impulses. Relaxation energy does not increase. Both supported bodies have one copy.
- Full public geometric30x1/120Hz330frames passes: peakdepth70.34µm, anchor0.4314mm, axis0.00817rad. Artifact `/tmp/colibri_support_star330.json` and copies NPZ.

## Initial result and limits

FrameGround had52points split into9chunks across3copies. The candidate assigns all9chunks and its incident joint to onecopy. Copy-to-physical residual differences disappear exactly. However, final loaded-contact mean slip is1.008mm/s versus1.019mm/s on the baseline trajectory, and maximum2.174 versus1.361mm/s. Later sequential contacts still disturb earlier contacts. These are different trajectories; they do not establish improved total drift.

The prototype takes roughly34ms/frame during correctness overlap, not an isolated timing claim. The simple regroup builder scans original rows for each partition and is intentionally not performance-ready. No promotion is justified. A same-input restored-state replay is prepared in `freeze_support_star.py`; it validates baseline replay twice before the candidate and again afterward, including body velocities, contact history and bilateral impulses.

## Frozen same-input comparison

`freeze_support_star.py` captures355reachable mutable device arrays at the public330 final post-relax state. It restores between reference/reference/support-star/reference replay, rebuilding ownership and factors and running exactly one biased dispatcher solve. The reference repeats are byte-identical for body velocities, angular velocities, contact impulses/history/derived fields and bilateral impulses. This is a diagnostic extra solve from post-relax state, not a trajectory stage or a new accepted iteration budget.

On identical witnesses, loaded-contact mean slip improves from1.021 to0.605mm/s. However, point27 separating normal velocity grows from2.20 to24.51mm/s. This is a tradeoff, not a demonstrated physical-quality improvement. The direct static-support momentum/work tests still pass; the stronger normal response arises in the biased coupled solve. Artifact `/tmp/colibri_support_star_frozen330.json`. Do not promote this grouping based on tangential slip alone.
