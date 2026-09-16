# Eight-lane joint RHS scheduling probe

A standalone frozen Colibri RHS microbenchmark evaluates188active rows across
38joints after330public frames. It materializes the actual wrench, copy-slot
twist, bias and dynamic-drive inputs once. Both implementations read identical
immutable inputs and produce byte-identical FP64 outputs after every timed
batch. No ablated output is continued as a trajectory.

| Scheduling | Mean kernel time |
| --- | ---: |
| One thread per joint, sequential rows | 8.207us |
| Eight lanes per joint, one lane per active row | 2.932us |

Thirty CUDA-graph batches contain100replays each; the first five batches are
excluded. This is2.80x for this materialized RHS kernel only. It excludes
materialization, native dependent gathers, grouped color barriers, the
triangular solve, impulse application and contact work. It is not a solver
speedup or a physics validation. No canonical source changed.

Each active lane calls the same _dot_double helper, accumulating the six FP64
components in their original order. Normalizing or parallel-reducing those
six terms would be a different experiment. The RHS probe requires no warp
shuffle because its rows are independent.

## Implications for an actual grouped callback

The parent audited38joint rows in13colors, located in only four independent
groups. Median joint occupancy is2of32lanes per color. Up to10joints in one
color require80lanes if expanded into eight-lane tiles. A wider group block
or a separate mapping is required; the existing32-thread launch cannot simply
substitute an eight-lane callback.

Existing reduced.py native helpers demonstrate width/mask-aware
__shfl_sync for FP32 and spatial vectors. A future FP64 shuffle helper must
use explicit valid eight-lane tile masks and all required participating lanes.

Exact triangular ordering remains mandatory:
- Forward elimination may broadcast solved rows in ascending order while
  each later row subtracts contributions in that same ascending order.
- Diagonal scaling is independent per row, using unchanged division.
- Back substitution currently subtracts higher-row terms in ascending index
  order for each descending solution row. Broadcasting solved values in
  descending order and immediately subtracting them changes arithmetic order.
  Do not claim such an elimination variant is bit-exact.
- Final wrench/impulse accumulation must preserve original row order too.

Reproduce: python -m local_studies.colibri.profile_cooperative_joint_rhs
Result: /tmp/colibri_cooperative_rhs.json.

## Local grouped-sweep integration

cooperative_bilateral_rhs.py maps eight lanes to each original constraint slot
in a256-thread group block. For joints, each active row retains the original
FP64 dot order; an eight-lane masked FP64 shuffle gathers RHS values.
Only the leader executes the unchanged triangular solve, impulse accumulation,
body writeback and inequality callback. Contacts run on their tile leader.
Existing color order, per-color barriers and physical copy mapping are unchanged.
No persistent scratch or cached wrench arrays are needed.

CUDA validation:
- Revolute/prismatic motion and finite property transitions are byte-exact
  against scalar execution, with all six momentum checks retained.
- One actual prepared Colibri sweep matches all355world arrays byte-exactly.
- Captured finite-drive bounds, contact energy/momentum, and mixed
  joint/friction per-phase momentum tests pass.
- Both330-frame candidate runs match all10saved canonical arrays byte-exactly,
  including complete q/qd histories.

Isolated candidate / fresh baseline / reverse candidate:
11.15298 / 12.34187 / 11.13667ms per frame.
Reverse candidate is9.77% less frame time:89.79FPS versus81.03FPS.
Each run excludes30warm-up frames and measures300actual public steps with
30temporal substeps per120Hz collision update. This gain includes the complete
step, unlike the earlier materialized microbenchmark.

Artifacts: /tmp/colibri_cooperative_rhs_paired_summary.json and corresponding
candidate330, baseline330 and reverse330 JSON/NPZ files.

This remains a local prototype. A production policy must preserve scalar CPU
execution (the native CPU shuffle fallback is not cooperative), avoid applying
the wider layout indiscriminately to contact-only scenes, and retain meaningful
finite-drive, lifecycle and physical invariant coverage. Longer trajectory and
unrelated-solver acceptance remain parent-owned gates before promotion.

## Canonical integration

The private grouped-sweep factory now has a cooperative specialization.
World selection enables it only for CUDA iterate/relax phases with bilateral
blocks. Preparation, contact-only scheduling and CPU remain scalar.
Scalar and cooperative callbacks share the same factor-solve/impulse helper.

The production launch-policy regression failed before the change because
only32-thread launches occurred, then passed with256-thread solve launches.
Four focused tests additionally cover39disjoint joints with counts0..6,
invalid and disabled rows, a partial final tile, both bias phases, global
disable, CPU/contact-only policy and live property transitions.
