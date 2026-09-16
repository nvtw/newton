# G1 remaining-cost audit

## What the 24% figure means

The latest saved isolated factor comparison is 40.182467 ms/policy step for
per-depth factorization versus 37.781217 ms for the existing cooperative factor.
The historical old-HEAD result is 30.469948 ms. The latter two numbers imply
23.99% higher latency (19.35% lower throughput), but were not collected as a
fresh adjacent current-versus-HEAD pair. They are 50 Hz controller-step units,
including inference and all physics, not display frames or individual substeps.
The fresh measurement below supersedes this historical current-cost estimate.

Earlier corrected physics initially cost 47.917816 ms versus 30.469948 ms on
HEAD, a 57.3% latency increase. Exact optimizations subsequently recovered part
of the cost:

| Controlled change | Reference ms | Candidate ms |
| --- | ---: | ---: |
| Skip unused dense impulse-response cache | 48.143033 | 41.779009 |
| Invocation-local packed cross-mobility cache | 41.592716 | 39.431273 |
| Existing cooperative factor policy | 40.182467 | 37.781217 |

These rows are separate comparisons, not additive contributions. Local
kinematics reuse also measured 39.553218 ms in its saved trial. Corrected
trajectory references remain byte-identical under the accepted exact changes;
old HEAD has different physics and is not an acceptable trajectory target.

Artifacts: `/tmp/colibri_g1_policy_{head,current}_timing.json`,
`/tmp/g1_response_{before,after}.json`,
`/tmp/g1_cache_only_{reference,candidate}.json`,
`/tmp/g1_factor_{depth,cooperative}_matched_timing.json`.

## Historical attribution and what remains applicable

The only saved detailed pair is `/tmp/g1_{head,current}_nodes.sqlite` and
`/tmp/g1_{head,current}_kernels.csv`: two warmed actual controller calls each.
It predates the above exact improvements. Read-only SQL resource extraction is
saved in `/tmp/g1_historical_profile_resources.json`.

| Work over two profiled policy calls | HEAD | Early corrected | Added time |
| --- | ---: | ---: | ---: |
| Dense body impulse response | 0 calls | 160 calls | 10.341 ms |
| Per-depth factor | 2240 calls / 8.023 ms | 4480 / 14.116 ms | 6.093 ms |
| Packed generalized row build | 56 / 1.505 ms | 216 / 6.222 ms | 4.717 ms |
| Contact gather | 56 / 0.860 ms | 216 / 3.111 ms | 2.251 ms |
| Local kinematics | 8 / 0.113 ms | 168 / 2.383 ms | 2.270 ms |
| Generalized contact tile | 320 / 17.956 ms | 320 / 22.733 ms | 4.777 ms |

Times contain instrumentation outliers; call counts give stronger causal
attribution than their sum. Dense unused response and redundant kinematics are
already guarded in current `ReducedPhoenXArticulation.solve_constraints`.
Cooperative factor selection already replaces the deep single-robot launch
chain. Packed cross mobilities are already cached within a solve invocation.
Do not propose these same optimizations again or quote their historical costs
as current residual overhead.

The required post-integration factor/Jacobian/contact-row refresh remains.
Current-pose response is necessary; simply restoring old refresh counts would
apply impulses using stale geometry. Reusing a factor across changed velocity
is also unsafe when predicted joint-limit activity changes its diagonal.
The mesh/SDF spatial-priority and optional search-floor studies do not explain
this G1 regression: the controller approximates meshes with convex hulls,
disables self-collision, and does not enable the Colibri search-floor overlay.

Historical resource data does not support a spill diagnosis. The contact tile
used 91 registers/thread on HEAD and 162 in the early corrected profile, but
both report zero local-memory bytes and only one block of 32 threads. Packed
row build used 64 registers, zero local memory, two blocks of 48 threads. A
single robot primarily exposes latency and limited available parallel work;
register count alone does not establish the limiting mechanism.

## Next concrete exact experiment

First refresh the current node profile and active-row metadata. If packed row
construction remains material, prototype cooperative preparation of independent
contact rows for the point-friction route. Current `_build_packed_generalized_row`
walks each row's ancestor path and then the entire articulation serially in
one thread. Its root multi-DoF dot products and matrix-vector rows can be
calculated by separate lanes while preserving each scalar dot/matrix summation
order; the leader must retain ordered wrench subtraction and inverse-mass
accumulation. The existing cooperative patch-row builder is a code precedent,
but switching G1 to patch friction would change physics and is not this proposal.

Start with frozen actual row inputs and measure only this preparation kernel,
including scratch/synchronization costs. Require complete byte equality of
Jacobian, response, row masses, state and scratch across changing point counts,
page order, finite limits, and disabled rows. Only then test the actual learned
controller. This is a viable bounded experiment, not a measured improvement.
It preserves post-integration refresh, point Coulomb friction, source mass,
controller parameters and every scalar accumulation order.

## Profiling needed before implementation

The old profile is sufficient to explain the original regression, not to rank
today's remaining costs. After the shared GPU is free:

1. Run a fresh canonical 20 checked + 10 warmup + 100 timed controller sequence
   with source/assets/environment metadata; retain the same 20-by-8 reduced
   schedule and learned controller.
2. Run `profile_g1_policy.py` with CUDA graph node tracing, collecting two warmed
   policy calls. Record active contact/row/page counts and selected factor path
   outside the measured range. Confirm unused-response and redundant-FK launches
   disappeared and cooperative factors are actually selected.
3. For a new exact candidate, use an isolated adjacent baseline/candidate pair
   plus full saved-state equality. If an updated old-HEAD percentage is needed,
   rerun HEAD adjacently with identical assets/settings and explicitly retain
   the different-physics caveat; do not attribute all timing difference to code
   overhead when the resulting contact trajectories differ.

No production edit or physics rollback was performed.

## Prepared isolated command (run only after GPU handoff)

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -u -m local_studies.colibri.validate_g1_current_cost --prefix /tmp/g1_current_cost_reaudit --profile
```

This invokes the existing timing and profiling scripts without changing the
policy or solver options. It hashes all current Newton Python sources and
already-cached unitree_g1 assets before/after, compares both checked20-frame
state histories bytewise to `/tmp/g1_cache_only_candidate.npz`, and records
pre-run GPU clocks/temperature metadata. The timing stage has20 checked,
10 additional warmup and100 timed policy steps. The node profile separately
has20 warmup andtwo profiled steps. Expected physics work is about5seconds
plus1second profile warmup, with setup/profiler overhead on top; no wall-time
promise is made. This fresh current profile is not a new HEAD comparison.


## Fresh current-source measurement

The isolated run passed source/asset hash checks and both corrected-reference
20-frame body pose/velocity byte gates. Twenty checked plus ten warmup and
100 timed learned-controller steps measured **36.751907 ms/policy step
(27.20947 steps/s)**, below the physical 50 Hz controller rate. Settings unchanged.

Two warmed Nsight calls give these kernel totals divided by two:

| Work | Calls over two steps | ms/policy step |
| --- | ---: | ---: |
| Contact tile | 320 | 9.393 |
| Advance and publish | 160 | 6.787 |
| Advance | 160 | 3.210 |
| Packed row build | 216 | 3.028 |
| Multi-DoF factor | 320 | 2.593 |
| Cooperative factor | 320 | 2.382 |
| Contact gather | 216 | 1.571 |
| Finish and publish | 160 | 1.350 |

Summed GPU kernels total32.500048 ms/step; separate profiling overhead means
the difference from isolated wall time is not an exact CPU/inference breakdown.
Dense body-response construction is absent, local kinematics has only eight
calls, and cooperative factors are selected. Dominant kernels report zero
local-memory bytes; register counts alone do not establish a spill bottleneck.

The bounded next exact experiment remains cooperative root-six-DoF work in
point-row preparation (3.028 ms/step,9.32% summed GPU time). Preserve each
scalar dot/matrix accumulation order and leader wrench/inverse-mass updates.
Capture actual row inputs first and require full output byte equality before
timing. Existing patch-row cooperative code is a precedent, not a reason to
change point Coulomb friction. No speedup is established. Contact tile remains
largest (28.90% GPU time), but its existing mobility cache must not be repeated.
Active point/row/page counts need that future frozen capture; launch size does
not provide those counts.

Artifacts: /tmp/g1_current_cost_reaudit.validation.json (hashes, hardware,
commands, byte gates), .json/.npz (timing/states),
/tmp/g1_current_cost_reaudit_nodes.nsys-rep and .sqlite (trace),
/tmp/g1_current_profile_resources.json (SQL aggregation).
The GPU was released after capture; attribution used CPU-only SQLite queries.
No fresh old-HEAD pair was run, and its differing physics remains unsuitable
as a current trajectory target or controlled overhead estimate.


## Follow-up outcome

The proposed eight-lane point-row experiment passed actual-input and expanded
storage/ownership byte gates, but frozen timing did not improve. Both variants
are rejected for promotion; see G1_COOPERATIVE_POINT_ROWS.md for resources and
complete-reset timing. Cooperative scalar-lane values removed a96-byte local
array depot but remained slower than the canonical kernel. No controller
settings or production source changed.
