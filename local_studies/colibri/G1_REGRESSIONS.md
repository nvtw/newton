# G1 regression checks during Colibri solver work

The comparison baseline is tracked HEAD `f0073be5332b1caf155d54a448943f924d856ef7`,
exported with `git archive` into `/tmp/newton-g1-head-baseline`. Both runs use the
same Python environment and cached robot assets. Current means the uncommitted
PhoenX Colibri worktree. No solver or controller settings were changed.

## Existing maximal/direct standing test

Run `python -m unittest newton._src.solvers.phoenx.tests.test_robot_policy_stiffness -v`.
This uses the real 29-DoF USD robot, fixed standing targets, sticky contacts,
5 internal steps, 2 iterations, and 40 control frames (0.8 physical seconds).

- The armature first-frame finite-state test passes on both revisions.
- The standing test fails on **both** revisions: mean tracking error is
  0.0553663 rad on HEAD and 0.0528800 rad currently, against the existing 0.04 rad
  threshold. This is a pre-existing failure, not a passing regression.
- The separate two-second MuJoCo/ankle parity tests already carry
  `expectedFailure`; they are not evidence of accepted parity.

Logs: `/tmp/colibri_g1_stiffness_{head,current}.log`.

## Production reduced training environment

`check_g1_regressions.py` calls the actual `EnvG1PhoenX.step`, preserving the
production reduced/reference articulation path, patch friction, explicit PD
joint torques, 50 Hz policy rate, 3 physics steps, and 2 solver iterations.
Four worlds use fixed zero actions, no reset noise, and no automatic resets.

The initial 0.4 seconds stays upright and finite. Over two seconds, both HEAD
and current terminate all four robots, first at control frame 64. These are
**open-loop pose holds without a learned balancing controller**; finite states
must not be reported as standing-policy success. The two-second minimum base
height is -0.132971 m on HEAD versus -0.133371 m currently; the foot-box training
recipe does not provide a general full-body floor collision model.

Artifacts: `/tmp/colibri_g1_reduced_{head_2s,2s}.{json,npz}`.

The older `bench_g1_drive_convergence` helper does not match current production
stepping: it queries contacts every internal step and omits the current explicit
force-scatter path. It was not used to make convergence or speed claims.

## Actual cached learned controller

`check_g1_policy.py` calls the existing `example_robot_policy.Example.step` with
`g1_29dof`, PhoenX, and the cached `mjw_g1_29DOF.onnx` Warp-NN controller. Its
existing reduced-coordinate settings remain 20 internal steps, 8 iterations,
2 velocity iterations, and `substep_end` readout. The command is zero velocity;
the policy actively balances. Each call advances 0.02 physical seconds despite
the example's display-time increment being 0.005 seconds.

Both revisions pass the existing all-body-origins-above-ground final assertion
after two physical seconds, with finite states and inference outputs:

| Metric at 2 s | HEAD | Current |
|---|---:|---:|
| Base height [m] | 0.781107 | 0.779860 |
| Minimum body-origin height [m] | 0.0350325 | 0.0350521 |
| Maximum body angular speed [rad/s] | 0.669899 | 0.678943 |

Artifacts: `/tmp/colibri_g1_policy_{head,current}_2s.{json,npz}`.
This is a bounded controller regression, not a long-run locomotion or contact
accuracy guarantee. No timings from concurrent correctness runs are quoted.

## Longer controller screen and isolated performance

Current learned controller also passed the existing final assertion at **10
physical seconds** (500 control steps), with all sampled states and policy
outputs finite. Final base height was 0.776753 m, minimum body-origin height
0.0353211 m, and maximum body angular speed 1.93957 rad/s.

An isolated comparison uses 20 checked steps, 10 additional warmup steps, and
100 timed actual `Example.step` calls. Timing includes Warp-NN inference and all
physics, excludes checks/setup, and preserves the 20×8 solver settings:

| | HEAD | Current |
|---|---:|---:|
| ms / policy step | 30.4699 | 47.9178 |
| policy steps / wall second | 32.8192 | 20.8691 |

Current is **57.3% slower**. Both final states remain finite and pass the
above-ground check at 2.6 physical seconds. These are 50 Hz policy-step units,
not the example's misleading 200 Hz display-time increments.
Artifacts: `/tmp/colibri_g1_policy_{head,current}_timing.{json,log}`.

## Captured-node cost attribution

`profile_g1_policy.py` profiles two actual policy steps after 20 warmup steps via
`cuProfilerStart/Stop`. Nsight uses `--cuda-graph-trace=node`; CUDA graphs and
ONNX inference remain enabled. Profiles and SQLite exports are
`/tmp/g1_{head,current}_nodes.*`; kernel CSVs are
`/tmp/g1_{head,current}_kernels.csv`.

Across two policy steps:

- Generalized contact solve: 320 calls on both revisions, 17.96 → 22.73 ms.
- Reduced depth factor: 2240 → 4480 calls, 8.02 → 14.12 ms.
- Body-space impulse response: **160 new calls**, 10.34 ms.
- Packed contact-row build: 56 → 216 calls, 1.51 → 6.22 ms.
- Contact block gather: 56 → 216 calls, 0.86 → 3.11 ms.

The source maps this to `ReducedPhoenXArticulation.solve_contacts` refreshing
kinematics, factorization, and body impulse response on every post-integration
relaxation, and preparing fresh contact rows there. Refreshing changed geometry
is physically motivated; simply restoring stale operators is not a valid fix.
A candidate investigation is whether the packed generalized path can avoid
building body-space impulse responses unused by that path, while retaining them
for deferred/fallback and loop consumers. No solver changes were made in this
audit. Profiler kernel totals include instrumentation outliers; use the isolated
wall-clock A/B for throughput, and call counts for attribution.

## Packed cross-mobility cache candidate

The generalized point-contact tile solve computes three off-diagonal mobility
terms from immutable packed Jacobian/response rows. Previously it repeated
those identical dot products during every sweep. The candidate stores their
first-sweep values in an invocation-local register tile of 32 `vec3` elements
(three cached scalar values per lane) and extracts them during subsequent sweeps.
There is no persistent cache, public option, new allocation, or preparation
ABI change. New rows, pages, body configurations, and invocation inputs always
start a new cache; alternating point order and all physical projection inputs
are unchanged.

`check_mobility_cache_frozen.py` clones the entire actual G1 kernel input state
including all nested Warp-struct arrays and replays both pre-cache and candidate
kernels. All arrays compare **bitwise equal** for:

- 52 generalized DoFs and 0, 1, 3, or 7 active points;
- 1, 2, or 8 alternating sweeps;
- a degenerate full 32-point page, checking the final cache lane as well.

All 13 cases passed. The complete 20-frame body pose and velocity histories
also match Kepler's guarded-response baseline bitwise.
The isolated candidate measured 40.8590 ms/policy step versus the immediately
preceding guarded-response result 41.779 ms, a modest 2.2% single-comparison
improvement requiring repetition before a strong performance claim.

Exact candidate patch: `packed_mobility_cache.patch`.
Pre-cache source snapshot: `/tmp/reduced_contact_block_before_mobility_cache.py`.
Frozen results: `/tmp/g1_mobility_cache_frozen_full.log`.
Actual-controller timing/history: `/tmp/g1_mobility_cache_timing.{json,npz}`.

The existing contact-coupling, reduced-response-refresh, and reduced-contact-
nullspace suites all passed (12 tests). The first cross-articulation fallback
module compilation took several minutes; the packed cache module itself
compiled normally. Log: `/tmp/g1_mobility_cache_regressions.log`.

A second isolated **cache-only** pair retains all current response/kinematics
changes and substitutes the pre-cache packed kernel only in the reference
process (no canonical source rewrite). Reference 41.5927 ms/policy step versus
candidate 39.4313 ms gives a 5.20% runtime reduction. Both 20-frame body pose and
velocity histories are bitwise identical; both final states remain finite and
upright after 2.6 physical seconds. This supports a modest speed improvement,
not recovery of the entire earlier 57% regression. Artifacts:
`/tmp/g1_cache_only_{reference,candidate}.{json,npz,log}`.

## After canonical color-group and fixed-bound joint preparation integration

The actual cached learned-controller check was repeated for 20 policy steps
(0.4 physical seconds), with its existing reduced 20-by-8 settings unchanged.
All 20 by 44 body poses and velocities are **byte-identical** to the saved
39.4313 ms cache-only candidate reference. States and policy outputs are finite;
the existing above-ground final assertion passes. This is a trajectory
regression check, not a new timing or long-run stability claim.

Artifacts: `/tmp/g1_color_static_regression20.{json,npz}` compared against
`/tmp/g1_cache_only_candidate.npz`. No G1-specific solver changes were needed.

## Existing cooperative factor kernel: single-robot local trial

A process-local override lowers the existing cooperative factor threshold from
32 articulations to 1. No kernel equations or controller settings change. Three
actual G1 states replayed through both factor paths produce byte-identical
implicit diagonals/forces, internal velocities, spatial inertia, Jacobians,
reduced inertia, response matrices, and inverse joint blocks. All 20 controller
frame body pose and velocity arrays also match the saved reference bytewise.

Proof artifacts: `/tmp/g1_cooperative_factor_frozen.json` and
`/tmp/g1_cooperative_factor20.{json,npz}`. The root multi-DoF kernel needed a
34.6-second first compilation; this is setup time, not simulation throughput.
No canonical threshold change has been made.

An isolated paired run, with 20 checked steps, 10 warmup steps, and 100 timed
policy steps, measured **40.1825 ms for per-depth factorization versus 37.7812 ms
for the existing cooperative kernel** (5.98% reduction). Both checked pose and
velocity histories remain byte-identical; the final base heights after the
timed sequence also match exactly, and both pass the above-ground assertion.
This does not recover the full difference from the older 30.47 ms HEAD result.
Artifacts: `/tmp/g1_factor_{depth,cooperative}_matched_timing.{json,npz}`.

### Canonical selection

The cooperative factor is now selected for CUDA trees with at least eight depth
levels, as well as the existing batches of at least 32 articulations. Shallow
small batches retain their prior selection. The two focused policy/limit tests
fail on the old selection and pass after integration; velocity-triggered joint
limit activation still rebuilds byte-identical factors under graph capture.
The existing two response-refresh tests also pass. No mass-factor reuse or
current-pose refresh was removed.
