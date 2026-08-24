# CI runtime decision record

## Measured baseline

Source: 2026-08-13 supplied CI logs.

| Job | Wall time |
|---|---:|
| Unit-test job | 30:06 |
| Unit-test execution | ~28:38 |
| ASV command | 42:36 |
| Benchmark worker | 43:31 |

ASV finished 14:56 after unit tests and was the measured critical path.

Latest PR #3966 run (2026-08-24): benchmark job 43:49; Windows and
Ubuntu test commands 38:54 and 38:42. Final unit-test reporting finished
1:13 before benchmarks. The benchmark commits are expected to make tests the
critical path.

## Commit `eb78b2831`

Expected ASV saving: 5.5-7.5 min; expected ASV wall time: 35-37 min.
This estimate is trace-based; it is not a completed CI measurement.

| Change | Reason | Coverage |
|---|---|---|
| PR ASV environment omits Torch | Selected benchmarks do not import Torch | Full ASV environment retains Torch |
| Install pinned simulation stack before Newton `[examples]` | Avoid dependency resolver churn and unnecessary `[dev]` packages | Same pinned Warp, MuJoCo, and MuJoCo Warp |
| Keep interleaved rounds | Avoid cold-start/order bias | Same base/head comparison method |
| Select 41 instead of 45 fast entries | Remove duplicate output/summary measurements | All benchmark definitions remain |
| Camera: normal combined+depth; pixel combined | Color is covered by combined; depth-only remains in normal order | Both render orders, color, depth, 4096 worlds retained |
| Teleop: omit duplicate mean track | `time_teleop_loop` supplies central runtime; p95 supplies tail | All four modes and p95 retained |
| Camera warm only selected outputs in PR mode | Avoid compiling excluded output modes | Full mode warms all outputs |

Removed PR entries only: normal color-only; pixel color-only; pixel
depth-only; teleop mean. No benchmark definition was deleted.

Unit-test optimizations preserve workloads and assertions:

| Test | Change | Preserved |
|---|---|---|
| Cable cross-slide | Reuse same-frame `body_q.numpy()` | 540 frames and all checks |
| Gimbal limits | Copy final q/qd only | 4 rollouts x 50 steps |
| Kamino loops | Remove per-step sync without progress output | Same steps and final comparisons |
| Pendulum | Record trajectories on device; copy once | Same steps, analytical and energy assertions |

Local CPU direct comparisons: gimbal limits 43.52 s -> 37.11 s
(-6.41 s, -14.7%); semi-implicit pendulum period 3.22 s -> 3.28 s
(no CPU gain; GPU synchronization reduction only).

## Follow-up benchmark commit

Expected additional saving: 2.0-2.5 min. A further 5-7 min is not
supported without removing a complete expensive case or weakening sampling
further.

| Benchmark | Before | After | Evidence |
|---|---:|---:|---|
| Allegro PR mode | 8192 worlds, 300 frames x 2 samples | 8192 worlds, 200 frames x 1 sample | Local setup 35.117 s -> 12.964 s (-63%); trace estimate ~1.4 min/job |
| Fast camera sampling | Adaptive repeat, 2 rounds | 1 repeat x 2 rounds | Each sample still executes 50 renders at 4096 worlds; estimate ~0.8-1 min/job |

Allegro remains selected for mean world-step time and p95 frame time. Its
validation, robot, solver, and world count are unchanged. Full mode remains
300 frames x 2 samples.
Fast camera classes use the explicit repeat count in all ASV runs; non-fast
camera benchmark definitions are unchanged.

## Pixel-priority camera profile

Local RTX GPU, explicit PR output subset:

| Setup | Wall time | Main costs |
|---|---:|---|
| Warm kernel cache | 5.414 s | finalize 2.251 s; URDF/mesh 1.378 s; replication 0.599 s |
| Isolated cold kernel cache | 11.780 s | kernel load/compile 7.024 s; URDF/mesh 5.530 s; finalize 2.737 s |

`numpy.asarray`: 0.709 s total across 14,737 calls, mostly inside model
construction. Mesh index remapping with `np.searchsorted` was 6.7x faster in
isolation, but alternating whole-setup timings were baseline 4.824 s versus
optimized 5.058 s. The change was reverted as noise-level and out of scope.

The pixel-priority cold cost is primarily compilation, mesh processing, and
model finalization. Frame reduction does not remove it. Sharing setup between
ASV spawn processes requires file caching, which is intentionally not used.

## Coverage and limits

- No unit test, test parameter, test step, assertion, benchmark class, robot,
  backend, render order, resolution, or world count is removed.
- PR ASV intentionally measures a representative metric subset, not every
  dashboard metric.
- Full ASV definitions retain all metrics; the repository has no automated
  scheduled full-ASV workflow.
- No model/file cache or implicit library cache was added.
- Rejected: non-interleaved rounds; it produced a false ~2x regression from
  cold-start ordering.
- Rejected: removing Allegro or pixel-priority camera; each is a distinct case.
- Rejected: NumPy mesh-loader change; whole-setup benefit was not measurable.
- Rejected: on-device joint-friction recording; identical CPU test regressed
  from 45.05 s to 62.84 s (+39%).
- Rejected: test-only null-viewer sampling; alternating warm cloth runs
  were both 3.54 s, and cable was 58.83 s baseline versus 60.23 s sampled.

## Validation

- New Allegro regression test failed before implementation: PR workload stayed
  at `(300, 2)` instead of `(200, 1)`.
- `uv run python asv/tests/test_benchmark_simulation.py`: 14 passed.
- `uv run asv check --config asv-pr.conf.json -E existing:same`: passed.
- `uvx pre-commit run -a`: passed after formatter application.
