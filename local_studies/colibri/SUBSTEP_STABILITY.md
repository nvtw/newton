# Colibri minimum-substep stability sweep

> Stationary-sculpture requirement: the recorded 300-second run is NOT an
> acceptable Colibri result. FrameGround drifts 437 mm and rotates 0.859 rad.
> It passes only the earlier scoped stability checks. See BASE_STATIONARITY.md.

## Conditions

All runs retain SI geometry, exact source offsets/filtering, source drives,
120 Hz collision updates, SOR 1, no global damping, one biased solver
iteration per temporal substep, and final-substep velocity relaxation.
`N` means substeps **per collision update**: 2N substeps per displayed
60 Hz frame; temporal timestep is 1/(120N) seconds.

Acceptance uses existing checks: fresh penetration below 1 mm, joint-anchor
error below 2 mm, joint-axis error below 0.03 rad, finite bounded motion,
and tail support. Saved orientation histories additionally check crank
motion. Passing finite-duration runs does not establish indefinite stability.

## Previous public schedule (before color-group integration)

| N | Result |
|---:|---|
| 5 | Fails at 0.0833 s, FrameGround/Frame anchor 2.838 mm |
| 10 | Fails at 0.1000 s, FrameGround/Frame anchor 2.305 mm |
| 11 | Fails at 0.1000 s, FrameGround/Frame anchor 2.065 mm |
| 12 | Passes 5 s; fails at 5.5833 s, gear/hypocycloid anchor 2.023 mm |
| 15 | Passes 5 s; fails at 11.9833 s, gear/hypocycloid anchor 2.081 mm |
| 18 | Fails at 12.0000 s, gear/hypocycloid anchor 2.033 mm |
| 20 | Passes 5 s; fails at 12.8000 s, gear/hypocycloid anchor exceeds 2 mm |

These are accuracy-limit failures, not claims that every failed run explodes.
The longer runs demonstrate why five-second screens are insufficient.
Reports: `/tmp/colibri_public_substeps_{N}_{frames}.json` and consolidated
`/tmp/colibri_public_substep_summary.json`. The minimum above 20 for that previous schedule was not narrowed by this sweep.

## Faster color-group schedule

The tested configuration uses groups of eight colors, public initial-twist
candidate admission and canonical parallel contact preparation. Static
factorization optimization is OFF during this sweep. The original sweep used a local adapter; the color-group implementation has
since been integrated into the public solver, with a canonical N=10
full-minute run matching all seven saved state arrays byte for byte.

N=5 fails at frame 4 (anchor 2.213 mm), N=8 at frame 5 (2.234 mm), and N=9
at frame 6 (2.144 mm). N=10 passes 20 s with peak anchor 1.923 mm and fresh
penetration 0.1612 mm. The crank completes 10.617 turns with mean speed
-3.3355 rad/s against -3.4907 rad/s target, with no stalled samples.
N=15 also passes the full60s in the canonical implementation: penetration
0.13886mm, joint anchor1.02691mm, axis0.014378rad
(`/tmp/colibri_canonical_groups15_3600.json`). N=20 was screened for5s.
The user subsequently requested optimization return to N=30; the public
example default remains30 and lower-N results remain a separate study.

N=10 also PASSES 3600 frames / 60 seconds: maximum fresh penetration
0.16118 mm, peak joint anchor 1.92263 mm, peak axis 0.020112 rad. Crank
motion is 31.806 turns, mean -3.33074 rad/s, with no stalled samples.
Reports: `/tmp/colibri_slab_sweep10_3600.json`, `.motion.json`, `.motion.npz`.
Thus 10 is the lowest tested count meeting these checks for one minute;
9 fails at startup. The anchor margin is only 0.07737 mm below the current
2 mm threshold. Longer durations and different initial conditions remain
unverified at this count.

This is a minimum search with
one iteration per substep, not a proof of the minimum attainable by trading
more iterations for fewer temporal substeps.

Reproduce from `/tmp/newton-colibri-phoenx`:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.sweep_slab_motion --body-count 36 --frames 3600 --warmup 30 --substeps 10 --iterations 1 --public-solver --parallel-prepare --mass-splitting --contact-chunk-size 6 --max-colored-partitions 8 --mass-splitting-batch-size 2 --contact-offset-map /tmp/colibri_physx_exact_contact_offsets_m.json --output /tmp/colibri_slab_sweep10_3600.json
```

## Isolated timing at the validated minimum

The same N=10 configuration (no static-factor optimization) measured
8.96868 ms/frame mean, 10.23846 ms/frame p95, or 111.50 FPS. Measurement
covers 300 frames after 30 warmup frames; other agent GPU jobs were held
and the compute-process list was empty before launch. Artifact:
`/tmp/colibri_slab_sweep10_isolated330.json`. The separate 3600-frame
run provides stability acceptance; the timing run is not a replacement.

The previous matched PhysX reference used 30 TGS position iterations per
120 Hz update. Do not use its 90-92 FPS result as a matched comparison
against this N=10 measurement. The matching PhysX10 run measured 5.53408 ms/frame (180.70 FPS), p95
6.6966 ms, but its final joint errors exceed this sweep's acceptance limits:
Gear_Large__3x_01/Hypocycloid_Gear__3x_01 has 3.46765 mm anchor error and
0.0719329 rad axis error. Thus this is a matched iteration-count comparison,
not equal accepted joint accuracy. No trajectory was saved for that timing
run, so it does not prove continuous drive motion or one-minute stability.
Reports: `/tmp/colibri_physx10_isolated300.json`, `.joints.json`.

## Optimization setting restored

Optimization continues at30 substeps per120Hz update and one biased iteration,
as requested. Current production-example color groups have size4; the minimum
study above used size8 and predates the endpoint/admission changes. Its10-step
result is a one-minute observation, not a validated minimum for the latest code.
The newer geometric-admission 30-step configuration subsequently passed300s:
maximum penetration0.753578mm, anchor0.938996mm and axis0.028822rad.
The assembly-relative escape check passed; the former absolute0.5m bound
was crossed at203.45s because the free assembly slides/yaws. Support drift
remains unresolved. See `/tmp/colibri_public_geometric_relative18000.json`.
This finite-duration result does not establish indefinite stability, and the
older minimum-substep sweep has not been repeated on this configuration.

## Current 30-step optimization result

With spatial-priority contact reduction and cooperative bilateral preparation,
the unchanged 30-step, one-iteration configuration passed 18,000 frames
(300 simulated seconds) at 98.6414 FPS. Peak penetration was 0.762952 mm,
anchor error 0.794343 mm, and axis error 0.0289467 rad against a 0.03 rad
limit. All ten saved arrays, including every pose and velocity sample, were
byte-identical to the pre-preparation-optimization reference. The latest
cooperative preparation reduces paired frame time by 8.83%.

Artifact: `/tmp/colibri_cooperative_prepare_velocity1_long18000.json`,
with `.npz` and `.source.json` companions. This does not revalidate the older
10-step minimum on current code; free-assembly drift remains unresolved.
