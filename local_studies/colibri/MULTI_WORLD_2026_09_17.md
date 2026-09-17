# Colibri replicated worlds

The PhoenX example defaults to four independent Colibris. Use
`uv run -m newton.examples phoenx_colibri --num-worlds N` (or `--world-count N`);
`--num-worlds 1` selects the previous single assembly. Nonpositive counts are
rejected. Viewer offsets arrange the worlds in a grid without translating
physics coordinates. The default four worlds form a 2x2 grid.

The existing temporal color-group scheduler now accepts replicated model
worlds. This uses `step_layout="single_world"`, the global coloring layout;
it does not switch to the fast-tail/block-world dispatcher used by DR Legs,
which does not implement temporal mass splitting. No solver arithmetic,
substep counts, collision frequency, materials, or density settings change.
Contact and audit capacities scale with world count. Assembly, drive, and
unpowered support checks cover every world. Render snapshots include all
worlds and lazily snapshot contacts when visible.

## Measurements

RTX PRO 6000 Blackwell. Rendering disabled unless stated. These measurements
include the independently requested adjacent-tail-feather filters; they must
not be compared against the original unchanged-policy optimization target.

- Powered four worlds, 3600 measured frames after 60 warmup: 22.8078 ms/frame
  (43.84 physics FPS), maximum penetration 0.476617 mm, maximum anchor error
  0.638848 mm, maximum axis error 0.0109348 rad. Assembly and drive checks
  passed throughout. All four pose trajectories were bitwise identical.
- Unpowered four worlds, 3600 measured after 60 warmup: 20.7151 ms/frame;
  penetration 0.240623 mm, anchor error 0.264054 mm, axis error 0.00358004 rad.
  No support-check failures. Relative to saved frame index 120 (after two
  seconds), maximum XY support drift was 1.29653 micrometers, final drift
  0.683124 micrometers, and maximum rotation 5.37996e-6 rad across all worlds.
- Single-world control/candidate, 180 measured after 60 warmup:
  14.6099 / 14.5325 ms. All 240 saved poses and velocities and the quality
  metrics matched bitwise. The 0.5% timing difference is not claimed as a gain.
- Headless 1920x1080 OptiX, priority overlap, 120 measured after 60 warmup,
  under Nsight: 24.6384 ms/frame (40.59 FPS). This is not the user's interactive
  28 FPS measurement. The temporal biased sweep accounts for 55.1% of summed
  GPU kernel time; OptiX launch accounts for 3.1%. Overlapping kernel durations
  must not be interpreted as a partition of wall-clock frame time.

## Reproduction

```bash
uv run -m local_studies.colibri.profile_temporal_performance --num-worlds 4 \
  --frames 3600 --warmup 60 --validate --output /tmp/colibri_multi4_minute
uv run -m local_studies.colibri.profile_temporal_performance --num-worlds 4 \
  --frames 3600 --warmup 60 --validate --motor-off --output /tmp/colibri_multi4_motor_off
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/colibri_multi4_render_trace uv run \
  -m local_studies.colibri.profile_optix_overlap --num-worlds 4 --mode priority \
  --frames 120 --warmup 60 --profile --output /tmp/colibri_multi4_render
```

The profiling tools continue to default to one world. Performance comparisons
reject different world counts, and joint metrics inspect every world.

New regressions fail against the prior implementation: multi-world temporal
construction is rejected; later-world support creep and stalled drives are
missed; independently translated assemblies are incorrectly diagnosed as
escaped. A three-world contact-wrench regression checks linear and angular
momentum changes under graph replay with different per-world gravity and
initial velocity, and verifies collision world isolation.

## Independent-world temporal sweep blocks

Keep the existing global coloring and mass-copy data, but give each independent
world its own block within each color slab. Every constraint is still visited
by its owner, in the same per-world color and contact/joint order. Preparation,
substeps, collision updates, friction anchors, and paired impulse arithmetic
are unchanged. No additional per-contact buffers or host/device transfers are
needed. Models containing global bodies retain the original schedule because
a shared moving body can couple the worlds. A global ground shape without a
body does not introduce such coupling.

| Worlds | Control ms/frame | Candidate ms/frame | Throughput ratio | Measured frames |
| --- | ---: | ---: | ---: | ---: |
| 4 | 22.8078 | 16.7698 / 16.8054 | 1.360x / 1.357x | 3600 after 60 warmup, repeated |
| 16 | 58.7368 | 23.6937 | 2.479x | 60 after 30 warmup |
| 64 | 206.7211 | 48.1909 | 4.290x | 60 after 30 warmup |

All saved poses and velocities, including warmup, are bitwise identical to
the corresponding control at each world count. All quality metrics match.
The 16/64-world runs are scaling screens, not full-minute quality acceptance.
The four-world result does not satisfy or replace the original single-world
9.4 ms optimization goal.

The shared-global-support momentum regression fails against the unguarded
prototype: simultaneous writes lose part of the shared body's paired impulse.
Keep the original sweep ordering whenever the model contains global bodies.
The existing contact-force regression also covers independent worlds with
distinct gravity and initial velocities, captured replay, and force reset.

The render trace uses DLSS, which bypasses OptiX CUDA graph capture. It contains
120 physics graph replays and 120 direct OptiX launches for 120 measured frames.
The two streams still overlap. The rendering harness now records actual OptiX
graph activity instead of assuming that rendering uses a captured graph.

The final four-world repeat uses the shared-body fallback and matches all 3660
saved control states and quality metrics bitwise. A fresh combined regression
run passed 34 tests, including the global support's paired linear/angular
momentum and four-world contact visualization. The same shared-body regression
fails without the fallback. One-world 240-state parity also passes; its
14.4694 ms/frame versus the earlier 14.6099 ms control is not claimed as a
single-world optimization.

The matching Nsight/headless OptiX comparison at 1920x1080 improves from
24.6384 to 18.8062 ms/frame (40.59 to 53.17 FPS). The final image and physical
snapshot match bitwise. The biased sweep still runs 5760 times in 120 measured
frames: median kernel time drops from 265.280 to 160.784 microseconds (mean
292.563 to 180.402 microseconds). The OptiX launch median remains about
741 microseconds. This is a scheduling improvement, not fewer solves or a
rendering-quality reduction. Interactive GUI FPS can differ from these
controlled headless measurements.

Artifacts for the scheduling comparison are under `/tmp/colibri_multi4_worldsplit_*`,
`/tmp/colibri_multi16_{control,worldsplit}_short.*`, and
`/tmp/colibri_multi64_{control,worldsplit}_short.*`. Use the same reproduction
commands above on the control and candidate revisions and check saved results
with `local_studies.colibri.check_temporal_performance --minimum-speedup 1.05`.

The optimized unpowered four-world run averages 15.4540 ms/frame
versus 20.7151 ms for the control over 3600 measured frames after 60 warmup.
All saved poses, velocities, and quality metrics match bitwise, including
the 1.29653 micrometer maximum settled support drift.

The optional `--fix-base` diagnostic has a pre-existing strict-pose failure:
both the control and optimized three-world runs exceed its 1e-5 tolerance
with the same maximum pose-component difference, 4.8514215e-5, on the first
checked frame. This acceptance check is not disabled or relaxed. The added
fixed-root scheduling regression compares all body poses and velocities
against the ordered reference instead of claiming that diagnostic passes.

The 60-frame, three-world fixed-root comparison passes with bitwise-identical
poses and velocities against a recaptured ordered reference. Independent
180-frame control repeats produce a 0.99836x throughput ratio and fail the
1.05x performance gate; the optimized screen passes at 1.35702x against the
same control. This verifies the performance regression detects the prior
implementation rather than comparing a control file to itself.
