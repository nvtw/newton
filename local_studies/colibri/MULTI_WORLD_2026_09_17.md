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
