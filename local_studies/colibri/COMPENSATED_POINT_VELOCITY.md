# Compensated point velocity: bounded conditioned-path control

This local control addresses the measured generic row-velocity bottleneck. It is not the primary D6 accuracy effort and has not established long-term stationarity.

## Change

Compute the external point velocity once using FP32 high/low arithmetic, then project onto the three unchanged contact directions. Products retain their explicit FMA residual; additions retain rounding residuals. CUDA multiplication uses `__fmul_rn`, with `__fmaf_rn` for the residual. No new FP64 operations are introduced. The actual point impulses, cones, normal softness, drive law, order, and native response remain unchanged. Existing generic internal/kinematic fallback remains; live dispatch is limited to a complete two-body component against static endpoints.

`COLIBRI_CACHE_COMPENSATED=1` selects this mode alongside `COLIBRI_CACHE_REGISTER=1` and `COLIBRI_CACHE_POINT_VECTOR=1` in `cached_component_live.py`.

## Precision gates

- Independent four-case suite passes CPU and CUDA: randomized root/child/reversed endpoints, common translation, nearly equal rotating levers, and a static rolling center.
- The static rolling case retains native `2.384185791015625e-7 m/s`; ordinary FP32 arithmetic returned zero.
- All 134 recorded biased/relax point vectors match native exactly on CPU and CUDA, in all three directions. This is measured coverage, not a universal arbitrary-scale bound.
- One-frame actual native/candidate comparison: maximum linear velocity difference `1.86265e-9 m/s`, angular velocity `7.45058e-9 rad/s`, drive impulse `1.81899e-12`, contact impulse `1.60071e-10`; history/broken state exact. These equal the prior serial/register control errors.

Commands: `python -m local_studies.colibri.check_compensated_point_velocity --device cuda:0` and `python -m local_studies.colibri.check_external_point_velocity --compensated --device cuda:0` through the shared uv environment. Raw-helper independent tests retain the known failing controls.

## Bounded timing

Same 65 frames, first30 warmup, 35 synchronized measured public steps; 30 substeps and120Hz unchanged. No parallel mobility refresh was included.

| Mode | FPS | ms/frame |
|---|---:|---:|
| Original no-skip native, previous600 control |20.4885|48.808|
| Register-local scalar rows,65 |23.7537|42.0986|
| Plain FP32 point vector,65 |36.9303|27.0780|
| Compensated FP32 point vector,65 |34.2623|29.1866|

Compensation adds2.109ms versus the insufficient-precision plain vector. Geometry and source gates pass. The1.083s run is too short to establish creep quality; no60s extension or production promotion follows from this result. Different code generation can change startup trajectories, so this table is an end-to-end control rather than a same-state kernel attribution. Independent frozen graph replay already established the dominant row-velocity cost.

Artifacts: `/tmp/colibri_compensated_{compare1,65}.*`, `/tmp/colibri_compensated_recorded_{cpu,cuda}.json`, `/tmp/colibri_compensated_cancellation_cuda.log`.
