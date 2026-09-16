# Cached complete component response prototype

Local files: `cached_component_response.py`, `cached_component_live.py`.
No production callsite/default changes. Work in progress; no stationarity or
performance acceptance yet.

## Replacement, not an extra correction

At each geometry refresh, apply the existing native factor to all 12 coordinate
wrenches of a complete two-body component. Save every body velocity response and
compliant joint-speed response. The contact sweep uses these cached responses
with the **actual loaded corrected native point-law source**: soft normal update,
static/dynamic metric Coulomb projection, normal/tangent coupling, predictive
policy, and broken-state writes. SOR remains one. The original joint-coordinate
row-velocity evaluation is retained.

The sweep updates scratch body velocities and accumulated compliant impulse;
commit these once after the complete contact phase. Aggregate all warm-start
point wrenches before one native factor application. All original point impulses,
application levers, targets, history, ordering and cones remain represented.

This first serial owner implementation prioritizes response validation. Scratch
arrays and fallback launches are not claimed to be an optimal final GPU layout.
The full scene needs a contacted-body response cache and one full-component
scatter, avoiding an all-body dense update for each contact.

## Explicit scope and fallback

- More than two bodies in a component: retain original entire callbacks.
- Within the bounded two-dynamic-body study, same-component internal contacts or
  contacts with another dynamic component: device detection dispatches the
  unchanged native mobility and iterate kernels.
- External static/kinematic endpoints contribute their existing row velocities;
  their inverse response is zero.
- No symmetrization, eigenvalue floor, regularization or rank truncation.

Internal fallback is mandatory. Root's independent regression demonstrates that
contracting large floating-root body blocks can destroy a tiny internal hinge
response. The cached dense contraction is not a universally stable replacement
for the native relative-coordinate mobility routine.

## Initial actual CUDA evidence

`/tmp/colibri_cached_component1.*`: one public frame, geometry/source/finite gates
pass; cache valid=1, actual two-body component selected. Against no-skip native
first-frame reference, max q difference 2.18e-11, max qd difference 1.86e-8.
This is roundoff-level comparison, not byte equivalence or long-time validation.

`/tmp/colibri_cached_component_frozen1.*`: every contact phase compared from
identical restored inputs; original warm-start retained for this first gate.
Maximum differences over the first frame:

| Quantity | Maximum absolute difference |
|---|---:|
| Body linear velocity | 4.89e-9 m/s |
| Body angular velocity | 2.42e-8 rad/s |
| Compliant drive impulse | 3.64e-12 Ns |
| Contact impulse | 1.60e-10 Ns |
| Persistent contact history including break state | 0 |

Comparison mode leaves native q/qd/time **byte identical** to the no-skip baseline
prefix. A full frozen dump includes all allocated native tree/response/body
arrays plus the new cache, for independent physical response checks.

Aggregate warm-start comparison has now been added separately; it is pending a
run. Independent native response P/L/work and high-mass checks are pending.
Only after these gates should the 600-frame accuracy/performance experiment run.

## Commands

Use the same Python environment as the current Colibri studies:

```
COLIBRI_CACHE_COMPARE=1 \
COLIBRI_STAGE_RUNNER=local_studies.colibri.cached_component_live \
COLIBRI_DIFFERENTIAL_REFERENCE=1 COLIBRI_DIFFERENTIAL_NORMALIZED=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
  -m local_studies.colibri.direct_relax_no_skip --native-path direct \
  --body-count 2 --frames 1 --substeps 30 --save-history \
  --output /tmp/colibri_cached_component_compare.json
```

Remove `COLIBRI_CACHE_COMPARE` for actual candidate execution. Comparison mode
runs both implementations and is unsuitable for timing.

## Independent response gate rejected the first cache forms

Root tested 128 random external point impulses on the captured native factor,
with an independent FP64 physical ledger (candidate arithmetic remains FP32).

- Independent dense body contractions: hard-joint delta residual 6.21e-8 versus
  native 3.37e-9; angular reaction error 1.76e-9 Nms versus native 1.45e-10.
- Shared-root/generalized cached coefficients with native forward reconstruction:
  hard-joint residual returns to native scale, but linear reaction error 1.10e-8
  Ns versus native 1.13e-9 and angular error 2.35e-9 Nms remain unacceptable.

Neither form passed; the tiny first-frame trajectory difference is insufficient
for acceptance. No 600-frame run was launched. A serial native two-body factor
control now preserves the original elimination and forward operation order in
local variables while removing per-contact global scratch and block barriers.
It is undergoing the same independent gate before any live use.

## Serial native control and first 600-frame result

Root's `/tmp/colibri_cached_component_frozen1.serial_native_physics.json` passed:
`serial_native_pair` produces **byte-identical** body increments and generalized
speeds for all 128 external random point impulses. P/L, hard constraints and
response work therefore match native exactly in that frozen test.

The live adapter now uses that local native elimination for impulse application;
it does **not** apply the rejected dense or generalized cached coefficients.
The basis cache remains only for external-contact mobility refresh. Exactly
root-plus-one-child components are supported; other sizes retain native dispatch.

Warm-inclusive actual microstep comparison, `/tmp/colibri_cached_serial_compare1.*`:

- Contact solve: max linear difference 1.86e-9 m/s, angular 7.45e-9 rad/s,
  drive impulse 1.82e-12 Ns, point impulse 1.60e-10 Ns; history exactly unchanged.
- Aggregated warm start: linear 9.31e-10 m/s, angular 1.49e-8 rad/s,
  drive impulse 1.36e-12 Ns.

`/tmp/colibri_cached_serial600.*` completed with geometry, finiteness and source
gates passing, but is **not accepted**:

| Metric | Current no-skip native | Serial component replacement |
|---|---:|---:|
| FPS, synchronized public step | 20.4885 | 20.4978 |
| Base drift, 5–10 s | 2.25892 µm/s | 2.96354 µm/s |
| Final hinge angle | 18.207503° | 18.207504° |
| Final drive constitutive residual | 1.37e-9 rad/s | 1.26e-9 rad/s |

The replacement ran (valid=1), and comparison instrumentation was disabled.
There is no measurable speed benefit and drift is worse. No 60-second extension
or default change. Root is profiling the actual replacement to locate its costs
before any subsequent algorithm change.
