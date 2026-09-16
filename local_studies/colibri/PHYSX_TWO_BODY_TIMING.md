# Measured two-body PhysX timing

The requested direct SDK measurement completed with all state/property byte gates:

- PhysX 0.5.11: **201.3567 FPS**, **4.966312 ms** per 60 Hz simulation frame.
- P95: 6.579693 ms.
- 30 warmup frames, 300 measured frames.
- Each frame is two completed synchronous 1/120 s calls, with authored
  30 position iterations and one velocity iteration.
- Existing awake two-body/1 mm plane overlay, source gravity/materials and
  source discrete drive law retained. No rendering or per-frame readbacks.
- Initialization, cooking, property/trajectory reads, warmup and output excluded.
- Initial, postwarmup and final q/qd plus all six physical-property arrays
  are byte-exact against the prior awake-plane1mm60 accuracy reference.
- GPU process list was empty before launch; parent and peers held GPU.

Artifacts: /tmp/colibri_physx_two_body_isolated300.{json,npz,log,provenance.json}.
Runner: benchmark_physx_two_body_timing.py.

The current native two-body result around20.5FPS is roughly9.8times slower
in throughput, but its measurement duration differs. This is not an interleaved
paired benchmark or a statement of identical physical/discrete drive behavior.
The full-scene historical PhysX87.68–91.52FPS samples are a different workload.

The earlier18.63-second accuracy-process duration included setup/readbacks and
must not be interpreted as isolated solve FPS; the number above was measured
directly around completed SDK step calls.

## Independent repeat and refreshed full scene

The exact two-body repeat passed all same byte gates:
**212.6318 FPS /4.702966ms**, P95 6.395262ms.
Two independent measurements span201.36–212.63FPS (5.3% frame-time variation).
Artifacts: /tmp/colibri_physx_two_body_isolated300_repeat.*.

The unchanged historical37-body benchmark now measures
**105.7821FPS /9.453400ms**, P95 11.420167ms. All10 non-timing state/property
arrays match its historical comparator byte for byte; source/runner hashes
also match. Artifacts: /tmp/colibri_physx_full37_isolated300_refresh.*.

**Sleep configuration differs:** the full scene authors zero sleep threshold
on only5of37bodies; the remaining32 use the pinned SDK schema default0.00005.
Its scene stabilization flag is unauthored, schema defaultfalse. The two-body
fixture explicitly disables sleeping on both bodies and stabilization on its
scene. Therefore the refreshed full-scene FPS is not an all-awake performance
target. Actual runtime sleep state has not been measured.

## Explicit all-awake full-scene companion

One additional serialized measurement sets sleepThreshold0 on all37 bodies,
stabilizationfalse, position min/max30 and velocity min/max1. Composed USD
comparison proves no other attribute, relation, API or geometry change.
Result: **119.6676FPS /8.356482ms**, P95 8.992219ms,300 measured after30warmup.
All six SDK physical-property arrays and paths are byte-identical to the prior
full-scene control. Initial/final q/qd also match byte-for-byte over330frames.
This is a single timing sample, not evidence that disabling sleep speeds up
physics; variation across independent short runs remains material.

Artifacts: /tmp/colibri_physx_full37_awake_isolated300.* and
/tmp/colibri_physx_full37_awake.fixture.json. Preparation script:
prepare_physx_full_awake.py. No drift/conservation acceptance inferred from FPS.
