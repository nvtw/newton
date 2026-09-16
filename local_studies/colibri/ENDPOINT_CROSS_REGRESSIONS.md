# Cross-example regression after mesh edge endpoint ownership repair

## Collision-path scope

Kapla builds box bricks and optionally a spherical camera collider, with a ground plane (example_kapla_tower.py). The cached G1 learned-controller example imports USD and immediately calls builder.approximate_meshes("convex_hull"), with self-collision disabled (example_robot_policy.py). These examples do not use the mesh edge/SDF endpoint search under repair. They do exercise canonical dispatch, mass-splitting, and reduced factor paths changed during Colibri work.

## Checks after source freeze

- Kapla: 30 warmup plus 180 measured frames, existing mass-split configuration, head-launch count 8. Compare position, orientation, linear/angular velocity, contact impulses and friction history bytewise to /tmp/kapla_head8.npz. Timing is diagnostic only while other correctness work runs.
- G1: actual cached learned-controller example, 20 policy steps (0.4 physical seconds), unchanged 20-by-8 reduced solver settings. Compare all body pose and velocity histories bytewise to /tmp/g1_cache_only_candidate.npz; require finite controller/state and the existing above-ground final assertion.

No physics parameters or collision approximations are changed for these checks.

## Results

G1 passes all existing checks after 20 controller steps; complete body pose and velocity histories are byte-identical to /tmp/g1_cache_only_candidate.npz. Artifact: /tmp/g1_endpoint_regression20.json and .npz.

Kapla passes finite-state checks after 30 warmup +180 frames and all six saved arrays are byte-identical to the newer accepted /tmp/kapla_rebased_packed_fixed.npz baseline. The first comparison against the older /tmp/kapla_head8.npz failed because that artifact predates the accepted packed-layout correction; this is not an endpoint-fix regression. Mean speed0.00624227m/s and maximum drift13.8471mm match the newer reference. Artifact: /tmp/kapla_endpoint_regression180.json and .npz; its comparison field records the deliberately retained older-baseline mismatch. Timings are excluded.
