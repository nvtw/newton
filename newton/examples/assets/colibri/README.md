# Colibri mesh assets

These local assets come from the supplied `Colibri Plans Rev FF` OBJ files and
`Colibri.usd`. The original files remain unchanged.

## Units and geometry

The original OBJ vertex coordinates are centimeters. These copies are meters,
including the USD assembly scale of 1.5: each source vertex is multiplied by
0.015. Do not apply another centimeter-to-meter conversion when loading them.
The affine transforms in the example preserve mesh placements and instances.

62 original OBJ files matched USD vertex positions within 0.00002 cm. Edited or mirrored
mesh instances, the two ball-joint spheres, and the slider are represented by 11 `*.usd.obj` fallback files extracted directly
from USD in body-local meters. The 83 cylinder meshes are reconstructed as
32-sided SDF meshes by the PhoenX example and require no mesh files.
The Kamino example also supports native cylinders for comparison.

## Assembly and simulation

All scene constants, joints, drives, and construction code are in
`newton/examples/kamino/example_kamino_colibri.py`; USD is not read at runtime.
The saved poses are retained, with initial velocities reset to zero.
Nine redundant/disabled joints are omitted (38 distinct enabled joints remain).
Joint targets that name a mesh are resolved to the owning rigid body.
The base is dynamic, with an optional `--fix-base` diagnostic. The flower is a
separate kinematic body because its USD hinge is disabled. The unjointed TailRack remains a free body.
The four genuine loop closures are retained outside the articulation tree.

Authored physics-material densities and friction are preserved, including the
60000 kg/m³ weighting of several small mechanism parts. Density is converted
from kg/cm³ to kg/m³; the assembly scale is already included in geometry volume.
Unbound shapes use density 1000 kg/m³ and friction 0.5. Only collision-enabled
shapes contribute mass, following USD automatic mass aggregation.
Joint positions are meters and angles are radians. Angular drive gains are
converted from the source torque-per-degree convention to SI torque per radian.
Self-collision remains enabled between unconnected bodies; joint-connected
body pairs are filtered to prevent bearings and pins fighting their joints.
Mesh contacts use cached signed distance fields to preserve gear/cam concavity.

## Incremental runs

```bash
uv run --extra examples -m newton.examples kamino_colibri --body-count 1
uv run --extra examples -m newton.examples kamino_colibri --body-count 2
uv run --extra examples -m newton.examples kamino_colibri --body-count 3
uv run --extra examples -m newton.examples kamino_colibri
```

`BODY_ORDER` in the example lists all 36 stages. `--body-count 1` builds
FrameGround and the separate flower, 2 adds Frame, and 3 adds the crank.
A headless stability check for any stage uses `--viewer null --test --num-frames 180`.
The first run builds SDF caches in the temporary `newton_colibri_sdf_cache` directory. CUDA is required for mesh SDF generation.

Asset ownership remains with the original Colibri design and supplied files;
this conversion does not assign a new license to those assets.

## Solver comparison

```bash
uv run --extra examples -m newton.examples phoenx_colibri
uv run --extra examples -m newton.examples phoenx_colibri --body-count 15
```

PhoenX uses source mesh cylinders and contact offsets, 120 Hz contact refresh,
and no body damping. At 30 substeps per contact update and one solver
iteration per substep, PhoenX passed a 300-second headless run: peak fresh
penetration 0.754 mm, joint-anchor error 0.939 mm, and joint-axis error
0.0288 rad. The free assembly still slides and yaws; the escape check is
relative to FrameGround. Support friction and gear engagement require further
validation. Kamino has not yet passed full-mechanism validation. Both solvers
remain experimental for this mechanism. The source gravity is 1 m/s².

Run the public PhoenX checks with:

```bash
uv run --extra examples -m newton.examples phoenx_colibri --viewer null --test --num-frames 18000
```

This checks 300 simulated seconds at 60 displayed frames per second. It is
not a timing command: fresh-contact and joint diagnostics add overhead.

## Reproduce the staged validation

The local development harness runs each prefix separately and stops at the first
failed stability check:

```bash
uv run --extra examples -m local_studies.colibri.check_stages --frames 180
```

Checks cover finite states, bounded speeds and displacement,
joint-anchor separation below 2 mm, and contact-buffer capacity. The final
assembly also requires a longer visual motion check; short stability checks
alone do not establish that every cam and gear moves correctly.
