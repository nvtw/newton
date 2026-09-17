# Experimental temporal rigid contacts

`SolverPhoenX(solver_scheme="tgs")` uses persistent two-anchor friction
patches, temporal joint springs, and one external-force update per collision
step. The existing `"soft"` scheme remains the solver default. The Colibri
example selects the temporal scheme with 24 substeps per 120 Hz collision
refresh and one biased solve per substep; velocity relaxation follows the
final substep. It uses `contact_matching="latest"` to refresh normal points
and normals while TGS retains its own material friction anchors. Sticky
normal geometry can become inconsistent on moving curved gear/pin surfaces.

Run the default OptiX viewer (the same renderer as the Kapla tower):

```bash
uv run -m newton.examples phoenx_colibri
```

Use `--viewer gl` to select OpenGL explicitly. OptiX requires `pyoptix` and
`warp_optix` from the local otk-pyoptix installation.

For a headless minute of simulation with physical checks:

```bash
uv run -m newton.examples phoenx_colibri --viewer null --num-frames 3600 --test
```

The implementation follows PhysX's normal-patch partitioning, anchor history,
normal-load friction budgets, and temporal spring equations. Impulses act at a
common world point on both bodies so they conserve paired linear and angular
momentum. Structural joint rows retain paired spatial wrenches. Mass copies are
averaged before static support is solved on physical bodies and broadcast back
to the copies. Spring impulses are disabled during velocity relaxation, as in
PhysX's `conclude1DStep`; structural velocity constraints remain active.
Within each color group, contact rows precede joint rows so later contact
updates cannot erase the freshly solved drive velocity before integration.
Every constraint still runs once per sweep, with unchanged copy ownership and
paired impulse application. This ordering reduced the Colibri crank's mean
tracking error from approximately 13% to 0.23% without retuning its drive.

Contact multipliers persist through the temporal substeps and reset at the
next collision step. Material anchors persist until correlation rejects them.
Normal rows are loaded cooperatively on CUDA, and friction geometry is prepared
in parallel; the ordered impulse solve is preserved. There is no added body
damping, base pinning, contact deletion, or extra support sweep.

## Colibri mass and attachment options

The Phoenx example defaults to `--counterweight-density-scale 1.0`, retaining
`Frame/Cylinder`'s authored 5,000 kg/m³ density. Adjusting the factor scales only
this cylinder before shape mass properties are accumulated. The containing
`Frame` body's mass, center of mass and inertia are consequently rebuilt
consistently. The base/frame axle's authored 20-degree position spring and damper
are disabled, allowing passive rotation under gravity and contact forces.
The factor must be finite and nonnegative.

Flower and slider helper geometry belong to the dynamic `FrameGround` body
in this example. The source USD instead nests a kinematic `Flower` beneath
the base and disables its connecting revolute joint. Combining those shapes
with the base gives a physical attachment, including their mass and inertia,
without kinematically anchoring the flower to the world. The shared Kamino
scene builder retains the authored behavior by default.

## Supported configuration

This experimental mode currently requires one rigid CUDA world, maximal
coordinates, `joint_solver="block_pgs"`, point-contact storage, mass splitting
with color groups, `solver_iterations=1`, `prepare_refresh_stride=1`, SOR 1,
and `velocity_readout="substep_end"`. It rejects armature, bounded active drives,
restitution, soft contact rows, sleeping, deformables, contact chunks, unrolled
dispatch and partition reuse. Property refresh validates supported model
properties before changing cached rows.

Requesting the `force` contact attribute before stepping enables optional
impulse-wrench accounting. `update_contacts` reports outer-step average force
and torque on body0 about its current COM, in collision-pipeline order. Anchor
friction is assigned to the first point in its patch. This point distribution
is a reporting convention; body-pair sums retain both the linear impulse and
the moment at the actual application points throughout the step. CUDA graph
capture supports this path. Force accounting is compiled out when not requested.

The scheme is a runtime solver policy, not an authored Model/USD property.
Its current SI tolerances are 0.00025 m for anchor correlation and 0.0004 m for
friction admission: PhysX's default factors evaluated at a 0.01 m length scale.
These are explicit limits of the experimental configuration, not a claim that
all PhysX scenes use those distances. This mode has not been validated across
arbitrary scene scales or large multi-world robot fleets.

## Validation of the revised scene

The following simulation results used density scale 0.9 with the axle drive
enabled. The current density scale 1.0 and passive axle configuration has not
been subjected to the same full-minute validation.

All eight scene tests and a 600-frame headless OpenGL run pass.
The density and attachment regressions confirm that counterweight density scales its
contribution to composite mass, COM and inertia, and all flower/slider shapes
share the dynamic base body. Shared-builder defaults preserve the Kamino scene.

The new physical configuration is not yet validated for a full minute. With
the unchanged 24-substep/four-color-group settings, it passes support and drive
checks but crosses the existing 1 mm gear-penetration limit at 29.32 seconds:
1.135 mm between `Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh` and
`Frame/Cylinders/Cylinder_11`. Increasing substeps or color-group size did not
resolve that contact failure; those experimental settings were not adopted.
The successful minute-long measurements below apply to the earlier mass and
attachment configuration, not the revised defaults.

## Colibri validation before the mass/attachment adjustment

On an RTX PRO 6000 Blackwell, the final 60-second public-example run passed
3,601 joint, contact, support and crank-tracking checks. From seconds 10 to 60,
net horizontal base motion was 1.04 micrometers and maximum excursion was
4.91 micrometers. Mean synchronized physics time was 21.72 ms/frame (46 FPS);
a fresh run of the previous soft configuration measured 21.13 ms/frame.
Rendering and diagnostic audits are excluded from these timings.

The crank completed 33.25 turns without stopping. Its mean speed was
3.4827 rad/s against a 3.4907 rad/s target (0.23% error). Authored drive gains
were not retuned. The example's test mode bounds settled support motion by
50 micrometers and 0.5 milliradians, and checks mean crank speed within 5% of
target after settling. Reverting the contact-before-joint ordering fails the
tracking check. OpenGL headless rendering also passed. These checks establish
readiness for visual inspection; they do not certify every gear tooth's
engagement throughout the mechanism.

CPU/CUDA component tests cover lossless partitioning and overflow, persistent
anchors, friction budgets, physical momentum after mass-copy averaging,
static support, contact removal, force timing, temporal springs, cached friction
geometry, and unsupported public configurations. Kapla and G1 retain their
existing solver settings and passed their regression runs.

## Penetration history

Use the optional CSV monitor during an interactive or headless run:

```bash
uv run -m newton.examples phoenx_colibri --penetration-log /tmp/colibri-penetration.csv
```

Each frame records simulation time, the deepest fresh contact and its shape
labels, and the deepest hypocycloid-gear/cylinder contact. Depths are in meters.
The monitor uses a separate collision pipeline and leaves solver contact history
unchanged. It adds collision detection and CPU synchronization overhead only
when enabled. The file is replaced when the example starts.

Inspect peaks over the whole trajectory, not just the final depth: a gear can
pass through a pin and subsequently report a shallow contact again. The
example's existing 1 mm penetration test remains an overlap screen, not proof
of correct tooth engagement. Joint attachment, support motion, and crank
tracking checks remain necessary.
