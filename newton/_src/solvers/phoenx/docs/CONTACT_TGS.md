# Experimental temporal rigid contacts

`SolverPhoenX(solver_scheme="tgs")` uses persistent two-anchor friction
patches, temporal joint springs, and one external-force update per collision
step. The existing `"soft"` scheme remains the solver default. The Colibri
example selects the temporal scheme with 24 substeps and one biased solve per
substep; velocity relaxation follows the final substep.

Run the standard interactive OpenGL viewer:

```bash
uv run -m newton.examples phoenx_colibri
```

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

## Colibri validation

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
