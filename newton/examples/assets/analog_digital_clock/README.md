# Analog-digital clock

Run `uv run -m newton.examples phoenx_analog_digital_clock` from the repository.
The example loads ignored links to the gzip-compressed OBJ files in
`~/Documents/Meshes/AnalogDigitalClock` and
`newton/examples/phoenx/analog_digital_clock_scene.py`; USD and `pxr` are not
needed at runtime. OptiX is the default viewer. Use `--viewer null` for headless
physics, `--motor-off` to disable the motor gear drive while retaining the seven
follower springs, and `--test` to monitor joint attachment and hinge alignment.

## Source fidelity

Extracted from `AnalogDigitalClock_SI_Units.usd`. The generated scene dictionary
records the SHA-256 of the source. It contains 25 dynamic bodies, 31 revolute
joints, and 54 mechanism shapes. The frame remains dynamic and rests on the
authored ground at -0.0114882954 m; it is not pinned to the world.

The stage is Z-up with meters and kilograms, but body transforms contain 0.01
scales. These scales are baked into mesh geometry and joint anchors. Body
rotations are extracted after removing scale and shear. OBJ vertices are in
meters, with authored normals retained and sharp edges split. Where CAD meshes
omit normals, extraction generates smooth normals within 30-degree surface
patches while preserving sharp edges. No mesh simplification is applied.

Physics material densities are preserved: 30 kg/m³ for the digit mechanisms,
5000 kg/m³ for followers and small gears, and the USD default 1000 kg/m³ for
unassigned shapes. The two density materials have the USD default zero friction;
unbound shapes and the ground use the PhysX default coefficient 0.5. Visual-only
shapes do not contribute mass. Seven follower
springs target 45 degrees with the authored stiffness. The motor gear
retains its +100 degree/s velocity drive and damping. The cam joint has dormant
-30 degree/s attributes without an applied DriveAPI; these remain inactive. Angular gains
are converted from degrees to radians. Connected-body collisions remain enabled
for the seven follower joints as authored; other connected pairs are filtered.
Authored body angular damping is retained as external drag torque `-d I_world ω`,
evaluated before each 120 Hz solve: 10 s⁻¹ for followers, 1 s⁻¹ for digit bodies,
and the PhysX default 0.05 s⁻¹ elsewhere. This drag intentionally removes angular
momentum; joint/contact impulses still use the momentum-conserving solver.
PhysX applies a clamped `1 - d dt` velocity factor before its constraint solve;
the example instead distributes the corresponding external torque across PhoenX
substeps, so this is not an exact reproduction of PhysX damping ordering.

SDF collision resolutions are retained, while the three authored convex shapes
use convex collision. Initial velocities are zero.

The default configuration refreshes contacts at 120 Hz, with 16 PhoenX
substeps, six mixed joint/contact iterations, two joint-only refinement
iterations, and one velocity iteration per refresh. The solver runs in
single-world mode. First launch builds and caches the SDFs under the system
temporary directory; subsequent launches reuse them. Joint-only refinement
uses the same paired impulses and mass-copy reconciliation as mixed sweeps, so
it retains momentum conservation while avoiding unnecessary contact work.

A 150-second run sampled every 0.1 seconds measured 66.55 headless physics FPS
on an RTX PRO 6000 Blackwell. Joint attachment error had 0.074 mm median,
0.852 mm p99, and 1.204 mm maximum; hinge-axis error had 0.0069 degree median,
0.0283 degree p99, and 0.0869 degree maximum. Settled penetration had 0.014 mm
median, 0.052 mm p99, and 0.097 mm maximum. The driven gear averaged
1.733 rad/s against its 1.745 rad/s target. The matched eight-mixed-iteration
baseline reached 60.62 FPS with slightly worse maximum attachment and hinge
errors. These are sampled measurements, not continuous bounds.

This scene exposed a general PhoenX compound-contact bug: mesh-plane and
convex-plane contacts can have opposite endpoint order within the same body-pair
column. Canonicalizing contact endpoints fixes the spurious frame impulses;
the example retains the source convex collision shapes.

## Regenerate assets

Only regeneration requires USD Python bindings and trimesh:

```bash
uv run python newton/examples/assets/analog_digital_clock/extract_usd.py \
    /path/to/AnalogDigitalClock_SI_Units.usd \
    --output ~/Documents/Meshes/AnalogDigitalClock
uv run ruff format newton/examples/phoenx/analog_digital_clock_scene.py
```

## Validation

`uv run python -m unittest newton.tests.test_analog_digital_clock` runs five CPU
tests covering joint frames, mesh normals/indices, density-based masses and
positive inertia, source drives/collision settings, and angular drag for a
rotated anisotropic body. All five also pass with `pxr` imports blocked.
The standard example framework passed a 120-frame null-viewer run and a
30-frame headless OptiX run with `--test`.

The standard example runner passed 100 frames with the one-velocity default.
At 1920x1080, headless OptiX with DLSS RR measured 54.95 FPS (18.20 ms/frame)
over 180 frames after 60 warmup frames. The matched eight-mixed-iteration
one-velocity baseline measured 53.52 FPS (18.68 ms/frame).
