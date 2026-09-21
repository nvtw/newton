# Tourbillon clock

Run `uv run -m newton.examples phoenx_tourbillon_clock` from the repository.
The example uses the adjacent OBJ links and
`newton/examples/phoenx/tourbillon_clock_scene.py`; USD and `pxr` are not
needed at runtime. OptiX is the default viewer. Use `--viewer null` for
headless physics, `--motor-off` to disable the escapement motor while retaining
the authored springs and damping, and `--test` to monitor hinge errors.

## Source fidelity

The assets were extracted from `TourbillonClock.usd`, whose SHA-256 is recorded
in the generated scene dictionary. The descriptor contains 24 rigid bodies, 53
mesh shapes, and 24 authored revolute joints. Geometry is converted from the
stage's Y-up centimeters to Z-up meters. Authored normals and UVs are retained;
missing normals are generated over 30-degree smooth patches while sharp edges
remain split. Meshes are not simplified.

The Newton model uses 23 revolute joints. The second of two coaxial bearings
between `Body` and `SmallConnectorWatchGear` is omitted because one revolute
already constrains all five relative degrees of freedom; keeping both produces
a redundant same-body-pair constraint. The other body poses, joint frames,
collision filters, SDF settings, visual materials, explicit masses, and angular
drives are preserved. Angular drive gains are converted from USD's degree
coordinates to SI radians. The wall mount is kinematic and gravity is zero, as
authored.

The mechanism is driven at the authored -800 degrees/s through
`TourbillonSpecialGear`. The tourbillon oscillator and stopper retain their
authored springs. PhoenX uses its direct maximal-coordinate D6 joint solve and
momentum-conserving paired contact impulses. Clock pivots use 0.01 N m of
Coulomb friction; the low-loss oscillator pivot uses 0.0005 N m. The equal and
opposite friction impulses preserve angular momentum. Contacts refresh at 120 Hz
and the renderer receives device-resident double-buffered transforms. Fresh
contact matching avoids carrying warm-start identity across successive ratchet
teeth.

OBJ files are copyrighted source assets and are deliberately ignored by Git.
The local links point to:

```text
/home/twidmer/Documents/colibri/Colibri Plans Rev FF/TourbillonClock obj
```

## Regenerate assets

Regeneration requires USD Python bindings and trimesh:

```bash
uv run python newton/examples/assets/tourbillon_clock/extract_usd.py \
    /home/twidmer/Documents/colibri/TourbillonClock.usd \
    --output "/home/twidmer/Documents/colibri/Colibri Plans Rev FF/TourbillonClock obj"
uv run ruff format newton/examples/phoenx/tourbillon_clock_scene.py
```

The extractor writes meter-scaled OBJ files with explicit normals to the
external directory and regenerates the Python descriptor in the repository.

## Validation

Run the CPU asset and joint tests with:

```bash
uv run python -m unittest newton.tests.test_tourbillon_clock
```

The default 128-cell collision SDFs can be replaced by authored resolutions
with `--sdf-resolution 0`. A 120-frame GPU regression at 120 Hz contact updates, eight substeps, four
mixed iterations, two direct joint projections, and one velocity iteration
kept the maximum joint gap below 1.5 micrometers and hinge-axis error below
0.002 degrees. A 10-second run remained finite with the complete gear train and
tourbillon oscillator active.
