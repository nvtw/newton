# BikeTransmission

These local OBJ meshes are extracted from the user-supplied
`BikeTransmission.usd`. All body poses, mesh placements, materials, revolute
joints, and drive settings are recorded in
`newton/examples/phoenx/bike_transmission_scene.py`. The USD and USD Python
libraries are **not required to run the example**.

The copyrighted OBJ files are not distributed with Newton. Keep generated
meshes in this directory beside the Colibri asset directory; `.gitignore`
excludes them from commits.

Run from the repository root:

```bash
uv run --extra examples -m newton.examples phoenx_bike_transmission
```

The source has 134 bodies (including two stationary kinematic mounts), 134
revolute joints, and a closed chain of 120 links. There are 492 mechanism meshes;
three environment meshes and the source lighting/OmniGraph are not imported.
The six authored joint drives remain active. The front drive defaults to a
60 rpm bicycle cadence, and the rear drive acts as a 0.020053523 N m s/rad
viscous dynamometer load. Use `--cadence-rpm` and `--rear-load-damping` to
change them. `--motor-off` disables only the front crank drive, leaving
derailleur springs and rear load damping active. The two rear-derailleur springs
keep their authored rates, while their preload angles default to three times
the unusually low source values and their damping defaults to twice the source
values. This stays clear of the revolute angle-wrap
boundary while reducing chain sag and spring oscillation. Use
`--derailleur-preload-scale 1 --derailleur-damping-scale 1` to reproduce the source values.
Interactive gear-changing OmniGraph logic is not reproduced.

The default viewer is OptiX. To run an accuracy smoke test without rendering:

```bash
uv run --extra examples -m newton.examples phoenx_bike_transmission --viewer null --num-frames 240 --test
```

At the default 60 rpm cadence, the configuration uses 120 Hz collision
detection, 8 physics substeps per collision refresh, 4 grouped contact
iterations, one velocity iteration, and the direct joint solver. Higher cadence
automatically raises contact refresh frequency by 60 Hz per 30 rpm and uses 6
substeps per refresh. This keeps sprocket travel near 5 mm per collision update.
The crank drive reaches its target over two seconds to avoid an unphysical
startup impulse. `--contact-updates-per-frame`, `--substeps`, and
`--startup-ramp-time` provide explicit experimental overrides.

Two-color mass-copy partitions expose independent GPU work while retaining
ordered Gauss-Seidel updates inside each partition. Grouped PGS alternates
forward and reverse color order between iterations to reduce directional bias
without adding work. Exact
mass-metric joint projections alternate with grouped mass-split contact PGS.
Contact columns are capped at 64 rows. Every retained contact row is solved.
Material-aware body-pair manifolds retain one deepest point in each of 12 normal
bins plus spatial extrema in all 26 nonzero directions of a local 3x3x3 stencil.
The diagonal support directions preserve separated regions of nonconvex compound
sprockets while reducing redundant graph colors. `--contact-chunk-size 0`
restores whole columns.
`--iterations N` permits additional convergence experiments.

Velocity-filtered speculative contacts cover up to 6 mm of predicted travel per
collision refresh, enough for one front-tooth transit at the default cadence.
Use `--speculative-contact-gap-max` to change that cap. The extension only
changes candidate acquisition; the solver still uses the physical separation.

Geometry is in metres and the scene is Z-up, converted from centimetres/Y-up.
Gravity is 9.8 m/s². Angular targets are converted from degrees to radians and
angular gains from source torque/degree to N·m/rad. The asset has no authored
mass or density values. Chain collision geometry uses steel density, gears use
aluminum density, and other collision geometry uses 1000 kg/m³; each density
sets both mass and inertia. Visual-only geometry contributes no mass. The
authored body poses are preserved; saved transient body velocities are replaced
with rest startup.

OBJ files retain the triangle geometry, explicit transformed normals, UVs when
present, and split vertices at authored normal discontinuities. Negative-scale
transforms reverse winding. Materials are stored alongside shape descriptions
in Python. Collision meshes retain SDF collision geometry, with source grid
resolutions where authored (otherwise 128). SDFs are built on the first run and
cached in the system temporary directory under `newton_bike_transmission_sdf`.
`--sdf-resolution N` is an explicit resolution override for experiments.

To regenerate from the original source (requires `pxr`):

```bash
uv run python newton/examples/assets/bike_transmission/extract_usd.py \
  /path/to/BikeTransmission.usd
uv run ruff format newton/examples/phoenx/bike_transmission_scene.py
```

The generated Python data records the source SHA-256.

A ten-second run of the ramped 60 rpm default ended at 44.0 crank rpm and
136.6 rear rpm, a 3.11 ratio, with 4.10 W absorbed by the rear dynamometer. Its
final chain span was 1.14 mm, joint attachment error was 0.042 mm, and hinge-axis
misalignment was 0.0025 degrees.

Cadence-scaled ten-second runs also remained engaged under the same rear load.
The 90 rpm target ended at a 3.79 ratio and 10.16 W with a 2.3 mm chain span and
0.058 mm joint gap. The 120 rpm target ended at a 4.15 ratio and 19.80 W with a
0.7 mm span and 0.053 mm joint gap. A non-mutating live-contact audit at 120 rpm
sampled at most 1.54 mm penetration, localized to rear-sprocket tooth contact.
The final crank-to-dynamometer power efficiencies were approximately 92%, 86%,
and 89% at the 60, 90, and 120 rpm targets. These diagnostics are sampled rather
than bounds over every substep, and do not validate interactive gear shifts.

On an RTX PRO 6000 Blackwell, the steady-state 60 rpm default measured 69.64
headless FPS, excluding startup and diagnostic reads. A separate 240-frame
1920x1080 OptiX run with asynchronous simulation/render overlap measured 60.09
FPS. The 90 and 120 rpm schedules measured 58.15 and 44.44 headless FPS,
and 52.09 and 40.84 FPS with asynchronous 1920x1080 OptiX rendering.
The expanded body-pair support stencil preserves disconnected sprocket regions
while grouped mass splitting parallelizes independent interactions with
equal-and-opposite impulses.

Use `--solver-stats` to print the actual color sizes, sequential color-group
sizes, and overflow count once per simulated second. This opt-in diagnostic
reads counters back from the GPU and is disabled during normal rendering.

The source requests PhysX TGS at 120 Hz with a minimum of 100 position
iterations. TGS position iterations advance temporal substeps; they are not
directly equivalent to PhoenX iterations within each substep. The current
configuration is selected by measured joint and contact errors.
