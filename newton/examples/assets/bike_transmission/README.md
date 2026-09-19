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
60 rpm bicycle cadence, and the rear drive acts as a 0.005729578 N m s/rad
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

The default configuration uses 120 Hz collision detection, 8 physics substeps
per collision refresh, 4 grouped contact iterations, one velocity iteration,
and the direct joint solver. Grouped PGS alternates forward and reverse color
order between iterations to reduce directional bias without adding work. Exact
mass-metric joint projections alternate with grouped mass-split contact PGS.
Parallel contact preparation is enabled on CUDA,
and contact columns are capped at 64 rows. Every contact row is retained; the
cap distributes long shape-pair columns across more solver work.
`--contact-chunk-size 0` restores whole shape-pair columns. `--substeps N`
and `--iterations N` permit convergence experiments.

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

A 270-frame measured run of the 60 rpm loaded default configuration sampled
joint gaps up to 0.120 mm, hinge-axis misalignment up to 0.0056 degrees,
contact penetration up to 1.10 mm after startup, and lateral chain span up to
2.77 mm. The rear dynamometer absorbed roughly 1.5--2 W while the chain stayed
engaged. The source pose begins with about 1.70 mm of contact overlap. The
penetration probe uses a separate collision pipeline so it cannot alter live
contact matching. These are sampled diagnostics, not bounds over every substep
or a validation of interactive gear shifts.

On an RTX PRO 6000 Blackwell, headless simulation measured 44.5--46.5 FPS,
excluding startup and diagnostic reads. The earlier 14-substep, 2-iteration
configuration measured 29.4 FPS. Symmetric body-pair contact refinement permits
the lower substep count while preserving equal-and-opposite impulses.

Use `--solver-stats` to print the actual color sizes, sequential color-group
sizes, and overflow count once per simulated second. This opt-in diagnostic
reads counters back from the GPU and is disabled during normal rendering.

The source requests PhysX TGS at 120 Hz with a minimum of 100 position
iterations. TGS position iterations advance temporal substeps; they are not
directly equivalent to PhoenX iterations within each substep. The current
configuration is selected by measured joint and contact errors.
