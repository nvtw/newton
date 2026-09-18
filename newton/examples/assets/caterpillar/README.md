# Caterpillar excavator

These local OBJ meshes are extracted from the user-supplied `Caterpillar.usd`.
Body poses, mesh placements, materials, joint frames, limits, and drives are
stored in `newton/examples/phoenx/caterpillar_scene.py`. The USD and USD Python
libraries are not required at runtime.

The copyrighted OBJ files are not distributed with Newton. Keep generated meshes in an external asset directory. Ignored symbolic links
in this directory can make those files available to the example without
copying or committing them. For example:

```bash
ln -s "/path/to/Caterpillar obj"/*.obj.gz newton/examples/assets/caterpillar/
```

Run from the repository root:

```bash
uv run --extra examples -m newton.examples phoenx_caterpillar
```

The source is a Z-up, metre-scale scene containing 131 rigid bodies, 129
revolute joints, three prismatic hydraulic joints, two fixed joints, and two
closed tracks with 46 links each. Its moving-body origins span approximately
7.0 by 6.5 by 3.1 metres. Saved transient velocities are intentionally reset;
the authored joint targets and gains remain active. `--motor-off` disables all
authored drives for passive diagnostics.

The 1,325 rendered meshes reduce to 453 unique compressed OBJ files after
geometry deduplication. They retain transformed vertices in metres, split
surface normals, UVs where authored, sharp edges for meshes lacking normals,
and corrected winding under negative transforms. The example uses 315 authored
collision meshes: 314 SDFs and one convex hull. The asset has no authored mass
or density, so closed collision volumes use 1000 kg/m³, matching USD's default
density. Visual-only meshes do not contribute mass.

The default configuration performs collision detection at 120 Hz, five
physics substeps per collision refresh, two contact iterations per substep,
and exactly one velocity iteration. It uses direct momentum-conserving D6 joint
projection and grouped mass-split PGS contacts. On an RTX PRO 6000 Blackwell,
an isolated 12-second headless benchmark measured 43.2 FPS. A 20-second
simulated-duration diagnostic remained finite; maximum joint translation error
was 0.099 mm and maximum angular error was 0.0018 degrees. The settling tracks
reached 87 mm of penetration in a 180-frame example test, then converged to
7--12 mm after three seconds. These are sampled diagnostics rather than bounds
over every substep.

The default SDF resolution is 128. The source requests resolutions up to 500,
which cost substantially more memory and startup time; pass
`--sdf-resolution 0` to reproduce authored resolutions. SDFs are cached under
`/tmp/newton_caterpillar_sdf` after the first run.

To regenerate the extracted data (requires `pxr`), run:

```bash
uv run python newton/examples/assets/caterpillar/extract_usd.py \
  /path/to/Caterpillar.usd \
  --output "/path/to/Caterpillar obj" \
  --scene-output newton/examples/phoenx/caterpillar_scene.py
uv run ruff format newton/examples/phoenx/caterpillar_scene.py
```

The generated descriptor records the source SHA-256 for provenance checks.
