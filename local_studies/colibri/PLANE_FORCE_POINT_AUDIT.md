# Known-plane force-point audit

CPU-only diagnostic; no production or live-candidate changes. Capture:
`/tmp/colibri_analytical_birth_capture.birth.npz`. Reproduce with:

```
OPENBLAS_NUM_THREADS=1 uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.audit_plane_force_points
```

Results: `/tmp/colibri_plane_force_points.json` and `.npz`. Higher precision is
used only for offline reconstruction/SVD; the plane-lever candidate is FP32.

## Native geometry

`newton/_src/solvers/phoenx/constraints/constraint_contact_cloth.py:690–698`
transforms both witnesses, includes margins, and chooses their midpoint as the
common impulse point. `:738` likewise initializes material anchors from the
midpoint. Normal gap remains the difference of the two original witnesses.
The scene's known plane is authored at z=0.002965275 m in
`newton/examples/kamino/example_kamino_colibri.py:2346`.

At the captured flat base, 52 ground rows have:

- Body witness penetration 33.06–48.43 nm, offline normalized reconstruction.
- Stored plane witness heights spanning 0.466 nm around the authored plane.
- Native common-point heights spanning 10.943 nm.
- Native tangent Jacobian singular values approximately
  `[7.21113, 7.21113, .64305, 2.408e-8, 2.289e-8, 3.23e-17]`.

Replacing **only the vertical lever component** by the one shared FP32 value
`plane_height - body_COM_z` leaves normal rows exactly unchanged and changes
tangent coefficients by at most 26.78 nm. Its tangent singular values are
`[7.21113, 7.21113, .64305, 9.07e-18, 5.61e-19, 0]`.
This is an explicitly different geometric operator, not a truncation of the
original small modes.

For a common plane-relative height h, tangent rows are
`[1,0,0, 0,h,-y]` and `[0,1,0, -h,0,x]`. They span the three physical combinations
`vx+h*wy`, `vy-h*wx`, and `wz`. Varying h adds two independent rotational
combinations. A single known plane therefore gives the three-mode structure
by construction; no singular-value threshold is needed.

The other 15 captured rows belong to Frame and have 8.19–11.96 mm positive
gaps. They are speculative rows, not evidence of a loaded sticking patch;
native preparation disables friction past 2 mm. Projecting their lever arms
would change them by up to 5.98 mm, so they must not be mixed into the flat-base
conditioning claim or used to justify activating friction.

## Momentum, geometry, and limits

Applying equal and opposite impulses at one point on the known plane preserves
linear/angular momentum in exact arithmetic. It does not require moving the
bodies, changing the collision witnesses, or changing gap/activation targets.
For this axis-aligned plane, shifting the common point only along its normal
also preserves the normal angular Jacobian exactly. Tangential torques change
explicitly by the normal shift crossed with the tangential impulse.

Separate FP32 COM subtraction still leaves ordinary rounding in the two
lever arms. The capture's per-unit-tangent world torque defect is 1.86e-9 m;
a straightforward projected-point FP32 construction also has 1.86e-9 m here.
Plane selection does not solve all scatter rounding or friction-history issues.

Using the known rigid surface is a physically defensible force-point convention
for a finite-penetration contact approximation. It is not algebraically identical
to the midpoint convention. A bounded trial must update both lever arms and
all associated mobilities consistently while retaining original witnesses,
normal recovery, contact activation, friction law and history semantics. A
stale point-only patch would be inconsistent. No live benefit has been tested.

## Actual PhysX GPU convention

The inspected source does **not** support claiming that PhysX uses a projected
plane point:

- `physx/source/gpunarrowphase/src/CUDA/trimeshCollision.cu:198–209` computes the
  plane separation but stores `worldPoint = trimeshTransform.transform(p)`,
  the mesh vertex, as its contact point.
- `:311–314` stores a separate plane-projected PCM witness. It is not the point
  exported to the solver: cached contact loading at `:341–344` transforms mesh
  `pointA`; output at `:397–401` exports that mesh point and separation.
- `physx/source/gpusolver/src/CUDA/contactConstraintBlockPrep.cuh:881–891`
  uses the exported point to make both normal-contact lever arms.
- The same file's `growPatches` at `:76–200` selects at most two friction anchors
  from exported contact points and transforms each common world anchor into
  both bodies. TGS friction preparation at `:1030–1040` rotates those persistent
  anchors and forms their positional error.

Thus known-plane projection is a separate geometry-consistency hypothesis.
PhysX's limited persistent patch anchors are another material difference, but
this audit does not establish either as the cause of the observed drift gap.
