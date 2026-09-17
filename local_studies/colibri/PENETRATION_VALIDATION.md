# Colibri gear/pin penetration investigation

Date: 2026-09-17. Configuration: passive base/frame axle, flower attached to
base, counterweight density scale 1.0 unless specified below.

## Cause and change

The first failing pair was
`Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh` against
`Frame/Cylinders/Cylinder_11`. It was detected from the beginning, retained
matched contact IDs, and accumulated normal impulse through the substeps.
Fresh penetration crossed 1 mm at 0.4667 seconds.

Sticky contact gathering replaces freshly generated local contact points and
world normals with historical geometry
(`newton/_src/geometry/contact_sort.py::_gather_full_kernel`). On the curved,
moving gear/pin surfaces this leaves inconsistent contact constraints. The
substep trace showed cached penetration and normal impulse increasing despite
the already-present contact.

Use `contact_matching="latest"` in the Phoenx example: keep frame-to-frame
matching but use fresh normal geometry at every original 120 Hz collision
refresh. The temporal friction solver retains material anchors independently
in `contact_tgs_anchors.py`; these do not require freezing normal geometry.
PhysX similarly constructs normal rows from the current contact buffer while
correlating persistent friction patches separately; see the contact-buffer
loop around lines 563-607 of
`/home/twidmer/Documents/git/physics/physx/source/lowleveldynamics/src/DyTGSContactPrep.cpp`.

This change leaves the paired common-point impulse solver intact. No contacts,
joint constraints, masses, damping, or tolerances were changed for the fix.
The example remains at 120 Hz collision detection and 24 physics substeps per
refresh. Higher rates below are diagnostic comparisons only.

## Full-minute measurements

Each case sampled independent fresh contacts at every 60 Hz frame, including
the initial pose (3,601 samples). Timings exclude validation and rendering,
but include synchronized physics/collision stepping.

| Normal geometry | Collision Hz | Density scale | Gear/cylinder peak [mm] | Samples above 1 mm | Physics [ms/frame] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sticky | 120 | 1.0 | 1.753 | 503 | 23.21 |
| Sticky | 120 | 1.2 | 1.753 | 454 | 23.24 |
| Sticky | 240 | 1.0 | 1.753 | 95 | 27.04 |
| Sticky | 480 | 1.0 | 0.281 | 0 | 36.42 |
| Fresh, adopted fix | 120 | 1.0 | 0.470 | 0 | 23.30 |

The original frame-pin pair's peak with the fix was 0.261 mm. The fixed run's
overall 0.470 mm peak involved `CamWheelBottom/Cylinder_40` and the lower
hypocycloid gear. The 1.2 weight factor did not prevent the original failure,
so the default remains 1.0. Reducing the constraint color-group size to one
or two also failed short contact and crank-tracking checks and was not adopted.

## Validation and remaining limitation

The new `newton.tests.test_colibri_contacts` regression failed before the
fix at 1.085 mm penetration and passed afterward, at the same 120 Hz update
rate. Nineteen focused tests passed, including scene, temporal policy,
dynamic contact, and friction tests with paired momentum checks.

The full-minute fixed run passed finite/bounded assembly, joint attachment,
crank tracking, and the existing 1 mm fresh-penetration screen. It still
failed the separate 50 micrometer base-support displacement limit:
the first failure was 0.170 mm at 6.75 seconds. The threshold remains in place.
This is not an all-clear for support stationarity or arbitrarily long runs.
Fresh point depths also do not prove all geometric intersections are absent.

## Monitor and artifacts

```bash
uv run -m newton.examples phoenx_colibri --penetration-log /tmp/colibri-penetration.csv
```

The optional log records time, deepest fresh contact, its shape labels, and
deepest hypocycloid gear/cylinder contact. Logging adds an independent collision
pass and CPU synchronization; it is disabled by default. A 120-frame comparison
confirmed monitored/unmonitored poses agree to 1e-7, with scene checks passing.

Temporary evidence:
- `/tmp/colibri_penetration_comparison.png` and `.json`
- `/tmp/colibri_penetration_s1.0_r2_n24{,_latest}.{csv,json}`
- `/tmp/colibri_penetration_s1.2_r2_n24.{csv,json}`
- `/tmp/colibri_gear_substeps.{npz,json}`
- `/tmp/colibri_contact_regression_before.log`
- `/tmp/colibri_contacts_after.log`

The CSV histories must be inspected over time: final poses alone can hide
a gear that has already passed through a pin.
