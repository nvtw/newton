# Ground residual before and after copy averaging

A read-only capture at public geometric30x1 frame330 records the last copy
velocities before `_mass_splitting_average_and_broadcast`. All330 final body
pose/velocity history frames remain byte-identical to the uninstrumented run.
The contact headers independently confirm FrameGround/world ownership,
static and dynamic friction0.5, and three dynamic base copies.

For loaded non-speculative contact27, tangential speed increases from
0.05885mm/s in its solved copy to0.97207mm/s in the averaged physical body;
normal velocity changes from-0.00520 to+1.34422mm/s. Its tangent/normal impulse
ratio is0.06746, below the0.5 friction limit. Contact26 has0.22315→0.90158mm/s
slip with an interior friction ratio0.01809. Contact42 has the reverse slip
change1.26826→0.77663mm/s but normal residual increases after averaging.

The three pre-average base linear velocities [m/s] are:

- (-0.001132,-0.001644,0.006278)
- (-0.000479,-0.000098,0.000167)
- (+0.000329,-0.001278,0.000098)

This identifies a concrete source of final contact residual: independently
solved copies are not mutually converged after the existing sweep budget.
It does not establish that all long-run travel is numerical, nor does it
justify pinning the base, damping it, or adding nonphysical friction loads.
Possible remedies must improve coupled convergence while preserving physical
impulses and the120Hz/30-substep comparison. The averaging operation itself
is required by the current mass-splitting formulation; removing it is invalid.

Artifacts:

- `/tmp/colibri_ground_pre_average330.npz`: physical state/contact snapshot
- `/tmp/colibri_ground_pre_average330_copies.npz`: pre-average copies
- `/tmp/colibri_ground_copy_residual330.json`: per-contact comparison
- `/tmp/colibri_geometric_ground_residual330.json`: loaded/non-speculative filter

Reproduce capture:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.capture_pre_average --copy-snapshot /tmp/colibri_ground_pre_average330_copies.npz --frames 330 --substeps 30 --geometric-candidates --save-history --save-contacts --output /tmp/colibri_ground_pre_average330.json
```
