# Five-minute gear motion audit

The canonical 18,000-frame trajectory continues rotating throughout the run.
Crank mean speed is -3.17343 rad/s versus the finite-gain source target
-3.49066 rad/s (about 9.1% slower). Every saved frame interval has negative
crank rotation: observed speeds range from -4.01483 to -2.12592 rad/s.
This rules out a sampled crank stall, not load-tracking error or subframe events.

After excluding the first second, least-squares phase fits give:

| Joint relative to Frame | Observed ratio to crank | Phase residual range [rad] |
|---|---:|---:|
| GearedSpinner | 0.999997774 | 0.313019 |
| CamWheelHead | 0.124999311 | 0.070540 |
| Gear_Large__3x_02 | -0.499999544 | 0.041228 |
| Gear_Large__3x_01 | 0.499999283 | 0.082235 |
| Gear_Large__3x_00 | -0.499998817 | 0.118822 |
| CamWheelBottom | 0.125000067 | 0.081726 |
| CamWheelTail | -0.124999225 | 0.081488 |

These fitted ratios are observations, not independently established design
ratios. In particular, fitting can absorb steady slip. The nearly rational
ratios and small variation across ten time intervals support sustained
coupling over this trajectory; they do not prove tooth-level engagement.
The spinner phase variation reaches 0.313 rad and still requires comparison
with the mechanical linkage geometry before being classified as error.

Reproduce with:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
  -m local_studies.colibri.audit_long_gear_motion \
  /tmp/colibri_cooperative_canonical_validation_colibri.npz \
  --output /tmp/colibri_long_gear_motion.json
```

No physics settings or acceptance tolerances were changed.

## Frozen drive convergence diagnosis

The independently saved final relaxation at frame330 separates finite motor
load response from residual equation error. The Frame/Crank row has
Jv=-2.99915903 rad/s and reference=-3.49065852 rad/s. Its physical accumulated
angular impulse is -1.1801494e-6 N m s; multiplying by compliance628.318508
gives a legitimate load lag of only0.00074151 rad/s. The executed constitutive
equation residual is0.49075798 rad/s. This single-state finding does not by
itself quantify the cause of every frame's average tracking error.

A second saved capture contains copies immediately before averaging. Its
per-body copy counts exactly match the relaxation capture, and independently
averaging those velocities reproduces physical velocities within7.45e-9 m/s
and2.38e-7 rad/s. Joint-first first-fit reconstruction places joints0/1 in
colors0/1, both in group0. Using the corresponding copy velocities and the
same final physical angular impulse gives:

| Drive | Final copy residual [rad/s] | Averaged physical residual [rad/s] |
|---|---:|---:|
| FrameGround/Frame | 0.03673231 | 0.13200489 |
| Frame/Crank | 0.22529142 | 0.49075798 |

The crank error already exists before averaging and increases by0.26546656
rad/s when copies are combined. These are end-of-sweep measurements, not
immediately after the drive row executes: later constraints can reopen the
row. This supports a coupled-convergence diagnosis and does not justify
removing averaging, rescaling the motor impulse or adding damping.

Inputs: /tmp/colibri_support_relax330.npz and
/tmp/colibri_ground_pre_average330_copies.npz. Outputs:
/tmp/colibri_native_drive_lag.json and /tmp/colibri_drive_copy_residual.json.
