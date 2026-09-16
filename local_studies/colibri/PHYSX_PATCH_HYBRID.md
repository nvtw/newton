# PhysX patch-friction hybrid: diagnostic, not production law

The isolated adapter retains native joint-conditioned response, compliant drive,
normal softness, witnesses, gravity timing and integration. It changes normal/friction
ordering, pools normal load across a ground/body manifold, retains up to two material
anchors, and applies the GPU TGS scalar tangent trial followed by radial disk clipping.
It currently supports only the two-body ground-contact study. No canonical source changed.

## Measured 60-second control

`/tmp/colibri_physx_patch_hybrid3600.*`, 30 substeps/120 Hz, one biased pass:

- 10–60 s horizontal net motion: 0.824723 µm, 0.0164945 µm/s.
- 50–60 s: 0.091084 µm, 0.0091084 µm/s.
- 10–60 s world yaw: +0.00102194 degrees.
- Final hinge: 18.20732867 degrees; drive equation residual −5.01e−10 rad/s.
- Two-body timing: 26.9797 FPS; not full-scene timing.
- First 600 frames q, qd and times byte-identical to separate 600-frame run.
- Last 128 friction phases: linear ledger error 8.44e−12 Ns, angular 1.19e−12 Nms;
  independent-anchor angular defect 4.70e−13 Nms. This defect is measured, not zero by design.
- Actual drive impulse change plus rank-five hard-reaction reconstruction closes
  midpoint work to 2.75e−15 J in that window. Startup is a separate ledger and has
  larger independent-anchor torque defect (2.50e−9 Nms).

The fixed source coefficient at 30 TGS position iterations is p8=2/sqrt(30)=0.36514837.
It multiplies velocity and recovery terms; this is explicitly a diagnostic underrelaxation.
The next controlled variant uses unit velocity coefficient while retaining the exact
source recovery impulse coefficient. Global SOR remains one throughout.

## Integration gates

Do not promote the wrapper or its ground/body keys. A general implementation requires
persistent patch identity independent of contact-column packing, unique point membership,
normal-compatible manifold grouping, arbitrary body pairs/contact counts, and explicit
split/merge/recontact rules. Every normal impulse must contribute to exactly one load
budget. Changing two point anchors into pooled friction can alter achievable torque and
is not automatically the original point-contact Coulomb law.

A production design must resolve wrench-capacity counterexamples, use paired common-point
wrenches or explicitly justify the model, preserve authored joint/drive response, and
check all supported backends. Static sticking, sliding, torsion, unequal masses/inertias,
normal-load transfer, contact permutation/chunking and disappearance/recontact need
physical tests. The favorable two-body drift result alone does not satisfy these gates.

## Unit velocity coefficient control

`/tmp/colibri_physx_patch_unit_velocity3600.*` is terminal PASS, source/geometry
checks unchanged; its 600-frame prefix is byte-identical to the separate control.
Velocity coefficient is 1.0; recovery impulse coefficient remains 0.36514837.
This changes the converged recovery target to p8 times the original target, so
it is not equivalent to applying unit multiplier to both terms.

- 10–60 s: 4.378665 µm, 0.0875733 µm/s; yaw −0.00381503 degrees.
- 50–60 s: 0.620337 µm, 0.0620337 µm/s.
- Hinge18.20742207 degrees; drive equation residual2.97e−9 rad/s.
- Two-body26.8895 FPS.
- Final128phase ledger: P error3.81e−11 Ns, L2.29e−12 Nms;
  independent-anchor defect3.77e−12 Nms, reconstructed work closure1.56e−14 J.

This control is worse than the source coefficient and is not accepted as
PhysX-quality stationarity. Neither result licenses pooled-anchor force capacity
for arbitrary manifolds. Next bounded study separates shared history from force
compression: retain original point cones/application points/metric solve and
only share a coherent history reference. Conservative complete-membership
matching must rebirth on new/rolling/split/merge points. A failed cheap stick
proposal is UNKNOWN, not a certified sliding/break event.

## Target-only compatibility control: bounded negative result

`shared_recovery_reference.py` independently reproduces corrected-native frozen
bias incompatibility: weighted RMS1.15886 µm/s, max5.71649 µm/s. No response rows
are removed: the chosen three-coordinate restriction applies only to numerical
recovery targets, even when the actual physical response has more modes.

`shared_recovery_live.py` uses prior completed-solve normal impulses only as
numerical projection weights. Every contact ingest invalidates those weights;
first-phase, zero-load, mu-zero and unresolved/ill-conditioned projections leave
original targets. Native current forces, point cones, metric solve, geometry,
ordering and history resets are unchanged. The FP32 condition*64eps<.01 and
backward-residual gates select whether to attempt this numerical correction;
failed gates preserve the original target, never clip a physical mode.

CPU and CUDA actual-container gates pass; CUDA FP32 target error versus offline
FP64 oracle7.16e−13 m/s. Impulse/history/normal/other rows byte-exact, target restore
exact, singular point-like field unchanged, graph replay passes.

`/tmp/colibri_native_shared_recovery600.*`: geometry/source checks PASS, but
5–10 s drift2.34460 µm/s versus corrected native2.44593 µm/s (only4.1% lower).
Final hinge18.2075255 degrees, drive residual9.01e−9 rad/s. Projection was active:
condition268.28, max target change4.84134 µm/s, cached load159.813 µNs.
Two-body19.5837 FPS. No 60-second extension or promotion is justified by this
bounded negative result. Target compatibility alone does not reproduce the
hybrid benefit; this experiment neither restores erased history nor changes
normal/friction ordering or pooled friction capacity.
