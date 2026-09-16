# Actual PhysX stage-order parity control

The root agent's installed-SDK Nsight trace establishes the repeated position
iteration sequence: `solveWholeIslandTGS`, average, `solveStaticBlockTGS`,
propagate static velocity, propagate average velocity. For the one-joint,
static-ground fixture this places the joint before ground contacts.
The ordinary PhoenX color order was ground contacts followed by the joint.

`colored_d6_static_last.py` changes only the single-world color permutation:
last color first, followed by the earlier colors in their original order.
The experiment is bounded to the two-body scene. Final device-buffer readback
asserts that the moved color contains exactly joint 0, all earlier IDs are
contact columns, there is one physical joint, no mass splitting, and forward
sweep direction. Actual imported source path and color buffers are recorded.
The scalar joint response, original implicit drive law, common force point,
point-contact equations, SOR 1, 30 substeps, and 120 Hz collision updates stay
unchanged. This is stage-order parity, not a complete PhysX TGS implementation.

## Result

`/tmp/colibri_d6_static_last600.{json,npz,physical_audit.json,static_last.json}`:
600 frames, 30 warmup frames excluded, 570 measured frames.

- 113.6576 FPS; 8.79835 ms/frame.
- Base horizontal creep over 5–10 seconds: 29.8077 micrometers/second.
- Final hinge 23.09005 degrees; analytical equilibrium 18.20732 degrees.
- Original hard-row residual maximum 0.0361043; drive residual 0.0329822 rad/s.
- Joint impulse linear momentum defect exactly zero; angular defect at most
  1.83e-12 N m s from stored FP32 common-point rows.
- Source and final color ownership checks pass. Finite/geometry checks alone
  do not establish physical acceptance.

**Rejected:** stage ordering alone does not cure the convergence error.
The two-frame smoke completed simulation and ownership checks but failed its
source-unchanged check because another approved canonical edit happened during
execution; it is not counted as a complete pass. The subsequent 600-frame run
used unchanged source and completed all provenance assertions.

## Next isolated parity step

The live scalar implementation still uses original native row order
(linear locks, angular locks, drive), whereas PhysX GPU preparation sorts
ordinary force drives first, angular equalities next, linear equalities last.
`colored_d6_physx_order.py` adds only that row permutation on top of static-last.
Earlier frozen tests explored this ordering but do not replace a corrected
live test. The new runner has not yet been measured.

### Actual row-order live result

`/tmp/colibri_d6_physx_order600.*` completed all source and ownership gates,
including each contact column having a static endpoint. It retains original
rows and visits drive, angular locks, then linear locks.

- 113.7860 FPS; 8.78842 ms/frame.
- 5–10 second creep: 54.7068 micrometers/second (worse).
- Hinge: 15.83924 degrees. This is closer to the measured PhysX hinge angle,
  but remains wrong against the independent 18.20732-degree equilibrium.
- Hard-row residual 0.00942750; drive residual 0.0181706 rad/s.
- P defect zero; L defect at most 3.03e-13 N m s.

**Rejected.** A matching row order is not proof of convergence or accuracy.

`colored_d6_preconditioned.py` adds separate angular/linear hard-row
mass-metric Gram-Schmidt groups in FP32. It keeps original Jacobians, targets
and accumulated impulses, stores the transformed prepared rows separately,
and maps each hard impulse back using the transform transpose. Soft drives
remain unmixed. This prepared candidate awaits compilation and live gates.

### Prepared hard-row live result

`/tmp/colibri_d6_preconditioned600.*` completed the two-frame smoke and
600-frame live run, including source/ownership/row-span/target gates.
The original drive row remains the identity basis row; det(T)=1.
Final Jacobian reconstruction error is 3.24e-9 and target error 8.88e-11.

- 105.8859 FPS; 9.44413 ms/frame.
- 5–10 second creep: 52.4502 micrometers/second.
- Hinge: 15.64158 degrees, still incorrect against analytical equilibrium.
- Original hard-row residual 0.0262334; drive residual 0.0242724 rad/s.
- Original paired-impulse ledger: P defect zero; L defect at most
  3.61e-13 N m s. This ledger is not a separate full-trajectory momentum audit.

**Rejected.** Prepared row groups improve neither sufficient coupled
convergence nor physical stationarity at one sweep per substep. No canonical
solver change or accepted-speed claim results from this experiment. The next
independent contact experiment should test all normal rows before all original
point-friction rows, preserving every original cone rather than pooling the
normal load across artificial anchors.
