# Maximal tree Colibri findings

## Verified defects

1. The direct-owned hinge path skipped legacy joint preparation, but the maximal contact tree gathered anchor levers and axes from legacy constraint storage. A two-link contact response then violated the current five equality rows by 28.226 m/s. Gathering directly from joint frames and current body transforms fixes this; the regression tolerance is 1e-5 m/s.
2. Computing internal contact mobility as four root/body cross terms loses small hinge mobility through floating-point cancellation. Combining paired wrenches at their lowest common ancestor before evaluating the quadratic form preserves the same positive semidefinite operator. The synthetic test fails before and passes after this change.
3. Equality position-recovery velocity can produce enormous physical contact impulses in nearly blocked directions. On free Colibri15, effective normal masses reach roughly one million kilograms and impulses exceed 10000 Ns. Disabling equality bias alone keeps the first20 microsteps bounded. A CPU reference that removes the minimum-energy structural bias velocity only during contact sweeps, then restores it before integration, also stays bounded with physical friction and SOR1. The corresponding two-sphere hinge regression starts with0.19mm joint anchor error and launches at94.1rad/s before the split. It additionally checks that anchor repair remains active.

## Geometry and source fidelity

The apparent4.36mm axial gear/pin penetration is real volume intersection with a nonminimal axial separating direction, not a simple false positive: the cylinder spans the full3.75mm thin gear plate. USD shape materials, transforms and geometry were audited independently. All175 resolved material tuples match the literals. The source has55 authored contact offsets (52 at1mm, three at2mm) and no authored rest offset other than zero. Source joint friction is only an inactive articulation attribute because the stage contains no articulation root.

## References

- maximal_split_bias.py: successful CPU split reference
- maximal_impulses.py: detailed contact effective mass/impulse diagnostic
- test_maximal_contact_mobility.py: numerical, geometry and bias regressions

Whole-scene stability and usable performance remain pending; these individual fixes do not establish either.


## Follow-up: projected warmstart and independent contact screening

- General direct Schur routing of free stage15 passes180frames of body/joint checks at source offsets, mesh cylinders, outer2/internal10/iterations8/SOR1. However fresh collision screening rejects this at frame16: LargeGear01/00 penetrates1.714mm. It is not physical acceptance.
- Specialized maximal with split bias, projected warmstart, paired PSD mobility, joint-coordinate velocity and exact prepared offsets still explodesframe10. Cold lambda diagnostic passed60frames; relative mobility cutoff diagnostic passed60frames but was reverted because it removes real constrained mobility.
- Prepared offset regression: reconstructing world point from COM+r then subtracting COM changes nearblocked inverse mobility4.25e-8 into5.02e-5 at translated1000m. Using prepared r directly fixes this, preserving actual applied impulse. All5 maximal mobility tests pass; old helper A/B fails.
- Original width6height6 cube-cloth regression fails current projectedwarm/no-raw in both eager and CUDAgraph, cubez=-1.275m at120frames. Exact oldraw prepare plus projectedwarm kernel skipped passes. Prepare refresh stride is1. Scripts coupling_six_eager.py and /tmp/colibri_test_raw_grid_no_projected.py establish this without sharedfile reverts.
- Frozen Kamino zero-bias snapshot /tmp/colibri_zero_bias_frozen_audit.npz:820contacts,72bilateralrows. FP64 contact A*lambda+f agrees GPUreported velocity within.000149m/s; Schur re-elimination within.00107m/s. Cached response relativeerror1.96e-7. Actual normal complementarity residual.06324m/s. See kamino_frozen_reference.py and /tmp/colibri_zero_bias_schur_reference.npz.
