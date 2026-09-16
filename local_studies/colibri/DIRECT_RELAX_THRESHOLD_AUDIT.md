# Direct relaxation residual skip: native-kernel audit

The existing threshold is 1e-4 on the EQUILIBRATED right-hand side, not a
uniform 1e-4 m/s or rad/s physical joint tolerance. Hard rows use
rhs = -row_scale * Jv, with row_scale = 1/sqrt(A_ii) after compliance and
regularization. Thus a skipped isolated row may retain
abs(Jv) <= 1e-4 * sqrt(A_ii). Translational and rotational row scaling has
different physical units; both normalized residuals are energy-metric
quantities (sqrt(J) for the usual velocity/impulse coordinates).

Source history:
- b405193de introduced _DIRECT_SOLVE_RESIDUAL_TOLERANCE=1e-4 to avoid a
  panel traversal after small equilibrated inequality residuals.
- 1cfac6ab3 renamed it _DIRECT_RELAX_RESIDUAL_TOLERANCE and excluded primary
  biased dynamics from the gate, because small velocity changes can carry
  significant physical impulse. The final relaxation gate remains.
- Current solve_active is one aggregate flag: any above-threshold row enables
  the full system. A system whose EVERY scaled residual is below threshold
  skips its entire correction. The gate is not a per-row clamp.

## Actual CPU kernel counterexample

test_direct_relax_threshold.py uses native diagonal preparation, RHS assembly,
impulse accumulation and body scatter. A one-row factor solve is independently
evaluated by division by its actual prepared diagonal; no production threshold
or solver source is changed. Two dynamic bodies have masses 1 and 4 kg,
one hard translational row, initial relative velocity 7.8e-5 m/s.

Native row_scale = .89442718, scaled RHS = -6.97653159e-5.
The final relaxation flag remains zero and all 7.8e-5 m/s is left unchanged.
Forcing that same local scalar correction through the actual native scatter
reduces the residual to exactly zero in stored FP32 velocities.
Impulse = -6.23999949e-5 Ns; kinetic/work change = -2.43360074e-9 J.
Stored momentum roundoff is -1.4552e-11 Ns, within the explicitly computed
FP32 storage bound 2.9755e-10 Ns. The mathematical equal/opposite impulse is
paired; no angular torque occurs for this collinear COM fixture.

Primary biased solving activates and removes the same residual. Scaling both
masses by 1e-3, 1 and 1e3 changes the activation decision for the same physical
velocity: this is a scaled numerical threshold, not a fixed physical criterion.
For a coupled near-singular system, the per-row scaled threshold alone is not
an energy or physical response certificate.

## Reproduction and scope

CPU-only module:
  local_studies.colibri.test_direct_relax_threshold

--require-correction exits 1 with "Nonzero hard-row residual skipped".
--require-correction --force-correction exits 0.
Reports/logs:
  /tmp/direct_relax_threshold_fail_before.{json,log}
  /tmp/direct_relax_threshold_pass_after.{json,log}

The force flag is a LOCAL fixture control, not a production fix or full native
factorization test. This proves a real skipped physical residual and that a
representable correction exists; it does not prove this threshold causes
Colibri creep. The parent's isolated live no-skip result improves joint
residual substantially but does not eliminate creep.
