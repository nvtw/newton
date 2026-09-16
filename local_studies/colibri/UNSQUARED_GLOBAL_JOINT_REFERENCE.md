> Correction: the first unsquared solve used the opposite correction sign. Its energy/contact conclusions are invalid; torque-defect magnitudes and nullspace findings remain informative. See REFORMED_JOINT_PIVOT_REFERENCE.md for corrected-sign results and explicitly changed geometry.

# Unsquared global joint correction reference

## Scope and result

CPU-only frozen diagnostic at the saved final relaxation state. The 188-row physical joint operator has exactly one pure-dual null direction and 187 independent directions. A 45-digit unsquared QR solve with only that exact gauge removed satisfies the original saved equations to 8.47e-18 and the physical impulse-response relation to 6.38e-32. It retains all five other tiny resolved modes.

This establishes a reliable linear response, not a physically accepted live correction. Maximum physical velocity increments are 0.0684382 m/s and 3.06836 rad/s; cancelling impulse coefficients reach 8.9228e6 in mixed SI units. Contact feasibility and physical work require separate checks before any acceptance.

## Formulation

Factor each nonzero physical inverse-mass block as W = L L^T. All 36 dynamic bodies have strictly positive blocks; two exactly zero static blocks contribute no response. Form B = J L and C = [B, sqrt(R)] with only the two existing nonzero drive-compliance columns. No artificial compliance is added.

Seek minimum-norm z subject to C z = rhs. An unsquared QR of the row-scaled C^T gives the primal solution; a second triangular substitution obtains physical impulse lambda. The body response is dv = L z_body. Verify independently:
- J dv + R lambda = rhs.
- dv = W J^T lambda.

No normal matrix is formed for this solve.

## Exact gauge certificate

Treat each saved binary float in [J^T; drive selector] as its exact rational value. Symbolic DomainMatrix nullspace has dimension one. Three independent modular ranks also return187. The nonzero null vector has exactly zero J^T response and exactly zero drive coefficients. Gauge lambda[15]=0 is permitted because that certified vector has a nonzero coefficient there.

The saved rounded RHS has a tiny exact-null inconsistency: n^T rhs = 8.47e-18 when max|n|=1. The omitted redundant equation retains this negligible mismatch. No finite singular-value threshold is used to classify or discard any other mode.

A separate 60-digit solve fixing lambda[20]=0 instead reproduces every physical velocity increment within 1.16e-12. Its original-equation residual is8.47e-18 and response residual8.03e-48. Thus the physical correction is stable under both increased precision and a changed certified gauge despite very large cancelling dual coefficients.

## Why ordinary FP64 was insufficient

Without the exact-null certificate, unsquared FP64 SVD/QR gave primal residuals around5e-15 while their reconstructed dual impulse response was inconsistent. Original constitutive residuals were0.123 and3.22e-5 respectively. A small primal residual alone is insufficient acceptance evidence. The high-precision original response check is essential.

## Reproduction

`uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.probe_joint_unsquared_mp`

Independent gauge/precision:
`uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.probe_joint_unsquared_mp --dps 60 --gauge 20 --output /tmp/colibri_joint_unsquared_mp60_gauge20`

Input: /tmp/colibri_global_joint_probe.npz.
Outputs: /tmp/colibri_joint_unsquared_mp.{json,npz}, /tmp/colibri_joint_unsquared_mp60_gauge20.{json,npz}.
Exact certificate: /tmp/colibri_joint_exact_null.json.
Rejected FP64 references: /tmp/colibri_unsquared_probe.json.

Each solve took about11.5 seconds for QR/substitutions, excluding setup. This is an offline reference, with no production performance claim or source-physics changes.
