# Joint-constrained contact mobility reference

## Construction

Use the geometry-consistent FP64 joint operator and its four exactly certified pure-dual dependencies. The physical 8.92836e-9 factor mode is retained. Factor W=L L^T and form C=[J L, sqrt(R)] using only the two existing drive-compliance columns.

A45-digit full QR of the independent C^T yields orthogonal complement Z. Its dimension is34. Let G=L Z_body. Then the physical joint-constrained response is P=G G^T. This avoids cancellation from W-WJ^T A^-1 JW and is symmetric positive semidefinite by construction. No eigenvalue floor, clipping or singular-value threshold is applied.

Drive impulse changes are recovered from the complement's drive rows:
delta_lambda_drive = Z_drive G^T f / sqrt(R).
They enforce J dv + R delta_lambda =0, retaining the actual drive compliance rather than adding damping.

The high-precision complement residual JG+sqrt(R)Z_drive is1.13e-43. Saved FP64 factors give a maximum3.09e-10 joint residual under32 unscaled random unit-force vectors, whose resulting quadratic works exceed6e6. The physical virtual-work identity
f dot P f = dv dot M dv + sum(R delta_lambda_drive^2)
holds to2.12e-15 relative error.

## Tiny actual frictionless subset

Select all four eligible unbiased contacts with source friction mu=0 between saved bodies9 and10, points1420/1428/1429/1442. Contact witnesses/normals are unchanged. Other contact impulses are held fixed. Start from the corrected joint-only state, subtract the selected old impulses through P, and enumerate the16 normal active sets of H=C_contact P C_contact^T.

One set passes nonnegativity and complementarity without rank truncation:
- Impulses [0.0003380408143,0,0,0] N s.
- Normal velocities [0,0.000747041,0.000746226,0.000749737] m/s.
- Joint-response residual7.15e-18.
- Total deltaP below3.1e-20 N s; deltaL below4.7e-21 N m s.
- DeltaKE=-4.22765e-7 J.
- Contact midpoint work=-4.25504e-7 J.
- Existing compliant drive midpoint work=+2.73895e-9 J.
- Work balance error6.18e-18 J.

The four-row mobility is exactly symmetric. Its two resolved eigenvalues are0.0009902153 and224.541692; a floating eigensolver reports near-zero values +/-3e-14. PSD is established by its Gram construction, not by clipping those values.

This validates a small coupled response and its physical accounting. It does not establish feasibility of every other held contact or provide a production global solver. The normal LCP is an unbiased rigid unilateral reference; it is not a claim of byte-equivalence to the native finite-iteration contact update.

## Reproduction and artifacts

Generate geometry with reform_joint_pivots.py. Run:
python -m local_studies.colibri.probe_joint_unsquared_mp --input /tmp/colibri_joint_reformed_input.npz --output /tmp/colibri_joint_constrained --save-mobility
python -m local_studies.colibri.audit_joint_constrained_mobility

Use the repository uv/interpreter invocation. All calculations are CPU-only.

Artifacts:
- /tmp/colibri_joint_constrained_mobility.npz: G, drive slack basis, active body IDs.
- /tmp/colibri_joint_constrained.json: high-precision complement certification.
- /tmp/colibri_constrained_contact_mobility.{npz,json}: selected contact mobility and physical audit.

No canonical solver changes.
