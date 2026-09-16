> Correction: the first unsquared solve used the opposite correction sign. Its energy/contact conclusions are invalid; torque-defect magnitudes and nullspace findings remain informative. See REFORMED_JOINT_PIVOT_REFERENCE.md for corrected-sign results and explicitly changed geometry.

# Torque defect in the global joint reference

The rejected angular-momentum change is fully explained by tiny torque imbalance in the saved FP32 linear joint wrenches, amplified by enormous cancelling oracle impulses.

## Independent row audit

For each row, evaluate total force and total world torque at the captured COM positions:
`f_total = sum_b J_linear[b]`
`t_total = sum_b (J_angular[b] + x_b cross J_linear[b])`.

- Every row's total force is exactly zero.
- Every angular-only row's total torque is exactly zero.
- Linear point rows have maximum torque coefficient 1.14087015e-8 m.
- Sum of lambda times these defects is [3.01281947e-5, 3.64067810e-4, 3.33172297e-5] N m s, reproducing the independent physical audit.
- Sum of absolute contributions is [0.0289428, 0.000691910, 0.00218592] N m s: substantial cancellation masks larger individual defects.

The largest individual terms are row18/joint3 with impulse -1.09548e6 and torque coefficient1.14087e-8, contributing -0.0124980 N m s along x; row13/joint2 nearly cancels it. Several wing-loop rows with impulses around1.63e6 contribute further millinewton-meter-second terms.

## Formulation and precision

Current direct_equality.py lines653-657 already shifts both anchor levers toward one shared midpoint. This audit does not identify a missing midpoint formulation. It stores the two lever cross products independently in FP32 spatial wrenches.

Independent least-squares reconstruction of each common lever from its three Cartesian point rows gives world application points agreeing within11.41nm at the saved body positions. This is compatible with FP32 lever rounding and strongly argues against an off-manifold anchor gap or a significant geometry-epoch mismatch as the observed cause.

Re-forming the same inferred common pivot and lever cross products in FP64 reduces total torque coefficient to6.94e-18. This is a diagnostic arithmetic control only; no solver coefficients were changed and no corrected trajectory was accepted. The snapshot has no embedded source hash, so its precise source epoch cannot be certified from the NPZ alone.

Native accumulated impulses at the same snapshot weight these defects to only [2.666e-14, 6.826e-13, 1.754e-13] N m s. The proposed global reference, with impulses approaching8.9e6, amplifies ordinary FP32 coefficient defects into an unacceptable physical angular-momentum change.

## Consequence

The high-precision solver accurately solves the saved operator, but that operator is insufficiently momentum-invariant for these extreme cancelling impulses. More solve precision alone does not fix this. A future global response must preserve torque balance in its wrench construction/storage and independently certify physical work and held-contact feasibility. No rank truncation, arbitrary compliance or live correction is justified by this result.

Evidence: /tmp/colibri_joint_torque_defect.json and /tmp/colibri_global_joint_physics_audit.json.
