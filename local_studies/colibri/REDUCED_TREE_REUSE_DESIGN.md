# Reusing reduced-tree machinery for the off-manifold operator

## Concrete reusable components

In newton/_src/solvers/phoenx/articulations/reduced.py:

- ReducedArticulationSystem.__init__ (3556 onward) builds stable tree depth ranges, child lists, body-to-joint mappings, and warp/depth execution eligibility. These topology schedules can be reused for an explicitly chosen spanning forest.
- _initialize_reduced_factor_kernel (470) initializes common-articulation-origin spatial inertias and joint motion subspaces.
- _factor_reduced_depth_kernel (540) and _make_factor_reduced_warp_kernel (963) implement deterministic child-inertia accumulation and articulated-body elimination.
- _compute_reduced_impulse_response_basis (621), its warp kernel (774), and _solve_articulated_system_kernel (2278) supply forward/backward response patterns and scratch organization.
- _shuffle_reduced_float/_shuffle_reduced_spatial and group synchronization helpers provide existing cooperative traversal mechanisms.

These are reusable algorithms and layouts, not drop-in numerically identical kernels. Current factors/spatial arrays are FP32, and the scalar factor contains a1e-20 denominator floor. The present retained tiny-mode reference requires explicit FP64 data and rejection/validation of invalid factors rather than silently inheriting that floor.

## Preserve actual geometry

Do not call the usual reduced local-kinematics reconstruction to replace the maximal body's captured pose: it reconstructs configurations from authored joint transforms and coordinates. Do not assume its authored motion subspace equals the off-manifold maximal row nullspace.

For the torque-balanced shared-pivot linear rows, a common-origin spatial representation is nevertheless valid. A rigid common twist satisfies every pair row because both total force and total torque vanish. Each tree's relative motion subspace can therefore be derived from its ACTUAL row nullspace: for these revolute rows, use the actual two angular-row axes and actual common pivot. This gives the same common-origin parent-plus-relative-twist structure used by the existing ABA machinery without requiring aligned authored anchors.

The current local prototype uses generic child-row QR to establish that equivalence. A future specialized initialization can produce actual-row-derived joint_s and transformed inertias, then use equivalent FP64 articulated-body elimination. The46x46 dense mass factor is a diagnostic intermediate, not an unavoidable production operation. Sparse ABA factors can generate the needed whitened loop rows/response basis; that square-root route must retain the unsquared small modes rather than form a cancellation-prone loop normal matrix.

## Affine biased RHS

For tree equations A v_child+B v_parent=t, build v=c+N u:
- Homogeneous transfer/nullspace N is unchanged by t.
- Propagate a particular velocity c using the same child QR and t.
- Project initial velocity into the affine tree space:
  v_tree=c+N M_reduced^-1 N^T M(v_initial-c).
- Solve the remaining loop/drive residual at v_tree with the same geometry/mass/compliance factor.

For the actual native bias convention, hard-row t=-bias; dynamic-row residual is Jv+bias+R accumulated-reference. Existing drive compliance and accumulated impulse accounting remain physical.

A bounded implementation is available through probe_tree_joint_mobility.py --biased. It uses saved actual row biases at the captured post-relax pose. Those biases were not applied during the captured relaxation, so this is a consistency stress test, not a reported live solver failure.

Result: tree affine residual5.83e-16, but full loop bias RHS has exact-null incompatibility up to4.81e-5. Solving independent rows leaves0.0061425 residual on the complete original equations and injects1.6247J. It is explicitly REJECTED. Dropping redundant equations is only valid when their RHS dependencies are also satisfied. No bias projection, extra compliance or damping was introduced.

## What can safely be cached

| Quantity | Valid lifetime |
|---|---|
| Spanning forest, parent/depth/child schedules, row ownership, allocation sizes | Until topology, enabled constraints, joint modes/DOF count, or prescribed/dynamic ownership changes |
| Certified loop row selection | Reuse only while rank/coverage remains valid; factor/all-row checks must reject a singular configuration rather than truncate it |
| Body-local physical mass and inertia | Until physical property updates |
| Source local anchor/axis definitions | Until corresponding model edits |
| Pose-dependent actual joint wrenches, shared pivots, motion subspaces, transformed inertias | Rebuild whenever geometry/orientation changes |
| Tree mass factor, whitened loop rows and QR/complement | Rebuild when J, physical W or drive R changes; velocities/contact impulses/bias alone do not invalidate them |
| Bias, target/reference, affine particular solution and accumulated-impulse RHS | Refresh at each phase according to current geometry, controls and substep lifetime |
| Contact witnesses, contact Jacobians and contact response rows | Refresh when contact geometry changes; this does not itself invalidate the joint factor |
| Existing factor during several contact/constraint sweeps at one fixed pose | Safe to reuse; only RHS/impulses change |

No general reuse across moving temporal substeps is justified: positions/orientations change. Position integration also invalidates a preceding biased factor before a later relaxed solve. Unchanged dt/control gains may keep R constant, but that does not keep J/W constant.

A deliberately scoped final-unbiased-relaxation strategy would build this operator only for the actual final relaxation phases (two120Hz updates per60Hz frame), retaining the native biased substeps. That is a scope choice preserving the original intermediate path, not freezing a factor across changed geometry. It remains conditional on coupled-contact convergence and live physical validation, neither of which is established yet.

## Status

Vendor48x14 setup cost is measured separately by the parent. No GPU or production implementation was added here. The source-based reuse design and affine prototype are local diagnostics. The unchanged unbiased tree formulation remains validated; the exact global biased recovery extension is blocked by demonstrated RHS inconsistency.

Artifact: /tmp/colibri_joint_tree_biased_mobility.json.
