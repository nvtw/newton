# Pointwise patch friction reference

Retain every contact capacity c_i = mu_i * normal_impulse_i. The exact fixed-load
wrench support for virtual velocity (u, omega) is sum_i c_i times the norm of
tangent(u + omega cross p_i). This is an independent separating certificate,
not a solution of normal contact coupling.

## Compression counterexamples

Two supports at +/-0.1 m with capacities 0.9 and 0.1 Ns permit only 0.02 Nms
pure yaw. Equal redistribution to two anchors permits 0.10 Nms: five times the
physical capacity. Central pressure has the same problem. A uniform square
and two opposite anchors disagree on combined wrench limits even when their
pure translation/yaw limits match. Material averaging independently of normal
pressure also fails.

## Fast STICK / UNKNOWN

For an actually planar patch, A maps all point tangents to two force components
and normal-axis torque. Solve physical SPD mobility H*w = -v for the zero-bias
stopping wrench. D repeats each original c_i. Propose
f = D*A.T*(A*D*A.T)^-1*w with a 3x3 Cholesky factor.

Accept only after every original disk, resultant wrench and stopping velocity
passes. A coupled implementation must also recompute all original normal,
joint and drive equations, paired momentum and work using actual scatter.
Use equal/opposite impulses at the SAME spatial point; material history must
not silently create different force application points.

Failed factor or over-capacity proposal means UNKNOWN, not proven sliding.
The tests exhibit a feasible distribution rejected by weighted minimum norm.
Fall back to the general point solver; never reset history from UNKNOWN or
one finite-iteration clipping event. A virtual velocity separating requested
wrench from exact support certifies infeasibility only for frozen normal loads.

Zero-load and zero-friction points receive zero impulse and cannot independently
break the patch. Preserve original materials and pressure; no synthetic torsion,
rank cutoff, diagonal regularizer or discarded nonplanar physical modes.
Production proposals should use FP32 plus strict physical acceptance/fallback.
This independent NumPy reference uses offline FP64.

## Recovery targets

Projecting material recovery targets onto a planar rigid field changes
stabilization, not physical mobility or cones. It diagnoses incompatible targets
but cannot restore erased history. Document weights and unloaded-point behavior.
Biased work needs an explicit recovery budget, not blanket dissipation claims.

## Evidence

Run module local_studies.colibri.test_point_friction_wrench_reference with uv
and the Newton venv Python. Nine tests cover compression counterexamples,
feasible acceptance, safe UNKNOWN, materials, unequal-body momentum/work and
nonzero external force plus torque balance.
Report: /tmp/point_friction_wrench_reference.json.
The --assume-pooled-exact control fails with the fivefold yaw counterexample.
This is a fixed-load reference, not a native live solver or full coupled solve.
