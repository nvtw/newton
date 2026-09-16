# Contact softness and material-history equilibrium

CPU/source audit only. No coefficient, production, or live-candidate changes.

## Actual equations

`constraints/constraint_container.py:621` computes Box2D-style coefficients.
For default hertz=1e9 and damping ratio1, omega clamps to pi/h. Therefore

- `beta_n = pi / ((2+pi)*h)`;
- `m = pi*(2+pi) / (1+pi*(2+pi)) = .9417004`;
- `i = 1 / (1+pi*(2+pi)) = .05829955`.

At h=1/3600 s, beta_n=2199.6557/s. The cached constants in
`constraint_contact_cloth.py:173–187` are the same coefficients. The scalar
update (`constraint_block.py:576`, contact_projection.py:53 onwards) is

`delta_lambda_n = -m*eff_n*(Jv_n + beta_n*gap) - i*lambda_n`,

followed by unilateral projection. At a converged active row this means

`Jv_n + beta_n*gap + gamma_n*lambda_n = 0`,

with `gamma_n = i/(m*eff_n) = .0619088*A_nn`. This is real timestep-dependent
normal softness, not only an iteration accelerator. With splitting, the row's
configured response contains its copy mass scaling; this audit does not claim
that changing splitting leaves the soft operator invariant.

The default coefficients DO NOT soften tangent rows. For interior sticking,
`contact_projection.py:430–453` and `:650–705` target

`Jv_t + beta_t*material_displacement = 0`, with `beta_t=.08/h=288/s`.

Normal/tangent cross terms are included in the local update. Mu itself is
unchanged. The canonical `_friction_normal_lambda` still subtracts a local
bias-derived amount from the normal load; the earlier `total_normal_friction`
overlay bypasses that independently known coupled-contact error. Claims about
which overlay actually ran must follow the corrected import-provenance audit.

Preparation clamps material displacement to +/-1mm per tangent before bias;
this is a correction-speed cap, not a zero-bias deadband. Nothing in these
coefficients depends explicitly on anchor age. Age affects accumulated material
error, changing geometry, and reset/matching state.

## Static equilibrium and biased integration

A loaded static row cannot have both zero penetration and zero velocity when
gamma>0. For one isolated mass under g, lambda_n=m_body*g*h and response1/m_body,
the equilibrium penetration is `-g*h*h/pi^2`. For the source g=1 and h=1/3600,
this is **7.818nm**. The isolated-mass result is illustrative; a jointed multipoint
system must satisfy all coupled rows, load balance, and drive equations together.
At .002965m world height, this is also close to ordinary FP32 geometry scales.

A static tangential load strictly within the cone requires no tangential creep:
with consistent zero material drift the tangent impulse cancels the external
force and Jv_t=0. An existing drift e is targeted by Jv_t=-.08e/h. In a single
consistent sticking degree of freedom, integrating this gives e_new=.92e.
A fixed measurement offset gives a fixed displaced equilibrium, not perpetual
creep by itself. One nanometre of material error corresponds to .288 micrometres/s
of biased target velocity.

`solver_phoenx.py:3065–3110` runs force integration, biased solve, pose integration,
then unbiased relaxation. With final_substep only, relaxation occurs after the
last of30microsteps. Relaxation uses normal(m,i)=(1,0), zero positional biases,
and skips positive-gap rows (`constraint_contact_cloth.py:1460–1500`). It changes
velocities after pose integration; it does not undo already integrated material
or penetration correction. Thus final nearly-zero velocity cannot certify zero
integrated drift. However, a fully converged consistent static solution can
satisfy both phases: gamma*lambda balances normal penetration bias and tangent
material drift is zero. The coefficients alone do not prove inevitable creep.

Useful attribution remains: measure biased pre-integration Jv_t + bias_t, actual
integrated displacement, and final relaxed Jv_t with correct callback provenance.
Increasing sweeps cannot remove a contradictory material target or wrong
constitutive equation. No recommendation to change coefficients follows from
this audit.

## PhysX contact slop caveat

The actual composed stage `/tmp/colibri_physx_two_body_awake_plane1mm.usda`
contains no authored `physxRigidBody:contactSlopCoefficient` on either enabled
body. PhysX source defaults it to zero:

- `physx/source/lowlevel/api/include/PxvDynamics.h:156`;
- `omni/extensions/runtime/source/omni.physx/plugins/usdLoad/PhysicsBody.cpp:341`;
- `omni/schema/source/physxSchema/schema.usda:741`.

The importer reads the property at PhysicsBody.cpp:436 and forwards it through
usdInterface/UsdInterface.cpp:478. This supports disabled angular-lever slop in
this control; no actual runtime getter was recorded. Do not copy nonzero
component truncation as a momentum-preserving contact fix.

## Supplemental material-derivative check

`audit_material_velocity_jacobian.py` reconstructs the normalized14-value
reference as two effective material anchors and differentiates it. It uses
`/tmp/colibri_native_direct_normalized3600.native_state.npz`, now explicitly a
**partial candidate** because root found that the owned callback was installed
too late. It is not proof of corrected combined-history behavior.

For22loadedbasepoints, the body material lever differs from the native fresh
geometric midpoint lever by up to11.67um. At the saved actual angular velocities,
the tangent velocity discrepancy is only1.62e-10m/s. Using the common midpoint
of the two material anchors instead has a2.57e-9m/s derivative discrepancy.
A separate-anchor scatter is not proposed: separated equal/opposite impulses
would generally create a torque couple. The geometry audit is useful but does
not establish a dominant creep cause at this sampled final state.

Some saved27-row references imply307.85um tangential error while stored bias is
zero on positive-gap rows. Phase/reset/provenance must be checked on the corrected
run before treating this as an implemented-equation mismatch.

Artifacts: `/tmp/colibri_contact_coefficient_audit.json` and
`/tmp/colibri_material_velocity_jacobian.json`.
