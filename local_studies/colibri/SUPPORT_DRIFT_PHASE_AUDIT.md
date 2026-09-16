# Native support drift: current phase audit

## Intended scene

The source FrameGround is dynamic (rigidBodyEnabled=true, kinematicEnabled=false). Its enabled revolute joint connects to dynamic Frame, not the world. The incident Flower joint is explicitly disabled; Flower itself is kinematic. There is no fixed world anchor or base filtered-pair override. The literal builder defaults fix_base=False, creates a free base, and retains the kinematic Flower. Current captured ground contacts have static/dynamic friction0.5. The user requires friction to hold the base; an artificial anchor is not a solution.

## Capture acceptance

Local capture_support_phases.py adds separate wp.copy launches around actual warm/biased/relax group sweeps, averaging, force integration, and pose integration. Production arithmetic kernels are unchanged. The current public330-frame run passed all ten independent trajectory arrays byte-exactly against /tmp/colibri_prepare_integrated_cooperative330.npz; source/reference hashes remained unchanged. Snapshot /tmp/colibri_support_phases330.npz and .capture.json; trajectory /tmp/colibri_support_phases330_trajectory.json/.npz. Instrumented timings are not performance evidence.

This stores the last temporal substep of each phase, not a cumulative300-second impulse history. Source-order callback state distinguishes the phases; captured before/after body and copy arrays verify actual boundaries. Contact multipliers are cc.impulses[normal,tangent1,tangent2], while cc.lambdas stores contact axes/anchors. Physical lambda is not divided by copy count: copy response is scaled before averaging. Warm-start applies full current lambda; subsequent sweeps apply its difference. Joint accumulated impulses are physical generalized impulses, reset at substep entry and retained into relaxation.

## Direct observed creep mechanism

At final substep (dt1/3600s), the velocity integrated into FrameGround position is (0.218117,-1.000808,-0.221580)mm/s. Measured COM displacement is (0.060536,-0.277534,-0.061467)micrometers, consistent with this velocity and FP32 pose storage. Final relaxation comes afterward, so it cannot undo already integrated drift.

Loaded ground points21/23/24/25/26/27 have normal impulses (10.185,39.173,19.907,33.616,21.947,6.562)micro-Ns during the biased sweep, but every tangent impulse is exactly zero. Source friction_load=clamp(lambda_n+mass_coeff*effective_normal_mass*bias,0,lambda_n). The corresponding estimated recovery components are (40.707,42.535,43.272,40.832,40.458,38.164)micro-Ns, exceeding every accumulated normal impulse; the computed friction load is exactly zero. The fixed Nyquist-clamped mass coefficient is0.9417003989219666. This is a measured consequence of the load estimator at the current unconverged copy state, not proof that full biased normal impulse is physical friction load.

Point27 retains1.06696mm/s physical tangential slip before integration. In final relaxation, its solved copy slip is45.87micrometers/s, but averaging produces825.61micrometers/s. Tangent/normal impulse ratio is about0.192, strictly inside mu0.5. Thus the final physical state also violates the sticking condition despite an interior friction reaction. Extra final relaxation cannot retroactively remove earlier position integration, and one copy solve is not a globally converged support solve.

## Impulse and work ledger

Local audit_support_phases.py reconstructs all contact and bilateral endpoint impulses at the cached common world contact point. It uses actual before/after copy averages and each phase mass/inertia, preserving both dynamic endpoints and separating world/Flower reactions. Maximum per-body velocity reconstruction error is5.82e-7 in native velocity units; aggregate unexplained linear impulse is below3.37e-10Ns and angular impulse below4.37e-11Nms. Flower reaction is zero in this substep. Internal contact and joint reactions cancel to numerical roundoff. No internal momentum-leak evidence is found in this captured substep.

Work accounting uses total phase impulse dotted with the phase midpoint physical velocity, so component contributions sum exactly to net kinetic change. This is net phase work attribution, not a claim that each sequential contact update is dissipative. Work closure error is below3.84e-11J. Warm kinetic change+8.5121e-6J, biased-7.1043e-6J, relax-2.4933e-6J. Drive work is separated from contact and hard-joint work in /tmp/colibri_support_phases330.audit.json. World reactions can change total dynamic momentum; nonzero aggregate momentum is not itself an internal defect.

## Limits and next physical question

A loaded, nearly recovered support manifold must obtain friction load from a coherent physical normal solve, not blindly from biased reaction or independently estimated single-row recovery. The observed subtraction can erase support capacity in copies while coupled joints/other contacts undo the local normal recovery. Parent controlled extra biased iterations strongly reduce drift at unchanged30substeps/120Hz: roughly14mm at10s default,1.6865mm at two biased sweeps,0.8110mm at four. This identifies convergence sensitivity, not an accepted computational-budget solution.

No anchoring, damping, contact offset inflation, friction coefficient change, extra physical load, or production solver edit was made. The captured diagnosis is one current late-substep witness; longer cumulative reaction attribution and independent controlled physical-load separation remain subsequent work.


## Total actual normal impulse control (local, not promoted)

The earlier requirement to use only an unbiased load applies to an explicit split-channel method. This solver applies stabilization impulses to its real velocity. Using the total applied normal impulse is a consistent single-channel stabilized Coulomb convention; it does not claim to recover an unbiased load. The process-local `total_normal_friction.py` changes only that friction-cap helper, retaining normal recovery, PD/motor semantics, physical gaps and the 30x1/120 Hz schedule. Four actual contact energy, momentum, bias and normal-first tests pass. Removing negative recovery instead failed fresh penetration at0.15s and is rejected.

The total-normal60s run passes the existing geometry checks and unchanged-source guard, and its first330 q/qd/history samples match the short candidate byte for byte. Stationarity fails: horizontal base displacement9.493mm and rotation16.314degrees at60s, versus baseline88.716mm and about9.05degrees. Translation improves but yaw worsens. No300s extension or promotion. Artifacts `/tmp/colibri_total_normal3600.json/.npz/.candidate.json`.

The candidate phase capture `/tmp/colibri_total_normal_support_phases330.npz` independently matches all ten short-candidate trajectory arrays. The audit accepts an explicit `--normal-load total` flag and stores base component impulses. Point42 in the biased phase has lambda_n3.675350microNs and tangent magnitude only22.12% of its mu0.5 disk radius. Its copy tangent speed is4.44e-11m/s, but physical averaging reopens0.301798mm/s slip and changes normal velocity from+17.3216mm/s to-0.260055mm/s. After final relaxation the same point goes from4.409microm/s copy slip to349.355microm/s physical slip while remaining inside the disk (55.02%). This establishes a remaining coupling residual even with restored friction capacity.

At this captured biased phase the ground contributes base COM torque impulse(-7.9851e-7 in z)Nms; hard joints add+1.4463e-7 and the drive-typed row adds-2.6796e-9Nms. Final relaxation ground adds+1.4839e-7, hard joints-2.0326e-8 and drive-typed row+1.3159e-9Nms. These are actual paired impulses; full P/L ledger residuals remain below6.5e-10Ns and4.0e-11Nms, and phase work closure below1.7e-11J. This fixed-substep ledger identifies no internal momentum leak; it does not integrate the complete60s torque history or establish per-update tangent dissipation.

Reusing the existing support-star scheduler with total-normal friction fails at5.333s: Frame/GearedSpinner anchor2.492mm exceeds2mm. Base horizontal displacement then1.456mm (maximum1.535mm), rotation0.01510rad. Artifact `/tmp/colibri_total_normal_star330.json/.npz`; production source unchanged. Grouping is therefore still unpromoted; grouping all support rows alone is not an accepted correction for the mechanism.

## Full-manifold biased physical support reference

`frozen_total_normal_support.py` now uses the corrected-total-normal biased snapshot, all52ground witnesses, actual base/Frame reactions, five hard rows and the actual compliant row. It removes owned total impulses once and holds unowned impulses fixed; this is not a live skipped-row replay. Normal recovery and tangent recovery are retained. Positive-bias rows use the native hard normal coefficients, with the native2mm speculative-friction admission. Explicit contact PD or unequal static/dynamic coefficients are rejected by assertions rather than approximated.

The first draft omitted tangential recovery and used overlap softness for predictive rows; its rejected2.3e-6 residual result is superseded. The corrected source-law root converges in56function/44Jacobian evaluations (fixed120cap), original natural residual2.78e-17 and joint residual2.09e-14. `/tmp/colibri_total_normal_support_frozen.json/.npz` stores operators, impulses, initial/final velocity, physical work and external joint effects.

This is an accepted local equation root, **not an accepted stationarity intervention**. Base horizontal speed becomes21.1mm/s and yaw-0.190rad/s. Hard-joint recovery contributes+2.85175mJ, contact normal-27.899microJ, tangent-2.597microJ and finite drive-20.165microJ. Thus energy growth here comes from isolated joint recovery, not positive friction work. Captured hard targets include0.5372rad/s and0.09485m/s; native residuals before replacement include-0.5281rad/s and-0.09132m/s. The exact local correction perturbs external Frame joint velocities by as much as0.327rad/s. All eight loaded contacts are retained, one carries about50mNs, and loaded contact slip reaches13.78mm/s. Physical P/L errors are below9.3e-12Nms and work balance below5e-19J.

This rejects a base-plus-one-joint correction in isolation. A bounded next diagnostic would constrain correction velocities through all existing hard joint rows (homogeneous response preserving their current residual), retain compliant-drive response, and solve the full support manifold with that whole-assembly mobility. It must audit unowned contacts and numerical rank; it is not a license to pin the base or drop recovery. No production solver or GPU change accompanies this reference.

## Homogeneous whole-assembly joint response

`frozen_global_support.py` implements the next CPU control, preserving current hard-joint velocities instead of independently imposing their large recovery targets. All186hard rows are reconstructed at shared pivots with exact rational cross products; exact rational row reduction certifies rank182. Complete QR uses those independent rows without removing a small physical mode. The resulting physical velocity nullspace has34dimensions. The two soft rows retain their actual compliant response. Geometry change from native rounded wrenches is explicitly6.52e-9 maximum; normalized independent factor condition is5.76e9.

All52support rows remain in the contact solve, including native speculative and tangent recovery semantics. It converges in22function/17Jacobian evaluations to3.47e-18 contact residual. Hard-row velocity changes are1.35e-14 for the certified geometry and7.11e-10 for original rounded rows; soft constitutive changes are zero.

This still fails stationarity: base horizontal velocity grows from0.388 to0.955mm/s, yaw changes from-0.005732 to+0.009954rad/s, and vertical velocity rises to13.987mm/s. External contact maximum normal/tangent residuals improve only slightly, from0.1202196/0.0049192 to0.1201284/0.0048817m/s; individual external contact velocities change by up to7.54mm/s. A support-only converged block therefore does not establish a converged whole mechanism.

Independent actual-velocity momentum change plus ground reaction is below1.1e-10Ns and1.54e-11Nms. Work closes within4.49e-14J, with kinetic change+17.37microJ. The separate hard reaction reconstruction has8.18e-7 velocity error, failing a1e-8 reconstruction gate despite the accurate nullspace velocity. That conditioning issue is retained explicitly, not fixed by dropping modes. Artifact `/tmp/colibri_global_support_frozen.json/.npz`; physical_intervention_accepted=false. No live or production promotion.

## Progressive authored scene: base plus Frame

The isolated base-only original-friction control supplied by root settles to0.324micrometers net horizontal motion from10to60s. Adding only the authored Frame and its one joint reintroduces sustained drift. `staged_base_mechanism.py` retains all authored body/material/joint parameters, a free dynamic base, and disables the same15Flower shape flags as the base-only control. It preserves the original wrapper and records source hashes and joint metadata.

The FrameGround/Frame revolute drive has ke=kd=0.5729577951308232, target angle0.3490658503988659rad, with authored unbounded effort. There is no Crank or gear train in this stage. Both runs use30x1/120Hz and existing group4.

| Body2 friction law | Initial-to60s base translation | Rotation | 10to60s translation | 10to60s rotation |
|---|---:|---:|---:|---:|
| Original diagonal-subtracted normal load |49.147mm|15.20degrees|41.542mm|12.936degrees|
| Total actual normal impulse |2.383mm|0.1855degrees|1.990mm|0.1134degrees|

Both geometry gates and source guards pass. Neither is a stationarity fix; the latter still creeps39.8micrometers/s after10s. The relative joint angle is nearly stationary by10s: original0.437782rad and total-normal0.437759rad, versus the20degree target. This target error alone is not a constitutive violation: captured impulse/compliance is needed to distinguish finite-gain loading from iterative error. The nearly equal joint angles while support drift differs strongly isolate the support friction response as material.

Artifacts `/tmp/colibri_base_frame_{original,totalnormal}3600.{json,npz,stage.json,motion.json}` and total-normal `.candidate.json`. The original `.stage.json` was recovered from its printed metadata plus a verified source hash after the wrapper's output-path handling was fixed; simulation data was not rerun or altered. No solver changes or stationarity acceptance.

## Two-body coloring and no-averaging control

Both native330phase captures match their respective saved60s trajectory prefixes across all ten arrays. The actual graph has one joint at color0/partition0. Base52points occupy nine chunks, colors1…9, partitions0,0,0,1,1,1,1,2,2. Frame has fifteen ground candidates in three chunks, colors1,2,3/partition0. Copies are Base3/Frame1. An independent dynamic-endpoint disjointness check passes every color in both captures (`/tmp/colibri_base_frame_graph_audit.json`). This is proper coloring, with intentional copy coupling between partitions—not a same-color race.

The Frame ground manifold contains one actually loaded point:45.99microNs original,52.51microNs total-normal in the captured biased phase. The simplified mechanism therefore includes two support interfaces plus the joint. Treating it as only base support misses an actual reaction.

`audit_staged_joint_support.py` reports the actual equations and distinguishes skipped speculative relaxation rows. Total-normal final-relax drive residual changes from0.0006857rad/s on its final copy to0.0453257rad/s at the physical average. The biased counterpart is0.08406→0.13221rad/s. These copy values are after later contacts, not immediately after the joint callback; no claim that the joint's own block solve is incorrect follows.

The corrected local physical-head prototype, ordinary cap12, overflow batch1024, total-normal friction, unchanged authored two-body scene and30x1 budget runs60s. It has one copy per dynamic body and zero final overflow rows. Geometry passes, but base translation reaches10.983mm, with9.067mm accumulated from10to60s. Rotation is0.002113rad from initial and0.001293rad after10s. Actual ordinary coloring puts the joint at color9 and supports at colors0…8, unlike the original grouped ordering. Thus this is a deterministic physical-GS/no-averaging control, not an order-identical comparison. It demonstrates that removing averaging alone is insufficient to hold the base; it does not prove all GS orderings equivalent.

Artifacts `/tmp/colibri_base_frame_physical_head3600.{json,npz,ownership.json,motion.json,candidate.json}`. No production promotion; remaining diagnosis must include both support manifolds and finite-sweep joint/contact coupling.

## Full two-body joint and BOTH ground manifolds

`frozen_two_body_full.py` owns the hinge plus both ground interfaces at the actual two-body330snapshot. Biased phase includes all67points with actual negative/positive normal bias, normal softness, tangent recovery, speculative-friction policy and the actual compliant drive. Relaxation retains38positive-bias contact impulses exactly as native does and solves the remaining29points, using hard normal coefficients and zero numerical bias. No contact is silently discarded or reassigned to the base alone.

The biased reference fails its fixed120evaluation budget at0.00390765m/s natural residual. It is not an accepted root and does not justify a velocity intervention. The relaxation reference converges in17evaluations to9.43e-20 contact residual and2.78e-17 joint residual; P/L error is below9.2e-13Nms. Base still moves at0.605mm/s horizontal, yaw-0.0001536rad/s. Normal/tangent/hard-joint/drive correction work is respectively-90.470/-79.369/-3.447/+156.470nJ. This is a frozen discrete relaxation root, not proof of static equilibrium or a fix to already-integrated creep. Artifacts `/tmp/colibri_two_body_full_{biased,relax}.json/.npz`.

`static_two_body_feasibility.py` separately asks whether the current captured pose can rest under authored gravity and motor load with rigid circular Coulomb support. It includes every overlapping point on both bodies; separated witnesses cannot carry at-rest support. Numerical recovery and algorithmic normal softness are explicitly excluded from this physical force-feasibility diagnostic, not altered in the simulation. Inscribed64gon and circumscribed64gon LPs both report infeasibility. The outer polygon contains the true circular disks; therefore its negative result is stronger than an inner-polygon rejection, subject to numerical LP tolerances1e-10.

The outer-cone admissible drive-impulse interval is+[4.20208,10.19502]microNms. Actual zero-velocity authored drive impulse is-14.11585microNms, or-0.0508171Nm, well outside the interval(+0.0151275…+0.0367021Nm). The source drive equation was checked independently: captured dynamic mass matches dt*kd+dt²*ke with zero armature, and reference implies joint coordinate0.43775835rad, matching the quaternion measurement. The sign mismatch is not inferred solely from target tracking error.

This establishes that the current apparently settled relative pose is not a static force equilibrium under these captured physical contacts. It does **not** show that the authored sculpture must slide or that its parameters should change: a correctly converged evolution may settle to a different Frame angle/contact configuration while keeping the base still. No friction increase, anchor, damping, artificial compliance or iteration-budget escalation is introduced. Static artifact `/tmp/colibri_two_body_static_feasibility.json`.

## Full-frame two-body hinge torque balance

`audit_two_body_frame_torque.py` reads the trajectory-verified frame330 ring,
including60 warm/biased microsteps and the two real relaxation phases29/59.
Physical copy-mean impulse reconstruction error is at most1.60e-11 Ns/Nms.
About the actual Frame hinge axis, mean torques are biased drive+0.02471636 Nm,
relaxation drive change−0.00120822 Nm, all ground contributions−0.00842024 Nm,
and gravity−0.01512756 Nm. Hard joint axial reaction is below6e-16 Nm.
Thus actual drive balances ground plus gravity; no hidden axial recovery torque
or sign mapping explains the mismatch. Authored drive requires−0.05081587 Nm
at rest, or−0.05084670 Nm at actual speed. Final biased constitutive residual
is0.12329–0.13738 rad/s; after relaxation it remains0.04468–0.04533 rad/s.
Independent native replay solves the joint immediately to zero residual;
subsequent contacts and averaging reopen it. This identifies unresolved coupled
drive/support convergence, not an isolated joint-kernel defect.
Artifact `/tmp/colibri_two_body_frame_torque.json` retains every phase and all
normal/tangent/speculative contributions. This is an instantaneous per-phase
hinge impulse ledger, not a moving-origin angular-momentum transport proof.
