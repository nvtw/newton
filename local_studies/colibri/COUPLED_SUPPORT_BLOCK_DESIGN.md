# Coupled support component: frozen reference proposal

This proposal does not change canonical runtime or the accepted30x1/120Hz configuration.

## Component and physical response

Select every witness in a static support manifold and all structural bilateral rows incident to its dynamic body. Include both endpoints of every selected joint: an attached Frame receives its actual reaction. Resolve overlapping components deterministically, with the same bounded ownership policy as the support-star study. Do not recursively absorb an entire articulated mechanism and do not drop contacts based on current impulse.

For physical inverse mass W, structural bilateral Jacobian Jb, free velocity u and bilateral target b, define B=Jb W Jb^T. Use a rank-revealing solve/pseudoinverse with explicit consistency checks; do not add fake compliance. Then

- vbar = u - W Jb^T B† (Jb u + b)
- P = W - W Jb^T B† Jb W
- contact mobility A = Jc P Jc^T
- contact free residual q = Jc vbar + contact_bias

After solving contact impulses lambda, recover beta = -B†(Jb u + b + Jb W Jc^T lambda) and v = u + W(Jb^T beta + Jc^T lambda). This keeps equal and opposite joint reactions and uses actual world-space inertia and common impulse points.

## Contact law

Reuse the multi-interface non-associated Coulomb formulation: normal complementarity stays independent of the friction-radius multiplier. A single associated cone-QP changes the normal law and is not a substitute. Retain all normal and tangent rows, with actual signed-gap speculative biases. Inactive rows may have zero solution; they remain checked for complementarity.

The existing CPU natural-map solver is a candidate, not an acceptance oracle. Independent normal, cone and tangential stick/slip KKT checks are required. The enumerated-mode reference currently scales only to tiny systems (roughly9contacts);52support witnesses are strongly rank-deficient and cannot be brute-force enumerated. Use a bounded solve with a clear failure result, never arbitrary contact pruning.

## Initial bounded scope

Start with a frozen unbiased relaxation component and structural bilateral constraints. Capture velocities before the local impulse update; reconstruct/remove warm-start contributions explicitly if the chosen capture already contains them. Compare against the same initial state and witnesses. Soft rows and finite drives require compliant/box-constrained equations; reject those inputs initially rather than treating them as hard equalities or dropping their effects silently.

Check bilateral residual, all normal complementarity residuals, all friction KKT residuals, cone feasibility, total linear/angular momentum including static reactions, and kinetic work/energy. Also verify translated/rotated and body-order-swapped fixtures. Reaction impulses may be nonunique under redundant constraints; velocity and work are the invariant comparison targets.

## Scheduling remains a separate question

The frozen reference uses physical W. A superrow inside the existing mass-splitting dispatcher would instead require the count-scaled copy response and subsequent averaging. Mixing physical W with copied velocities without a derivation is invalid. First establish whether the physical coupled block resolves the frozen residual. Only then design a supported live scheduling scheme and prove its copy-count, ownership and momentum semantics.

## Accepted frozen physical reference

`capture_support_relax.py` records the actual last pre-relax state of public330frames with byte-identical pose/velocity histories. `coupled_support_reference.py` retains all52support witnesses:27native-active relaxation points are variables;25positive-gap points retain their exact native frozen impulses. Five structural rows are eliminated via the physical mass nullspace. The sixth joint row is an actual unbounded compliant drive (captured mass/reference/old impulse), retained through its positive compliance, not rigidified or dropped.

The CPU Coulomb solve converged in98evaluations. Normal residual5.19e-9m/s, tangent residual5.31e-15m/s, cone violation7.07e-16Ns, structural joint residual6.68e-17 and drive equation residual0. Reaction reconstruction error7.20e-16; physical momentum minus static reaction is below2.1e-13. Contact work−4.446e-7J, hard-joint work−1.138e-6J, drive work−2.876e-8J sum to kinetic change−1.611e-6J within3.6e-21J. Nonzero drive work is explicit.

On the same witnesses, native final relaxation gives normal residual1.575e-3m/s, tangent natural-map residual1.796e-4m/s and joint residual2.700e-3. The coupled solution releases separating contacts rather than forcing their tangential velocity to zero. The native full sweep also updates outside-component rows; this is local frozen feasibility evidence, not a live replacement or global equivalence claim.

Artifacts: `/tmp/colibri_support_relax330.npz`, `/tmp/colibri_coupled_support_reference.{json,npz}`, `/tmp/colibri_coupled_support_comparison.json`.


## Copy-scaled frozen bridge: accepted local equations, rejected physical replacement

`support_bridge_counts.py` reconstructs every captured body's original copy count exactly before creating the support-star ownership. In this particular frozen state the support has one copy and Frame has 41 (not the 43 copies in the earlier trajectory). The component owns partition45, support slot0 and Frame slot41.

`run_support_copy_reference.py` solves the same equations with Wcopy=diag(copy_count)Wphysical. Actual compliance and drive reference stay unchanged. Initial least squares stopped at normal residual7.55e-7m/s; scaling the numerical residual and Jacobian by1e4, without changing physical equations or acceptance thresholds, lets the zero seed converge in231 evaluations (after38 unsuccessful warm-seed evaluations). Accepted normal residual4.56e-13m/s, tangent1.21e-16m/s, hard joint1.36e-16, reaction reconstruction4.13e-16. This is an offline reference, not a production performance result.

`support_copy_bridge.py` broadcasts the actual frozen input to all copies, replaces only the two owned copy velocities with that solution, and calls canonical CPU average/writeback. No native constraint sweep occurs before or after this component, so owned rows are not double-solved; outside rows are held fixed. CUDA averaging remains pending; no live integration is made.

The actual canonical physical velocity agrees with the independent averaged result within2.44e-9. All six physical momentum components minus static reaction are below2.49e-10 in SI momentum units. Copy kinetic change +6.8791e-7J includes actual drive work +7.6307e-7J, contact work -8.2074e-8J and hard-joint work +6.9183e-9J. Averaging dissipates7.6044e-7J, leaving physical kinetic change -7.2534e-8J; the independent work balance closes within3.2e-22J. Thus energy increase in the copied solve is legitimate motor work, not an energy failure.

However, physical averaging reopens the hard joint and compliant drive equations. The five hard-row residuals are:

- Anchor x/y/z velocities: [-1.30782,-0.618669,-2.54078] mm/s.
- Two constrained angular velocities: [-0.0196623,-0.00379889] rad/s.
- Compliant angular drive equation:0.0968254rad/s.

The support contact normal/tangent residuals remain4.78e-11/1.29e-13m/s because that body has one copy. This is a concrete scheduling obstruction: exact local copied equations plus valid momentum/work do not imply a valid physical coupled block after averaging. The candidate is rejected as a live replacement.

Artifacts: `/tmp/colibri_support_copy_reference_scaled_residual.{json,npz}`, `/tmp/colibri_support_copy_bridge_cpu.{json,npz}`, `/tmp/colibri_support_bridge_counts.npz`.

## Proposed physical post-average phase (CPU design only)

A physical response can be applied coherently to all copies, but it changes the global splitting schedule and needs an external-row audit:

1. Build deterministic component ownership before the relaxation phase. Union components sharing any dynamic body, or process them in a deterministic conflict-free sequence. Record owned contact columns and joint IDs explicitly. Preserve all witness rows and finite drive/limit/friction semantics; unsupported components stay entirely on the native route.
2. During that relaxation phase, suppress the owned rows in both ordinary callbacks and any separate joint-bound resolver. Leave the biased solve and geometry preparation unchanged. Carry current accumulated impulses as the reference unknowns; do not apply an extra warm start merely because ownership changed.
3. Solve all unowned copied rows and average normally. At that exact physical state, recompute the owned physical residual, including the existing accumulated impulses, actual compliant drive and unchanged speculative-row policy. Solve the component with physical W. The component is executed once, replacing its skipped native relaxation update.
4. Apply equal/opposite component impulse increments to the physical bodies with physical W, then broadcast the resulting physical velocities identically to *every* copy of each participating body. Do not inject a physical-W increment into a single copy and do not average the same impulse a second time. Update each owned accumulated impulse once. Position/orientation and contact history stay unchanged.
5. Audit every unowned incident row using the new physical velocities and its actual limits/compliance/friction law. In particular, Frame's many other joints and contacts may lose convergence. Record each pre/post residual, impulse-cone feasibility and work contribution, not merely the selected support manifold. An external residual worsening cannot be hidden by the selected block's successful KKT check.

This schedule preserves impulse accounting and physical momentum algebraically; it does not establish global convergence. Adding a component last can reopen neighboring constraints just as averaging reopened the local joint. The next bounded proof must freeze the actual post-unowned-average input, measure all incident-row residuals and explicit drive/contact work, and reject the local schedule if the external coupling requires a larger component. No extra iterations, body pinning or live shortcut is authorized by this proposal.


## Physical post-average replacement and external audit

`post_average_support_reference.py` reconstructs captured native joint/contact impulse increments, removes the owned joint0/support increments once, and replaces them with the physical coupled result. Removal plus reapplication reproduces the original velocity exactly in FP64; reconstructing the complete native sweep from recorded FP32 impulse differences is accurate within8.34e-7 in velocity components. Unowned impulses retain their exact captured values, including their dependence on the native row ordering. This is therefore an accounting diagnostic, **not** a replay of skipped owned rows.

The replacement's own KKT checks pass (normal6.31e-16m/s, tangent2.65e-15m/s, hard joint8.88e-17). Relative to the native output, net replacement kinetic change is-1.42998e-6J: contact work-1.12446e-6J, hard-joint work-6.02365e-7J, drive work+2.96845e-7J. Work balance closes within3.18e-21J and six momentum components minus the changed static reaction within1.95e-13. Owned native increments are removed before replacement; no impulse is applied twice.

The external audit includes61 joint rows and944 contact witnesses touching either participating body. Of the contacts,15 are native-active relaxation rows and929 are fixed speculative rows whose velocities are reported but whose impulses remain unchanged. All active friction coefficients match static/dynamic and there are no bounded drives in this snapshot. Some neighboring residuals worsen: angular rows by about0.00908rad/s and contact186 closing error7.675→9.026mm/s. Such local worsening is expected in block Gauss–Seidel and is not by itself a convergence rejection. It motivates the following full frozen sweep experiment.

## Four-cycle physical block-GS experiment

`physical_support_block_gs.py` starts from the actual captured **pre-relax** state and executes either ordinary physical blocks or one support coupled block plus the remaining physical blocks. Ownership is now a true skip: joint0 and all support point updates are absent from the ordinary sweep when the coupled block is selected. Each cycle runs the support block first, then remaining joint IDs, then contact columns/points. All52support points remain represented;25 keep their native fixed speculative impulses. Across the whole scene there are2543points,70native-active relaxation points,188joint rows and no soft contact PD/bounded drive rows. Positive-bias speculative rows stay frozen exactly as in native relaxation.

The CPU implementation uses the native physical equations in FP64: simultaneous joint block including actual compliant drives, normal-first unilateral contact, then the exact2D metric friction disk. This is not a native kernel bitwise-equivalence assertion. It uses physical masses rather than copy-scaled masses, a deterministic serial order rather than the colored copied schedule, and extra solve cycles at one fixed pose. **No extra temporal steps or live solver changes are made.**

| Residual maximum | Native post-relax, all rows | Plain physical4cycles | Coupled physical4cycles |
|---|---:|---:|---:|
| Normal (mm/s) |37.538|13.125|13.116|
| Tangent map (mm/s) |15.490|see artifact|3.163|
| Hard anchor (mm/s) |14.928|8.965|8.961|
| Hard angular (rad/s) |1.38166|0.52687|0.52825|
| Drive equation (rad/s) |0.49076|0.40744|0.40707|

The earlier0.196rad/s native value covered **only unowned rows incident to Frame/base**. The1.38166rad/s value above covers **all world joint rows at the same captured native post-relax state**. Comparing0.196 directly with the global0.53 reference value is invalid. Native one-sweep and physical four-cycle results also have different work budgets and schedules; this table does not claim a like-cost improvement.

Per cycle, plain performs38joint blocks plus70point blocks. Coupled performs37joint blocks plus43point blocks plus one nonlinear support solve (27variable contacts,5hardjoint rows and1compliantdrive). Its four support solves require98,47,8,9evaluations. Coupled global maxima and L2 norms decrease substantially from the initial state but remain close to the cheaper plain result after4cycles. No promotion is justified by this small global benefit.

All per-cycle residual categories (max/L2/count and worst5row IDs), cumulative work, six momentum checks and evaluation counts are saved in `/tmp/colibri_physical_block_gs_{plain,coupled}.json`. Reporting-only repeats produce byte-identical final state/impulse arrays. Coupled momentum minus external reaction stays below1.68e-12 and work-balance error below3.3e-19J. The following work is incremental per cycle, not cumulative:

| Cycle | Contact work (J) | Hard-joint work (J) | Drive work (J) |
|---|---:|---:|---:|
|1|-8.63952778e-06|-1.04043561e-05|5.46830367e-06|
|2|-3.05250999e-06|-2.59678745e-06|2.86814682e-06|
|3|-2.42603764e-06|-1.55631335e-06|3.06668789e-06|
|4|-2.38666433e-06|-1.29019852e-06|3.43988221e-06|

## Dominant remaining coupling structure

- Joint33, TailMount31.494g / TailPinion3.187g (mass ratio9.88), retains0.52825rad/s constrained angular residual.
- Joint1, Frame552.363g / Crank9.880g (ratio55.90), dominates8.961mm/s anchor and0.40707rad/s compliant-drive residuals; its other hard angular row is0.26927rad/s.
- Normal points2025/2005, Tail_Feather_A1.514g / TailRack3.754g (ratio2.48), retain13.116/11.473mm/s residuals.
- Tangent points1148/1127, HummerBody39.956g / TailPinion3.187g (ratio12.54), retain3.163/1.878mm/s natural-map residuals.

Thus the support/base component does not include the dominant tail gear/contact loop or the Frame–Crank drive coupling. A larger connected component preconditioner should be selected using these actual joint/contact interfaces, with explicit bounded-size/rank policies and the same physical Coulomb/drive equations. No further iteration increase was performed merely to reduce the reported residual.


## Bounded tail component and global four-cycle result

`tail_component_size.py` selects the diagnosed tail mount/feather/pinion/rack core, retains every incident joint and contact witness, and includes both reaction endpoints. The resulting component has10bodies (including HummerBody and CamFollower boundary bodies),60physical DoF,7joints/35hard rows of rank35, and25nullspace DoF. All916contact witnesses are retained;902are fixed speculative rows under native relaxation,14are variables (42Coulomb unknowns). There are no soft or bounded drive rows inside this component; both actual world drives remain active in the external ordinary sweep. The component has7external joints and8external contact columns.

The mass-weighted joint factor condition is410.7, physical mobility condition8.11e5, and retained contact factor condition1.09e10 across19numerical directions. All directions remain in the factorized nullspace response; there is no artificial regularization or rank cutoff beyond a documented machine-precision numerical rank diagnostic. `tail_component_reference.py` passes the unchanged1e-8physical tolerance in113warm evaluations without extra residual rescaling: normal1.73e-9m/s, tangent1.13e-16m/s, joint1.20e-15. Contact work-1.0197e-5J and joint work-1.8701e-5J sum to kinetic change-2.8899e-5J; six momentum components stay below4.92e-12.

The tail has mixed per-point friction0/.5 among the active rows. Local `coulomb_semismooth.py` now broadcasts scalar or per-point coefficients consistently into the natural map and derivative. The prior scalar98-evaluation reference remains byte-identical. Two analytical tests check scalar/vector identity and independent frictionless/sliding/separating contacts, including normal complementarity and dissipation.

A local tail solve reopens external shoulder angular constraints and a HummerBody contact. This spillover is assessed through the same true-skip full physical schedule rather than treating individual worsening as rejection. In `physical_support_block_gs.py`, the tail block replaces joints27–33 and all owned contact updates exactly once; the remaining31joint and56active point blocks run normally. The fixed pose, source masses, actual drive equations and speculative-row policy are unchanged. Extra work is explicit: one nonlinear42-unknown solve per cycle, with113/99/70/48evaluations, instead of7ordinary joint and14point updates.

| Cycle | Normal max (mm/s) | Tangent max (mm/s) | Hard anchor max (mm/s) | Hard angular max (rad/s) | Drive max (rad/s) |
|---|---:|---:|---:|---:|---:|
|1|13.3156|4.5944|11.1268|0.223300|0.338077|
|2|4.5421|0.8867|7.4020|0.245016|0.339834|
|3|3.1808|1.0837|8.6592|0.261558|0.389631|
|4|2.2341|0.8707|8.9682|0.270107|0.407328|

At four cycles, versus plain physical GS, tail coupling reduces global normal maximum13.125→2.234mm/s (83%), tangent3.165→0.871mm/s (72%), and hard angular0.52687→0.27011rad/s (49%). Anchor and drive errors remain essentially unchanged, now dominated by Frame–Crank. The angular maximum rises after the first tail cycle because the independent Frame–Crank mode grows; this is not claimed as monotone convergence of every category.

Across all cycles, work balance closes within6.7e-18J and six momentum components minus external reactions within1.08e-11. The additional nonlinear work has no measured GPU implementation or speed claim. These are promising frozen convergence results, not live stability evidence or a promoted production policy. Artifacts: `/tmp/colibri_tail_component_{size,reference}.json`, `/tmp/colibri_tail_component_operator.npz`, and `/tmp/colibri_physical_block_gs_tail.{json,npz}`.

Tail schedule incremental physical work:

| Cycle | Contact work (J) | Hard-joint work (J) | Drive work (J) |
|---|---:|---:|---:|
|1|-1.68696031e-05|-3.31594656e-05|5.95329495e-06|
|2|-6.87678829e-06|-8.20671711e-06|3.08076520e-06|
|3|-4.22419858e-06|-2.66748206e-06|3.02035736e-06|
|4|-3.19046970e-06|-1.57722434e-06|3.42789044e-06|


## Frame–Crank size and bounded combined-cycle obstruction

The second component uses the Frame/Crank core and every incident joint/contact, retaining all reaction endpoints:17bodies/102DoF,13joints with65hard rows and2actual compliant drives,993witnesses (16active,977native-fixed speculative). Hard rank65 leaves37DoF; joint-factor condition949 and contact-factor condition4.55e10. There are no overlapping owned rows with the tail component: joint-ID and witness-point intersections are empty. Only HummerBody and CamFollower reaction bodies are shared, so the CPU schedule executes Frame–Crank first and tail second, never concurrently.

Both compliant drive masses/references/accumulated impulses are retained. An initial unscaled natural-map solve fails acceptance. Explicit optimizer residual/Jacobian scaling1e4 (numerical units only, not physical tolerance) converges in130warm evaluations: normal3.42e-12m/s, tangent1.18e-17m/s, hardjoint3.81e-15 and drive0. The component includes Frame-to-ground reactions; subtracting those leaves six momentum errors below1.09e-11. Positive kinetic change1.2207e-4J is accounted for by actual motor work3.9342e-4J, contact work-2.1579e-4J and hardjoint work-5.5560e-5J.

The combined global schedule completes cycle1 with normal maximum11.977mm/s, anchor12.816mm/s, hard angular0.15482rad/s and drive0.006829rad/s. For comparison, plain cycle1 angular/drive are0.61693/0.33761rad/s. This addresses the dominant drive mode, while some anchor/contact errors remain larger than desired. The cycle performs18ordinary joint blocks and40ordinary point blocks plus the two nonlinear component solves (130 and84evaluations).

**Cycle2 is rejected before applying its Frame–Crank update.** Warm/zero least-squares and bounded hybrid attempts cannot meet the unchanged1e-8physical criterion: best normal1.73e-6m/s and tangent1.02e-5m/s. Hardjoint/drive and momentum/work checks pass, so this is a nonlinear-contact root-finding obstruction, not a reason to relax physical acceptance. No four-cycle combined result or live stability claim is made. A reporting-only replay reproduces the same accepted cycle and rejected block, preserving the partial report rather than advancing with an inaccurate update.

Completed cycle1 budget (relative to initial frozen state):
- Kinetic change: 5.7341471157e-05J.
- Contact/hardjoint/drive work: -2.5474789741e-04, -8.1327414916e-05, 3.9341678348e-04J.
- Work balance: 1.531e-18J; six momentum-minus-external components below 1.161e-11.

Artifacts: `/tmp/colibri_frame_crank_component_size.json`, `/tmp/colibri_frame_crank_reference_scaled.json`, `/tmp/colibri_physical_block_gs_combined.{json,npz}`, and exact failed operator/state `/tmp/colibri_physical_gs_combined_frame_crank_2.{json,npz}`. A possible bounded next numerical step is a fixed-active-face solve seeded from the candidate mode and checked against full heterogeneous-friction KKT; exhaustive3^16mode enumeration is not proposed.


## Exact-Jacobian polish of rejected Frame–Crank cycle2

`polish_frame_crank_newton.py` diagnoses and polishes the exact saved failed48-unknown system using the same natural-map residual and analytic piecewise Jacobian as the previous solver. The evaluator was factored into `natural_map_evaluator`; the prior scalar support solution remains byte-identical after refactoring.

The failed point has two sticking contacts and several separating contacts with tiny negative normal impulses. The Jacobian has one machine-null impulse direction; the next column-scaled singular value is9.65e-8 and is retained. The initial Newton step solves the linearized equation to1.64e-15, but its0.0515Ns norm exceeds the current0.0182Ns impulse norm and a full step raises the physical residual to8.72m/s. This motivates deterministic Armijo backtracking, rather than additional least-squares iterations or physical regularization.

One attempt is capped at40Newton iterations and30halvings per step. It succeeds in10steps. The first8steps cross the sticking/sliding boundary using fractions down to3.81e-6; the smallest retained singular value then rises from about6.9e-7 to0.00302, and two full steps finish. No modes are removed based on the squared mobility; the least-squares linear solve removes only a machine-null Jacobian direction while checking its actual linear residual. Rank becomes48 after the mode transition.

Reconstructing the original physical component asserts byte-identical A/rhs/initial impulse/friction arrays before inserting the polished result. Independent checks pass: normal6.94e-18m/s, tangent8.89e-10m/s, hardjoint3.55e-14, drive equation0, cone violation4.13e-12Ns, reaction reconstruction9.27e-12. Momentum minus static reaction is below4.22e-12. Kinetic change+5.4593e-5J is accounted by contact work-8.1857e-5J, hardjoint work-2.9430e-5J and actual drive work+1.6588e-4J, with8.2e-17J balance error. Fixed speculative impulses remain unchanged.

This resolves that specific rejected block without weakening the original1e-8physical criterion. It does not yet establish a completed combined four-cycle run or a production algorithm. Artifacts: `/tmp/colibri_frame_crank_newton_diagnosis.json`, `/tmp/colibri_frame_crank_newton_polish.{json,npz}`, `/tmp/colibri_frame_crank_polished_validation.{json,npz}`.


## Combined continuation with bounded Newton fallback: stop at cycle3

The authorized four-cycle run was resumed with the validated Newton/Armijo fallback enabled only for failed Frame–Crank blocks. Cycle2 now passes its complete physical checks, and both components complete two global cycles. No tolerance is changed. The third Frame–Crank block is rejected before its velocity/impulse update is applied, so the completed partial state remains cycle2.

| Completed cycle | Normal (mm/s) | Tangent map (mm/s) | Anchor (mm/s) | Angular (rad/s) | Drive (rad/s) |
|---|---:|---:|---:|---:|---:|
|1|11.9767|12.1481|12.8160|0.154819|0.006829|
|2|11.3974|6.9025|12.7886|0.172932|0.005539|

Cycle3's fallback takes13Newton attempts. Its rank transitions42→44 and the linearized residual remains about6.3e-12, but physical merit acceptance requires successively smaller fractions down to1.86e-9, then fails the30-halving limit. The original contact residual remains normal4.18e-5m/s and tangent2.00e-5m/s, above1e-8. Joint/drive/momentum/work equations still pass. This is a further contact-mode globalization obstruction; no additional budget or approximate update is used to bypass it.

The completed two cycles preserve six momentum components minus external reaction below1.61e-11 and work balance within7.96e-17J. Each cycle includes18ordinary joint and40ordinary point blocks plus two nonlinear components. Frame–Crank's second block uses the original bounded least-squares/hybrid attempts plus10Newton steps; this extra cost is recorded in `support_solver_evaluations`, not hidden in the cycle count. The third rejected attempt also remains recorded separately.

Drive and angular errors improve strongly versus plain physical GS, but normal/tangent/anchor errors remain worse than tail-only coupling at the same cycle count. Therefore neither completed four-cycle convergence nor a viable global preconditioner is established. Production/default physics remains unchanged.

Authoritative terminal evidence: `/tmp/colibri_physical_block_gs_combined.json` has `failure.cycle=3`, `failure.component=frame_crank`, and exactly two completed cycles. `/tmp/colibri_physical_gs_combined_frame_crank_3.{json,npz}` preserves the failed block. The earlier cycle2-failure section above is superseded only for that particular block by the successful Newton polish; it is not erased.


## Fixed auxiliary-friction ladder and terminal combined cycle4 failure

Root authorized exactly four auxiliary coefficient fractions0,0.25,0.5,1 at the same original A/rhs/drive/pose. These are seed-generation problems only. The failed cycle3 block is resolved: stage residuals7.45e-14,1.79e-14,9.53e-6,3.10e-14m/s. The half-friction stage is not accepted, but its candidate is a numerical seed for the final original-friction solve; no auxiliary velocity is applied. Reconstructing the original physical block verifies byte-identical A/rhs/initial/friction and passes normal3.10e-14, tangent1.18e-17, hardjoint4.97e-14, drive0, six momentum-minus-static errors below3.42e-12, and work balance1.45e-16J. Actual drive work1.32467e-4J is retained.

The same fixed ladder was enabled as the final bounded Frame–Crank fallback in the combined schedule. The run now completes three cycles, then **rejects Frame–Crank cycle4 before applying it**. Its original-friction normal4.47e-6m/s and tangent1.82e-6m/s still exceed1e-8 after all bounded least-squares, Newton/Armijo, and four-stage seed attempts. No extra fractions or weaker acceptance are used.

| Completed cycle | Normal (mm/s) | Tangent (mm/s) | Anchor (mm/s) | Angular (rad/s) | Drive (rad/s) |
|---|---:|---:|---:|---:|---:|
|1|11.9767|12.1481|12.8160|0.154819|0.006829|
|2|11.3974|6.9025|12.7886|0.172932|0.005539|
|3|10.4150|3.9824|10.0136|0.184433|0.003944|

The complete run retains original material friction and all native fixed speculative impulses. Three completed cycles use54ordinary joint blocks and120ordinary point blocks plus six accepted nonlinear component solves; the rejected seventh component attempt is counted separately. Numerical solve cost (reported least-squares/hybrid function evaluations; Newton linear attempts, including failed line searches):

| Attempt | Component/cycle | LS/hybrid evaluations | Newton linear attempts |
|---|---|---:|---:|
|1|Frame–Crank / 1|130|0|
|2|tail / 1|84|0|
|3|Frame–Crank / 2|251|10|
|4|tail / 2|73|0|
|5|Frame–Crank / 3|1191|29|
|6|tail / 3|80|0|
|7|Frame–Crank / 4|1375|81|

Completed-cycle3 kinetic change1.13689956e-04J equals contact-4.43157624e-04 + hardjoint-1.34916981e-04 + drive6.91764561e-04J within2.232e-16J. Six momentum-minus-external errors are below1.955e-11.

Global drive/angular improvement is strong; global contact/anchor benefit is mixed relative to the cheaper tail-only schedule. A completed four-cycle convergence result is **not established**. Authoritative terminal artifact `/tmp/colibri_physical_block_gs_combined.json` now contains exactly three completed cycles and `failure.cycle=4`. Failed data are `/tmp/colibri_physical_gs_combined_frame_crank_4.{json,npz}`. No production change or speed/stability promotion follows from this CPU reference.


## Cycle4 diagnosis: energy convexity and one original-friction globalization test

The exact non-associated Coulomb equations do **not** in general minimize the usual cone-constrained contact quadratic. With Q(lambda)=0.5 lambda^T A lambda+rhs^T lambda, a sliding cone-QP KKT equation includes normal term `g_n - mu*nu = 0`, where nu is the friction-disk multiplier. Native non-associated contact requires `g_n = 0` for positive normal impulse, independently of that multiplier. Fixed-normal tangential minimization is convex, but changing the normal impulse changes the disk radius; that does not establish a global energy-monotone coordinate-descent potential for the full law. Actual compliant motor work also means total kinetic energy is not a valid universal descent merit.

Concrete counterexample: A=I, rhs=(-1,2,0), mu=0.5. Exact non-associated solution lambda=(1,-0.5,0) has Q=-1.375 and satisfies normal complementarity. The associated cone-QP returns(1.6,-0.8,0), Q=-1.6, but has positive normal impulse and separating normal residual0.6, violating the requested law. Accepting the lower quadratic would silently change physics.

The exact rejected cycle4 candidate has a full-rank48 natural-map Jacobian, column-scaled condition1.44e10 (smallest singular value3.53e-10 retained). Points795/796 sit within about1e-9Ns of a stick/slip boundary; several other normals are within1e-9Ns of activation, and point774 has a small negative normal impulse. The failed merit gradient is nonzero. This supports testing a different equivalent complementarity merit rather than assuming a missing physical mode or convex energy minimizer.

`frame_crank_fb_trust_region.py` makes one bounded alternative attempt at **unchanged original friction**: Fischer–Burmeister normal complementarity `sqrt((s*lambda_n)^2+g_n^2)-s*lambda_n-g_n`, with the original non-associated tangent disk residual, analytic generalized Jacobian, and Powell dogleg trust-region merit globalization. The80-iteration cap and initial radius are numerical algorithm choices; no model compliance, material coefficient, contact row, pose, drive, or acceptance tolerance changes. The analytic Jacobian passes an independent central-difference check within5.70e-10 on a smooth mixed-friction fixture.

Result: original natural-map residual improves4.47e-6→1.01e-6m/s but **fails1e-8**. Late iterates retain full rank48 with linear solve defects around1e-18; the remaining obstruction is nonlinear convergence near contact-mode boundaries. No extra iterations or second alternative were attempted. The candidate was reconstructed against byte-identical original A/rhs/initial/friction for the physical audit and was not applied to the global schedule.

Artifacts: `/tmp/colibri_frame_crank_cycle4_diagnosis.json`, `/tmp/colibri_frame_crank_fb_trust_region.{json,npz}`, `/tmp/colibri_frame_crank_fb_physical_audit.{json,npz}`. The global run remains at three completed cycles and no production promotion follows.
