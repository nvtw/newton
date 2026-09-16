# Momentum-preserving local/global solves for PhoenX

## Decision and scope

Study the local/global solver architecture, not VBD as a replacement for PhoenX.
Keep rigid-body impulse dynamics, physical SI masses/inertia, the intended
Coulomb contact law, source drives, SOR 1, and no global damping. Speed/FPS must
never sacrifice quality, behavior, or correctness. Twelve ordinary colors plus
an explicit overflow strategy are the practical baseline, not a demand that
all conflicting rows fit in twelve independent colors.

Recommendation: establish unsplit physical-mass local sweeps as the reference,
then add a coupled impulse correction for slowly converging structural modes.
Treat mass splitting as optional scheduling/preconditioning, never a prerequisite
for momentum conservation. The implementation and performance are unproven.

## What the supplied research contributes

### JGS2 (Lan et al., 2025)

The paper models how the complementary system responds to a local perturbation.
Its exact quadratic derivation uses elimination of complementary degrees of
freedom; its practical elastodynamics method approximates that response with
precomputed, rotated subspaces and sampled energy contributions. This is more
specific than merely reducing a relaxation factor. Near-Newton convergence
claims concern the studied elastodynamic formulation and its approximations;
they do not establish convergence for our changing unilateral contact set or
nonassociated Coulomb friction. We should borrow coupling-aware correction,
not assume its position-space updates preserve PhoenX momentum.

Primary paper: https://wanghmin.github.io/publication/lan-2025-jgs/Lan-2025-JGS.pdf
Local download: /tmp/colibri_research/Lan-2025-JGS.pdf
SHA256: 7540a54035d28a542eee1bc01b004e49ae370032c2c73b5cf5d540a6dacea86c
Sections 3–5 provide the relevant derivation. Its mathematical correction is
not artificial global velocity damping, despite damping terminology in the paper.

### Supplied local/global report

The report inserts an early structural correction among local sweeps. Its
fixed-protocol DR Legs comparison improves closure and passes the posture gate,
but costs more per step. Other cases show mixed benefit, including a cross-slide
case without improvement. These results support testing information propagation;
they do not establish universal speedup or momentum conservation. The separately
tuned comparison changes settings and is not an algorithm-only ablation.

Source: https://jumyungc.github.io/newton/reports/vbd_local_global_comparison/
Downloaded HTML/text: /tmp/colibri_research/vbd_local_global_comparison.*
Exact implementation branch/commit is being investigated. G1 in this report
means one global correction, not the Unitree G1 robot.

## Our proposed transfer: stay in impulse space

The following is our derivation/proposal, not a claim made by either reference.
At fixed geometry, let v denote physical generalized velocities, W the physical
inverse mass/inertia operator, J the constraint Jacobian, and lambda physical
constraint impulses. Apply every solver increment by

    delta_v = W J^T delta_lambda.

For internal rows let C map generalized momentum to total linear/angular
momentum. Require C J^T = 0 for the actual assembled rows. Then every accepted
increment conserves total momentum at that frozen configuration, regardless
of whether delta_lambda came from a local sweep, a global solve, or a coarse
approximation. Static/kinematic contacts have external reactions; include those
reactions in the balance instead of demanding constant dynamic-body momentum.
Use actual common-point wrenches and moments, not a verbal equal/opposite
argument for inconsistent lever arms. Pose integration needs a separate audit.

This condition does not imply contact feasibility, energy correctness, or good
convergence. A momentum-balanced iteration can still overshoot, violate a
friction cone, creep, or leave joints separated. Neither velocity averaging
nor a final global momentum repair substitutes for the constraint law.

### Exact structural elimination as a reference

For bilateral rows j and a fixed linearization of contact rows c, assemble
A = J W J^T + R, where R contains only intended compliance. With residual r,
eliminate the joint block:

    delta_lambda_j = -A_jj^-1 (r_j + A_jc delta_lambda_c)
    S_cc = A_cc - A_cj A_jj^-1 A_jc
    corrected_r_c = r_c - A_cj A_jj^-1 r_j.

The contact solve must use both S_cc and corrected_r_c, and scatter the induced
joint reaction as well as contact impulses. Otherwise the mechanism response
and/or joint residual are wrong. Redundant joints require a rank-aware solve
and consistency checks; arbitrary diagonal regularization changes the model.
Changing active contacts or sliding directions requires refreshing the relevant
linearization. Friction is not assumed to be a symmetric quadratic minimization.

### Practical local/global schedule

1. Prepare current geometry and physical response; apply each warm impulse once.
2. Run twelve conflict-free physical-mass colors, followed by a deterministic
   physical overflow reference. Keep all contact points and the existing law.
3. Measure the coupled residual; form a structural or interface correction in
   impulse space. Begin with the exact small-system reference above.
4. Apply joint and contact reaction updates together. Reconcile contact
   feasibility, sticking/sliding, drive bounds, and joint residuals afterward.
5. Accept based on the original equations, not only a smaller coarse objective.

Later, a basis Z in impulse space can target slowly converging modes. For the
smooth bilateral subproblem, solve (Z^T A Z) delta_a = -Z^T r and apply
J^T Z delta_a. This keeps the internal momentum nullspace intact, provided the
original J passes its wrench audit. Use it as a correction inside the complete
solver, not as a permanent reduced model that discards unresolved motion.
For nonsmooth contacts use an appropriate residual/active-set globalization;
projecting a coarse impulse into the cone alone is not a convergence proof.

## Why this addresses the current evidence

Our copied-head audit found ordinary rows use N-scaled mobility when a body has
N overflow copies. Averaging restores physical impulse accounting but leaves
large finite-iteration residuals. Physical head followed by overflow broadcast
removes that dilution, but does not remove long-range coupling error.

The latest uncorrected twelve-color control failed at 0.8833 s with a 2.015 mm
Frame/CamWheelBottom anchor separation. Its final graph had 606 rows, 529 in
overflow, and up to 110 body copies. Base horizontal motion reached 4.754 mm.
Evidence: /tmp/colibri_total_normal_ordinary12_before.json and ownership sidecar.
The local total-normal friction candidate was active; this is not an accepted
production result. The earlier all-one-copy control also failed, so eliminating
mass splitting alone cannot be assumed to solve Colibri.

## Candidate comparison and order of experiments

| Candidate | Momentum mechanism | Expected benefit | Main unresolved cost/risk |
| --- | --- | --- | --- |
| Physical colored GS + physical overflow | Paired native impulses | No copy dilution; correctness reference | Serial overflow; long-range convergence |
| Physical head + split overflow | Paired impulses and consistent copy average | Protect regular constraints | Overflow still converges slowly |
| Coupled joint/contact Schur solve | Joint reaction reconstructed with contacts | Mechanism response propagated globally | Factor cost, rank, changing contact modes |
| Local sweeps + coarse impulse correction | Corrections remain in J^T range | Target slow structural modes | Basis quality; contact globalization |
| Independent additive Jacobi impulses | Deterministic sum of paired impulses | Broad parallelism without copied masses | Overshoot and energy/contact residuals |

Prioritize physical reference, exact coupled reference, then coarse acceleration.
Keep one contact/joint model across strategies. A deterministic per-island choice
can be pragmatic, but must pass the same physical gates at strategy boundaries.

## Validation before any FPS acceptance

- Native closed-system off-center impulses: full linear/angular momentum balance
  per phase, unequal masses, anisotropic inertia, nonzero warm starts.
- Ground-supported and driven systems: explicit external reaction and motor work.
- Coulomb law: nonpenetration, complementarity, sticking residual, sliding
  direction/dissipation and friction-cone bounds; separate recovery work.
- High-mass chains and the existing difficult 400:1 case: residual versus work,
  not merely finite states or softened constraints. Add rank-deficient loops.
- Colibri: absolute base translation/yaw, joint closure/axes, fresh penetration,
  motor motion and long duration, at 30 substeps and 120 Hz contact updates.
- Kapla and G1 robot: unchanged physical fixtures, conservation where applicable,
  contact/joint behavior and deterministic repeated trajectories.
- Timings only after physical acceptance; compare matched temporal budgets and
  report setup, correction, collision and solve costs separately.

Pending work: native physical-head prototype and tests; source-branch retrieval;
small impulse-space local/global experiment; coupled contact transfer proof.
No reference currently proves that this approach resolves Colibri drift or
matches PhysX performance.

## Executed local/global proof experiment

CPU FP64, fixed poses: 25 bodies, 400:1 mass ratio, 24 shared-pivot ball
joints (72 bilateral rows), twelve conflict-free colors, physical inertia.
After eight local GS cycles the residual L2 is 0.0925186 m/s. Adding one
exact 72-by-72 coupled impulse correction reaches 1.62e-15 after one cycle.
Across 27 audited states, total linear/angular momentum error is at most
2.27e-13 and kinetic-work closure error is at most 2.89e-14.

A six-dimensional constant/linear multiplier coarse basis does not help:
residual after eight cycles is 0.0985784 m/s. Nonpositive correction work and
momentum conservation alone therefore do not establish improved convergence.
A useful coarse basis must represent the actual slow coupled modes.

Reproducer: local_studies/colibri/check_local_global_impulses.py.
Results: /tmp/local_global_impulses.json and .log. This example has no unilateral
contacts or Coulomb friction and makes no GPU or speed claim.

Terminology caution: exact coordinate GS on a fixed convex quadratic reduces
that full objective. Slow convergence is not itself proof of energy overshoot.
Simultaneous local updates or inconsistent local objectives can overshoot; the
PhoenX diagnosis must identify which mechanism actually occurs in captured data.

## Published VBD comparison: implementation provenance audit

CPU/network-only audit on 2026-09-16; this is architecture research, not a proposal to adopt VBD.

Primary published metadata: [raw repeated results](https://jumyungc.github.io/newton/reports/vbd_local_global_comparison/dr_legs_local_global.json), keys `repositories`, `fixed_vbd_routes`, and `routes`; [publication manifest](https://jumyungc.github.io/newton/reports/vbd_local_global_comparison/manifest.json).

- Report G1 checkout: detached base `d711ddf377d642ff040ed887df4adebe16707504`, local path `/home/jumyungc/JC/git/newton_g1_latest_integration`, with upstream `811b7b1ac803e193f063819d6e5d085cebdcddf6` **merged without commit**. The measured implementation is therefore not identified by a reproducible immutable commit.
- Publication snapshot: `gh-pages` tip `11030dfc6d94c9c84dc2ced72956f6426f4cfa2e`; manifest generated 2026-09-11T09:35:30.740227Z. The manifest hashes report builders/results/media, not the measured solver source. The page separately names newer upstream droop control `4129afde71`; do not conflate it with the DR Legs control revision.
- Exact base is unavailable from the public repository: `git fetch origin d711ddf377d642ff040ed887df4adebe16707504` returned `not our ref`; GitHub commit API returned422, raw `solver_vbd.py` returned404. No exact implementation checkout or source-level global-solver attribution is possible from these published artifacts.
- An isolated exploratory checkout exists at `/tmp/newton-jumyungc-vbd-reference`; its checked-out `vbd-lrc-report` commit `fcadb9828df1390fb226ee29b0f0c0c925b570ab` is an older, DIFFERENT report and must not be used as G1 implementation evidence. Existing working trees were untouched.

What is directly established about the architecture: controlled DR Legs comparison uses eight local sweeps, with one global structural correction **after local sweep1**; tuned G1 uses one local sweep, one global correction, then one local reconciliation sweep. It reports one structural island with36joints and31bodies, including six loop-closure joints; contact generation remains Newton. Thus the transferable concept is an early connected structural correction followed by local reconciliation, rather than only more local sweeps. Published metadata does not establish whether the global solve is direct/iterative, its factorization, constraint space, contact coupling, momentum properties, or exact recovery equations. These must not be inferred or copied into PhoenX/Kamino without actual source and independent physical-law validation.

Downloaded metadata: `/tmp/colibri_research/dr_legs_local_global.json` and `/tmp/colibri_research/vbd_manifest.json`. No GPU execution or solver changes.

### Public branch search scope

`git ls-remote --heads https://github.com/jumyungc/newton.git` enumerated the public branches. The29 VBD/cable-related branch tips below were checked through their raw solver entry points for global-correction/global-solver/structural hooks:27 available files yielded no matching hooks; two older paths returned404. This bounded search does not prove no implementation exists under another name or in old history. Exact ref inventory and results are in `/tmp/colibri_research/vbd_branch_audit.json`.

- `VBD-cable-dev` → `cb1184a91e9a4425cd16f1faaf9a9969ba4fd762`
- `fix/vbd-kinematic-state-handling` → `9857fce9e3d5990b85feb475ea4a0b1dd0a53247`
- `fix-VBD-cloth-example` → `42c335f47ebe3d37dbd1edf93bf4f34fdceca596`
- `horde-avbd-stack` → `6556f6adf53548061a9d61bbe1f6cf8194863c94`
- `jumyungc/clarify-cable-slot-docs` → `64e268bdf8c9035dd7cce371743691502b569dbf`
- `jumyungc/support-cable-rest-pose` → `4432efc0f00bb63eba1ccc8c09f5d8bd17b2c0ba`
- `jumyungc/vbd-angular-contact-friction` → `e809616268578b036c5b4df996adce6050ab3a3f`
- `jumyungc/vbd-compliance-kkt` → `d418045a6d4eed176081702bf1df988e6a7f1116`
- `jumyungc/vbd-compliant-alm-joint-friction` → `2a4048945a90fceb5853ecea7cdf7242b3d4f130`
- `jumyungc/vbd-rigid-block-dim` → `fefa4e89ffe9e902cca42424431a358e4073bc27`
- `optimize-cable-bend-twist-jacobian` → `103f759b3a3b574b9a99a265ab826b46ed198458`
- `remove-vbd-dahl-defaults-compat` → `6cf8d8393b368306faf2d6571d1336fca0405dd4`
- `rigid-body-chain-rod-xpbd` → `5b080bd93be1d7bd6c0d03d132646ce6a18ba3a7`
- `vbd-alm-scale-factor` → `6f7e2534de6d127c16928a2bc58b6657db1baee6`
- `vbd-cable-eval-fk` → `f3a8ab663ff372f8921aa494cd74eda12fb8db7c`
- `vbd-compliant-alm` → `a78350f6ecd32680efcd1d34b49e5d3448dcf2c4`
- `vbd-conveyor-belt` → `c1754a0099913d294eb17361b34572057eb0a31a`
- `vbd-custom-attribute` → `46c66e3e9103c1778b3b79712967ad50e1d3813a`
- `vbd-evalfk` → `6f3b9707e6ea39ee4399b699b02f651dfb1bbc74`
- `vbd-ground-particle` → `f9c230165e12907658a12b8c7a338c5fae87f53f`
- `vbd-history-warning` → `dfd2358d96aab5e3ee18d999c4d127602c35ebe7`
- `vbd-inplane-force-hessian-damping` → `346956bbee6f37359599dd4b4c0563c25b3bdbf7`
- `vbd-joint-friction` → `adccb0b697da50709205b7764b6135976d99daae`
- `vbd-lrc-report` → `fcadb9828df1390fb226ee29b0f0c0c925b570ab`
- `vbd-rigid-contact-auto-default` → `4ffc65c98b33df9d4cc95b74737dd292c2893506`
- `vbd-rigid-determinism` → `e7ff2ed161d662ae98d3610394991fa3840f8697`
- `vbd-rigidbody-cable` → `1591263bca8f93da309e162ef14bc3358b246809`
- `vbd-support-reset` → `bf9778059bfe36d16f699d1084640b09a36fd3a7`
- `vbd-tension-reaction` → `62c5b1fd8553f17c9472c045d5bd6a260c1daaac`

Additional recursive-tree checks covered main `b6fce29dc68c7cfe13bef64ad0e982d903526537`, compliance-KKT, compliant-ALM-joint-friction, VBD-cable-dev, and report branch `codex-sol/robust-contact-report` (`aac8a374afb3ec6fb3fb5ac50121c3d8eaf21efe`). The report branch contains published research artifacts, not the G1 builder/solver implementation. The inaccessible exact base cannot be tested as an ancestor of these refs. Published repository metadata gives local filesystem paths and `upstream/main`, but no alternative source-fork URL. Therefore exact implementation provenance remains unresolved beyond the reported detached/uncommitted base; no algorithmic implementation claim is warranted.

## Build on existing PhoenX references

This is not the first coupled-response idea in this worktree.
JOINT_CONSTRAINED_MOBILITY.md already records a geometry-consistent joint
nullspace response and a four-contact frictionless reference with momentum
and work audits. COUPLED_SUPPORT_BLOCK_DESIGN.md records a full support
component and a whole-scene physical block-GS comparison. The latter shows
that solving the support locally can worsen neighboring residuals, and that
plain and coupled four-cycle results remain globally unconverged.

GLOBAL_JOINT_TORQUE_DEFECT.md documents a rejected global reference where
huge cancelling impulses amplified tiny FP32 wrench inconsistencies into
angular-momentum error. Its initial sign error is explicitly corrected in
the linked followup report. A high-precision linear solve alone is therefore
not sufficient: preserve torque-consistent wrench construction and certify
rank/conditioning before applying any global correction.

The new paper motivates improving this existing coupled-response path and
its preconditioning/globalization. It does not justify repeating a rejected
global solve, clipping physical modes, or declaring the toy proof a live fix.

## Current native prototype status

The local physical-head/split-overflow prototype passes its small native
joint/contact momentum, warm-history and relaxation-energy fixture. However,
its first cap12 Colibri run produces nonfinite state in the first frame
(0.0166667 s). It is REJECTED pending debugging and is not a validated solution.
Artifact: /tmp/colibri_physical_head12_first.json/.log/.npz.
No production solver changes or FPS acceptance follow from this prototype.
The CPU local/global proof is independent and remains limited to bilateral
fixed-pose equations.
