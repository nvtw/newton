# PhysX GPU D6 source study and bounded adaptation design

Source inspected: `/home/twidmer/Documents/git/PhysX`, commit
`ed6e5ca2474c9c80ad4f4826591b88476779c6ef`. CPU/source study only;
no GPU execution or solver changes. This identifies the actual GPU path,
not an analogy to CPU PGS. The installed ovphysx SDK need not have this exact
source revision; no binary/source identity claim is made.

## Actual preparation and solve

Paths below are relative to the PhysX checkout.

1. **Revolute represented as D6.**
   `physx/source/physxextensions/src/ExtD6JointCreate.cpp:116` constructs
   frames with their X axis on the revolute axis, leaving translation and
   swing locked and setting twist free/limited/locked. This is model setup;
   the actual GPU preparation is
   `physx/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu:516`,
   `D6JointSolverPrep`. It emits drive rows first (linear at554, twist at593),
   limits next, then locked axes at733. Dispatch at1144 calls this device
   routine. The comment at1217 explicitly distinguishes D6 device pre-prep
   from other joint types whose pre-prep occurs on CPU.
2. **Five hard rows plus optional drive for a hinge.**
   `physx/source/gpusolver/include/PxgConstraintHelper.h:162`
   `prepareLockedAxes` emits three translation rows and two angular rows.
   Angular rows use the quaternion-error Jacobian (`computeJacobianAxes`,43),
   rather than blindly using fixed world Y/Z axes away from closure.
3. **GPU preprocessing reorders and preconditions rows.**
   `physx/source/gpusolver/src/CUDA/jointConstraintBlockPrep.cuh:171`
   `preprocessRows` insertion-sorts by solve hint, forms square-root-inertia
   angular responses, and mass-metric orthogonalizes eligible equality groups
   (group IDs4/8, at226–237). `orthogonalize`,73, transforms Jacobian rows,
   geometric error and velocity target together. It is not arbitrary mass
   replacement. TGS invokes this same device preprocessing at
   `jointConstraintBlockPrepTGS.cuh:311`.
   `physx/include/PxConstraintDesc.h:79` supplies actual ordering values:
   ordinary drive hint0, angular equality1024, linear equality2048.
   Thus a normal force-mode twist drive precedes angular hard rows, which
   precede linear hard rows after sorting; do not infer order merely from
   row emission. Acceleration-drive/SLERP hints have their own values.
4. **The GPU solve is ordered scalar GS inside each joint, not a dense
   simultaneous five/six-row inversion.**
   `physx/source/gpusolver/src/CUDA/solverBlockTGS.cuh:744`
   `solve1DBlockTGS` holds both bodies' velocities in registers and loops
   sequentially over rows at813. Each row computes current relative velocity,
   effective response, clamped accumulated impulse, then immediately updates
   both bodies at899–920. The next row sees those updates. One lane handles
   one joint; a warp batches independent constraints. There is no SOR>1
   factor in this update.
5. **Temporal geometry and residual targets.**
   The same solve rotates cached anchor levers by accumulated body rotation
   (774), computes their accumulated motion, and rebuilds angular levers from
   the current arm cross linear axis (830). It projects eligible rows against
   angular equality bases and adjusts their bias consistently (843–862).
   `computeResolvedGeometricErrorTGS` updates errors using accumulated
   displacement and elapsed drive motion (871). Effective response is
   recomputed with the current levers (876). Hard bias is bounded. Springs
   and bounded drives retain separate coefficient/impulse semantics.
6. **Contacts and joints share the partition schedule.**
   `physx/source/gpusolver/src/CUDA/solverMultiBlockTGS.cu:1745` walks
   partitions in order, dispatches contact or D6 row solve at1810, writes
   body velocities to shared storage, and synchronizes between partitions
   at1831. There is no global exact joint elimination followed by an isolated
   contact solve. `PxgTGSCudaSolverCore.cpp:1106` chooses one256-thread
   whole-island kernel for a rigid-only position iteration when
   `(bodies including offset)*numSlabs <= 944`; otherwise it launches the
   partition path. Do not assert the benchmark necessarily selected this
   branch without a profile.
7. **TGS position iterations really advance time.**
   `PxgTGSCudaSolverCore.cpp:1269` divides dt by position-iteration count;
   the loop at1331 solves constraints, averages/propagates body updates,
   integrates, and advances accumulated time at1501. The final velocity
   loop begins1623. `solverBlockTGS.cuh:1091` concludes joint rows: ordinary
   bias is removed unless KEEP_BIAS; spring row update coefficients are set
   to zero. This is not a claim that PhoenX should disable its physical
   drives, or that a final unbiased pass alone guarantees correct friction.

## Mass handling and nonconservative exceptions

- Per-joint inverse-mass/inertia scale factors are explicit:
  `jointConstraintBlockPrepTGS.cuh:265–286`. Unequal user scales change
  physical momentum exchange and must stay1 in a faithful adaptation.
- GPU slab splitting additionally multiplies inverse response by each
  body's active slab count (`solverMultiBlockTGS.cu:1088`,
  `solverBlockTGS.cuh:769,783`). The average stage sums slab velocity changes
  and divides by the same active count (`solverMultiBlockTGS.cu:395–410`).
  This is an aggregate cancellation identity under consistent membership,
  not permission to mass-scale ordinary coloring rows without matching
  writeback. It also does not prove aggregate Coulomb closure after averaging.
- **Anchor handling is not automatically momentum conservative away from
  joint closure.** The helper constructor at85 starts both arms at world
  anchor B. For locked translation, `prepareLockedAxes:180` adds the locked
  error correction only to A's arm. With all three translations locked this
  restores A's own anchor while B retains B's anchor. A force row can then
  exert net internal torque `(pA-pB) cross impulse` when anchors disagree.
  Do not port this convention while claiming exact angular momentum.
- `DyCpuGpu1dConstraint.h:50` returns zero inverse response below a configured
  minimum, invoked by GPU prep at `jointConstraintBlockPrepTGS.cuh:224`.
  Do not transplant that cutoff to remove real near-blocked Kamino modes.
- TGS geometric-bias clamps, drive saturation, kinematic reactions and
  positional recovery are explicit; a reference's stability does not imply
  exact energy conservation. They require separate accounting.

## Concrete bounded PhoenX experiment

First compare an **ordered, locally preconditioned hinge row kernel** against
the current bilateral block on frozen small islands. This is an optional
reference experiment, not a default/API change or an exact-equivalence claim
for finite-sweep trajectories.

1. Retain PhoenX's physical masses, existing source joint frames/drive gains,
   SOR1, contact law, common-force-point convention, and ordinary coloring.
   Keep splitting only overflow. No global damping, response cutoff or mass
   boost. Reuse existing colored joint/contact ownership.
2. Prepare up to five hard hinge rows in square-root-mass coordinates.
   Form a small invertible mass-metric row transformation `T`, applying it
   to Jacobian and hard target together. Preserve every original mode; reject
   unresolved rank cases rather than truncating them. Keep bounded/soft drive
   rows in their original coordinates. Store `T` so transformed hard impulses
   map back by `lambda_original = T.T lambda_transformed`.
3. Process drive then angular then linear hard rows locally, immediately
   scattering each update to register body velocities using physical
   `M^-1 J.T`. Verify the transformed physical impulse is exactly the original
   row-span impulse and that the chosen common point retains net P/L.
   The optional angular-to-linear projection must transform the target too;
   changing only effective mass is invalid.
4. Rebase geometry at each existing substep and preserve current recovery
   and final-relax policies for the first comparison. A separate comparison
   can test accumulated-displacement target refresh; do not combine this
   with a recovery-policy rewrite or the open support-load correction.
5. Gate on frozen off-center unequal-mass hinges, two-joint chains, closed
   loops, finite/bounded drives, rotated poses, contacting supports, and
   original-row residuals. Measure P/L, impulse work, joint/normal/friction
   residuals before speed. Then bounded Colibri/G1/Kapla/cloth checks under
   unchanged physical budgets. No adoption unless quality is preserved.

The transferable GPU lessons are local mass-metric row preprocessing,
immediate joint/contact feedback through a common color schedule, current
temporal geometry, and batching independent joints. PhysX does not provide
evidence here for replacing all contacts/joints with a global dense solve,
using nonphysical mass scales, or weakening momentum requirements.

## Executed frozen two-body comparison

`check_d6_frozen_rows.py` runs CPU-only using the captured total-normal
two-body phase at frame330. It retains all67 biased or29 eligible relaxed
contacts and the six original joint rows. Four controls use identical
contact ordering and equal joint/contact sweeps: simultaneous block,
drive/angular/linear scalar order, PhysX-style within-group hard-row
orthogonalization, and all-hard-row orthogonalization. The production
metric-friction projection runs as a native Warp CPU helper. Joint solves
are FP64 evaluations of the native block equations, **not** a replay of the
full production dispatcher or its FP32 arithmetic. There is no copy
averaging; this is the physical unsplit two-body response.

Hard-row transformation T maps targets and Jacobians together and maps
impulses back with T transpose. The compliant drive is never mixed into
the hard basis: transformed compliance T R T transpose equals the same
single diagonal drive coefficient. An independent simultaneous-solution
gate verifies physical response equivalence of transformed/original
operators. No response rank cutoff, damping, geometry modification, or
SOR overrelaxation is used. Forces and reactions are reconstructed from
original rows; maximum aggregate P/L error over saved sweep checkpoints
is3.49e-11. Impulse-response and midpoint-work identity gates also pass;
the latter is an accounting identity, not a dissipation guarantee.

Result: `/tmp/colibri_d6_frozen_rows.json`.

| Phase / sweep | Block drive residual | Group-orthogonal drive residual | Block / grouped contact KKT |
|---|---:|---:|---:|
| biased /1 | .222307 | .096188 | .010402 / .010259 |
| biased /16 | .138154 | -.013311 | .018684 / .019068 |
| relax /1 | .005116 | .021026 | .001112 / .000943 |
| relax /16 | .001801 | .000193 | .000218 / .000224 |

Drive residuals are rad/s. Scalar sweeps leave a nonzero joint residual
before contacts; a smaller residual afterward can partly reflect
cancellation with the next contact update, rather than better satisfaction
at every phase. At biased sweep16 the grouped joint residual is .151955
before contacts and .024810 afterward, whereas the block closes it before
contacts to rounding. Thus this result does not justify production
promotion or predict stationary-base behavior. It motivates, at most, a
bounded native schedule comparison with the same complete physical gates.

Reproduce without accessing CUDA:
`uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python
-c 'import os,runpy; os.environ["CUDA_VISIBLE_DEVICES"]="";
runpy.run_module("local_studies.colibri.check_d6_frozen_rows",
run_name="__main__")'`.

### Joint-last and interleaved schedule controls

The same executable now writes `/tmp/colibri_d6_frozen_schedules.json`
with all16 sweep checkpoints, full original joint residual vectors, hard
and drive norms, contact KKT, P/L and work accounting, and exact joint-row
and contact-point visit counts. Joint norms combine translation (m/s) and
rotation (rad/s); they are descriptive, not a dimensionless acceptance
metric. The individual original-row vectors retain the distinction.

Additional schedules are C-J (joint last), J-C-J (symmetric), and each
actual captured contact column followed by J. Captured column sizes are
[6,6,6,6,6,6,6,6,4,6,6,3] in bias and
[1,4,6,6,4,6,1,1] in relaxation. The latter preserves the omission of
native-ineligible positive-bias points, whose captured impulses stay fixed.
No point within the eligible set is dropped.

At16 sweeps, biased contact KKT is .018684 for baseline J-C,
.026082 for C-J, .025656 for J-C-J, and .011946 for column-interleaving.
The latter uses12 joint solves per contact sweep, versus1 baseline.
Relax contact KKT is .0002182, .0003005, .0002954, and .0003541,
respectively; interleaving uses8 joint solves per relaxed contact sweep.
All three joint-last variants close original joint equations to rounding,
but reopening contacts remains measurable. J-C-J doubles joint work and
does not offer a substantial benefit.

D6 grouped ordering at biased16 retains hard-row error .024810 versus
baseline .024773 despite a smaller drive error. Thus the apparent aggregate
improvement is predominantly redistribution toward the drive, not closure
of the full contact/joint system. These controls do not establish a robust
production convergence improvement. No live, default, source geometry or
physical material changes were made.
