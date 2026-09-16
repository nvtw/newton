# Native final relaxation iteration controls

No prior Colibri final-substep count2/4 control was found in the existing local
studies. These process-local adapters set the existing public
SolverPhoenX velocity_iterations option. Canonical source is unchanged.

All controls retain 30 substeps per 120Hz collision update, one biased iteration,
SOR1, source geometry/materials/drives and the current public geometric
candidate policy. Counts1/2/4 mean 2/4/8 final relaxation sweeps per60Hz frame:
the added work is 0/2/6 sweeps. This is not a matched PhysX velocity budget.

## Identical frozen state

The captured final330 pre-relax arrays match the established reference byte
for byte. Restoring that state and executing one native sweep also reproduces
all four saved output arrays byte for byte. Each count begins with the same
physical velocities, joint accumulation and contact state; native code performs
copy broadcast, sweeps, averaging and writeback.

| Final relaxation count | Normal residual m/s | Tangent residual m/s | Crank equation residual rad/s | Base drive residual rad/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | .0375384 | .0154903 | .490758 | .132005 |
| 2 | .0332766 | .0125179 | .395787 | .0828788 |
| 4 | .0270197 | .00818159 | .293378 | .0352968 |

The four-sweep control improves these errors but does not approach a converged
solution. Contact residuals use the original physical inverse mass scaling;
joint residuals include actual accumulated impulse and compliance.

## Live330 controls

All three pass the unchanged public bounds for5.5s. Their source hashes match.
Their trajectories and eligible contact sets differ, so the final residual
comparison is supplementary to the frozen causal comparison.

| Count | Final normal/tangent m/s | Crank residual rad/s | Mean crank speed rad/s | Peak depth um | Peak anchor mm | Peak axis rad |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | .0375384 / .0154903 | .490758 | -3.20081 | 67.22 | .49206 | .010683 |
| 2 | .0305694 / .000706964 | .404845 | -3.20130 | 72.97 | .44937 | .011288 |
| 4 | .0236260 / .00285030 | .298237 | -3.20663 | 79.56 | .47695 | .007798 |

Mean synchronized step times were12.0634/12.0585/12.3193ms, with30 warmup
and300 timed frames each in an isolated GPU window. These include the identical
phase capture instrumentation. Count2's difference is noise; count4 costs
about2.1% more in this single comparison. These are not replacement clean
production FPS measurements. Mean drive motion changes little despite better
final-relax residuals: only two end-of-update relaxation phases are affected.

This is a useful low-cost accuracy control, not a validated accuracy solution
or a long-duration stability result. No default was changed.

## Reproduce

Use the project interpreter through uv run --no-project.

- python -m local_studies.colibri.check_native_velocity_iterations
- python -m local_studies.colibri.check_live_velocity_iterations --velocity-iterations 2 --output /tmp/colibri_native_velocity2_live330
- python -m local_studies.colibri.check_live_velocity_iterations --velocity-iterations 4 --output /tmp/colibri_native_velocity4_live330

Reports: /tmp/colibri_native_velocity_iterations.json,
 /tmp/colibri_native_velocity_summary.json. Each live prefix also saves the
trajectory, final phase snapshot, and source hashes. No rendering or audit
time is included in reported step timing.

## Long stability gate: four sweeps rejected

The clean public run requested18000 frames/300s and stopped at77.45s
(step4647). TailMount/Tail_Feather_A__2x__mirrored exceeded the original axis
limit: .037042rad > .03rad. The independent failed-pose joint audit reports
1.84473mm anchor error and .0370442rad axis error. Prior successfully checked
poses reached .91405mm peak penetration. All final state values were finite.

Canonical solver, geometry, scene and assets remained hash-identical throughout.
The first330 q/qd histories are byte-identical to the short instrumented
velocity4 control, so removing phase capture did not change that trajectory
prefix. The4646 successful poses end at77.433333s; the separate final q/qd
arrays contain the first rejected pose. No threshold was relaxed.

The interrupted clean run measured11.3035ms/frame (88.47FPS), excluding physical
audits/rendering and without phase-copy hooks. This is a failed candidate's
cost measurement, not a stable-performance claim. The additional final sweeps
must not be promoted on the short residual improvement.

Reproduce:
python -m local_studies.colibri.check_clean_velocity_iterations --velocity-iterations 4 --frames 18000 --output /tmp/colibri_native_velocity4_clean18000

Artifacts: /tmp/colibri_native_velocity4_clean18000.json/.npz, .source.json and
.prefix.json. No production source or default was changed.

## Bounded onset and fresh geometry diagnosis

Saved60Hz poses show an abrupt local event, not a slowly approached assertion:

| Time s | Mirrored-A anchor mm | Mirrored-A axis rad | Tail kinetic energy mJ |
| --- | ---: | ---: | ---: |
| 77.416667 | .17829 | .001084 | .09291 |
| 77.433333 | .24703 | .007078 | .08873 |
| 77.450000 | 1.84473 | .037044 | .31539 |

The final angular-velocity changes are largest for TailRack (11.96rad/s) and
mirrored-B feather (11.10rad/s). Total kinetic energy rises from .5744mJ to
.8231mJ. This is a trajectory diagnostic, not an energy-conservation failure:
contact correction and actual motor work were not captured.

Fresh collision queries use the exact public source scene, contact offsets,
SDF geometry and pair filtering. Compare ordinary fresh audit, current
geometric candidates, and legacy velocity-filtered candidates at the last
four successful poses and rejected pose.

- At77.416667s mirrored-A feather versus TailRack mesh has166um penetration.
- At77.433333s mirrored-A versus mirrored-B feather has914um penetration.
- At77.45s the nonmirrored-A/B pair has455um penetration, and mirrored-A/B
  has268um.
- Rack/Pinion remains near10--17um penetration in these samples, except a
  positive145um gap at77.416667s. This is not the old deep gear-flank event.
- At77.416667s current geometric admission already produces89 mirrored-A/B
  witnesses with minimum initial gap47.7um. Transporting them to the next
  saved60Hz pose gives a minimum projected gap of-1.070mm. The harmful
  direction is represented in this fresh query.

These sibling-feather and feather/Rack pairs are emitted by the unchanged
source filter policy. They must not be removed to make the diagnostic pass.
The solver still owns the same block joints, all source contacts and group4
copy schedule; only final relaxation count changed.

The60Hz saved poses do not expose the intermediate120Hz contact refresh,
native matched witnesses, impulses or color/copy states. Transported
witness gaps are projections, not fresh SDF samples. Consequently these
queries establish a feather/Rack collision cascade and rule out simply
relabeling the final assertion as instability, but do not yet prove whether
native history/activation at the missing midpoint or finite-iteration
response is the causal defect. An actual120Hz pre/post-contact ring around
this onset would resolve that distinction.

Artifacts: /tmp/colibri_velocity4_tail_onset.json,
 /tmp/colibri_velocity4_tail_geometry.json and per-query NPZs.
Scripts audit_velocity4_tail_onset.py and probe_velocity4_tail_geometry.py.

## Actual120Hz ring isolates midpoint reduction loss

The completed native ring reproduces all4646 successful q/qd histories,
timestamps and rejected final q/qd byte for byte, with unchanged source/assets.
All8 ring updates and32 final-substep phase entries pass completeness checks;
selected contact IDs have exactly matching header body ownership.
Earlier last2-only trace controls are preserved separately: allocations made
during CUDA capture invalidated their older entries. The final trace allocates
all buffers before capture and first passed a four-frame completeness/prefix
test. This was a diagnostic bug, not a solver change.

At step9290 (77.416667s generation) the actual mirrored-A/B manifold has89
contacts. At step9291 (77.425s generation) it has41. At the9291 post pose,
retained contact normals and witnesses report only41.1um penetration while
fresh geometry finds914um. The normal stored by the solver equals the
corresponding generated view normal exactly. Body copy counts are10/12.

Exact9291 generation re-query:
- Raw geometric:3634 pair contacts;518 transport below-500um at the actual
  post pose, minimum-1075.56um.
- Reduced geometric:42 contacts, none below-500um, minimum-41.1um.
- Ordinary reduced:40 contacts, none below-500um.

Thus raw geometry contains the missed directions. Reduction loses them;
a small live/fresh retained-count difference does not explain the common
loss. The next refresh admits the deep overlap, with summed normal impulse
about7.5mNs and substantial closing velocity remaining after four relaxation
sweeps. Extra final iterations did not repair the prior reduction loss.

All518 lost witnesses are initially closing, but none would cross under the
instantaneous generation twist: initial gap296--507um, predicted gap after
1/120s177--486um. The deepest has initial405.0um, vn=-24.264mm/s, predicted
+202.8um, actual transported post gap-1075.6um. Later response changes motion;
future pose must not determine production admission.

Exact voxel radii for shapes36/37 are212.405255/212.405124um, with unit scales,
zero margins and authored pair gap700.093253um. All518 lost witnesses are
therefore outer-tier; the reducer keeps inner-tier candidates at107--209um
in the same normal bins7/8/10. Every inner candidate competes for all six
spatial slots with unconditional priority over outer candidates. Depth and
voxel coverage is inner-only. Geometric mode does not use the instantaneous
predictive guards. This explains the loss even though the witnesses lie
inside the authored physical contact offsets. Both feathers have source mu0.

## Geometry-only spatial alternatives

CPU emulation uses current generation geometry for selection and post pose
only for retrospective coverage:

| Spatial selection | Distinct bank winners | Retained below-500um | Minimum transported post gap |
| --- | ---: | ---: | ---: |
| Current inner-tier priority | 29 | 0 | -41.1um |
| A: independent outer bank | 38 | 2 | -949.2um |
| B: all candidates spatial plus depth | 40 | 2 | -949.2um |
| C: prioritize only actual overlap | 37 | 2 | -949.2um |

C keeps the same six spatial slots and leaves inner max-depth/voxel coverage
available. It is the smallest candidate supported by this frozen evidence.
Raw witnesses680/737 win spatial extrema without reference to the post pose.

These are spatial-bank counts, not exact full export counts: voxel ownership
and packed-score tie breaks are not emulated. Conservative unions with the
current42-contact set are at most80/82/79 for A/B/C. They bound this pair,
not every possible pair's allocation. A would require extra architectural
slots; C does not. Production selection and regression remain parent-owned.

Artifacts: /tmp/colibri_velocity4_native_tail.trace.npz/.analysis.json,
 /tmp/colibri_velocity4_midpoint_coverage.json,
 /tmp/colibri_velocity4_lost_witnesses.json,
 /tmp/colibri_velocity4_feather_sdf_metadata.json and
 /tmp/colibri_velocity4_spatial_alternatives.json.

## Canonical spatial priority correction

The focused test_spatial_contact_priority.py regression failed before the fix
in all eight CPU/CUDA, fast/deterministic, and insertion-order cases: a separated
witness at 150 um excluded a spatial extreme at 400 um, although both were
inside the 700 um contact gap. It passes after separating overlap spatial
priority from the 212 um depth/voxel coverage threshold. Overlapping support
still supersedes separated contacts. The existing two-depth wrapper retains
its previous priority threshold.

The frozen 9291 CUDA query now exports 50 geometric reduced pair points, including
two with transported post-gap below -500 um, minimum -949.184 um. Before the fix
it exported 42 points, none below -500 um, minimum -41.056 um. The raw set remains
3634 points, including 518 below -500 um. See
/tmp/colibri_velocity4_priority_fixed_midpoint_coverage.json. Selection uses
generation geometry only; transported post-pose gaps are an audit, not selection
inputs. No live stability acceptance follows from this frozen proof.
