# Fused contact gap audit

Frozen hard-contact stage 8 fails at frame 7: fresh depth 1.399 mm on Crank/SmallGear versus LargeGear02, with normal closing speeds 0.177–0.190 m/s. Refreshed joint residuals reach 0.01345 rad/s and 0.00653 m/s.

Ordinary preparation incorrectly floored live separation at positive generation-time gap. Removing that floor lets speculative contacts activate during temporal steps, as PhysX TGS current separation updates do (DyTGSContactPrep.cpp:1527–1530).

The new test_rigid_speculative_contacts has no joints: a 20 mm mesh cube starts 0.5 mm above a plane, descending at 0.1 m/s. After one collision query and ten 1 ms steps, old preparation leaves center z=9.5000 mm. After removing only the floor, the unchanged z>=9.999 mm assertion passes. Physical masses, SOR 1, friction classification, and geometry are unchanged.

Default-path validation: 30 tests pass, covering six maximal momentum cases, width-six cloth coupling and mass-splitting variants, contact mobility, and direct/D6 drives. Ruff passes. Colibri rerun remains required before acceptance.

## Live-gap follow-up (2026-09-15)

Unchanged fused 30 temporal steps × 1 sweep, collision120Hz, source mesh cylinders and contact offsets, SOR1, no ether damping:

| Stage | Frames | Maximum fresh depth | Final depth | Warm mean |
|---|---:|---:|---:|---:|
| 8 | 180 | 76.8µm | 13.7µm | 18.50ms incl cold |
| 15 | 180 | 164.6µm | 35.7µm | 44.80ms |
| Full36 + Flower | 180 | 365.2µm | 60.5µm | 92.25ms |

All stages pass existing joint/finite-state checks and fresh collision depth screen. These are three-second physical checks, not long-duration acceptance. The full scene remains slow at10.84FPS. Reports /tmp/colibri_bilateral_fused{8,15,36}_livegap_180.json preserve settings/results. Hard-contact stage8 also passes30frames, establishing that changing normal softness is unnecessary to obtain the improvement.

Reducing only temporal steps from30 to15 also passes180frames: stage15 max95.1µm/final61.0µm, warm23.13ms43.23FPS; full37 max342.3µm/final74.4µm, warm47.77ms20.93FPS. Reports append _15x1_180. These runs precede the independent dimensionally invalid load_boost removal candidate; do not combine their acceptance evidence with later physics changes.

Performance inspection: both preparation and iteration execute a contact column on one lane, serially walking all its contact points. Prepare does not reach the optional FP64 bilateral solve, so its roughly38ms/frame profile cost has a separate cause from joint matrix register pressure. Full37 has approximately2200contact points over77pairs. A separate parallel per-point preparation and colored warmstart could address preparation work imbalance, but requires independent equivalence tests.

## Parallel preparation prototype

Rigid per-point preparation reads no body velocities before its final warmstart scatter. Opt-in pointwise_geometry shares the exact source factory, then invokes existing cached warmstarts in the same colored order. The mixed free-root/hinge/motor fixture compares every prepared row, impulse, joint data and velocity bitwise; eight phases pass. Eighteen fused conservation/drive semantics also pass unchanged.

An initial prototype omitted the active column bound and failed full scene at frame52. Recycled inactive headers could write reused live point slots. A regression replacing two sphere/plane columns with one four-point mesh/plane column fails without the guard: reused point1 effective mass2.0 is overwritten as1.0029. Restricting cid to num_contact_columns makes both fixed and recycled-contact A/B tests pass bitwise. Initial failed report is preserved; full guarded screen remains pending.

Guarded point-parallel preparation passes full37 at15×1 for300frames (5s): max265.471µm exactly matches the accepted serial-preparation baseline maximum, final68.91µm; warm31.998ms31.25FPS versus45.94ms21.77FPS serial. Report /tmp/colibri_bilateral_fused36_parallelprep_guarded_15x1_300.json. This is a30.3% time reduction while preserving numerical/physical checks, still slower than the PhysX reference. Guarded two-fixture strict A/B passes, and all18 fused semantics pass.

At8 temporal steps ×1 sweep, stage15 and full37 both pass300frames (5s). Stage15: max439µm/final204µm, warm9.805ms101.99FPS. Full37: max539.3µm/final246.7µm (ground), warm18.212ms54.91FPS, p95 18.90ms. This improves speed but increases penetration relative to15×1; the PhysX reference final41.9µm remains more accurate. Reports /tmp/colibri_bilateral_fused{15,36}_parallelprep_8x1_300.json.

Actual runtime profiling correction: root verified ovphysx uses solveBlockUnified batches, not the possible whole-island source path. Split Phoenx profile: iterate21.63ms/frame, cachedwarm preparation4.64ms, parallelgeometry0.168ms. Remaining optimization target is solve scheduling/work distribution and contact manifold point count, not collision detection.

## Primary matched PhysX work budget

The primary comparison keeps120Hz collision updates and30 temporal steps per update (matching30 PhysX TGS position iterations). Full37 parallelprepare30×1 passes300frames (5s): max257.9µm, final38.81µm, warm61.846ms16.17FPS. Report /tmp/colibri_bilateral_fused36_parallelprep_matched30x1_300.json. This is the pre-two-anchor baseline; lower8/15-step timings are secondary tradeoffs, not equal-work speed comparisons.

NSYS resources: Phoenx hot fusediterate154registers/thread,0localbytes,1KBshared,256threads in ONEblock; PhysXsolveBlockUnified128registers/thread,0local/shared,64threads/block across21–24blocks. Cached Phoenxprepare102registers,0localbytes. No evidence of local-memory spills. PhysX actualcontactreport2103normalpoints/944frictionanchors/68headers is similar to Newton~2200points/77pairs, so large pointcount reduction does not explain speedgap.

PhysX count-scaled copy solver is momentum-compatible in principle: each of n copies uses n times original inverse mass/inertia; averaging accumulated copyvelocity changes by1/n gives exactly original inverse mass times totalphysicalimpulse. solveBlockTGS.cuh187–191 scalescontactmass/angularresponse,767–780 scalesjointresponse; solverMultiBlockTGS.cu395–410 averagesbyreciprocal active slab count. Consistent counts and common impulse locations are required. This motivates an opt-in Newtoncopy_state-based prototype, with physical masses unchanged and conservation tests before scene acceptance.

## Mass-split bilateral prototype validation

Opt-in install_fused(...,mass_splitting=True) now factors each joint using current copy_state.count_per_node weights and updates the same per-partition slots as existing joint/contact kernels. Existing dispatcher averaging restores physical body velocities. Original mass/inertia arrays are unchanged; finite-drive active-set corrections use the original physical-body block sweep after averaging.

21 checks pass:18 existing finite-drive/D6/contact+motor P/L tests forced toJacobi copies; unequalcopy root3/leaf1 with reversedallocation preserves all6momentum andnonincreasing energy; alternatingbalanced contact rows changescopycount1↔2 across12steps withoutmomentumleak; unequalcopy boundedmotor obeys actualeffort_limit×dt while producingnonzeroimpulse and preservingmomentum.

Heavy-light-heavy masses1000,1,1000 stillconvergepoorly: maxvelocityerror.4995/.4978/.4840m/s at1/8/64Jacobi sweeps, whilemomentumerror≤8.3e-5kgm/s andenergy500→484J. This remainsanopt-in route; direct/global strategiesare requiredforpoorlyconditionedjointchains. Report /tmp/colibri_bilateral_mass_split_heavy_light_heavy.json. Colibri mass-split stagedscreen pending.


## Matched mass-split stage screens (2026-09-15)

The opt-in bilateral copy-response path passed all three source stages for 300 frames (5 seconds), at 120 Hz collision updates and 30 temporal position steps per update with one sweep, SOR 1, and no ether damping. Configuration: `--fused --mass-splitting --max-colored-partitions 8 --mass-splitting-batch-size 8`; parallel preparation was disabled because its current prototype only supports unsplit contacts. Physical masses and inertias were unchanged.

| Stage | Maximum fresh depth | Final fresh depth | Warm physics time | FPS |
| --- | ---: | ---: | ---: | ---: |
| 8 | 76.8 µm | 13.0 µm | 19.92 ms | 50.19 |
| 15 | 76.8 µm | 41.9 µm | 48.37 ms | 20.67 |
| Full 37 | 290.3 µm | 32.9 µm | 101.69 ms | 9.83 |

Reports: `/tmp/colibri_bilateral_ms8_stage8_30x1_300.json`, `/tmp/colibri_bilateral_ms8_stage15_30x1_300.json`, `/tmp/colibri_bilateral_ms8_full_30x1_300.json`. Full final cached solve had 2,148 generated points, 39 solver columns, 65 nonzero normal impulses and 54 nonzero tangent impulses. This active-normal sparsity is already comparable to the PhysX report (59 of 2,103); dense nonzero force distribution is not the measured primary difference.

The route is stable in these screens but slower than the accepted unsplit parallel-preparation run (~61 ms/frame). This does not isolate mass-splitting overhead because preparation architecture also differs. Actual active copy and overflow counts must be recorded before attributing performance to the copied-state path: small stages may fit entirely in eight colored partitions. Keep this opt-in; no default strategy change.


### PhysX solver descriptor granularity

The GPU source processes one contact patch per active solver lane, not an entire contact-report shape-pair header. `physx/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu:884` reads `partitionIndexData[uniqueIndex].mPatchIndex`; lines985–997 select that patch and assert `nbContacts <= PXG_MAX_NUM_POINTS_PER_CONTACT_PATCH`. `physx/source/gpucommon/include/PxgCommonDefines.h:36` defines that maximum as6. The same descriptor stores its partition/slab index before entering `solveBlockUnified`.

Thus the measured 2,103 PhysX contact points require at least351 six-point descriptors (a source-derived lower bound, not a directly measured active-descriptor count). The68 contact-report pair headers cannot be used as the solver work-item count. Newton's final full-scene copied-state screen grouped2,148 points into39 columns, about55 points per column. This establishes a material granularity difference independent of contact reduction or normal-impulse sparsity. Any smaller-column prototype must keep all physical rows and validate momentum, energy, and finite-iteration convergence; splitting columns changes solve ordering and is not an exact equivalence claim.


### Mass-split phase attribution and chunk prototype entry tests

`/tmp/colibri_ms8_phase_profile.json` profiles one eager full-scene frame at the matched30×1 schedule:49 active copy slots, at most2 copies/body,9 colors,19 overflow constraints in only3 batches. The two60-call fused preparation/iteration phases cost46.45+42.26ms; averaging122calls costs7.03ms. Total instrumented GPU kernel time108.12ms. This is attribution, not an FPS measurement.

The opt-in `local_studies/colibri/contact_chunks.py` reserves a column per possible point, then splits already-ingested contiguous contact ranges into at most6 points. It preserves material grouping, all point arrays/history indices, and updates current per-point column IDs. It does not implement PhysX friction-patch reduction or discard points. A box-face regression splits4points into4 independently copied columns and checks full point coverage, reversed body allocation, all6 momentum components at2e-6 absolute tolerance, and nonincreasing kinetic energy; it passes. All18 existing fused drive/conservation assertions also pass with six-point chunks and forced Jacobi copies. Source-scene acceptance remains pending; smaller chunks change finite-iteration ordering/convergence.


### Chunking exposed equal-priority graph-coloring races

The first six-point stage8 screen failed at frame39; its range splitter omitted the expanded pair-source metadata. Correcting that map and adding a multi-pair/recycling coverage regression moved failure to frame89 but did not resolve it. An explicit colored-endpoint audit then failed at frame1: color1 contained constraints7 and8 sharing dynamic FrameGround, with only one copy/body and no overflow. These failed screens are invalid convergence comparisons.

Root cause: all JP, greedy, speculative, and endpoint-owner algorithms treated equal packed priorities as simultaneous winners. Chunks from the same pair share both contact-count cost and shape hash. The new repeated-edge regression failed before the fix for all four algorithms (15–16 adjacent constraints in one color). The fix imposes deterministic (priority, element-ID) ordering; endpoint-owner uses a uint64 key preserving its previous unsigned priority ordering, while other algorithms preserve their previous signed ordering. No contact rows, physical parameters, or convergence tolerances changed.

The regression and five existing speculative checks pass, as do37 existing overflow/general/warm-start checks. CPU multi-pair metadata/recycling coverage and box-face momentum/energy checks pass. Source chunking must be screened again after the coloring fix. Production graph source is frozen for root's baseline launch-size comparison.


### Corrected six-point chunk stages

After fixing graph priority ties, the point-owner audit exposed a second local metadata issue: warm-start gather restamped original pair offsets after the splitter ran. The prototype now restamps final chunk owners after gather. A compound-body box-face fixture explicitly exercises this path and verifies pair ranges, exact ownership, momentum and energy. The earlier one-frame artifact with an empty assertion was explicitly relabeled failed; the runner now treats any non-None failure as failure.

All corrected stages passed300frames with zero shared dynamic endpoints within colored partitions and exact point-to-column ownership checked every frame:

| Stage | Maximum fresh penetration | Final penetration | Diagnostic warm time |
| --- | ---: | ---: | ---: |
|8|372.7µm|0µm|20.68ms|
|15|439.8µm|14.1µm|28.53ms|
|Full37|110.0µm|73.4µm|47.65ms|

The full scene had369 columns for2,124points,302 overflow constraints across38batches,156copy slots and at most19copies/body. This is actual finer-grained parallel work, unlike the original39-column route. Timing is diagnostic rather than a controlled A/B, and includes the current normal-first contact optimization. Settings remain120Hz collision updates,30temporalsteps/update, SOR1, no ether damping, authored contact offsets plus1mm fallback. These are not the measured automatic PhysX offsets. Reports use `/tmp/colibri_chunks6_tiefix_restamp_{stage8,stage15,full}_30x1_300.json`. A20-second full-scene robustness screen is in progress.


The corrected full scene subsequently passed1,200frames (20 simulated seconds) with all per-frame endpoint/point-owner/joint/fresh-depth checks enabled. Maximum fresh penetration260.6µm; final51.6µm. Diagnostic46.93ms/frame, with small peer correctness checks allowed to overlap, so not an isolated timing claim. Report: `/tmp/colibri_chunks6_tiefix_restamp_full_30x1_1200.json`. The baseline gap distinction remains: authored values plus1mm fallback, not measured automatic PhysX contact offsets.


### Bounded batch-size comparison on the corrected chunk route

Keeping30×1/120Hz, six-point chunks, eight colored partitions, source-authored offsets plus1mm fallback, and frozen production equations:

| Overflow batch size | Stage8 (300frames) | Stage15 (300frames) | Full37 |
| --- | --- | --- | --- |
|2|PASS, max156.0µm|PASS, max76.8µm|PASS300, max198.8µm, final14.1µm; diagnostic28.48ms|
|1|PASS, max76.8µm|PASS, max261.3µm|FAIL frame30: gear/hypocycloid joint anchor2.093mm|

Full batch2 had380 columns,157 overflow batches,384 copy slots and at most74 copies/body. Full batch1 had311 overflow batches,655 slots and at most150 copies/body. Batch1 had no preceding coloring or point-owner audit failures; the failing invariant was joint anchor accuracy, while fresh contact penetration remained28µm at that frame. Finer copied-state parallelism therefore has a real finite-iteration accuracy limit; retain batch2 as the best passing5-second candidate and batch8 as the20-second baseline. No damping, SOR, mass, contact-row, or tolerance concessions were introduced. Reports: `/tmp/colibri_chunks6_batch{2,1}_{stage8,stage15,full}_30x1_300.json`.


Batch2 also passed1,200frames/20seconds with all race/ownership/joint/depth audits: maximum305.7µm, final61.1µm. Report `/tmp/colibri_chunks6_batch2_full_30x1_1200.json`. Its48ms concurrent-run timing is not comparable to the earlier isolated~28ms result. This extends robustness evidence for the authored+1mm-fallback configuration; exact-offset predictive-search combination remains pending independent validation.


## Combined exact-offset predictive-search acceptance

Independent tests first passed20seconds for both the corrected chunk6/batch2 route and the exact-offset velocity-predictive collision search. The combined full37-body configuration then passed1,200frames/20seconds with every-frame coloring exclusivity, point ownership, joint, fresh-penetration, and unchanged-physical-gap assertions enabled.

- Exact per-shape contact offsets: `/tmp/colibri_physx_exact_contact_offsets_m.json`.
- Collision refresh120Hz;30temporalsteps/update; one sweep; SOR1; no ether damping.
- Six-point chunks; eight colored partitions; overflow batch size2.
- Velocity-derived search extension capped at5mm; physical shape gaps remain unchanged.
- Maximum fresh penetration156.1µm; final60.9µm.
- Maximum cached contacts1,860 of8,192; final303columns/122overflow batches, at most69copies/body.
- Diagnostic27.70ms/frame; an isolated fair timing/profile remains the parent agent's next task.

Report: `/tmp/colibri_chunks6_batch2_exact_predictive_30x1_1200.json`. Reproduce the complete audit by importing `local_studies.colibri.check_chunk_coloring` before running `local_studies.colibri.check_speculative_tail` through `runpy`, with `--search-cap 0.005 --body-count 36 --frames 1200 --substeps 30 --iterations 1 --fused --mass-splitting --max-colored-partitions 8 --mass-splitting-batch-size 2 --contact-chunk-size 6 --contact-offset-map /tmp/colibri_physx_exact_contact_offsets_m.json`.

This opt-in local PGS route is not a replacement for the direct/reduced strategy on high-mass-ratio mechanisms: the heavy-light-heavy chain convergence limitation remains documented above.


## Joint-first priority diagnostic

A local wrapper assigns enabled bilateral rows coloring cost 127, preserving their random tie-break field; six-point contact chunks retain cost 6. Physical rows, masses, impulses, and eight-color budget are unchanged. This changes finite-iteration ordering, so it is a convergence experiment rather than an equivalent optimization. Eighteen drive and momentum assertions pass with batch size 1. Stage 8 passes 300 frames with continuous coloring and ownership audits; report `/tmp/colibri_jointfirst_batch1_stage8_300.json`. Stage 15 rejects this candidate at frame 15: Frame/HummerBody anchor separation 2.258 mm, with fresh contact depth only 23.29 µm and no coloring or ownership failure. The ordinary-priority batch-size-1 stage 15 control passed 300 frames. No full-scene run is justified; prioritizing joints does not repair averaging-induced convergence loss. Report `/tmp/colibri_jointfirst_batch1_stage15_300.json`. The accepted batch-size-2 configuration remains unchanged.


## Local graph-color slab reference

The saved full graph has 345 active rows. Greedy complete coloring requires 148 colors. Grouping eight sequential colors per independent slab gives 19 slabs, 115 body-copy slots with cid order or 122 with current CSR visitation order, and a maximum of 19 copies per body. Current batch-2 scheduling uses 306 slots and a maximum of 71 copies. These are work-count estimates, not performance or convergence results. `/tmp/colibri_full_slab_schedule.json` records the width-4/8/16 estimates.

Three CPU tests verify exact row coverage, color exclusivity, rebuilding after topology shrink, unequal-copy momentum and energy, and single-slab equivalence to sequential projection. A CUDA spatial reference tests shared-point contact rows and bounded internal torque across widths 1, 2, and 8. Its one-block-per-slab implementation is array-identical to the serial-per-slab reference; all six momentum components pass an absolute 3e-6 tolerance, passive energy does not increase, and the motor obeys its impulse bound. This is an isolated row reference, not yet the actual Phoenx contact/joint dispatch.

The high-mass 1000:1:1000 chain still converges slowly: 64 row sweeps with one shared slab leave a 0.440 m/s velocity error. Direct equality solving remains necessary for that regime. `/tmp/colibri_slab_heavy_light_heavy.json` records this limitation. No production defaults or kernels changed.


### Actual callback slab adapter

`slab_adapter.py` now executes the actual rigid Phoenx prepare/iterate/relax callbacks in one block per slab, with a barrier between internal colors. It replaces only the eager local copy graph and row-to-copy lookup, retains existing endpoint-specific slot-cache code, and uses the existing averaging/writeback stages. CPU complete coloring and explicit device-array assignment make this a correctness prototype; CUDA graph capture is intentionally rejected.

`python -m local_studies.colibri.test_slab_adapter` passes nine tests: the three existing mass-split fixtures at slab widths 1, 2 and 8. Six-momentum, passive energy, and bounded-drive impulse assertions are unchanged. Expected copy counts are adapted to slab grouping. Reversing body order and changing active self-contact/history are covered. No production solver defaults changed and no full-scene slab acceptance has been claimed.

## Predictive normal-family coverage repair

The captured slab trace at outer step 746 contains 76 raw witnesses already
predicted to close during the 120 Hz update, but the old seven-slot predictive
reducer retains none of their gear flank. Reusing existing icosahedral normal-bin
depth slots for eligible predictive candidates retains that flank: 18 reduced
contacts include a witness transported to -2.754 mm in the captured failed motion,
versus 16 old contacts all above +2.001 mm. This is a coverage diagnostic, not
a claim of accepting penetration. Deeper regular contacts retain priority;
capacity and physical offsets are unchanged.

The generic closing-flank test failed all 12 CPU/CUDA, fast/deterministic,
rotation cases before the fix; all pass afterward, including replacing the
predictive normal winner with a deeper physical witness and re-submitting it.
Eight coarse octant guards were investigated and rejected: a nearly axial
normal won over the flank within the same octant. The final fix uses existing
20-bin resolution and adds no slots. Outer 745 still lacks a future penetrating
feature even in raw prediction, so that earlier coverage issue remains open.

Frozen-input check: local_studies.colibri.probe_slab_tail_normal_bins.
Reports use /tmp/colibri_slab_tail_normal_bins_*; original fail-before data
remain /tmp/colibri_slab_tail_*.

### Follow-up validation

All 32 predictive reduction tests and three actual mesh-SDF tests pass unchanged.
The new closing-normal fixture also verifies a deeper physical contact replaces
the predictive depth winner, which cannot reclaim that slot on re-submission.
The full slab8 scene still fails at frame448 with Rack/MountRight penetration
1.3705 mm, despite clean schedule and ownership audits. Slabs remain experimental.

Actual pre-generation SDF queries at the material witnesses of fresh post745
penetration show they are initially receding: Rack-target distance slopes are
approximately +0.062 to +0.105 m/s; reverse Pinion-target slopes are +0.091 to
+0.100 m/s. Texture and exact triangle distance agree within about7 micrometers.
These initial-twist predictions cannot anticipate the subsequent reversal.
The one initially closing Rack-target sample is not penetrating at the end.
The earlier745 omission therefore differs from the confirmed746 reduction bug.

Query source: local_studies/colibri/query_slab_pre_sdf.py.
Data: /tmp/colibri_slab_745_sdf_queries.json.
Slab result: /tmp/colibri_slabs8_normalcoverage_full_1200.json.
