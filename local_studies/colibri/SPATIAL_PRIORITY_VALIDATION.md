# Overlap-only spatial priority validation

The mesh/SDF reducer now separates spatial priority from depth/voxel coverage.
Actual overlaps retain priority. Separated points inside the contact search
range compete on spatial projection, including points farther than one voxel
radius. The existing six spatial slots, inner depth/voxel eligibility,
physical offsets, geometry, filters, and solver equations are unchanged.
The legacy two-depth wrapper retains its prior behavior.

## Gates completed

- Eight focused CPU/CUDA, fast/deterministic, insertion-order cases fail
  behaviorally before the fix and pass afterward.
- All 63 existing reducer/predictive tests pass.
- Captured update9291 retains two previously lost geometric witnesses.
  Their minimum transported gap is -949.184um; previously no retained
  witness transported below -41.056um. Selection uses only current geometry.
- Clean330-frame controls for final relaxation counts1 and4 pass all public
  bounds. Each repeat matches all ten saved arrays byte for byte.
- Kapla180 matches six reference arrays; G1's20-step controller history
  matches both reference arrays. These are bounded regression checks.
- CPU and GPU stress each pass96 cases,1024 candidates, six orders,
  reclamation on/off, both reduction modes and nonzero margin translation.
- Three mixed joint/contact and relaxation conservation tests pass.

## Baseline final-relaxation count1: 300 seconds PASS

All18000 frames pass at30 substeps per120Hz update, one biased iteration,
SOR1, no global damping, group size4 and geometric candidate admission.
Production source and scene assets remain hash-identical throughout.

| Metric | Result |
| --- | ---: |
| Maximum penetration |0.762952mm|
| Maximum joint-anchor error |0.794343mm|
| Maximum joint-axis error |0.0289467rad|
| Mean physics frame time |11.174356ms|
| p95 physics frame time |12.353267ms|
| Physics FPS |89.4906|
| Mean crank speed |−3.183971rad/s|
| Crank target |−3.490659rad/s|

Timing synchronizes the public physics step, excludes rendering and audits,
and uses17970 frames after30 warmups. The other reducer test job finished
around launch; later compute-process inspection shows only this run. No
paired speedup claim is made versus the earlier90.2295FPS baseline.

The axis maximum remains close to the existing0.03rad limit. The free assembly
still drifts; maximum final body displacement is0.5589m, while the existing
assembly-relative escape test passes. Drive lag and convergence remain open.
This pass does not establish indefinite stability or validate extra relaxation.
The corrected count4 traced run fails at248.666667s on a different pair:
GearLarge02/GearLarge01 fresh penetration1.709mm. Joint errors at the failed
pose remain below limits (anchor0.633mm,axis0.007054rad). All ring completeness
checks and the330-frame reference prefix pass, with unchanged source/assets.
The clean14920-frame control reproduces all ten arrays byte for byte,
including all14919 successful pose/velocity histories and the failed final
pose. Failure, time and source/asset hashes also match. This establishes
faithful late-event tracing; it does not validate extra relaxation. The clean
failed candidate costs11.4615ms/frame (87.2484FPS), not a stability-qualified
performance result.

Raw generation witnesses also omit the future closing direction. Independent
CPU triangle/winding queries show all62 later deep samples initially OUTSIDE
the opposing gear, at1.5377–3.3827mm. Reconstructed per-shape velocity-derived
search gaps sum to1.20724mm. The deepest eventual point initially has exact
clearance1.64101mm and only−0.4976mm/s normal approach, but later exact depth
is1.84723mm (stored SDF depth1.70854mm). GearLarge01 rotates0.108679rad in the
update, far beyond its initial-twist prediction. This is evidence of search
envelope escape under changing velocity, not another proven reducer loss.
A local wider computational search now passes the bounded 300-second control
below; authored physical gaps, drives and solver equations remain unchanged.


## Local 2 mm search floor, final-relaxation count4: 300 seconds PASS

This is a process-local search experiment, not a production API or default.
After the existing shape-velocity calculation, each dynamic shape receives
`search_gap = max(search_gap, physical_gap + 0.002 m)`. The configured maximum
velocity-derived extension remains 5 mm per shape. Static shapes are unchanged.
The existing geometric AABB expansion then encloses the increased search gap.
The physical activation gap, margins, state velocities and contact equations
are not modified. The implementation adds two GPU kernels per rendered frame;
count4 also adds six final relaxation sweeps per frame relative to count1.

The frozen failed-update control verifies actual search arrays and AABB padding.
Raw candidate count grows from 53,925 to 114,137, below its diagnostic capacity
of 262,144. The gear pair grows from 176 to 837 raw points and 27 to 48 reduced
points. It retains seven reduced witnesses that transport below -0.5 mm at the
failed post-pose, versus none without the floor. This is coverage evidence,
not a guarantee against arbitrary acceleration.

| Configuration, 30 substeps/120 Hz | Result | Physics FPS | Peak depth | Peak anchor | Peak axis |
| --- | --- | ---: | ---: | ---: | ---: |
| Count1, no floor | PASS 300 s | 89.4906 | 0.762952 mm | 0.794343 mm | 0.0289467 rad |
| Count4, no floor | FAIL 248.6667 s | 87.2484* | 1.70854 mm failure | 0.633 mm at failure | 0.007054 rad at failure |
| Count4, local 2 mm floor | PASS 300 s | 81.5803 | 0.708282 mm | 1.39849 mm | 0.0236303 rad |

*The failed run's speed is not stability-qualified. These separate long runs
are not a paired performance-isolation study. The floor/count4 pass averages
12.257861 ms per physics frame (p95 13.306355 ms), over 17,970 samples after
30 warmups. More search candidates also change copy grouping and finite-sweep
convergence: peaks are 3,869 live contacts, 668 contact columns, 64 copies on
one body, and 394 total body copies. The cost is not just the extra kernels.

The independent CPU integrity audit passes all 18 checks: 18,000 finite,
ordered pose/velocity samples; final state matching the final history sample;
all first330 q/qd/time samples byte-identical to the short floor/count4 pilot;
initial poses, labels, masses, inertias and COMs byte-identical to the no-floor
baseline. All 607 saved source/asset hashes still match disk and the pilot.
The live harness separately asserts byte-unchanged physical gaps/margins and
unchanged collision source; gap arrays are not independently stored in the
trajectory NPZ. No source or physics-integrity failure was found.

Motion continues: crank mean speed is -3.180034 rad/s versus target
-3.490659 rad/s, with 151.827 turns and no sampled near-zero crank speeds.
The free assembly still drifts (maximum final body displacement 0.561594 m).
Neither drive lag nor static-support accuracy is solved. Count1 with the floor
has only the 330-frame pilot so far; its 300-second behavior is not established.

Artifacts share prefix `/tmp/colibri_search_floor2mm_velocity4_long18000`:
`.json`, `.npz`, `.source.json`, `.floor.json`, and `.integrity_audit.json`.
The frozen control is `/tmp/colibri_gear_search_floor.json`.

### Generic CPU force-activation fixture

The local `test_search_floor_force_activation.py` reproducer passes without
textures, mesh assets or GPU execution. Two equal dynamic spheres begin at rest,
1 mm apart, with 0.1 mm authored gap each. Their contact axis is diagonal and
the assembly is translated away from the origin. Collision candidates are
created once; after two resting substeps, actual equal/opposite central forces
apply a one-substep pulse. Velocities are never reassigned.

Three independent controls distinguish detection from physical response:

- No floor generates no candidate and ends with 8.9999 mm penetration.
- A 2 mm floor without the pulse generates one candidate but leaves every
  contact impulse, velocity and pose exactly unchanged throughout eight steps.
- The floor plus pulse retains contact and stops at a +5.96 nm gap. The stored
  normal/tangent impulse reproduces each body's momentum change after subtracting
  the known force impulse. Linear momentum remains zero; maximum angular error
  is 1.24e-8 N m s. Contact midpoint work is -0.3351032257 J, accounting for the
  pulse's 0.3351032257 J kinetic kick; final energy is 1.71e-11 J.

The physical response uses the actual production `SolverPhoenX.step` kernels
(single-world, one substep per call, eight solver iterations, SOR1). There is
no custom contact projection or velocity update. Production `CollisionPipeline`
uses geometric admission, sticky matching and the same shape-velocity/search/AABB
route as Colibri; only the local floor insertion is substituted. Its narrowphase
is analytic sphere-sphere, not mesh/SDF reduction. Thus it proves common candidate
versus activation semantics and production impulse response, but does not test
mesh feature generation, reducer capacity or grouped-copy convergence.

The fixture checks all six momentum components, paired impulse response, the
contact-work identity, no positive contact work beyond FP32 tolerance, and
unchanged physical gaps/margins. It is a local CPU proof, not yet a canonical
API regression or a GPU/torque-driven coverage test. Artifact:
`/tmp/search_floor_force_activation_cpu.json`.

### Minimum remaining gates for an optional search-floor API

1. Define a disabled-by-default, finite nonnegative per-shape search-extension
   floor, with explicit cap and geometric-admission requirements. Test invalid
   values, zero floor, static shapes, and calls without a prediction horizon.
   It must never alter authored physical offsets or fabricate velocities.
2. Replace the local launch interception with normal pipeline/kernel arguments.
   Verify CPU/CUDA and captured/eager equivalence to this local control, all
   search/AABB writer paths, deterministic repeated contact output, and zero-floor
   byte equivalence to current defaults. Retain the frozen29839 fail-before
   coverage proof and capacity checks.
3. Exercise a small force-driven collision with actual contact activation,
   stopping, momentum and work/energy checks. An expanded search must retain
   witnesses without applying premature physical impulses or losing existing
   penetrating support.
4. Revalidate the canonical positive-floor implementation against this local
   300-second trajectory and run bounded default-disabled Kapla/G1 checks.
   Before any Colibri default adoption, validate the intended final-sweep count
   for 300 seconds and assess contact/copy growth and cost. Optional availability
   does not imply an acceleration-independent collision guarantee.

Reproduce the local long control:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.check_live_search_floor --velocity-iterations 4 --frames 18000 --output /tmp/colibri_search_floor2mm_velocity4_long18000
```

## Reproduce

From the PhoenX worktree, use the project interpreter through uv:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.validate_spatial_priority --output /tmp/colibri_spatial_priority_validation.json
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.check_clean_velocity_iterations --velocity-iterations 1 --frames 18000 --output /tmp/colibri_spatial_priority_velocity1_long18000
```

Artifacts: /tmp/colibri_spatial_priority_validation.json,
/tmp/colibri_spatial_priority_short_summary.json,
/tmp/colibri_spatial_priority_velocity1_long18000.json/.npz/.source.json,
/tmp/colibri_velocity4_priority_fixed_midpoint_coverage.json,
/tmp/spatial_priority_concurrency_cpu.json,
/tmp/spatial_priority_concurrency_cuda.json,
/tmp/colibri_spatial_priority_conservation.log.

Late-event evidence: /tmp/colibri_spatial_priority_velocity4_clean_trace_equivalence.json,
/tmp/colibri_spatial_priority_velocity4_trace18000.query.json,
/tmp/colibri_spatial_priority_velocity4_trace18000.exact_geometry.json.
