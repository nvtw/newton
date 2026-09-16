# Current integration regression plan

The integrated 300-second gate passed at 98.6414 FPS with all ten trajectory
arrays byte-identical. The rigid contact/conservation stage below also passed:
16 isolated modules, 66 tests, no skips or expected failures, with full Newton
Python-source and Colibri-asset hashes unchanged. Reports:
`/tmp/colibri_integrated_rigid_regressions.json` and `.summary.json`; per-module
logs retain every unittest result. The geometry/candidate-selection stage also passed all 18 tests in eight
isolated modules without skips or source/asset changes; see
`/tmp/colibri_integrated_geometry_regressions.json` and `.summary.json`.
The joint/drive/response batch also passed: 108 tests in nineteen isolated
modules; `/tmp/colibri_integrated_joint_regressions.json` and `.summary.json`
record all individual summaries and unchanged source/asset hashes. Other stages remain pending
except the explicitly completed preparation/CPU/Kapla/G1 gates. Keep production frozen
during comparisons. No full pre-commit run has been started.

The local `run_regression_modules` command runs one subprocess per module,
records exit codes/counts/summaries/source hashes, and stops on failure by
default. It accepts an `@FILE` containing one module per line. Reproduce the
completed rigid stage with:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.run_regression_modules --output /tmp/colibri_integrated_rigid_regressions.json @/tmp/colibri_rigid_regression_modules.txt
```

## Execution

Use one subprocess per module to isolate CUDA capture/module state. For each suffix below:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m unittest -v newton._src.solvers.phoenx.tests.TEST_MODULE
```

Geometry tests use `newton.tests.TEST_MODULE`. Preserve individual exit codes, test counts, skips, expected failures, and logs. First-time reduced fallback compilation previously took several minutes (12 tests completed in roughly 491 seconds); silence during compilation is not a failed or stalled numerical test. Do not overlap performance measurements with these runs.

## 1. Collision geometry and candidate selection

Run newton.tests modules: `test_contact_sdf_gradients`, `test_sdf_contact_coverage`, `test_predictive_contact_normals`, `test_predictive_normal_coverage`, `test_geometric_candidate_admission`, `test_spatial_contact_priority`.

These cover analytical local-cell derivatives, mesh endpoint coverage, normalization/reuse consistency, predictive normal coverage, geometric candidates with unchanged physical gaps, dormant/reversing contacts, and spatial reducer priority. Run PhoenX `test_geometric_mesh_activation` and `test_contact_ingest_large_shape_count` too. The public 300-second replay alone cannot exercise all primitive/convex/raw/reduced collision paths.

## 2. Joint preparation, drives, and articulation response

PhoenX modules: `test_bilateral_preparation`, `test_block_joint_policy`, `test_inactive_joint_prepare`, `test_cooperative_joint_rhs`, `test_direct_drive`, `test_d6_direct_drive`, `test_direct_joint_types`, `test_direct_equality`, `test_direct_setup`, `test_armature_live_update`, `test_joint_friction`, `test_joint_modes`, `test_reduced_factor_policy`, `test_reduced_response_refresh`, `test_reduced_contact_nullspace`, `test_reduced_speculative_contacts`, `test_reduced_forest`, `test_direct_contact_nullspace`, `test_direct_contact_coupling`, `test_maximal_contact_mobility`, `test_maximal_contact_conservation`.

Preparation and block-policy six-test gate already passes after cooperative integration; actual CPU scalar compilation/output smoke also passes. Broader rerun still needed for joint property transitions, finite effort/velocity/position limits, armature/kinematics, kinematic-body ownership, current-pose response refresh, and fallback/loop consumers. No caching of physically necessary velocity-dependent mass factors is authorized by a speed regression result.

## 3. Rigid contact law, physical conservation, and history

PhoenX modules: `test_contact_coupling`, `test_contact_relax_momentum`, `test_rigid_normal_first`, `test_rigid_frictionless_fastpath`, `test_rigid_contact_bias`, `test_rigid_contact_energy`, `test_rigid_friction_anchors`, `test_rigid_speculative_contacts`, `test_rigid_speculative_momentum`, `test_rigid_split_prepare`, `test_contact_patch_friction`, `test_contact_coefficients`, `test_contact_force`, `test_momentum_conservation`, `test_energy_conservation`, `test_kinematic`.

Important distinct regressions: packed relaxation must preserve live solve impulses without an extra gather; free-pair relaxed impulses must share one world point; mixed normal/tangent mobility must obey the existing non-associated Coulomb law. Keep all six momentum components, physical work/energy, and common-point lever checks. Small residual alone is not a substitute for these gates.

## 4. Grouping, copy ownership, and dispatch

PhoenX modules: `test_color_groups`, `test_color_group_grid`, `test_color_group_policy`, `test_color_group_conservation`, `test_rigid_color_rows`, `test_color_first_fit`, `test_graph_coloring_priority_ties`, `test_contact_chunks`, `test_compound_contact_grouping`, `test_partition_slot_cache`, `test_contact_prepare_grid`, `test_colored_contact_rows`, `test_block_copy_round_trips`, `test_tail_first_dispatch`, `test_velocity_relaxation_schedule`, `test_multi_world_contact_isolation`, `test_multi_world_pgs_order`.

Also all five modules under `newton._src.solvers.phoenx.mass_splitting.tests`: `test_access`, `test_copy_state`, `test_interaction_graph`, `test_broadcast_average`, `test_solver_integration`.

The three-body/two-manifold `test_color_group_conservation` explicitly covers unequal copy counts, moving/rotated COMs, actual biased and relaxed callbacks, and averaging. Tail-first tests include a real hanging cloth contact, not only mocked scheduling. Slot-cache tests independently verify natural endpoint mapping and explicit partitions.

Expected rejected configurations are PASS only when the specified exception occurs: block policy unknown/reduced/multiworld/patch combinations; invalid group sizes, grouping without mass splitting, unrolled/sleeping combinations, SOR other than one; invalid chunk sizes and incompatible contact modes. Do not label these unsupported-scope checks as skipped physics tests.

## 5. Existing general-scene and unsupported-path coverage

PhoenX modules: `test_kapla_contact_regression`, `test_stack_stability`, `test_high_mass_ratio`, `test_cloth_contact_partitioner`, `test_cloth_mass_splitting`, `test_cloth_mass_splitting_determinism`, `test_soft_body_mass_splitting_momentum`, `test_soft_body_mass_splitting_determinism`, `test_multi_world`.

Current scoped integration already preserves Kapla180 six saved arrays and learned G1 twenty-frame body_q/body_qd bytes. These short controls do not replace the suites above. Add the existing two-second learned policy and production reduced environment screens if reduced-side changes need a refreshed broader baseline; use the original controller settings and report physical time, not display time.

## Known limitations and baseline failures

- G1 `test_robot_policy_stiffness`: armature finite-state passes both revisions, standing threshold fails both old/current (0.0553663/0.0528800 rad against 0.04). Keep as baseline failure, not a newly accepted result. Separate MuJoCo/ankle parity tests already carry expectedFailure. See G1_REGRESSIONS.md.
- Four-world reduced open-loop pose hold falls on both revisions around control frame64; this is not the learned-controller test. Do not use finite short states as a standing guarantee.
- High-mass 400:1, 8x8-budget quality remains unresolved. Existing `test_high_mass_ratio` uses a different 100:1 stack and 40x32 budget; a pass does not close the observed 400:1 RMS regression. CPU nonlinear experiments remain local and unpromoted.
- Earlier contact-admission prototype rotated-box coverage failure remains historical negative evidence; canonical geometric-admission focused tests and current source trajectories are the relevant implemented-policy evidence. Do not promote local support-star or nonlinear oracle variants implicitly.
- Whole-solver CPU simulation is not claimed. Actual scalar preparation CPU smoke is scoped to module compilation and valid/ragged/invalid factors; most full solver tests legitimately require CUDA graph capture.

## Final broad sweep

Only after targeted failures are understood, use the existing direct `test_run_all` entry point for isolated modules and timing reports. It discovers many training, RL, parity, and benchmark modules beyond this change, so it is not the first bounded readiness gate. Run public example registration/final tests plus `newton.tests.test_colibri_assembly_bounds` and `test_viewer_packed_upload` for the example/viewer changes. Before committing, run the required pre-commit suite only after source freeze is released; formatting may change files and invalidate trajectory source hashes.

## Grouping gate fixture correction

The first grouping batch passed 13 modules (31 tests), then stopped at
`test_tail_first_dispatch`: seven subcase errors came from its synthetic
`SimpleNamespace` lacking `_color_group_data`. The production world initializes
this field; the test now explicitly selects the ungrouped path with `None`.
No production solver code or acceptance threshold changed. The original failure
is preserved in `/tmp/colibri_integrated_grouping_regressions.json` and its
module logs. All four tail-first tests then passed in 1.234 seconds, including
the real cloth case (`/tmp/colibri_tail_first_fixture_fixed.log`).

The remaining nine modules, including the corrected fixture, are running under
`/tmp/colibri_integrated_grouping_remaining.json`; completion is not yet claimed.

The follow-up grouping run completed: 21 of 22 modules pass (82 tests in
passing modules). The remaining mass-splitting integration module ran 11
tests with four errors, all from the obsolete positional `add_shape_plane`
transform argument. Its fixture also needs inspection for physical mass
initialization and continuous state evolution before accepting a repaired
pass. The failure is preserved in
`/tmp/colibri_integrated_grouping_remaining_08_test_solver_integration.log`;
consolidated evidence is `/tmp/colibri_integrated_grouping_summary.json`.
The grouping gate remains incomplete.

## Completed fixture repair and remaining general-scene failure

The repaired mass-splitting integration module passes11/11 tests in3.920s.
Negative controls detect both original defects: omitted mass initialization
and resetting state every frame. It now verifies cumulative motion, actual
contact response, captured/eager agreement, and a displaced pendulum.
Only the fixture changed; no production solver changes or looser thresholds.
Evidence: `/tmp/mass_splitting_fixture_negative_controls.json` and the
agent's fixture94715 test result. Grouping modules therefore have82 earlier
passing tests plus11 repaired integration tests; retain both original failed
reports rather than overwriting them.

General-stage run `/tmp/colibri_integrated_general_regressions.json` passes
Kapla4, stack4, high-mass4, and cloth partitioner1 tests. It then stops on
`test_cloth_mass_splitting`: the split-enabled cube falls below the cloth
(cube z0.3166m; cloth mean z1.9929m), while the disabled control passes.
This is a real unresolved regression until a source-controlled comparison
shows otherwise. No failed assertion is removed. Remaining general-stage
modules have not run. The suite's100:1 high-mass pass still does not close
the separate400:1 low-budget convergence problem.

CPU-only public example/bounds and packed-viewer tests also pass6/6, log
`/tmp/colibri_public_cpu_regressions.log`.

## Overflow-joint response regression

Added test_overflow_joint_factor_matches_unequal_copy_response to the existing
newton._src.solvers.phoenx.tests.test_bilateral_preparation module. It uses
native factor/iterate/average paths, unequal endpoint copies, and an independent
FP64 impulse/velocity reference with momentum accounting. An intentional
count-one factor control is rejected; altered state is restored in finally.
All four module tests pass through actual unittest discovery; Ruff passes.
Logs: /tmp/production_overflow_factor_regression.log and
/tmp/production_bilateral_preparation_discovery.log.

Local prototype suite additionally forces a joint into overflow with counts
[5,1], checks warm-history and phase P/L plus support reactions and relaxation
energy. Both tests pass; missing-cache clone fails with counts[1,1].
Evidence: /tmp/physical_head_overflow_joint_regression.json.
Run local tests without extra environment configuration:
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m unittest local_studies.colibri.test_physical_head_physics -v
Production solver source is unchanged. These tests protect response accounting;
they do not establish full-scene stability. Sequential-overflow Colibri control
is the next diagnostic.
