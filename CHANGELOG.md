# Changelog

## [Unreleased]

### Added

- Add opt-in averaged double-Q targets and actor objectives to experimental `ConfigSAC`, matching the FastSAC estimator used for massively parallel humanoid training.
- Add experimental `SolverPhoenX(contact_friction_model="patch")` for maximal rigid PGS scenes. It preserves every point normal while coupling convex shape-pair friction into one central 2D Coulomb block with full tangent effective-mass coupling, contact-matched world-space warm starting, CUDA graph capture, and conservative point-friction fallback for raw meshes, heightfields, and compound body-pair columns.
- Add `SolverPhoenX(solver_flavor="simple")`, an experimental rigid-body flavor that replaces graph-colored block PGS with fixed-capacity, one-thread-per-scalar-equation Jacobi rows and atomic body-delta accumulation and copy-free mass splitting that normalizes arbitrary row fan-in while conserving momentum. `jacobi_max_colors` defaults to 10 and scales the effective Jacobi substep count without constructing a constraint graph. Matched normal and tangent contact multipliers, together with stable Cartesian joint-row multipliers, persist as Jacobi initial guesses.
- Add `example_motorized_cable_chain` as a one-for-one cable-joint counterpart to the PhoenX motorized hinge-chain example.
- Add pure-Warp Muon optimizer support to `newton.rl` PPO trainers so experimental G1 runs can match nanoG1 optimizer settings without PyTorch.
- Add pure-PhoenX `newton.rl` environments and CUDA-graph PPO training scripts for Ant, Unitree Go2 and H1 flat locomotion, classic 21-action Humanoid, and closed-loop DR Legs hold-pose/walking tasks.
- Add `ConfigEnvG1PhoenX` scheduler knobs for `threads_per_world`, `multi_world_scheduler`, and `prepare_refresh_stride` so experimental G1 RL runs can benchmark PhoenX solver schedules without monkeypatching.
- Add experimental `SolverPhoenX(articulation_mode="reduced")` support for linear-time reduced-coordinate articulation dynamics and rigid contacts through the common Newton `State` and `Control` API, including CUDA graph capture and multi-world execution.
- Add experimental `SolverPhoenX(articulation_mode="hybrid")` support, keeping maximal PGS joint rows active while using the exact armature-aware articulated-body response as their preconditioner. This improves stiff robot training stability while preserving CUDA graph capture and multi-world execution.
- Add experimental `SolverPhoenX(articulation_mode="maximal_projected")` support for exact mass-metric projection of eligible maximal-coordinate robot joint trees, including common mixed rigid joints, recovered warm-start reactions, CUDA graph capture, multiple articulations per world, and pure reduced fallback for unsupported topologies.
- Add opt-in `SolverPhoenX(articulation_mode="maximal_articulated")` research support for deterministic exact tree-constrained hard-contact GS on free-root revolute forests, with CUDA graph capture, multiple articulations per world, and colored PGS fallback for external dynamic contacts.
- Add experimental pure-Warp `newton.rl` PPO and SAC training utilities, including a PhoenX Anymal sparse-target locomotion environment, deterministic evaluation metrics, and PPO checkpoint resume helpers.
- Add manual critic-backward PPO controls for the experimental PhoenX G1 RL training path, including `ConfigPPO.manual_critic_backward` and `newton.rl train-g1-ppo --no-manual-critic-backward`.
- Add strict walking validation metrics and configurable upright termination for the experimental PhoenX Anymal RL environment.
- Add public `SolverPhoenX` construction for model particles, cloth triangles and bending edges, and soft tetrahedra; deformable contacts use the same `model.contacts()` / `model.collide()` flow as rigid PhoenX scenes.
- Add `body_qdd` extended state attribute support to `SolverPhoenX`, populated as a finite difference of pre-step and post-step COM-frame velocity over the outer `dt`. Enables `newton.sensors.SensorIMU` on top of PhoenX (accelerometer reports specific force; gyroscope reports body angular velocity)
- Add `prepare_refresh_stride="auto"` to `SolverPhoenX` for graph-capture-safe reuse of cached rigid contact/joint prepare data in high-substep rigid scenes. Fixed integer strides remain available; the default `1` still rebuilds prepare data every substep, while `"auto"` falls back to `1` for unsupported deformable, mass-splitting, or sleeping scenes.
- Add `multi_world_scheduler` to `SolverPhoenX` for graph-capture-safe selection between the default fast-tail multi-world solver and the block-world scheduler; `"auto"` keeps fast-tail for robot-style fleets and selects block-world for dense contact-only multi-world stacks.
- Add `joint_friction` (Coulomb friction on the axial DoF of revolute / prismatic joints) support to `SolverPhoenX` via a regularized saturated soft constraint on the same scalar row as the drive, matching MuJoCo's `dof_frictionloss + actuator` decomposition. Total axial impulse per substep is the sum of the clamped drive PD term and a clamped friction term capped at `±μ * dt`. Regularization is sized from `PHOENX_FRICTION_SLIP_VELOCITY` (default 1 mm/s or 1 mrad/s) so the slip velocity at saturation is invariant across joint impedances
- Add D6 joint auto-dispatch to `SolverPhoenX`: `JointType.D6` configurations whose per-DoF lock pattern matches FIXED (all locked), BALL (3 linear locked + 3 angular free), REVOLUTE (3 linear locked + 2 angular locked + 1 angular free), or PRISMATIC (2 linear locked + 1 linear free + 3 angular locked) are now routed to the corresponding specialized constraint modes. Unlocks MJCF / USD imports that emit D6 for these patterns. Cylindrical / planar / generic configurations remain unsupported with a descriptive error
- Add `JOINT_MODE_UNIVERSAL` (Hooke joint) to `SolverPhoenX`: D6 configurations with 3 linear locked + 1 angular locked + 2 angular free dispatch to a new mode composing the BALL_SOCKET 3-row positional lock with a 1-row angular lock about the user-specified axis. Reuses the existing axial drive/limit machinery in rigid-Box2D mode (`min == max == 0`, rigid `hertz_limit`) for the lock
- Add `JOINT_MODE_CYLINDRICAL` to `SolverPhoenX`: D6 configurations with 2 linear locked + 1 linear free + 2 angular locked + 1 angular free, with the two free axes parallel, dispatch to a new mode. The constraint set is PRISMATIC's 4-row tangent block (anchor-1 + anchor-2 perpendicular locks) minus PRISMATIC's anchor-3 scalar lock that gates rotation about the slide axis; the Schur complement collapses (no anchor-3 row → no complement needed). MVP is kinematic-only: both free DoFs are unactuated; a follow-up may add paired axial drives for slide and spin independently
- Add `JOINT_MODE_PLANAR` to `SolverPhoenX`: D6 configurations with 1 linear locked + 2 linear free + 2 angular locked + 1 angular free, with the locked-linear and free-angular axes parallel (= plane normal), dispatch to a new mode. 3 constraint rows operate directly on relative body motion (no anchor lever arms): 1 linear lock on relative COM velocity along the plane normal + 2 angular locks on relative angular velocity perpendicular to the normal. 3x3 K matrix inverted directly. MVP is kinematic-only (3 free DoFs unactuated)
- Add framework-level `joint_gear` per-DoF field and `ModelBuilder.JointDofConfig.gear_ratio` parameter to express motor-to-joint gear ratios. `gear_ratio == 1.0` (default) is a no-op back-compat path. `SolverPhoenX` consumes the field: motor-side `effort_limit` is amplified by `gear`, motor-side `armature` reflects through the gearbox as `gear**2 * armature`, motor-side `friction` is amplified by `gear`. Lets URDF / MJCF importers carry through the gear ratio without having to pre-multiply each affected quantity
- Add parallel PGS-kernel compile to `SolverPhoenX`: when `wp.config.load_module_max_workers > 1`, `PhoenXWorld.__init__` eagerly instantiates the six single-world (or two multi-world fast-tail) `module="unique"` dispatcher kernels for the active scene spec and hands the modules to `wp.force_load` so NVRTC compiles them across the worker pool. Cold-cache start-up on a representative soft-body scene drops from ~82 s to ~19 s with eight workers. No-op when the config is unset, 0, or 1
- Add `newton.use_coord_layout_targets` opt-in flag exposing `Model.joint_target_q` / `Control.joint_target_q` shaped `(joint_coord_count,)` (matching `joint_q`) and `joint_target_qd` shaped `(joint_dof_count,)` (matching `joint_qd`); solvers, the actuator library, and `ModelBuilder.finalize()` honor the flag. Defaults to `False` for backwards compatibility; will flip in a future release.
- Add opt-in `validate_mesh` parameter to `ModelBuilder.add_cloth_mesh()`, `ModelBuilder.add_soft_mesh()`, and `style3d.add_cloth_mesh()` that warns on degenerate geometry; add public `newton.utils.validate_triangle_mesh()` and `newton.utils.validate_tet_mesh()` utilities
- Warn from `SolverMuJoCo` when a `JointType.FREE` joint has a non-world parent; MuJoCo requires free joints to attach directly to the world.
- Document loop closure in the articulations concept page, covering the omit-from-`add_articulation` pattern and USD `excludeFromArticulation` with per-solver caveats
- Add `ViewerGL.show_loading_splash()` / `ViewerGL.hide_loading_splash()` displaying a stylized Newton's-cradle overlay while the GL viewer waits on Warp kernel compilation; raised automatically by `newton.examples.init()` for visible GL viewers
- Add `ModelBuilder.mesh_edge_lower_angle_threshold_rad` (default 0.1 degree) to drop near-coplanar internal edges when packing precomputed mesh edges for SDF-mesh contact generation. Boundary and non-manifold edges are always kept; set to `0` to disable filtering
- Add edge-simplification options to `Mesh.build_sdf()`: `edge_lower_angle_threshold_rad`, `edge_upper_angle_threshold_rad`, opt-in `edge_box_absorption`, plus mutually exclusive absolute (`edge_box_half_height` / `edge_box_half_width`) and relative (`edge_box_half_height_rel` / `edge_box_half_width_rel`) box half-extents. The simplified edge set is cached on the `Mesh` and consumed by `ModelBuilder.finalize()` for SDF-mesh contact generation
- Add 8-node trilinear-hex soft-body constraints to `SolverPhoenX` (`CONSTRAINT_TYPE_SOFT_HEXAHEDRON`, schema in `newton._src.solvers.phoenx.constraints.constraint_soft_hexahedron`). The default model uses xpbd-fem-style integrated trace strain at eight Gauss points plus a reduced center volume row, projected as a coupled 2x2 XPBD block. `PhoenXWorld.populate_soft_hexahedra_from_arrays()` accepts `strain_model="arap"` to use integrated per-Gauss-point ARAP strain coupled with the same center volume row, avoiding the center-point hourglass pitfall. Exposes per-hex `(k_mu, k_lambda, beta_h, beta_d)` Lame + damping, a `num_soft_hexahedra` parameter on `PhoenXWorld`, and a 6th `StepReport.time_us_total_soft_hexahedra` timing bucket. Includes a minimal pinned-face example (`example_soft_hex_pinned`) and CUDA unit tests covering rest-pose boundedness, ballistic free fall, compression volume recovery, rigid rotation, hourglass correction, elevated-stiffness stability, shared-face two-hex patches, and top-face-pin hang. No collision support yet
- Add edge-simplification options to `Mesh.build_sdf()` that drop near-coplanar internal edges from the mesh-edge set used by SDF-mesh contact generation: `edge_lower_angle_threshold_rad` (default 0.1°; pass a negative value to opt out and keep the full edge set), `edge_upper_angle_threshold_rad`, opt-in `edge_box_absorption`, and box half-extent controls `edge_box_half_{normal,lateral}` / `edge_box_half_{normal,lateral}_rel`
- Add `cable_cross_slide_table` example demonstrating a cable-driven XY table
- Parse `NewtonSDFCollisionAPI` attributes from USD in `ModelBuilder.add_usd()`, including the `newton:hydroelasticEnabled` toggle, absolute narrow band / margin, texture format, hydroelastic stiffness (`newton:hydroelasticStiffness`), and applied-API schema defaults. Hydroelastic configuration is folded into `NewtonSDFCollisionAPI` and opted into via `newton:hydroelasticEnabled` (default `false`). SDF generation is opt-in by applying the API; for primitive shapes the SDF is only built when hydroelastic is also enabled.
- Add `cloth_stiff_material_hanging` and `cloth_stiff_material_stretch` examples regression-guarding the new Neo-Hookean triangle material (stability under gravity at extreme stiffness, and bulk area-preservation across a Poisson-ratio sweep)
- Add three VBD contact examples — `vbd_rigid_rigid_contact`, `vbd_soft_rigid_contact`, and `vbd_soft_rigid_mix_contact` — demonstrating rigid-rigid, soft (particle-rigid), and mixed cloth-bag contacts
- Add viewer layer system to overlay multiple solvers/models in supported rendering viewers; call `ViewerBase.activate(layer_id)` to route subsequent `set_model` / `log_state` / `log_*` calls into a named layer, `ViewerBase.set_layer_visible()` to toggle layers independently, and `ViewerBase.set_layer_transform()` to position layers side-by-side. See `example_basic_multi_solver_overlay.py`
- Add `viewer.set_picking_linear_only_bodies()` and `viewer.clear_picking_linear_only_bodies()` to mark bodies that should receive only the linear component of mouse-picking force, suppressing offset-induced torque.
- Add opt-in `body_frame_origin="com"` to `ModelBuilder.add_rod()` and `ModelBuilder.add_rod_graph()` for COM-centered cable capsule body frames.
- Add user-defined pressure laws to hydroelastic SDF contact via `HydroelasticSDF.Config.pressure_func` (a `@wp.func` mapping `(signed_depth, shape_idx, data) -> pressure`) and `pressure_data` (a `@wp.struct` carrying per-shape state). The contact patch is the iso-pressure surface `p_a == p_b`; the default linear law `pressure = -kh * signed_depth` is preserved when no callback is supplied.
- Add `SensorTiledCamera.utils.assign_checkerboard_material(shape_indices=...)` for applying the checkerboard texture to selected shapes.
- Add `--render-fps` to cap example rendering rate without changing simulation frame timing
- Add `ModelBuilder.BvhConfig` for selecting Warp BVH constructors during model finalization for mesh, Gaussian, and shape BVHs.

### Changed

- Normalize experimental SAC observations with automatically tracked running moments by default, improving held-out PhoenX G1 tracking without reducing measured training throughput. Pass `ConfigSAC(normalize_observations=False)` to preserve raw observations.
- Change the experimental PhoenX G1 train-to-gate benchmark to switch PPO replay ratio to 2.0 after the frozen 78.64M-sample screen, reducing the validated zero-training wall time while preserving the unchanged full gate. Pass `--late-replay-ratio 0` to disable the late replay schedule.
- Improve reduced-coordinate PhoenX CUDA split-dynamics stepping by using the fused advance/publish path when safe and skipping redundant packed contact-row clearing for unchanged articulated bodies.
- Change experimental PhoenX G1 RL defaults to use the reduced-articulation recipe path, recipe-aligned G1 env-step benchmark defaults, graph/no-readback G1 training benchmark defaults, and a 0.84 train-to-gate full-gate promotion screen. G1 training benchmark nanoG1 comparisons now require an explicit measured reference via `--nanog1-train-result` or `--nanog1-reference-env-sps` instead of document-derived defaults. Pass `articulation_mode="maximal"`, explicit benchmark `--sim-substeps`/`--steps-per-graph` values, `--execution-mode eager`, `--readback-diagnostics`, or `--screen-trigger-battery-perf 0.88` to preserve previous settings.
- Change experimental PhoenX G1 train-to-gate restarts to use frozen held-out screens at 39.32M and 78.64M samples, rejecting clearly weak attempts earlier while preserving the final quality gate.
- Switch the experimental PhoenX G1 train-to-gate final phase from three 6.67 ms physics steps to two 10 ms steps after 99,614,720 samples, preserving the hard-Hertz contact and PGS schedule while reducing validated total time-to-gate. Pass `--angular-fine-tune-start-samples 0` to disable both final-phase timestep and regularization changes.
- Change experimental PhoenX G1 RL defaults to the validated 6.67 ms schedule (`sim_substeps=3`) and strengthen angular-command tracking to 2.5, increasing graph-leapfrog training throughput while retaining repeatable full walking-gate success. Pass `--sim-substeps 4 --w-track-ang 1.25` to preserve the previous 5 ms recipe.
- Derive block-owned reduced-coordinate PhoenX contact effective masses from exact generalized response rows and skip the redundant link impulse-response traversal when model collision ownership proves that deferred contacts are impossible.
- Publish only reduced-coordinate PhoenX velocities after bias-free relaxation, reusing the unchanged configuration and motion subspaces instead of repeating pose integration and kinematic reconstruction.
- Schedule reduced-coordinate PhoenX local kinematics with topology-sized 8-, 16-, or 32-lane warp groups, increasing parallelism across independent narrow articulations without changing transforms.
- Iterate reduced-coordinate PhoenX generalized contact solves over each resident page actual contact count, eliminating graph-captured no-op work while preserving exact hard-Hertz projection and arbitrary contact paging.
- Schedule reduced-coordinate PhoenX state publication with topology-sized 8-, 16-, or 32-lane warp groups, packing independent narrow articulations without changing integration or velocity reconstruction.
- Accelerate experimental PhoenX G1 PufferMinGRU training with tiled aligned FP32 projections, linear-time recurrent backpropagation that reuses stored forward states, and deterministic FP32 or BF16 split-K weight gradients.
- Extend experimental PufferMinGRU BF16 backward contractions to hidden input gradients, retaining FP32 parameters and accumulation while accelerating PhoenX G1 training.
- Add an explicit reduced-coordinate PhoenX continuation-state step path that preserves authoritative state between external temporal substeps while still importing forces and controls.
- Store dense reduced-coordinate PhoenX generalized contact responses in a coalesced ABA layout and tile-transpose them for PGS consumption, while retaining direct row-major writes for sparse pages.
- Bypass the reduced-coordinate PhoenX schedule radix sort when collision-eligible model pairs and the active contact-ingest ordering prove that rigid contact columns are already grouped by articulation; retain deterministic sorting for inter-articulation and otherwise unproven scenes.
- Align reduced-coordinate PhoenX generalized contact-row launches to one fixed 96-row articulation page per CUDA block, improving row-builder locality without changing contact arithmetic.
- Skip reduced-coordinate PhoenX fallback contact coloring when the model's collision-eligible shape pairs prove that every reduced contact is against a static or kinematic body, preserving deterministic fallback for self-contact and dynamic-body interactions.
- Reuse exact reduced-coordinate PhoenX articulated contact responses across bias and velocity-relaxation passes with a bounded graph-capture-safe cache, accelerating contact-rich scenes without changing hard-Hertz contact projection.
- Reuse bounded reduced-coordinate PhoenX contact geometry during velocity relaxation, recomputing velocity-dependent rows while retaining exact overflow paging and hard-Hertz contact behavior.
- Schedule experimental reduced-coordinate PhoenX ABA advance with topology-sized 8-, 16-, or 32-lane warp groups, increasing parallelism across independent narrow articulations while retaining full-warp execution for wide trees.
- Reject cable joints and D6 angular limits in experimental simple PhoenX instead of solving different equations; use `solver_flavor="standard"` for these constraints.
- Alternate PhoenX PGS color traversal direction between iterations to reduce ordering bias without changing constraint equations or CUDA graph capture.
- Change PhoenX cable bend constraints to a physically damped angular Schur block coupled with the point attachment, so high stiffness approaches the revolute lock without SOR or additional constraint storage.
- Change experimental PhoenX G1 RL actuation to explicit clamped PD torques matching nanoG1/MuJoCo; pass `actuation_model="constraint_drive"` or `--actuation-model constraint_drive` to use the previous implicit PhoenX drive-row formulation.
- Change experimental PhoenX G1 RL defaults to the validated 5 ms schedule (`sim_substeps=4`, two position iterations, and one velocity iteration), strengthen action-rate and roll/pitch regularization, apply the validated final angular-stability phase in the train-to-gate benchmark, and embed the complete G1 environment and training recipe in new checkpoints. Pass `--sim-substeps 10 --solver-iterations 8 --velocity-iterations 2 --w-action-rate -0.01 --w-ang-vel-xy -1.3 --angular-fine-tune-start-samples 0` to preserve the previous defaults.
- Change the default CoACD convex decomposition threshold from `0.5` to `0.05` to match CoACD's default; pass `remeshing_kwargs={"threshold": 0.5}` to preserve the previous coarse decomposition.
- **Breaking change (experimental `SolverVBD`):** VBD now interprets all damping coefficients as absolute physical units instead of dimensionless stiffness-relative (Rayleigh) multipliers (`D = kd · ke`). Existing `kd`-family values will produce different damping. Affected parameters: tetrahedral `k_damp` [Pa·s], `tri_kd`, spring `kd` [N·s/m], cable `stretch_damping` [N·s/m] and `bend_damping` [N·m·s/rad] in `add_joint_cable()`/`add_rod()`/`add_rod_graph()`, `joint_target_kd` and `joint_limit_kd` (including `JointDofConfig.limit_kd`), shape contact `kd`/`shape_material_kd` and `soft_contact_kd` [N·s/m], and `SolverVBD(rigid_joint_linear_kd=…, rigid_joint_angular_kd=…)`. To preserve previous behavior, set `kd_new = kd_old · k`, where `k` is the stiffness or penalty coefficient the value was previously paired with, and pass the product to the same field.
- Change `SolverKamino.reset(world_mask=...)` to accept `wp.bool` arrays instead of `wp.int32`; callers passing `wp.int32` masks must switch to `wp.bool` (e.g. `wp.array([False, True, False], dtype=wp.bool)` or `wp.ones((num_worlds,), dtype=wp.bool)`). (#2934)
- Change `SolverFeatherstone` FREE/DISTANCE `joint_qd` to use the public Newton convention: callers must pass six values per joint as child-COM linear velocity followed by angular velocity, both expressed in the joint parent frame; floating-root FREE/DISTANCE articulations now use a root-COM-centered internal solve frame for improved stability far from the world origin.
- Assign one default visual color to capsule segments generated by `ModelBuilder.add_rod()` or `add_rod_graph()`; pass `color=` to choose an explicit cable color.
- Rework the API of the `reset()` operation in `SolverKamino` to use an explicit `ResetConfig` instead of many keyword arguments.

### Deprecated

- Deprecate passing solver constructor options positionally after stable positional inputs such as `model` and explicit solver configs; migrate calls such as `SolverVBD(model, 10)` to `SolverVBD(model, iterations=10)`.
- Deprecate passing option-heavy helper API parameters positionally, including `ModelBuilder.ShapeConfig`, `ModelBuilder.JointDofConfig`, `Contacts`, `ArticulationView`, and selected `ModelBuilder` body, joint, shape, rod, cloth, soft-body, and FEM helpers. Keep stable identifiers such as `body`, `parent`/`child`, capacity counts, and topology indices positional; migrate calls such as `add_shape_box(body, xform, hx=...)` to `add_shape_box(body, xform=xform, hx=...)`.
- Deprecate omitting `body_frame_origin` in `ModelBuilder.add_rod()` and `ModelBuilder.add_rod_graph()`; the implicit behavior still uses the existing start-node body-frame convention during the deprecation window, but the implicit default will change to `body_frame_origin="com"` in a future release. Pass `body_frame_origin="start"` to preserve the legacy frame or `body_frame_origin="com"` to opt into the future COM-centered frame.
- Change VBD Neo-Hookean membrane/tet damping to an objective metric based on the rate of `C = FᵀF`, so rigid-body rotations no longer generate damping force.
- Change VBD spring damping to act only along the spring axis (damping edge-length rate), so transverse and rigid-rotational motion is no longer damped by springs.
- `SolverVBD` now applies each shape's `ShapeConfig.margin` (`model.shape_margin`) to particle-rigid (soft) contacts, widening the soft-contact detection shell and reducing penetration depth per shape; previously only the global `soft_contact_margin` and particle radius were used. Re-check VBD scenes that set per-shape margins. (#2994)

### Fixed

- Fix experimental PhoenX Ant rollouts to terminate and reset non-finite or physically exploded states instead of retaining corrupted simulation state.

- Fix experimental PhoenX recurrent PPO replay to preserve the exact detached MinGRU state at rollout boundaries, avoid mutating persistent state during bootstrap evaluation, and synchronize recurrent state across graph-leapfrog trainers.
- Fix experimental PhoenX recurrent PPO replay to reset MinGRU state and cut recurrent gradients at terminal episode boundaries, including trajectory minibatches and CUDA graph capture.
- Fix experimental PhoenX G1 held-out evaluation to preserve fixed commands across fall resets and accumulate metrics in deterministic CUDA graphs without per-step host synchronization.
- Fix the experimental PhoenX time-to-policy suite truncating G1 trials at the nanoG1 75M-sample comparison point instead of the frozen 170.39M-sample restart horizon.
- Fix the pure-PhoenX Ant environment applying the MJCF Z-up to Y-up rotation twice, which made every nominal reset pose immediately terminal.
- Fix the pure-PhoenX H1 environment omitting ankle colliders, and add source-matched foot-contact, air-time, sliding, soft-limit, spawn-height, and graph-safe command behavior.
- Fix the pure-PhoenX DR Legs walking task missing contact matching, swing clearance, foot orientation, touchdown-speed, gait-period, and graph-safe randomized-command behavior.
- Fix `SolverPhoenX.prepare_refresh_stride` for reduced and hybrid articulations so internal substeps reuse cached block, deferred, and fallback contact preparation at the requested cadence instead of silently forcing a refresh every substep.
- Fix the reduced-coordinate PhoenX fallback fast-path proof to retain deterministic fallback coloring for unrelated maximal rigid-body contacts.
- Fix experimental PhoenX G1 replacement foot boxes bypassing the MJCF no-self-collision filters, preventing unintended robot self-contacts during policy exploration.
- Fix the experimental PhoenX G1 recipe to match nanoG1's 8192-world batch and full-range command distribution; the pinned build does not enable its optional command curriculum.
- Fix maximal-coordinate PhoenX revolute armature by adding motor-side stator inertia to the parent and gear-reflected rotor inertia to the child, preserving rigid-body momentum and gyroscopic dynamics without auxiliary constraint rows.
- Fix `SolverPhoenX` state imports leaving world-space inverse inertia at the previous orientation, which produced incorrect first-frame impulses and angular-momentum jumps after resets or externally modified poses.
- Stabilize the full-coordinate PhoenX `robot_g1` example with a fine temporal schedule.
- Fix reduced-coordinate PhoenX motion-subspace evaluation after the common Featherstone API gained configuration-dependent D6 angular axes.
- Fix reduced-coordinate PhoenX FREE and DISTANCE descendants to preserve child-COM velocity under rotated anchors and accelerating parents, and accept tree DISTANCE joints through `articulation_mode="reduced"`.
- Fix reduced-coordinate PhoenX direct forward dynamics to evaluate body wrenches in articulation-local coordinates and refresh cached ABA inertia after live body-property changes.
- Fix experimental simple PhoenX warm starting and velocity relaxation so separated or removed contacts and stale conditional joint rows cannot affect bodies.
- Fix PhoenX PGS iteration ordering so every solve and relaxation iteration traverses all constraint colors before processing any constraint again.
- Fix PhoenX primitive stack contacts so dense Kapla tower examples remain stable while preserving SDF speculative-contact safeguards.
- Fix `ViewerFile.is_running()` to return `False` after `ViewerFile.close()` so headless recording loops can terminate like interactive viewers. (#3094)
- Fix `SolverVBD` rigid contact injecting kinetic energy for yawed finite-radius contacts (e.g. small-radius cables blowing up). The normal response now acts at the geometric skeleton point rather than the rotating surface anchor, which was non-conservative under reorientation; friction still uses the surface anchor to preserve finite-radius slip. (#3125)
- Fix `SolverKamino` contact filtering and constraint stabilization so gap/margin contacts are handled consistently, positive-distance contacts can be filtered as configured, and converted contact forces/wrenches populate matching Newton contact slots for `SensorContact`. (#2908)
- Fix memory growth in the Style3D solver when CUDA Graph capture is disabled
- Fix `newton.eval_jacobian`, `SolverFeatherstone`, and the IK analytic Jacobian building `JointType.D6` angular motion-subspace columns from raw axes, so `J @ joint_qd` now matches `State.body_qd` for two- or three-angular-DOF joints at non-identity configurations.
- Fix mesh inertia computation to produce deterministic results across repeated CUDA runs. (#3136)
- Fix `SolverImplicitMPM` whole-step CUDA graph capture failing when the rheology inner solver is an iterative linear method such as `solver="cg"`. (#3155)
- Fix USD import so non-unit `metersPerUnit` and `kilogramsPerUnit` warn as unsupported, and stop scaling `PhysicsScene` gravity magnitude by `metersPerUnit`.
- Fix swapped kinetic and potential energy labels in the `basic_plotting` example, and report per-world values directly so the live plot overlays match the side-panel readouts
- Fix `SolverMuJoCo` reporting incorrect `State.body_qd` angular velocity for `JointType.D6` joints with two or three angular DOFs at non-identity configurations.
- Fix VBD collision damping to use relative normal gap rate so uniform contact-stencil motion and tangential sliding do not create artificial normal damping.
- Fix `ModelBuilder.add_usd` so a stray authored joint outside any articulation root no longer suppresses base-joint (articulation) creation for unrelated floating rigid bodies in the same call. (#3002)
- Fix `RenderContext` triangle mesh construction by removing the unsupported `device=` keyword from `wp.Mesh(...)`.
- Fix MJCF `euler` producing wrong orientations for multi-component angles by treating angles as intrinsic rotations. (#3030)
- Fix `newton.eval_fk` / `newton.eval_ik` producing wrong rotations and joint velocities for `JointType.D6` with three angular DOFs whose axes form a left-handed orthonormal basis.
- Fix MJCF parsing so attributes from multiple `<compiler>` elements, including `<include>`-expanded children, are merged in document order. (#3030)
- Fix MJCF worldbody static geoms bypassing the visual/collider class filter, so `parse_visuals=False` drops visual-class geoms attached directly to `<worldbody>` too. (#3030)
- Fix `cable_cross_slide_table` example stability so the cable-driven table reliably tracks its rectangular path and catches drift during regression runs.
- Fix URDF `package://` mesh fallback resolution without `resolve-robotics-uri-py` so package names only match full path components instead of unrelated directory-name substrings
- Fix `ModelBuilder.collapse_fixed_joints()` crashing with `IndexError` when a `mujoco:equality_constraint` row omits optional fields (`anchor`, `relpose`) that carry defaults. (#3054)
- Fix `ViewerGL.set_model()` resetting headless/interactive camera and wind state when switching between models that use the same up-axis. (#2658)
- Fix bend force calculation error in Style3D solver

### Removed

- Remove the deprecated Style3D `CollisionHandler`; use `Collision` instead
- Remove the deprecated `state=None` and `refit_bvh` arguments of `SensorTiledCamera.update()`; pass a `State` and refit BVHs explicitly via `Model.bvh_refit_shapes()` / `Model.bvh_refit_particles()` before rendering frames that change geometry
- Remove support for the legacy `newton_actuators`-style `ModelBuilder.add_actuator(actuator_class, input_indices=...)` signature; use `add_actuator(controller_class, index=..., ...)` with `newton.actuators` controllers (e.g. `ControllerPD`, `ControllerPID`) instead
- Remove the deprecated `newton-actuators` package dependency; all actuator functionality is built into `newton.actuators`
- Remove the deprecated implicit MPM `collider_velocity_mode` aliases `'finite_difference'` and `'instantaneous'` (deprecated in 1.1.0); use `'backward'` and `'forward'` instead
- Remove the deprecated `Viewer.update_shape_colors()`; write to `Model.shape_color` directly instead
- Remove body armature: drop `ModelBuilder.default_body_armature`, the `armature` parameter of `ModelBuilder.add_link()` and `add_body()`, and USD-authored body `newton:armature` (deprecated in 1.1.0); add any isotropic artificial inertia directly to `inertia` instead. All `add_link()` and `add_body()` parameters are now keyword-only
- Remove support for passing a `Gaussian` as the second positional argument to `ModelBuilder.add_shape_gaussian()` (deprecated in 1.1.0); pass it via the `gaussian=` keyword instead
- Remove support for `worlds_per_row=0` in `SensorTiledCamera.flatten_*_to_rgba()` (deprecated in 1.2.0); pass `worlds_per_row=None` for automatic layout (values below 1 now raise `ValueError`) (#3149)
- Remove the deprecated SensorTiledCamera 1.1 API surface: `Config`, `RenderContext`, `render_context`, image helper methods, and random color helpers; use `RenderConfig`, `render_config`, `SensorTiledCamera.utils.*`, and per-shape colors instead
- Remove the deprecated `SensorRaycast`; use `SensorTiledCamera` (`SensorTiledCamera.utils.compute_pinhole_camera_rays()` and `create_depth_image_output()` for single-camera depth) instead
- Remove the deprecated `max_worlds` parameter of `ViewerBase.set_model()`; call `ViewerBase.set_visible_worlds()` after `set_model()` instead
- Remove the deprecated `a`, `b`, `c` parameters of `ModelBuilder.add_shape_ellipsoid()` (deprecated in 1.1.0); use `rx`, `ry`, `rz` instead
- Remove the deprecated and ignored `rigid_enable_dahl_friction` argument of `SolverVBD`; Dahl friction is auto-detected from `model.vbd.dahl_eps_max` / `model.vbd.dahl_tau`

## [1.3.0] - 2026-06-11

### Added

- Add `newton.use_coord_layout_targets` opt-in flag exposing `Model.joint_target_q` / `Control.joint_target_q` shaped `(joint_coord_count,)` (matching `joint_q`) and `joint_target_qd` shaped `(joint_dof_count,)` (matching `joint_qd`); solvers, the actuator library, and `ModelBuilder.finalize()` honor the flag. Defaults to `False` for backwards compatibility; will flip in a future release. (#2965)
- Add `joint_damping` model attribute and `JointDofConfig.damping` for velocity-proportional damping that is always active. (#2304)
- Add `SolverBase.reset()` method for in-place solver state resets with optional `world_mask` and `StateFlags`; default implementation is a no-op. (#2657)
- Implement `SolverMuJoCo.reset()` to reset selected worlds: it resets `State.joint_q` / `State.joint_qd` to the model defaults (gated by `StateFlags`) and clears the per-world MuJoCo buffers that persist between steps (`qacc_warmstart`, `qfrc_applied`, `xfrc_applied`, `act`, `ctrl`), so a world recovers cleanly on the next step after a divergence (e.g. NaNs); accepts an optional `world_mask`. (#3062)
- Add `StateFlags` enum to control which state attributes are reset. (#2657)
- Add `ModelFlags` as the canonical name for model-change notification flags. (#2657)
- Add opt-in `validate_mesh` parameter to `ModelBuilder.add_cloth_mesh()`, `ModelBuilder.add_soft_mesh()`, and `style3d.add_cloth_mesh()` that warns on degenerate geometry; add public `newton.utils.validate_triangle_mesh()` and `newton.utils.validate_tet_mesh()` utilities
- Add edge-simplification options to `Mesh.build_sdf()` that drop near-coplanar internal edges from the mesh-edge set used by SDF-mesh contact generation: `edge_lower_angle_threshold_rad` (default 0.1°; pass a negative value to opt out and keep the full edge set), `edge_upper_angle_threshold_rad`, opt-in `edge_box_absorption`, and box half-extent controls `edge_box_half_{normal,lateral}` / `edge_box_half_{normal,lateral}_rel`. (#2894)
- Add `ModelBuilder.ShapeConfig.sdf_padding` and `ModelBuilder.shape_sdf_padding` for setting the per-shape SDF AABB padding [m] used when building primitive texture SDFs and deferred mesh SDFs
- Add `contact_reduction_hashtable_size_factor` to `CollisionPipeline`, `NarrowPhase`, and `HydroelasticSDF.Config` for increasing contact reduction hashtable capacity when fill/failure warnings appear. (#2978)
- Add `Model.bvh_build_shapes()`, `Model.bvh_refit_shapes()`, `Model.bvh_build_particles()`, and `Model.bvh_refit_particles()` for managing model BVHs, with optional `bvh_constructor` selection on the build methods. (#3039)
- Add functional `newton.intersect_ray()` shape query helper for composing custom raycast sensors. (#2971)
- Add `Model.heightfield_count` for the number of `GeoType.HFIELD` shapes in a model. (#3039)
- Document loop closure in the articulations concept page, covering the omit-from-`add_articulation` pattern and USD `excludeFromArticulation` with per-solver caveats
- Add `model.mujoco.equality_constraint_objtype`, `_target_kind`, and `_target` fields, recording the object kind a MuJoCo equality references and whether it was projected onto a native loop joint or mimic constraint for solver portability. (#2959)
- Add `newton.actuators.SchemaNames` exposing the canonical USD schema token constants used by `parse_actuator_prim` for actuator USD parsing
- Add `basic_dzhanibekov` example demonstrating free rigid-body intermediate-axis instability across VBD, XPBD, and MuJoCo solvers
- Parse URDF `<material>` colors (inline `<color rgba>` and named material references) during import and apply them to `ModelBuilder.shape_color` for all shape types
- Add opt-in `collapse_massless_fixed_root` to URDF and MJCF importers to collapse massless fixed-root chains for maximal-coordinate solvers while preserving topology by default
- Parse `NewtonSDFCollisionAPI` attributes from USD in `ModelBuilder.add_usd()`, including the `newton:hydroelasticEnabled` toggle, absolute narrow band / margin, texture format, hydroelastic stiffness (`newton:hydroelasticStiffness`), and applied-API schema defaults. Hydroelastic configuration is folded into `NewtonSDFCollisionAPI` and opted into via `newton:hydroelasticEnabled` (default `false`). SDF generation is opt-in by applying the API; for primitive shapes the SDF is only built when hydroelastic is also enabled. (#2533)
- Add USD parsing for `NewtonSiteAPI` to mark shapes as sites
- Add USD parsing for `NewtonMassAPI`: shell mass model (`newton:massModel`), shell thickness (`newton:shellThickness`), and compact inertia tensor (`newton:inertia`). (#2951)
- Add USD parsing for contact response attributes (`ke`, `kd`, `kf`, `ka`) from `NewtonMaterialAPI` on bound materials. (#3005)
- Add `ViewerGL.show_loading_splash()` / `ViewerGL.hide_loading_splash()` displaying a stylized Newton's-cradle overlay while the GL viewer waits on Warp kernel compilation; raised automatically by `newton.examples.init()` for visible GL viewers
- Add an optional `kernel_block_dim` argument to `SensorTiledCamera.update()` for tuning the Warp ray-tracer's `render_megakernel` launch shape
- Add `rec_id` parameter to `ViewerRerun` for specifying the recording ID, enabling multiple processes to share a single Rerun recording
- Add `ViewerRTX`, a real-time ray-traced viewer powered by NVIDIA OVRTX. (#1861)
- Add edge overlay toggle (`renderer.draw_edges`) for wireframe visualization on top of solid geometry
- Add `newton.utils.ColorSpace`, `color_srgb_to_linear()`, `color_linear_to_srgb()`, and `SensorTiledCamera.RenderConfig.output_color_space` for color-space boundaries. (#2411)
- Add `cable_cross_slide_table` example demonstrating a cable-driven XY table
- Add robotics tutorial notebook covering ModelBuilder, solvers, CUDA graphs, IK, and pick-and-place
- Add rigid-soft contact example covering a rigid sphere dropping onto an XPBD tetrahedral soft grid

### Changed

- Change the experimental PhoenX G1 PPO defaults to use nanoG1-style Muon optimizer hyperparameters; pass `--optimizer adam` to restore Adam.

- Change the experimental PhoenX G1 PPO primitive-collision environment to cap rigid-contact buffers at 32 contacts per world by default; pass `rigid_contact_max_per_world=0` or `newton.rl train-g1-ppo --rigid-contact-max-per-world 0` to restore SolverPhoenX automatic sizing.

- Change `SolverPhoenX` fast-tail multi-world robot fleets to use per-color
  joint/contact family ranges at 512+ worlds, reducing branchy mixed constraint
  dispatch in high-throughput PhoenX RL scenes.

- Change the experimental PhoenX G1 PPO defaults to use BF16 inputs with
  FP32 accumulation and 128-row dense tiles for manual CUDA MLP backward tile
  matmuls and large-minibatch hidden-layer forward tile matmul; pass
  `--manual-mlp-weight-grad-dtype float32` or
  `--manual-mlp-forward-dtype float32` to restore exact FP32 manual kernels.

- Change the experimental PhoenX G1 PPO defaults to use nanoG1-style
  V-trace replay clips; pass `--vtrace-rho-clip 0.0 --vtrace-c-clip 0.0`
  to keep replay uncorrected.

- Change the experimental PhoenX G1 PPO training diagnostics to read back one
  compact device-side metric buffer instead of full rollout arrays; pass
  `--no-readback-diagnostics` for strict no-diagnostic-readback benchmark runs.

- `ModelBuilder.finalize()` no longer writes the deferred mesh SDF back to `Mesh.sdf` on shared `Mesh` instances. The SDF data is retained on the finalized `Model` (`model.shape_sdf_index`, `model.texture_sdf_data`). Call `Mesh.build_sdf()` directly when you want the SDF stored on a `Mesh`.
- `ModelBuilder.add_shape_convex_hull()` (and any path producing `GeoType.CONVEX_MESH`) now raises `ValueError` if `ShapeConfig.sdf_*` or `ShapeConfig.is_hydroelastic` are set, matching `add_shape_mesh()`. Build and attach the SDF on the underlying `Mesh` via `Mesh.build_sdf()` instead.
- `GeoType.HFIELD` shapes now use a `wp.Mesh` BVH for raycasting (built during `ModelBuilder.finalize()`), replacing the per-thread DDA grid traversal; the raycast kernel signature no longer accepts `shape_heightfield_index`, `heightfield_data`, or `heightfield_elevations` — those arrays are still present on `Model` for collision kernels
- `ModelBuilder.finalize()` no longer writes the deferred mesh SDF back to `Mesh.sdf` on shared `Mesh` instances. The SDF data is retained on the finalized `Model` (`model.shape_sdf_index`, `model.texture_sdf_data`). Call `Mesh.build_sdf()` directly when you want the SDF stored on a `Mesh`.
- `ModelBuilder.add_shape_convex_hull()` (and any path producing `GeoType.CONVEX_MESH`) now raises `ValueError` if `ShapeConfig.sdf_*` or `ShapeConfig.is_hydroelastic` are set, matching `add_shape_mesh()`. Build and attach the SDF on the underlying `Mesh` via `Mesh.build_sdf()` instead.
- Build shape and particle BVHs automatically during `ModelBuilder.finalize()`. Call `Model.bvh_refit_shapes()` and `Model.bvh_refit_particles()` after state changes; use `Model.bvh_build_shapes()` and `Model.bvh_build_particles()` only when explicitly rebuilding, selecting a custom constructor, or working with manually populated models.
- Treat `NewtonSDFCollisionAPI` and `NewtonMeshCollisionAPI` as independent collision representations in the USD importer. Co-applying both APIs on the same prim emits a warning and SDF configuration is used. `physics:approximation` (inherited from `PhysicsMeshCollisionAPI`) is ignored on SDF prims with a warning. `ModelBuilder.approximate_meshes()` raises `ValueError` for mesh-replacing methods (`convex_hull`, `coacd`, `vhacd`, `bounding_box`, `bounding_sphere`) when a target shape carries deferred SDF configuration or the `HYDROELASTIC` flag.
- The USD importer warns and degrades on invalid or under-specified SDF configuration instead of aborting the whole import. A hydroelastic `Mesh` prim without an SDF source (no `newton:sdfMaxResolution` / `newton:sdfTargetVoxelSize`) imports as a plain mesh collider with hydroelastic disabled. An `sdfMaxResolution` not divisible by 8, an unknown `sdfTextureFormat`, and both `sdfMaxResolution` and `sdfTargetVoxelSize` authored on the same prim each warn with the prim path and fall back to defaults (`sdfTargetVoxelSize` takes precedence over `sdfMaxResolution`). The Python API path (`ShapeConfig.validate()`) still raises.
- Users of heightfield ray queries, `newton.intersect_ray()`, or sensor rendering should call `Model.bvh_refit_shapes(state)` after shape state changes. `GeoType.HFIELD` ray queries now use the mesh BVH built during `ModelBuilder.finalize()` instead of per-thread DDA grid traversal; only code manually launching internal raycast kernels is affected by the removed heightfield-specific kernel buffers. (#2971, #3039)
- Users who update shape or particle state after finalization should call `Model.bvh_refit_shapes()` or `Model.bvh_refit_particles()` before BVH-backed queries. `ModelBuilder.finalize()` now builds shape and particle BVHs automatically; use `Model.bvh_build_shapes()` and `Model.bvh_build_particles()` only when explicitly rebuilding, selecting a custom constructor, or working with manually populated models. (#3039)
- Users who relied on `ModelBuilder.finalize()` populating `Mesh.sdf` on shared `Mesh` instances should call `Mesh.build_sdf()` directly when reusable SDF data must live on the `Mesh`. The finalized model now retains deferred SDF data internally for contact generation; do not depend on `Model.shape_sdf_index` or `Model.texture_sdf_data`. (#2533)
- Users configuring SDF or hydroelastic options on convex-hull shapes should build and attach SDF data on the underlying `Mesh` with `Mesh.build_sdf()` instead. `ModelBuilder.add_shape_convex_hull()` (and any path producing `GeoType.CONVEX_MESH`) now raises `ValueError` if `ShapeConfig.sdf_*` or `ShapeConfig.is_hydroelastic` are set, matching `add_shape_mesh()`. (#2533)
- USD importer users who co-apply `NewtonSDFCollisionAPI` and `NewtonMeshCollisionAPI` on the same prim should expect a warning and SDF configuration to be used. The importer now treats these as independent collision representations, ignores `physics:approximation` on SDF prims with a warning, and `ModelBuilder.approximate_meshes()` raises `ValueError` for mesh-replacing methods (`convex_hull`, `coacd`, `vhacd`, `bounding_box`, `bounding_sphere`) when a target shape carries deferred SDF configuration or the `HYDROELASTIC` flag. (#2533)
- USD users with invalid or under-specified SDF configuration now get warnings and degraded imports instead of whole-import failures. A hydroelastic `Mesh` prim without an SDF source (no `newton:sdfMaxResolution` / `newton:sdfTargetVoxelSize`) imports as a plain mesh collider with hydroelastic disabled. An `sdfMaxResolution` not divisible by 8, an unknown `sdfTextureFormat`, and both `sdfMaxResolution` and `sdfTargetVoxelSize` authored on the same prim each warn with the prim path and fall back to defaults (`sdfTargetVoxelSize` takes precedence over `sdfMaxResolution`). The Python API path (`ShapeConfig.validate()`) still raises. (#2533)
- Users comparing SDF-mesh contact distances or normals may see small shifts below typical `contact_threshold` and `shape_margin` settings. The SDF-mesh narrow phase now uses hardware-filtered SDF texture sampling with centred-difference gradients; hydroelastic SDF sampling is unchanged. Pass a negative `edge_lower_angle_threshold_rad` (e.g. `-1.0`) to `Mesh.build_sdf()` to disable the new edge-simplification pass and reproduce the pre-optimisation behavior with the full edge set. (#2894)
- Support negative (mirrored) scale on mesh, convex hull, and SDF shapes, so a single `Mesh` instance can be shared across shapes with different signed scales without re-baking. (#2837)
- Users passing negative scale to symmetric primitive shapes (sphere, box, capsule, cylinder, ellipsoid, plane, gaussian) should apply mirrors through the shape transform (`xform`) instead if the sign matters. `ModelBuilder.add_shape*` now normalizes these negative scale components to `abs()` because the shapes are point-symmetric. (#2837)
- Users passing negative scale to cones or heightfields must now mirror cones through the shape `xform` and pre-mirror height data before passing positive heightfield scale. `ModelBuilder.add_shape_cone()` and heightfield shapes now reject negative scale components that were previously silently accepted and produced invalid geometry. (#2837)
- MJCF/USD import users who need the legacy MuJoCo equality-constraint arrays should pass `convert_mjc_equality_constraints=False`. MuJoCo equality constraints now convert to Newton loop joints and mimic constraints by default while preserving MuJoCo metadata for `SolverMuJoCo` round-trips. (#2959)
- Change `SolverMuJoCo` joint-limit `jnt_solref` conversion so Newton force-space `joint_limit_ke` / `joint_limit_kd` match the configured limit stiffness/damping at the joint limit instead of MuJoCo's acceleration-space default. The conversion uses MuJoCo's per-DOF `dof_invweight0`, so effective dynamics depend on asset mass/inertia. (#2610, #3132)
- `SolverMuJoCo` users with Newton-authored joint-limit gains should retune `joint_limit_ke` / `joint_limit_kd` per asset mass/inertia and timestep. The same numeric stiffness/damping can produce different dynamics than Newton 1.2, and previously damped assets may become underdamped or hit MuJoCo's timestep safety clamp. MJCF/USD-authored native `mjc:solreflimit` / `model.mujoco.solreflimit` values stay in raw MuJoCo mode (`SOLREF_MODE_RAW`) and are not rescaled; unauthored MJCF defaults stay in `SOLREF_MODE_MJCF_DEFAULT` until their Newton gains are edited. To keep native MuJoCo behavior, author/pass raw `solreflimit`, or set `joint_limit_ke <= 0` or `joint_limit_kd <= 0` to restore MuJoCo's default `solreflimit = (0.02, 1.0)`. (#2610, #3132)
- Warn from `SolverMuJoCo` when a `JointType.FREE` joint has a non-world parent; MuJoCo requires free joints to attach directly to the world
- `SolverKamino.reset()` users should migrate `state_out=` calls to `state=` and pass reset targets by keyword, such as `base_q=...`. The method now resets in place on `state`, matches `SolverBase.reset()`, and no longer accepts beta `state_out=` or legacy positional reset-target compatibility. (#2657)
- Accept plain `int` flag bitmasks in solver reset and model-change notification APIs so users can define extension bits beyond `StateFlags` and `ModelFlags`. (#2657)
- Mark `SolverVBD` as experimental. (#3068)
- Mark `SolverKamino` as experimental. (#3065)
- Mark the actuator API as experimental in its docstring. (#3067)
- Mark the `"sticky"` contact-matching mode as experimental in its docstring. (#3067)
- `SensorTiledCamera` users who relied on previous linear-byte packed `color` and `albedo` outputs should pass `RenderConfig(output_color_space=ColorSpace.LINEAR)`. Packed `color` and `albedo` now default to sRGB-encoded bytes so authored display colors render at the expected display brightness. (#2411)
- Auto-scale `ViewerGL` contact arrows, joint axes, and COM markers by `Viewer.scene_scale`; to approximate the previous fixed sizes after `set_model()`, set `viewer.renderer.arrow_length_scale = 0.1 / viewer.scene_scale`, `viewer.renderer.joint_scale = 0.1 / viewer.scene_scale`, and `viewer.renderer.com_scale = 0.1 / viewer.scene_scale`. (#2856)
- Make the `ViewerGL` left control panel movable (drag the title bar) and resizable (drag the bottom-right corner); a vertical scrollbar appears automatically when contents overflow and is operable with the mouse wheel or two-finger trackpad gestures. The initial dock-on-left position is unchanged. (#2926)
- Remove the `cbor2` `<6` dependency ceiling after updating recorder deserialization to accept mapping-like decoded containers
- Users configuring Warp logging should use Newton's `--quiet` flag or `--warp-config log_level=...` instead of legacy Warp `verbose` or `quiet` config keys. Newton now requires Warp 1.14 and configures Warp logging through `warp.config.log_level`. (#2900, #3046)
- Rename `SensorContact` sensing-object API names: `sensing_obj_idx` to `sensing_indices`, `sensing_obj_type` to `sensing_type`, `sensing_obj_transforms` to `sensing_transforms`, `sensing_obj_bodies` to `sensing_bodies`, and `sensing_obj_shapes` to `sensing_shapes`; the old names remain deprecated aliases for compatibility. (#3120)
- Bump `newton-usd-schemas` to `>=0.3.1`

### Deprecated

- Users of joint target aliases should migrate `Model.joint_target_pos` / `Control.joint_target_pos` and `Model.joint_target_vel` / `Control.joint_target_vel` to `joint_target_q` / `joint_target_qd`. The legacy names emit a `DeprecationWarning` and raise `AttributeError` when `newton.use_coord_layout_targets = True`. Set that flag before building a model to switch `joint_target_q` to `joint_coord_count` shape (matching `joint_q`); the default `False` keeps the legacy `joint_dof_count` layout. `ModelBuilder.joint_target_pos` / `ModelBuilder.joint_target_vel` are now read-only deprecated aliases (the writable attributes are removed) — set per-axis targets via `JointDofConfig.target_pos` / `target_vel` or write directly to `ModelBuilder.joint_target_q` / `joint_target_qd`. (#2965)
- Actuator users should pass `control_target_pos_attr="joint_target_q"` (and the velocity counterpart) explicitly to adopt the future default now. The implicit `Actuator` default that resolves `control_target_pos_attr` / `control_target_vel_attr` to legacy `joint_target_pos` / `joint_target_vel` when `newton.use_coord_layout_targets` is `False` is deprecated and will switch to `joint_target_q` / `joint_target_qd` in a future release. (#2965)
- Deprecate `SensorTiledCamera.utils.assign_checkerboard_material_to_all_shapes()` in favor of `SensorTiledCamera.utils.assign_checkerboard_material(shape_indices=...)`.
- Deprecate `SolverNotifyFlags` in favor of `ModelFlags`; migrate calls such as `SolverNotifyFlags.MODEL_PROPERTIES` to `ModelFlags.MODEL_PROPERTIES`. (#2657)
- `SolverVBD.register_custom_attributes()` users who want Dahl cable friction should pass `dahl_defaults_enabled=False` and explicitly author positive `model.vbd.dahl_eps_max` and `model.vbd.dahl_tau` values instead of relying on registered default values. Implicit positive Dahl defaults are deprecated. (#2921)
- Deprecate `Model.mujoco.dof_passive_damping` in favor of `Model.joint_damping`; MuJoCo `damping` import now populates `Model.joint_damping` and the old namespaced attribute remains a warning alias. (#2304)
- Deprecate the `ls_parallel` argument of `SolverMuJoCo`; parallel line search is being removed from `mujoco_warp`. Remove `ls_parallel` from solver construction. (#3096)
- USD authors should put contact response attributes on bound `NewtonMaterialAPI` materials instead of shape prim custom attributes. `newton:contact_ke`/`newton:contact_kd`/`newton:contact_kf`/`newton:contact_ka` on shape prims are deprecated in favor of `newton:contactStiffness`/`newton:contactDamping`/`newton:contactFrictionGain`/`newton:contactAdhesion` on the bound material. (#3005)
- USD authors should put material-level contact stiffness/damping on bound `NewtonMaterialAPI` materials, or use per-shape `mjc:solref` (`MjcGeomAPI`) when per-shape MuJoCo settings are needed. Material-level `mjc:solref` is deprecated in favor of `newton:contactStiffness`/`newton:contactDamping` on the bound material. (#3005)
- Deprecate module-level BVH helpers `newton.geometry.build_bvh_shape()`, `refit_bvh_shape()`, `build_bvh_particle()`, and `refit_bvh_particle()` in favor of `Model.bvh_build_shapes()`, `Model.bvh_refit_shapes()`, `Model.bvh_build_particles()`, and `Model.bvh_refit_particles()`. (#3039)
- Deprecate `Model.has_heightfields` in favor of `Model.heightfield_count`; use `model.heightfield_count > 0` for boolean checks. (#3039)
- Users reading `SDF.texture_block_coords` should stop depending on it. The attribute now always returns `None` and will be removed in a future release because the hydroelastic broadphase derives block coordinates arithmetically from each SDF's coarse-texture dimensions. (#2701)
- Users reading `Model.sdf_block_coords` and `Model.sdf_index2blocks` should stop depending on these internal broadphase caches. Both attributes are now lazily recomputed from each SDF's coarse-texture dimensions (matching the new broadphase semantics) and will be removed in a future release. (#2701, #3072)
- Users reading public SDF storage attributes should stop depending on them directly. Public access to `Model.shape_sdf_index`, `Model.texture_sdf_data`, `Model.texture_sdf_coarse_textures`, `Model.texture_sdf_subgrid_textures`, and `Model.texture_sdf_subgrid_start_slots` is deprecated because these are implementation details of the SDF contact pipeline. The public aliases keep working during the deprecation window and will be removed in a future release. (#2701, #3072)

### Removed

- Remove `SensorContact.net_force` (deprecated in 1.1.0); use `SensorContact.total_force` and `SensorContact.force_matrix` instead. (#2945)
- Remove `SensorContact(include_total=...)` (deprecated in 1.1.0); use `SensorContact(measure_total=...)` instead. (#2945)
- Remove `SensorContact.sensing_objs` (deprecated in 1.1.0); use `SensorContact.sensing_indices` and `SensorContact.sensing_type` instead. (#2945)
- Remove `SensorContact.counterparts` and `SensorContact.reading_indices` (deprecated in 1.1.0); use `SensorContact.counterpart_indices` and `SensorContact.counterpart_type` instead. (#2945)
- Remove `SensorContact.shape` (deprecated in 1.1.0); use `total_force.shape` / `force_matrix.shape` instead. (#2945)
- Remove `SensorContact.ObjectType` enum (deprecated in 1.1.0); use the `sensing_type` and `counterpart_type` attributes instead. (#2945)
- Remove internal `raycast_kernel_no_hfield`; raycast code now uses `raycast_kernel` with heightfield raycasting routed through the mesh BVH path. (#2971)
- Remove the examples helper `newton.examples.compute_world_offsets`; use `ModelBuilder.replicate()` when duplicating a model, or `newton.utils.compute_world_offsets(world_count, spacing, up_axis=...)` if you only need placement offsets. (#3075)

### Removed

- Remove the `SolverPhoenX` constraint clustering pipeline (`ConstraintClusterBuilder`, `SupernodalElements`, `ClusteringPipeline`), the `enable_clustering` parameter on `SolverPhoenX` and `PhoenXWorld`, the `--enable-clustering` CLI flag on `example_soft_body_drop`, the `cluster_aware` axis and `cluster_members` parameter on the single-world PGS kernels, and the `StepReport.num_clusters` field. The cluster-aware dispatch path was 10x slower than the non-clustered path on dense soft-body scenes and never produced a speedup on any tested scene; the parallelism loss from serializing up to 4 cluster members per thread could not be recovered without implementing body-fetch amortization (the paper's core mechanism), which would still not flip the sign on dense graphs

### Fixed

- Fix the experimental PhoenX G1 PPO environment so `sim_substeps` is
  the total number of PhoenX physics steps per policy step. Previously the
  environment loop and internal SolverPhoenX substep loop were nested, so the
  default recipe ran 25 internal substeps when it requested 5.

- Restore ModelBuilder triangle and tetrahedron shape helpers so primitive examples run.
- Fix `SolverPhoenX` single-world joint-only scenes so rigid joints remain active when no contact columns are allocated.
- Fix `SolverPhoenX` sleep-enabled contact stacks so graph-stable contact history does not carry cross-frame warm-start impulses that can prevent quiet towers from resleeping.
- Fix `SolverPhoenX` D6 ball/universal reductions to preserve angular limits, so MJCF humanoid joints no longer dispatch through unconstrained ball sockets.
- Fix `SolverPhoenX` joint stiffness and multi-world contact isolation regressions in robot scenes.
- Fix `SolverPhoenX` fast-tail multi-world solve dispatch to recover robot-fleet performance without regressing joint stiffness.
- Fix `example_recording` PhoenX mode to use unified model contacts and avoid nesting internal PhoenX substeps inside the MuJoCo outer loop.
- Fix `SolverPhoenX` drive target handling to use `Model.joint_target_q` / `Control.joint_target_q`, including coord-layout target indices after free joints.
- Fix `SolverPhoenX` ignoring `Model.joint_damping` on revolute/prismatic and D6 ball/universal joints, which destabilized damped MJCF humanoids.
- Refresh PhoenX's baked-in armature on `SolverNotifyFlags.JOINT_DOF_PROPERTIES`: `SolverPhoenX.notify_model_changed` now resets body inertia to `Model.body_inv_inertia` and re-bakes `model.joint_armature` whenever joint-DOF properties change, so domain-randomization edits to armature between episodes actually take effect
- Fix `ModelBuilder.finalize(skip_shape_contact_pairs=True)` so large SAP/NXN benchmark scenes can skip O(n^2) explicit shape-pair precomputation.
- Fix `SolverMuJoCo` returning `State.joint_qd` in world frame for root `FREE` joints with non-identity `parent_xform`, violating the documented parent-frame contract and corrupting derived `body_qd`.
- Fix `SolverPhoenX` infinite loop in `IncrementalContactPartitioner.build_csr_greedy_with_jp_fallback()` when `speculative_coloring=True`, `capture_while_greedy_coloring=True`, and `max_colored_partitions=0` (the all-overflow mass-splitting configuration used by `example_soft_body_drop`). `speculative_validate_commit_kernel` now exempts the overflow colour from neighbour conflict checks — mass splitting resolves intra-bucket conflicts via copy states, so multiple neighbouring elements are allowed to share that colour. Previously the first iteration coloured one element per neighbourhood and every subsequent iteration rejected the rest forever, hanging the partitioner inside `wp.capture_while`
- Fix `SensorTiledCamera` not rendering heightfield (`HFIELD`) shapes, which were missing from the render BVH. Heightfields are now rendered through the existing mesh path (they are triangulated `wp.Mesh` shapes), which also resolves a tiled-camera render-performance regression caused by the unused heightfield branch lowering the render kernel's GPU occupancy.
- Fix `SolverMuJoCo` passing `numpy.bool_` scalars for the `mocap` and `actgravcomp` parameters when building the MuJoCo spec, causing a `DeprecationWarning` under NumPy 2.2 and silent behavioral breakage under NumPy 2.5 where boolean scalars are no longer interpreted as integer indices.
- Fix `eval_fk()` overwriting VBD-simulated `JointType.CABLE` body poses.
- Fix hydroelastic SDF contact surfaces dropping the central region under deep interpenetration. The broadphase used to skip subgrids whose centers were deeper than the SDF narrow band, leaving a hole in the contact patch when overlap exceeded the narrow-band thickness. Broadphase now visits every subgrid in the SDF coarse grid (block coordinates are derived arithmetically from per-shape SDF coarse-texture dimensions); sampling at far-inside locations is correct because the coarse SDF is dense and accurate everywhere. On-disk SDF caches written by earlier versions are transparently re-cooked on first load (`_sdf_cache.CACHE_FORMAT_VERSION` bumped to `2`)
- Fix `SolverXPBD` `body_parent_f` reporting to include `Control.joint_f` contributions and accumulate multiple inbound joint contributions, matching the `SolverMuJoCo` and `SolverFeatherstone` convention.
- Fix MJCF `xyaxes` parsing to treat the second vector as Y and derive Z from X cross Y.
- Fix MJCF `euler` producing wrong orientations for multi-component angles (used extrinsic instead of intrinsic rotation).
- Fix MJCF reading only the first `<compiler>` element; attributes are now merged across all elements in document order including `<include>`-expanded children.
- Fix MJCF worldbody static geoms bypassing the visual/collider class filter, so `parse_visuals=False` now drops visual-class geoms attached directly to `<worldbody>` too (previously only filtered geoms inside `<body>` elements).
- Fix `ViewerFile` playback dropping namespaced custom attributes (e.g. `model.mujoco.geom_solimp`) when restoring into a fresh `Model`.
- Fix mesh-SDF contacts with positive contact gaps by making contact reduction prefer margin-depth contacts over gap-only directional fallbacks. `SolverPhoenX` now treats positive-gap contacts as detection-only rows: they remain available for later substeps, but apply no impulse until the current substep pose is penetrating.
- Fix `brick_stacking` example contact gaps to avoid oversized contact envelopes around the robot, table, and ground.
- Fix `ModelBuilder.collapse_fixed_joints()` producing a NaN center of mass when collapsing joints between zero-mass bodies.
- Fix mesh and convex-mesh contact sign classification for watertight meshes with nearby opposing surfaces or inconsistent triangle winding.
- Fix `SolverMuJoCo` returning `State.joint_qd` in world frame for root `FREE` joints with non-identity `parent_xform`, violating the documented parent-frame contract and corrupting derived `body_qd`.
- Fix `SolverVBD` custom attribute setup so `vbd:joint_is_hard` can be authored without implicitly enabling Dahl cable friction by calling `SolverVBD.register_custom_attributes(..., dahl_defaults_enabled=False)`.
- Fix `example_softbody_gift` emitting spurious non-manifold edge warnings caused by mismatched 5-tet diagonals across adjacent cubes in the soft body mesh.
- Fix `basic_conveyor` example emitting a spurious inertia validation warning at finalize.
- Fix `SolverKamino` contact anchors being shifted off the geometry surface by `ShapeConfig.margin`, which biased friction.
- Fix `ViewerGL.get_frame()` crashing when a CPU model is rendered while a CUDA context is active.
- Fix `SensorTiledCamera` not rendering heightfield (`HFIELD`) shapes, which were missing from the render BVH. Heightfields are now rendered through the existing mesh path (they are triangulated `wp.Mesh` shapes), which also resolves a tiled-camera render-performance regression caused by the unused heightfield branch lowering the render kernel's GPU occupancy. (#3088)
- Fix hydroelastic SDF contact surfaces dropping the central region under deep interpenetration. The broadphase used to skip subgrids whose centers were deeper than the SDF narrow band, leaving a hole in the contact patch when overlap exceeded the narrow-band thickness. Broadphase now visits every subgrid in the SDF coarse grid (block coordinates are derived arithmetically from per-shape SDF coarse-texture dimensions); sampling at far-inside locations is correct because the coarse SDF is dense and accurate everywhere. On-disk SDF caches written by earlier versions are transparently re-cooked on first load (`_sdf_cache.CACHE_FORMAT_VERSION` bumped to `2`). (#2701)
- Fix mesh and convex-mesh contact sign classification for watertight meshes with nearby opposing surfaces or inconsistent triangle winding. (#3004)
- Fix `ModelBuilder.collapse_fixed_joints()` producing a NaN center of mass when collapsing joints between zero-mass bodies
- Fix USD import losing authored negative scales on shape and parent xforms, so mirrored primitives and meshes are now imported with the correct signed scale. (#2936)
- Respect USD color-space metadata for scalar material colors and convert linear-authored USD color textures to display space when loading them. (#2411)
- Fix USD import of orphaned body-to-world fixed joints not accounting for ancestor xform offsets, so pinned bodies now FK to the correct world pose (env origin + spawn xform). (#2974)
- Fix USD import of revolute and D6-angular joint `limit_ke` / `limit_kd` from `mjc:solreflimit` being over-scaled by ~57x. (#2736)
- Fix `SolverMuJoCo` passing `numpy.bool_` scalars for the `mocap` and `actgravcomp` parameters when building the MuJoCo spec, causing a `DeprecationWarning` under NumPy 2.2 and silent behavioral breakage under NumPy 2.5 where boolean scalars are no longer interpreted as integer indices. (#3098)
- Fix MJCF `xyaxes` parsing to treat the second vector as Y and derive Z from X cross Y
- Fix `SolverMuJoCo` returning `State.joint_qd` in world frame for root `FREE` joints with non-identity `parent_xform`, violating the documented parent-frame contract and corrupting derived `body_qd`. (#2871)
- Fix mesh-SDF contacts with positive contact gaps by making contact reduction prefer margin-depth contacts over gap-only directional fallbacks.
- Fix MJCF joint `damping` attribute being ignored by `SolverFeatherstone`
- Fix `SolverMuJoCo` generated MuJoCo joint names for multi-axis D6 joints to avoid duplicate names
- Fix `SolverMuJoCo` ball-joint frame conversion: `joint_q` and position-target quaternions were applied in the wrong basis when `child_xform` had a non-identity rotation, and `joint_qd` / velocity targets / applied / actuator torques were applied and read back in the wrong basis whenever the ball was away from its rest pose. (#2981)
- Fix `SolverMuJoCo` to preserve authored zero-valued USD `mjc:solreflimit` values as raw MuJoCo data and avoid writing limit `solref` values to unlimited joints in saved MJCF
- Honor authored `mujoco.solreflimit_mode` even when a non-zero `mujoco.solreflimit` is also present, so the explicit mode (force-space or raw) is authoritative
- Fix `SolverMuJoCo` CPU backend overwriting `mj_model.body_iquat` with Newton's eigendecomposition result on every `BODY_INERTIAL_PROPERTIES` notify; the compiled principal-axes basis is now preserved, fixing single-contact box equilibrium (incorrect normal force) and stiff WELD-loop instabilities (`Nan, Inf or huge value in QACC`) caused by basis ambiguity on repeated eigenvalues. (#3018)
- Fix `SolverMuJoCo` CPU backend dynamics for asymmetric MJCF `diaginertia` models whose principal moments are reordered during Newton/MJWarp synchronization
- Fix `SolverMuJoCo` honoring force-space `shape_material_ke` / `shape_material_kd` for contacts (`use_mujoco_contacts=False`); authored `mjc:solref` is preserved via new `mujoco.solref` / `mujoco.solref_mode` per-shape custom attributes. Force-space scaling is unsupported on `use_mujoco_contacts=True` and the MuJoCo CPU backend. (#2964)
- Fix MJCF importer rejecting MuJoCo's one-value `solreflimit`/`solref` shorthand emitted by `mujoco.MjSpec.to_xml()` for default-valued joints, which broke `SolverMuJoCo(save_to_mjcf=...)` round-trips
- Fix `eval_fk()` overwriting VBD-simulated `JointType.CABLE` body poses
- Fix `SolverVBD` custom attribute setup so `vbd:joint_is_hard` can be authored without implicitly enabling Dahl cable friction by calling `SolverVBD.register_custom_attributes(..., dahl_defaults_enabled=False)`
- Fix `SolverKamino` contact anchors being shifted off the geometry surface by `ShapeConfig.margin`, which biased friction
- Fix rigid-rigid friction in `SolverVBD` for contacts with nonzero `rigid_contact_offset0/rigid_contact_offset1`
- Fix `SolverXPBD` `body_parent_f` reporting to include `Control.joint_f` contributions and accumulate multiple inbound joint contributions, matching the `SolverMuJoCo` and `SolverFeatherstone` convention
- Fix `SolverXPBD` tetrahedral constraints reading static model activations instead of runtime control activations
- Fix `SolverXPBD` tetrahedral constraints ignoring FEM material stiffness and damping
- Fix `ViewerFile` playback dropping namespaced custom attributes (e.g. `model.mujoco.geom_solimp`) when restoring into a fresh `Model`
- Fix `ViewerGL` GUI rendering at half size on HiDPI / Retina displays by scaling the ImGui style, fonts, sidebar width, and `log_image` window/tile/spacing constants with pyglet's `window.scale` (with framebuffer-to-window ratio as a fallback). DPI changes are tracked at runtime via the pyglet `on_scale` event so the GUI follows the window across displays with different scaling. (#2926)
- Fix ground grid clipping in the Viser renderer
- Fix `brick_stacking` example contact gaps to avoid oversized contact envelopes around the robot, table, and ground
- Fix `example_softbody_gift` emitting spurious non-manifold edge warnings caused by mismatched 5-tet diagonals across adjacent cubes in the soft body mesh
- Fix `basic_conveyor` example emitting a spurious inertia validation warning at finalize

## [1.2.1] - 2026-06-04

### Added

- Add `ArticulationView.joint_labels`, `link_labels` (aliased as `body_labels`), and `shape_labels` exposing the full template-articulation labels alongside the existing leaf-only `*_names`, so callers can disambiguate selected entries whose leaf names collide.

### Fixed

- Fix mesh-convex and heightfield-convex contacts missing when shapes are separated by margin but still within the contact envelope.
- Fix `ArticulationView` link selections for closed-loop joints so BODY-frequency accessors expose each physical body once.
- Fix USD visual mesh imports to preserve face-material `UsdGeom.Subset` colors and textures by splitting material subsets into separate render meshes; collision/physics import behavior is unchanged

## [1.2.0] - 2026-05-12

### Added

- Add linear HDR color output support to `SensorTiledCamera` via `hdr_color_image`.
- Add composable actuator subsystem with pluggable `Controller` (`ControllerPD`, `ControllerPID`, `ControllerNeuralMLP`, `ControllerNeuralLSTM`), `Clamping` (`ClampingMaxEffort`, `ClampingDCMotor`, `ClampingPositionBased`), and `Delay` components; supports per-DOF delays, CUDA graph capture, and masked environment reset
- Add heatmap rendering for scalar arrays logged through `ViewerGL.log_array()`
- Add Blender-style orbit, pan, and dolly controls to the GL viewer using middle-mouse drag combinations
- Add `SolverXPBD.update_contacts()` to populate `contacts.force` with per-contact spatial forces (linear force and torque) derived from XPBD constraint impulses
- Add `body_parent_f` extended state attribute support to `SolverXPBD` so it populates per-body incoming joint wrenches in world frame at the body's COM (matches `SolverMuJoCo`'s convention; values are approximate due to XPBD's relaxation and non-momentum-conserving nature)
- Add `body_parent_f` extended state attribute support to `SolverFeatherstone` populated directly from the RNEA backward pass (per-body net spatial wrench translated to the body's COM, matching the `SolverMuJoCo` convention)
- Add public `newton.geometry.build_bvh_shape()`, `build_bvh_particle()`, `refit_bvh_shape()`, and `refit_bvh_particle()` helpers for managing model BVHs
- Raise process priority automatically in `--benchmark` mode for more stable measurements; add `--realtime` for maximum priority.
- Import per-shape authored color from USD stages into `ModelBuilder.shape_color`
- Add `TRIANGLE_PRISM` support-function type for heightfield triangles, extruding 1 m along the heightfield's local -Z so GJK/MPR naturally resolves shapes on the back side
- Add `ViewerGL.log_scalar()` for live scalar time-series plots in the viewer
- Add `Mesh.is_watertight` property (cached) that reports whether every geometric edge is shared by exactly two triangles
- Add `HydroelasticSDF.Config.mc_edge_clamp_min` to expose the marching-cubes edge-interpolation clamp; default `0.02` matches the previous hard-coded value. Set to `0.0` to disable the clamp and recover faithful contact-surface dynamics for threading-style scenarios (#2702)
- Add `deterministic` flag to `CollisionPipeline` and `NarrowPhase` for GPU-thread-scheduling-independent contact ordering via radix sort and deterministic fingerprint tiebreaking in contact reduction
- Add `shape_pairs_max` override on `CollisionPipeline` to cap the SAP/NXN broad-phase candidate-pair buffer below the worst-case `N*(N-1)/2` per-world bound, avoiding multi-GB allocations on large sparse scenes (a too-small value triggers a runtime overflow warning)
- Add fast parity-based SDF construction path for watertight meshes in `SDF.create_from_mesh`, using `wp.mesh_query_point_sign_parity` instead of winding numbers; selected via the new `sign_method` argument (`"auto"` — the default — picks parity when `Mesh.is_watertight` is true, or `"parity"` / `"winding"` to force either strategy)
- Add `Viewer.log_image()` for displaying single or batched images in `ViewerGL`; other backends inherit a no-op. Also add `SensorTiledCamera.utils.to_rgba_from_color()`, `to_rgba_from_normal()`, `to_rgba_from_depth()`, and `to_rgba_from_shape_index()` (hash palette or caller-provided RGB lookup) adapters producing output consumable by `log_image()`.
- Add on-disk caching of cooked texture-based SDFs via the new `cache_dir` argument on `SDF.create_from_mesh` and `Mesh.build_sdf`. Cached entries are content-addressed by mesh and build parameters, written atomically as a single uncompressed `.npz`, and versioned via `CACHE_FORMAT_VERSION` so format changes invalidate stale caches transparently
- Enable CPU execution of the collision pipeline, including mesh–mesh and mesh–heightfield SDF contacts and contact reduction (`reduce_contacts`) that were previously CUDA-only, by replacing the CUDA `__shared__` fast paths in `sdf_contact.py`, `multicontact.py`, and `collision_core.py` with portable `wp.tile_stack` / `wp.tile_mesh_query_aabb` primitives. CPU runs now execute the same kernels as CUDA; the previous `"NarrowPhase running on CPU: mesh-mesh contacts will be skipped"` warning is no longer emitted.
- Add `ViewerBase.log_arrows()` for arrow rendering (wide line + arrowhead) in the GL viewer with a dedicated geometry shader
- Add frame-to-frame contact matching via `CollisionPipeline(contact_matching=...)` with modes `"latest"` (populates `contacts.rigid_contact_match_index`) and `"sticky"` (experimental; additionally replays previous-frame contact geometry on matched contacts — the sticky update strategy may change without warning). Optional `contact_report=True` exposes new/broken contact index lists on `Contacts`.
- Add JP-MIS + greedy graph coloring to PhoenX (`SolverPhoenX`) via `IncrementalContactPartitioner.build_csr_greedy()` and `_per_world_greedy_coloring_kernel` for `step_layout="multi_world"`, default-on through `solver_config.PHOENX_USE_GREEDY_COLORING`. Each MIS commit picks the smallest color not used by its already-colored neighbors instead of the round number, dropping color counts to (or near) the `max_body_degree` lower bound — Kapla Tower goes from 78 to 28 colors (3.04x → 1.00x optimum), Kapla 30 frames runs in 7.7 s vs 10.4 s for the round-based JP. Drops the per-round `remaining_ids` compact kernel and uses a persistent grid-stride loop sized 1 block/SM with a fused single-thread post-pass (count + exclusive scan) for the CSR build. Capped at 64 colors via a single-`int64` forbidden mask; falls back to a descriptive error if exceeded.
- Add `max_thread_blocks` parameter to `SolverPhoenX` (and underlying `PhoenXWorld`) for capping the persistent grid used by the single-world PGS sweeps (constraint prepare, main iterate, velocity relax). Defaults to `None` (auto-size as `clamp(ceil(cap / 256), 32, 4 * sm_count)` blocks); when set, sizes the grid to `min(ceil(cap / 256), max_thread_blocks)` blocks, bypassing the 32-block floor and SM-derived ceiling. Useful for sharing the GPU with a co-resident workload or measuring SM occupancy. No effect on `step_layout="multi_world"`.
- Add `enable_multiccd` parameter to `SolverMuJoCo` for multi-CCD contact generation (up to 4 contact points per geom pair)
- Reimplement PhoenX `JOINT_MODE_CABLE` as a positional 3+2+1 anchor split with PD-soft bend / twist rows (previously a rigid ball-socket plus angular log-map rows). Newton `JointType.CABLE` joints are routed through this formulation, which avoids the prior log-map convergence degradation at high bend stiffness. Adds an analytical CABLE validation suite (`test_cable_joint.py`) covering bend / twist stiffness and damping vs damped-harmonic-oscillator theory, gain conversion across `rest_length`, multi-segment cantilever Euler-Bernoulli static deflection, and chain torque transmission, plus Newton-side adapter wiring tests (`test_cable_joint_newton.py`)
- Add VBD rigid-contact warm-starting via `rigid_contact_history`, backed by `Contacts.rigid_contact_match_index` from `CollisionPipeline(contact_matching="latest")`.
- Add VBD hard/soft controls for body-body contacts and structural joint slots, including `rigid_contact_hard`, `SolverVBD.set_joint_constraint_mode()`, and `SolverVBD.JointSlot`
- Add AVBD contact/joint alpha overrides and linear/angular beta overrides to `SolverVBD` for stabilization and penalty-ramping control
- Add `enable_multiccd` parameter to `SolverMuJoCo` for multi-CCD contact generation (up to 4 contact points per geom pair)
- Warn when `SolverMuJoCo` detects installed `mujoco` or `mujoco-warp` versions that do not satisfy `pyproject.toml` requirements
- Support `<joint type="ball"/>` in the MJCF importer, and preserve authored damping, stiffness, and frictionloss when exporting ball joints to MuJoCo specs (previously silently dropped)
- Add `ViewerViser.log_scalar()` for live scalar time-series plots via uPlot
- Honor `UsdGeomImageable` visibility (including inherited `invisible`) on USD prims imported via `ModelBuilder.add_usd()`; visual shapes, gaussian splats, and collider shapes are imported with `ShapeFlags.VISIBLE` cleared when the prim is effectively invisible, while collision behavior is preserved
- Add import of `UsdGeom.TetMesh` prims as soft meshes through `ModelBuilder.add_usd()`
- Add site-targeted actuator support to MuJoCo solver
- Add USD parsing support for equality constraints based on the `MjcEquality` schema
- Add more solver options to implicit MPM: `gs-soa` (or `gauss-seidel-soa`) for improved memory coalescing, `gs-batched` (or `gauss-seidel-batched`) merging GS colors with Jacobi-style mass-split parallelism, plus `cr` (Conjugate Residual) and `gmres` linear solver options.
- Add frame-by-frame step support to `ViewerGL`: press `.` while paused to advance one simulation frame
- Add ViewerBase.should_step() — call once per frame to determine whether the simulation loop should advance; returns True when not paused.
- Add Kamino-specific simulation examples in `newton/examples/kamino`
- Add per-mesh `color` override to `ViewerBase.log_mesh()` for tinting individual meshes without authoring per-vertex colors
- Add per-mesh `roughness` and `metallic` PBR overrides to `ViewerBase.log_mesh()`

### Changed

- Use pre-computed local AABB for `CONVEX_MESH` shapes in `compute_shape_aabbs`, avoiding a per-frame support-function AABB computation
- Build mesh SDFs via the texture-based sparse path only; sample via `SDF.texture_data` instead of `SDF.sparse_volume` / `SDF.coarse_volume`.
- Change implicit MPM default `solver` from `"gs"` to `"auto"`, which selects `"gs"` for trilinear bases and `"gs-batched"` for higher-order ones. Set `solver="gs"` explicitly to restore the previous behavior.
- Change `SolverImplicitMPM.Config.solver` warmstart syntax from `+`-separated strings to ordered sequences; use `solver=("cg", "gauss-seidel")` instead of `solver="cg+gauss-seidel"`.
- Change implicit MPM default `collider_basis` from `"Q1"` to `"S2"` for improved contact quality; set `collider_basis="Q1"` explicitly to restore the previous behavior.
- Change GL viewer scroll to dolly toward the orbit pivot; use Ctrl+scroll for FOV zoom
- Render all GL viewer lines (joints, contacts, wireframes) as geometry-shader quads instead of ``GL_LINES`` for uniform width across zoom levels and non-square viewports
- Adjust grouping of `reset`, `step`, and `pause` controls so they appear together
- Bump `Pillow` floor to `>=11.3.0`
- Bump `jupyterlab` lower bound to `>=4.5.7` to pick up the fix for CVE-2026-40171
- Replace `ModelBuilder.add_actuator(actuator_class, input_indices=..., output_indices=..., **kwargs)` with `ModelBuilder.add_actuator(controller_class, index=..., clamping=[...], delay_steps=..., pos_index=..., **ctrl_kwargs)` where each call registers a single DOF
- Change `ArticulationView.get_actuator_parameter(actuator, name)` and `set_actuator_parameter(actuator, name, values)` to require a `component` argument identifying the owning `Controller`, `Clamping`, or `Delay` instance: `get_actuator_parameter(actuator, actuator.controller, "kp")`
- Update default environment map texture in GL viewer (source: https://polyhaven.com/a/brown_photostudio_02)
- Remove the implicit-MPM rasterized collider's reliance on Warp's `warp.fem` module (behavior unchanged)
- Replace the StVK VBD triangle membrane material with the stable Neo-Hookean form (Smith et al. 2018, adapted to 2D shells). The upstream two-constraint Rayleigh damping model is preserved unchanged
- Bump `mujoco` and `mujoco-warp` dependencies to `~=3.8.0` (`mujoco-warp` requires `>=3.8.0.3`)
- Bump `GitPython` lower bound to `>=3.1.47` to pick up the fix for GHSA-x2qx-6953-8485 (`multi_options` argument injection in `Repo.clone_from`)
- Bump `open3d` floor to `>=0.19.0`
- Bump `meshio` floor to `>=5.3.5`; `5.3.0` calls `np.string_` which was removed in NumPy 2.0
- Bump `newton-usd-schemas` to `>=0.2.0` introducing new experimental actuator schemas & re-aligning friction defaults
- Restrict `usd-core` to `<26.5` to avoid deprecation warnings introduced in 26.5
- Require explicit `SensorTiledCamera` BVH lifecycle management instead of implicit camera maintenance: call `newton.geometry.build_bvh_shape()` / `build_bvh_particle()` once after setup, then `refit_bvh_shape()` / `refit_bvh_particle()` before rendering frames that change geometry
- Increase conveyor rail roughness in `example_basic_conveyor` to reduce mirror-like reflections
- Remove visual-only procedural terrain from `example_robot_anymal_c_walk`
- Migrate all raycast logic to `geometry.raycast`, all raycast functions now return distance and normal information
- Disable process reuse in the test runner on multi-GPU systems to prevent CUDA errors from cascading across test suites, keeping process reuse enabled on single-GPU systems for faster throughput
- Default `python -m newton.examples` with no argument to launch `basic_pendulum`; use `--list` to print available examples
- Reduce default `stretch_stiffness` from `1.0e9` to `1.0e5` in `add_joint_cable()`, `add_rod()`, and `add_rod_graph()`
- Treat `stretch_stiffness` and `bend_stiffness` in `add_rod()` and `add_rod_graph()` as direct per-joint stiffness values, matching `add_joint_cable()` and other joint stiffness APIs
- VBD solver uses augmented-Lagrangian hard constraints for body-body contacts by default (`rigid_contact_hard=True`)
- Reduce collision-pipeline overhead in `SolverMuJoCo` via incremental contact conversion when the contact set is unchanged (~6× speedup on `example_robot_anymal_d` with 4096 worlds)

### Deprecated

- Deprecate the top-level `Model.equality_constraint_*` arrays and `Model.equality_constraint_count`, the `ModelBuilder.equality_constraint_*` accumulators, `ModelBuilder.add_equality_constraint{,_connect,_weld,_joint}()`, and the `Model.AttributeFrequency.EQUALITY_CONSTRAINT` enum, in favor of the namespaced `model.mujoco.equality_constraint_*` fields (custom attributes on the `"mujoco:equality_constraint"` frequency). Migrate reads and writes to `model.mujoco.equality_constraint_*`, and construct rows via `ModelBuilder.add_custom_values(**{"mujoco:equality_constraint_*": ...})`. The deprecated names forward to the namespace during the deprecation window and will be removed in a future release.
- Deprecate `SensorRaycast` in favor of `SensorTiledCamera`; migrate to `SensorTiledCamera.utils.compute_pinhole_camera_rays()` and `create_depth_image_output()` for single-camera depth rendering — see the `SensorRaycast` class docstring for a complete migration example
- Deprecate and ignore `rigid_enable_dahl_friction` in `SolverVBD`; Dahl friction is now auto-detected from model attributes (`model.vbd.dahl_eps_max` / `model.vbd.dahl_tau`)
- Deprecate `newton-actuators` package dependency; all actuator functionality is now built into `newton.actuators`. The dependency is kept for backward compatibility and will be removed in a future release; migrate imports from `newton_actuators` to `newton.actuators`

### Fixed

- Fix `remesh_convex_hull` raising `QhullError` on degenerate (coincident, collinear, or coplanar) point clouds; it now returns a zero-volume fallback mesh with a `UserWarning`, raises `ValueError` on empty input, and retries Qhull with `QJ` joggle as a last resort on the 3D path
- Fix narrow-phase CPU launches using GPU-sized block dimensions with kernels that observe `wp.block_dim() == 1`, avoiding out-of-bounds tile and strided-loop indexing until Warp GH-1413 is fixed
- Fix `ViewerGL` Step button remaining clickable while the simulation is running; the button is now greyed out when not paused
- Fix `ViewerGL` `Plots` window opening on top of the `Performance Stats` overlay by anchoring its default position to the bottom-right corner; user-dragged positions persisted in `imgui.ini` are unaffected
- Fix the example viewer's Reset button discarding user-provided CLI options (e.g. `--world-count`) and rebuilding the example with parser defaults instead
- Fix `SolverMuJoCo` Newton-contact conversion to use geometry-surface contact anchors
- Fix `ModelBuilder.finalize()` crashing with 3+ articulations after `collapse_fixed_joints()` reordered `articulation_start` and dropped per-articulation metadata
- Fix Sphinx docs builds to auto-discover bundled ``pypandoc_binary`` pandoc so notebook tutorials build without manual PATH configuration
- Fix `SolverStyle3D` initialization to precompute its fixed PD matrix from the finalized model
- Fix connect constraint anchor computation to account for joint reference positions when `SolverMuJoCo` is the chosen solver.
- Fix joint-synthesized CONNECT constraint anchors not updating when `dof_ref` or `joint_X_p` changes at runtime via `notify_model_changed()`
- Fix WELD constraint data corruption when a model contains both FIXED and revolute/ball loop joints
- Fix `SolverMuJoCo` passing non-zero geom/pair margins to `mujoco_warp.put_model()`, which fails when NATIVECCD is enabled. Margins are forced to zero when MuJoCo handles collisions (`use_mujoco_contacts=True`); the Newton collision pipeline (`use_mujoco_contacts=False`) is unchanged
- Fix `SolverMuJoCo` failing to compile planar mesh colliders with MuJoCo's convex-hull path when `use_mujoco_contacts=False`; use MuJoCo contacts only with non-planar mesh colliders, primitive planes, or thick proxy geometry
- Fix GPU illegal-memory-access in `SolverMuJoCo` Newton-contacts fast path when `notify_model_changed(BODY_INERTIAL_PROPERTIES | JOINT_DOF_PROPERTIES | MODEL_PROPERTIES)` was called between substeps (e.g. mass randomization in IsaacLab), or when the bound `Contacts` instance / MJWarp `naconmax` changed without invalidating the cached `tid_to_cid` mapping. The fast path is now invalidated on any property notify that affects cached MJWarp contact fields, and bounds-checks `cid` against `naconmax` defensively
- Fix `State.assign` not copying namespaced extended and custom state attributes
- Fix mesh-convex back-face contacts generating inverted normals that trap shapes inside meshes and cause solver divergence (NaN)
- Fix triangle-mesh-vs-convex collisions silently dropping all contacts under non-uniform (and even large uniform) mesh scale: the BVH AABB query in `mesh_vs_convex_midphase` is now performed in unscaled mesh-local space (matching the BVH built over `mesh.points`), with the per-axis contact gap converted accordingly. Previously the query was performed in scaled mesh-local space, so any convex shape whose unscaled-space AABB lay outside the BVH bounds would receive 0 triangles and 0 contacts.
- Fix finite plane geometry 2x too large in collision, bounding sphere, and raytrace sensor
- Fix MPR convergence failure on large and extreme-aspect-ratio mesh triangles by projecting the starting point onto the triangle nearest the convex center
- Fix MPR/GJK reporting wrong contacts for `CONVEX_MESH` shapes whose authoring origin lies outside the hull, and tighten heightfield-vs-convex midphase to use the convex's local AABB instead of an origin-centered bounding sphere
- Fix O(W²·S²) memory explosion in `CollisionPipeline` shape-pair buffer allocation for NXN and SAP broad phase modes by computing per-world pair counts instead of a global N²
- Fix non-determinism in `CollisionPipeline(contact_matching="sticky")` where the matcher's `atomic_min` claim tie-break used the unsorted narrow-phase thread id (which `wp.atomic_add` makes non-deterministic) instead of the contact's sort key, so two runs of the same scene could pick different winners and diverge across frames
- Fix the deterministic narrow-phase sort buffer being sized to the broad-phase candidate-pair bound (`N*(N-1)/2` per world for NXN/SAP) instead of `rigid_contact_max`, which wasted multi-GB of VRAM on scenes with thousands of shapes
- Fix `SensorRaycast` ignoring `PLANE` geometry
- Fix `nut_bolt_hydro` example threading regression where some nuts were pinned in static friction; nuts now thread reliably down the bolt under both MuJoCo and XPBD solvers (#2702)
- Fix VRAM leak when resetting examples that allocate large GPU state (e.g. `diffsim_bear`)
- Fix `SensorRaycast` and viewer picking ignoring `HFIELD` (heightfield) geometry
- Fix `SensorTiledCamera` textured albedo output rendering flat colors when color and normal outputs are disabled
- Fix URDF Collada visual meshes dropping diffuse texture bindings
- Fix `contacts_rj45_plug` example crashing on reset
- Fix `SolverMuJoCo` dependency version-mismatch warning being silently skipped when Newton is installed from a wheel
- Fix `ViewerGL.log_image()` windows persisting across example-browser switches and failing to re-open on re-entry after manual close, by clearing the image logger in `ViewerGL.clear_model()`
- Fix `ModelBuilder.add_shape_heightfield` `scale` being ignored by narrow-phase collision and raycast
- Fix `collision_filter_parent` silently ignored on joints to world (`parent=-1`); now honored for all `add_joint_*` methods, with `add_joint_fixed(parent=-1, ...)` defaulting to filter child shapes against world-static shapes
- Fix multi-world `qfrc_actuator` conversion using the wrong body center of mass for worlds with `worldid > 0`
- Fix `SolverMuJoCo.__init__` time scaling with `world_count × actuators_per_world` instead of `actuators_per_world` by vectorizing the template-world filter for site-targeted actuators
- Fix compressed tets in `evaluate_volumetric_neo_hookean_force_and_hessian` producing an indefinite Hessian by clamping the cofactor-derivative coefficient to `max(0, s)`, removing a contribution that could corrupt the VBD inner solve
- Fix SDF hydroelastic broadphase scatter kernel using a grid-stride loop with binary search instead of per-pair thread launch
- Fix box support-map sign flips from quaternion rotation noise (~1e-14) producing invalid GJK/MPR contacts for face-touching boxes with non-trivial base rotations
- Fix `SolverPhoenX.update_contacts()` to produce correctly-signed forces aligned with `rigid_contact_shape0/1`/`rigid_contact_normal`: negate to match Newton's "force on shape0" convention (was reporting force on shape1), and honor the body-pair-grouping sort permutation so `contacts.force[k]` lines up with the narrow-phase contact slot. Enables `SensorContact` to work with `SolverPhoenX`, including for compound bodies
- Fix USD import of multi-DOF joints from MuJoCo-converted assets where multiple revolute joints between the same two bodies caused false cycle detection; merge them into D6 joints with correct DOF label mapping for MjcActuator target resolution
- Fix USD `MjcActuator` import so position and velocity actuators populate Newton's joint target arrays and can be driven via `Control.joint_target_pos` / `Control.joint_target_vel`
- Fix MJCF importer creating finite planes from MuJoCo visual half-sizes instead of infinite planes
- Fix USD custom-frequency parsing to respect `ModelBuilder.add_usd(root_path=...)`, avoiding rows from sibling subtrees
- Fix USD import of joint limit stiffness/damping from `MjcJointAPI`: `SchemaResolverMjc` now reads the schema-correct `mjc:solreflimit` attribute instead of the generic `mjc:solref`, which was never authored on joints
- Fix MJCF importer in `compiler.angle="degree"` mode: (1) stop multiplying joint `damping`/`stiffness` by `180/π` (MuJoCo stores these in `N·m·s/rad` and `N·m/rad` regardless of `angle`); (2) stop `deg2rad`-scaling the default `±MAXVAL` sentinel for joints without an explicit `range=`, which was turning unlimited hinges into bounded joints with `~1.75e8 rad` range
- Fix MJCF importer ignoring explicit `mass=` on visual geoms loaded via `parse_visuals=True`; authored visual-only mass now contributes to body mass and inertia like visual-only density already does
- Fix ViewerViser mesh popping artifacts caused by viser's automatic LOD simplification creating holes in complex geometry
- Fix ViewerViser notebook recording playback to load the matching browser client from the installed `viser` package and bind the playback HTTP server to loopback only
- Fix rendering of planes in ViewerViser as finite grids of line segments to prevent flickering artifacts
- Fix degenerate zero-area triangles in SDF marching-cubes isosurface extraction by clamping edge interpolation away from cube corners and guarding against near-zero cross products
- Fix multi-world coordinate conversion using the wrong body center of mass for replicated worlds
- Fix MJCF importer ignoring `<default><equality/></default>` attribute defaults (e.g. `solref`, `solimp`) for `<connect>`/`<weld>`/`<joint>` equality constraints
- Remove incorrect body-level `mjc:damping` -> `rigid_body_linear_damping` mapping from `SchemaResolverMjc`; `mjc:damping` is defined on `MjcJointAPI`, not on bodies
- Fix `target_voxel_size` being silently ignored on the texture-SDF path of `SDF.create_from_mesh()` and on the primitive-mesh path in `ModelBuilder`; the requested voxel resolution is now honored end-to-end and matches the sparse-SDF path
- Fix material-combination inconsistency in the Newton-to-`mujoco-warp` contact converter so combined friction / solref / solimp values match native MuJoCo
- Fix `eq_objtype` mismatch for joint equality and mimic constraints in `SolverMuJoCo` so compiled models match native MuJoCo XML behavior
- Fix implicit-MPM rheology solver launch-dim handling under `warp-lang` 1.13's templated `launch_bounds` (formerly produced out-of-bounds reads)
- Fix `SolverKamino.reset` clobbering `q_j_p`, `q_j`, and `dq_j` for worlds outside `world_mask` when `joint_q`/`joint_u` targets were provided. The previous unmasked writes broke TWOPI revolute-joint angle unwrapping after partial-mask resets.

## [1.1.0] - 2026-04-13

### Added

- Add repeatable `--warp-config KEY=VALUE` CLI option for overriding `warp.config` attributes when running examples
- Add 3D texture-based SDF, replacing NanoVDB volumes in the mesh-mesh collision pipeline for improved performance and CPU compatibility.
- Parse URDF joint `limit effort="..."` values and propagate them to imported revolute and prismatic joint `effort_limit` settings
- Add `--benchmark [SECONDS]` flag to examples for headless FPS measurement with warmup
- Interactive example browser in the GL viewer with tree-view navigation and switch/reset support
- Add `TetMesh` class and USD loading API for tetrahedral mesh geometry
- Support kinematic bodies in VBD solver
- Add brick stacking example
- Add box pyramid example and ASV benchmark for dense convex-on-convex contacts
- Add plotting example showing how to access and visualize per-step simulation diagnostics
- Add `exposure` property to GL renderer
- Add `snap_to` argument to `ViewerGL.log_gizmo()` to snap gizmos to a target world transform when the user releases them
- Expose `gizmo_is_using` attribute to detect whether a gizmo is actively being dragged
- Add per-axis gizmo filtering via `translate`/`rotate` parameters on `log_gizmo`
- Add conceptual overview and MuJoCo Warp integration primer to collision documentation
- Add configurable velocity basis for implicit MPM (`velocity_basis`, default `"Q1"`) with GIMP quadrature option (`integration_scheme="gimp"`)
- Add plastic viscosity, dilatancy, hardening and softening rate as per-particle MPM material properties (`mpm:viscosity`, `mpm:dilatancy`, `mpm:hardening_rate`, `mpm:softening_rate`)
- Add MPM beam twist, snow ball, and viscous coiling examples
- Add support for textures in `SensorTiledCamera` via `Config.enable_textures`
- Add `enable_ambient_lighting` and `enable_particles` options to `SensorTiledCamera.Config`
- Add `SensorTiledCamera.utils.convert_ray_depth_to_forward_depth()` to convert ray-distance depth to forward (planar) depth
- Add `newton.geometry.compute_offset_mesh()` for extracting offset surface meshes from any collision shape, and a viewer toggle to visualize gap + margin wireframes in the GL viewer
- Add differentiable rigid contacts (experimental) with respect to body poses via `CollisionPipeline` when `requires_grad=True`
- Add per-shape display colors via `ModelBuilder.shape_color`, `Model.shape_color`, and `color=` on `ModelBuilder.add_shape_*`; mesh shapes fall back to `Mesh.color` when available and viewers honor runtime `Model.shape_color` updates
- Add `ModelBuilder.inertia_tolerance` to configure the eigenvalue positivity and triangle inequality threshold used during inertia correction in `finalize()`
- Add `ViewerBase.set_visible_worlds()` for runtime control of which worlds are rendered, replacing the static `max_worlds` parameter
- Add `compute_normals` and `compute_uvs` optional arguments to `Mesh.create_heightfield()` and `Mesh.create_terrain()`
- Pin `newton-assets` and `mujoco_menagerie` downloads to specific commit SHAs for reproducible builds (`NEWTON_ASSETS_REF`, `MENAGERIE_REF`)
- Add `ref` parameter to `download_asset()` to allow overriding the pinned revision
- Add `total_force_friction` and `force_matrix_friction` to `SensorContact` for tangential (friction) force decomposition
- Add Gaussian Splat geometry support via `ModelBuilder.add_shape_gaussian()` and USD import
- Add configurable Gaussian sorting modes to `SensorTiledCamera`
- Add automatic box, sphere, and capsule shape fitting for convex meshes during MJCF import
- Add color and texture reading to `usd.utils.get_mesh()`
- Export `ViewerBase` from `newton.viewer` public API
- Add `custom_attributes` argument to `ModelBuilder.add_shape_convex_hull()`
- Add RJ45 plug-socket insertion example with SDF contacts, latch joint, and interactive gizmo

### Changed

- Require `mujoco ~=3.6.0` and `mujoco-warp ~=3.6.0` (previously 3.5.x)
- Replace `plyfile` dependency with `open3d` for mesh I/O. Users who depended on `plyfile` transitively should install it separately.
- Switch Python build backend from `hatchling` to `uv_build`
- Switch mesh-SDF collision from triangle-based gradient descent to edge-based Brent's method to reduce contact jitter
- Unify heightfield and mesh collision pipeline paths; the separate `heightfield_midphase_kernel` and `shape_pairs_heightfield` buffer are removed in favor of the shared mesh midphase
- Replace per-shape `Model.shape_heightfield_data` / `Model.heightfield_elevation_data` with compact `Model.shape_heightfield_index` / `Model.heightfield_data` / `Model.heightfield_elevations`, matching the SDF indirection pattern. Use `Model.heightfield_data` indexed via `Model.shape_heightfield_index` instead.
- Standardize `rigid_contact_normal` to point from shape 0 toward shape 1 (A-to-B), matching the documented convention. Consumers that previously negated the normal on read (XPBD, VBD, MuJoCo, Kamino) no longer need to.
- Replace `Model.sdf_data` / `sdf_volume` / `sdf_coarse_volume` with texture-based equivalents (`texture_sdf_data`, `texture_sdf_coarse_textures`, `texture_sdf_subgrid_textures`). Use `Model.texture_sdf_data`, `texture_sdf_coarse_textures`, and `texture_sdf_subgrid_textures` instead.
- Render inertia boxes as wireframe lines instead of solid boxes in the GL viewer to avoid occluding objects
- Make contact reduction normal binning configurable (polyhedron, scan directions, voxel budget) via constants in ``contact_reduction.py``
- Upgrade GL viewer lighting from Blinn-Phong to Cook-Torrance PBR with GGX distribution, Schlick-GGX geometry, Fresnel-weighted ambient, and ACES filmic tone mapping
- Change implicit MPM residual computation to consider both infinity and l2 norm
- Change implicit MPM hardening law from exponential to hyperbolic sine (`sinh(-h * log(Jp))`), no longer scales elastic modulus
- Change implicit MPM collider velocity mode names: `"forward"` / `"backward"` replace `"instantaneous"` / `"finite_difference"`. Old names are no longer accepted.
- Simplify `SensorContact` force output: add `total_force` (aggregate per sensing object) and `force_matrix` (per-counterpart breakdown, `None` when no counterparts)
- Add `sensing_obj_idx` (`list[int]`), `counterpart_indices` (`list[list[int]]`), `sensing_obj_type`, and `counterpart_type` attributes. Rename `include_total` to `measure_total`
- Replace verbose Apache 2.0 boilerplate with two-line SPDX-only license headers across all source and documentation files
- Improve wrench preservation in hydroelastic contacts with contact reduction.
- Show Newton deprecation warnings during example runs started via `python -m newton.examples ...` or `python -m newton.examples.<category>.<module>`; pass `-W ignore::DeprecationWarning` if you need the previous quiet behavior.
- Reorder `ModelBuilder.add_shape_gaussian()` parameters so `xform` precedes `gaussian`, in line with other `add_shape_*` methods. Callers using positional arguments should switch to keyword form (`gaussian=..., xform=...`); passing a `Gaussian` as the second positional argument still works but emits a `DeprecationWarning`
- Rename `ModelBuilder.add_shape_ellipsoid()` parameters `a`, `b`, `c` to `rx`, `ry`, `rz`. Old names are still accepted as keyword arguments but emit a `DeprecationWarning`
- Rename `collide_plane_cylinder()` parameter `cylinder_center` to `cylinder_pos` for consistency with other collide functions. The old name is no longer accepted.
- Add optional `state` parameter to `SolverBase.update_contacts()` to align the base-class signature with Kamino and MuJoCo solvers
- Use `Literal` types for `SolverImplicitMPM.Config` string fields with fixed option sets (`solver`, `warmstart_mode`, `collider_velocity_mode`, `grid_type`, `transfer_scheme`, `integration_scheme`)
- Migrate `wp.array(dtype=X)` type annotations to `wp.array[X]` bracket syntax (Warp 1.12+).
- Align articulated `State.body_qd` / FK / IK / Jacobian / mass-matrix linear velocity with COM-referenced motion. If you were comparing `body_qd[:3]` against finite-differenced body-origin motion, recover origin velocity via `v_origin = v_com - omega x r_com_world`. Descendant `FREE` / `DISTANCE` `joint_qd` remains parent-frame and `joint_f` remains a world-frame COM wrench.

### Deprecated

- Deprecate `ModelBuilder.default_body_armature`, the `armature` argument on `ModelBuilder.add_link()` / `ModelBuilder.add_body()`, and USD-authored body armature via `newton:armature` in favor of adding any isotropic artificial inertia directly to `inertia`
- Deprecate `SensorContact.net_force` in favor of `SensorContact.total_force` and `SensorContact.force_matrix`
- Deprecate `SensorContact(include_total=...)` in favor of `SensorContact(measure_total=...)`
- Deprecate `SensorContact.sensing_objs` in favor of `SensorContact.sensing_obj_idx`
- Deprecate `SensorContact.counterparts` and `SensorContact.reading_indices` in favor of `SensorContact.counterpart_indices`
- Deprecate `SensorContact.shape` (use `total_force.shape` and `force_matrix.shape` instead)
- Deprecate `SensorTiledCamera.render_context`; prefer `SensorTiledCamera.utils` and `SensorTiledCamera.render_config`.
- Deprecate `SensorTiledCamera.RenderContext`; use `SensorTiledCamera.RenderConfig` for config types and `SensorTiledCamera.render_config` / `SensorTiledCamera.utils` for runtime access.
- Deprecate `SensorTiledCamera.Config`; prefer `SensorTiledCamera.RenderConfig` and `SensorTiledCamera.utils`.
- Deprecate `max_worlds` parameter of `ViewerBase.set_model()` in favor of `ViewerBase.set_visible_worlds()`
- Deprecate `Viewer.update_shape_colors()` in favor of writing directly to `Model.shape_color`
- Deprecate `ModelBuilder.add_shape_ellipsoid()` parameters `a`, `b`, `c` in favor of `rx`, `ry`, `rz`
- Deprecate passing a `Gaussian` as the second positional argument to `ModelBuilder.add_shape_gaussian()`; use the `gaussian=` keyword argument instead
- Deprecate `SensorTiledCamera.utils.assign_random_colors_per_world()` and `assign_random_colors_per_shape()` in favor of per-shape colors via `ModelBuilder.add_shape_*(color=...)`

### Removed

- Remove `Heightfield.finalize()` and stop storing raw pointers for heightfields in `Model.shape_source_ptr`; heightfield collision data is accessed via `Model.shape_heightfield_index` / `Model.heightfield_data` / `Model.heightfield_elevations`
- Remove `robot_humanoid` example in favor of `basic_plotting` which uses the same humanoid model with diagnostics visualization

### Fixed

- Fix GL viewer crash when enabling "Gap + Margin" for soft-body-only states with no rigid body transforms
- Fix inertia validation spuriously inflating small but physically valid eigenvalues for lightweight components (< ~50 g) by using a relative threshold instead of an absolute 1e-6 cutoff
- Restore keyboard camera movement while hovering gizmos so keyboard controls remain active when the pointer is over gizmos
- Resolve USD asset references recursively in `resolve_usd_from_url` so nested stages are fully downloaded
- Unify CPU and GPU inertia validation to produce identical results for zero-mass bodies with `bound_mass`, singular inertia, non-symmetric tensors, and triangle-inequality boundary cases
- Fix `UnboundLocalError` crash in detailed inertia validation when eigenvalue decomposition encounters NaN/Inf input
- Handle NaN/Inf mass and inertia deterministically in both validation paths (zero out mass and inertia)
- Update `ModelBuilder` internal state after fast-path (GPU kernel) inertia validation so it matches the returned `Model`
- Fix MJCF mesh scale resolution to use the mesh asset's own class rather than the geom's default class, avoiding incorrect vertex scaling for models like Robotiq 2F-85 V4
- Fix articulated bodies drifting laterally on the ground in XPBD solver by solving rigid contacts before joints
- Fix `hide_collision_shapes=True` not hiding collision meshes that have bound PBR materials
- Filter inactive particles in viewer so only particles with `ParticleFlags.ACTIVE` are rendered
- Fix concurrent asset download races on Windows by using content-addressed cache directories
- Fix body `gravcomp` not being written to the MuJoCo spec, causing it to be absent from XML saved via `save_to_mjcf`
- Fix `compute_world_offsets` grid ordering to match terrain grid row-major order so replicated world indices align with terrain block indices
- Fix `eq_solimp` not being written to the MuJoCo spec for equality constraints, causing it to be absent from XML saved via `save_to_mjcf`
- Fix WELD equality constraint quaternion written in xyzw format instead of MuJoCo's wxyz format in the spec, causing incorrect orientation in XML saved via `save_to_mjcf`
- Fix `update_contacts` not populating `rigid_contact_point0`/`rigid_contact_point1` when using `use_mujoco_contacts=True`
- Fix MPR anti-flicker inflate biasing contact distances and witness points for convex-convex pairs, causing phantom overlap in stacking scenarios
- Fix VSync toggle having no effect in `ViewerGL` on Windows 8+ due to a pyglet bug where `DwmFlush()` is never called when `_always_dwm` is True
- Fix loop joint coordinate mapping in the MuJoCo solver so joints after a loop joint read/write at correct qpos/qvel offsets
- Fix viewer crash when contact buffer overflows by clamping contact count to buffer size
- Decompose loop joint constraints by DOF type (WELD for fixed, CONNECT-pair for revolute, single CONNECT for ball) instead of always emitting 2x CONNECT
- Fix inertia box wireframe rotation for isotropic and axisymmetric bodies in viewer
- Implicit MPM solver now uses `mass=0` for kinematic particles instead of `ACTIVE` flag
- Suppress macOS OpenGL warning about unloadable textures by binding a 1x1 white fallback texture when no albedo or environment texture is set
- Fix MuJoCo solver freeze when immovable bodies (kinematic, static, or fixed-root) generate contacts with degenerate invweight
- Fix forward-kinematics child-origin linear velocity for articulated translated joints
- Fix `ModelBuilder.approximate_meshes()` to handle the duplication of per-shape custom attributes that results from convex decomposition
- Fix `get_tetmesh()` winding order for left-handed USD meshes
- Fix contact force conversion in `SolverMuJoCo` to include friction (tangential) components
- Fix URDF inertial parameters parsing in parse_urdf so inertia tensor is correctly calculated as R@I@R.T
- Fix Poisson surface reconstruction segfault under parallel test execution by defaulting to single-threaded Open3D Poisson (`n_threads=1`)
- Fix overly conservative broadphase AABB for mesh shapes by using the pre-computed local AABB with a rotated-box transform instead of a bounding-sphere fallback, eliminating false contacts between distant meshes
- Fix heightfield bounding-sphere radius underestimating Z extent for asymmetric height ranges (e.g. `min_z=0, max_z=10`)
- Fix VBD self-contact barrier C2 discontinuity at `d = tau` caused by a factor-of-two error in the log-barrier coefficient
- Fix fast inertia validation treating near-symmetric tensors within `np.allclose()` default tolerances as corrections, aligning CPU and GPU validation warnings
- Fix URDF joint dynamics friction import so specified friction values are preserved during simulation
- Fix `requires_grad` not being preserved in `ArticulationView` attribute getters, breaking gradient propagation through selection queries
- Fix duplicate Reset button in brick stacking example when using the example browser
- Cap `cbor2` dependency to `<6` to prevent recorder test failures caused by breaking deserialization changes in cbor2 6.0
- Clamp viewer picking force to prevent explosion when picking light objects near stiff contacts, configurable via `pick_max_acceleration` parameter on the `Picking` class (default 5g of effective articulation mass)
- Fix `cloth_franka` example Jacobian broken by COM-referenced `body_qd` convention change; adjust robot base height, gripper orientations, and grasp targets for improved reachability (a follow-up PR will migrate the example to `newton.ik`)

## [1.0.0] - 2026-03-10

Initial public release.
