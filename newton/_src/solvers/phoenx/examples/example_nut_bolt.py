# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Nut & Bolt
#
# PhoenX variant of :mod:`example_nut_bolt`. Same scene (an SDF nut
# dropped onto a fixed SDF bolt) and the same ``CollisionPipeline``
# driven contact ingest, but drives the simulation through
# :class:`PhoenXWorld` instead of the jitter ``World``. The two
# examples produce visually similar threading behaviour, so this is
# also a side-by-side validation of the PhoenX contact path on a
# dense-manifold (multi-column) scene.
#
# Requires CUDA (SDF narrow phase is CUDA-only), ``trimesh``, and
# the IsaacGymEnvs asset cache (auto-downloaded on first run).
#
# Run:  python -m newton._src.solvers.phoenx.examples.example_nut_bolt
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.materials import (
    COMBINE_MIN,
    Material,
    material_table_from_list,
)
from newton._src.solvers.phoenx.picking import (
    Picking,
    register_with_viewer_gl,
)
from newton._src.solvers.phoenx.solver_phoenx import (
    PhoenXWorld,
    pack_body_xforms_kernel,
)

# Contact matching mode. ``"latest"`` uses fresh narrow-phase
# normals each frame so the helical-thread torque drives nut
# rotation; ``"sticky"`` pins frame-1 normals and produces no
# rotation about +Z but is more numerically stable under low
# friction.
_CONTACT_MATCHING = "latest"

# Default for ``--bolt-static``. ``False`` exercises the dynamic-bolt
# case (xpbd handles this; phoenx must too).
BOLT_IS_STATIC_DEFAULT = False

# Per-shape Coulomb friction. ``COMBINE_MIN`` makes bolt-ground
# ``min(1.0, 1.0) = 1.0`` (bolt clamps) and bolt-nut
# ``min(1.0, 0.01) = 0.01`` (threads engage).
BOLT_GROUND_FRICTION: float = 1.0
NUT_FRICTION: float = 0.01

# Assembly + asset source -- identical to the jitter / XPBD / MuJoCo
# nut-bolt examples so the same mesh pair is compared across solvers.
ASSEMBLY_STR = "m20_loose"
ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

# 2D grid of nut/bolt pairs. ``(1, 1)`` reproduces the original
# single-pair scene; larger grids tile the pair in the XY plane,
# scaling body / shape / contact counts linearly. The grid can be
# instantiated either inside a single Newton world (``--no-multi-scene``,
# the default "single-scene" mode -- one Newton/PhoenX world that
# contains every pair) or as one world per pair (``--multi-scene`` --
# uses :meth:`ModelBuilder.replicate` so each pair sits in its own
# world, matching the parallel-environment layout used by other
# Newton examples).
GRID_DIMS_DEFAULT: tuple[int, int] = (80, 80)

# Hard frame cap (see :meth:`Example.step`). The GL viewer's
# ``is_running()`` doesn't honour ``--num-frames`` in interactive
# runs, so we exit explicitly after this many simulation steps. Bump
# (or set ``MAX_FRAMES = 10**9``) if you want the example to run
# indefinitely.
MAX_FRAMES: int = 10**9
# Centre-to-centre spacing [m] applied at ``scene_scale=1.0``. The
# M20 nut/bolt assembly is ~5 cm across; 0.1 m leaves ~5 cm clearance
# between neighbours so their SDF narrow bands don't overlap.
GRID_SPACING_DEFAULT: tuple[float, float] = (0.1, 0.1)

# ``mu = 0.01`` is deliberate: only Coulomb friction (not the
# axially-symmetric normal impulse) can produce torque about the
# bolt axis, so it drives the threading rotation. ``gap`` / SDF
# narrow band are tight (0.5 mm / 1 mm); wider speculative ranges
# create off-axis nut-bolt SDF contacts that PhoenX's PGS turns
# into a horizontal impulse the threading dynamics can't damp out.
SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    margin=0.0,
    mu=0.01,
    ke=1e7,
    kd=1e4,
    gap=0.0005,
    density=8000.0,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=False,
)

MESH_SDF_MAX_RESOLUTION = 512
MESH_SDF_NARROW_BAND_RANGE = (-0.001, 0.001)


def _load_mesh_with_sdf(mesh_file: str, gap: float) -> tuple[newton.Mesh, wp.vec3]:
    """Load a trimesh, center its bounding box at the origin, and bake
    an SDF inside the configured narrow band.

    The SDF is baked from unscaled mesh vertices; any ``scene_scale``
    the caller applies ends up as a runtime shape scale in
    :meth:`add_shape_mesh`, and the runtime narrow-phase rescales SDF
    distances / gradients by ``min_scale`` on each query. Keeping the
    SDF in mesh units preserves the asset's authored resolution
    regardless of ``scene_scale``.

    Returns the :class:`newton.Mesh` and the pre-centering world
    offset so the caller can compensate by shifting the shape origin
    (the SDF is generated around the mesh's local origin, not its
    original AABB centre). ``center_vec`` is in mesh (unscaled) units.
    """
    import trimesh

    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)
    min_extent = vertices.min(axis=0)
    max_extent = vertices.max(axis=0)
    center = (min_extent + max_extent) / 2
    vertices = vertices - center
    center_vec = wp.vec3(float(center[0]), float(center[1]), float(center[2]))

    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=gap,
        # Drop edges whose oriented box is fully covered by another
        # edge's box -- mesh-SDF narrow phase iterates one edge less
        # without losing contact coverage. Bump the in-plane box width
        # to ~2 % of the AABB diagonal (4x the default) to absorb more
        # of the bolt's smooth-shaft edges; per-mesh edge counts drop
        # bolt 22 560 -> 15 359 (-32 %), nut 12 773 -> 6 483 (-49 %),
        # total 35 333 -> 21 842 (-38 %).
        edge_box_absorption=True,
        edge_box_half_lateral_rel=2e-2,
    )
    return mesh, center_vec


class Example:
    """Drop-an-SDF-nut-onto-an-SDF-bolt scene for :class:`PhoenXWorld`.

    Pipeline per frame mirrors the tower example -- sync state,
    run Newton's :class:`CollisionPipeline`, apply picking force,
    call :meth:`PhoenXWorld.step`, sync back -- so the same
    CUDA-graph capture pattern works.
    """

    def __init__(self, viewer, args):
        # Timing + iteration budget matches the XPBD nut-bolt scene in
        # :mod:`newton.examples.contacts.example_nut_bolt_sdf` so the
        # two solvers can be compared head-to-head: fps=120, 5
        # substeps, 10 solver iterations per substep. XPBD also runs
        # with ``angular_damping=0`` (no rotational drag); PhoenX
        # keeps damping multipliers at 1.0 by default
        # (``body_container_zeros``), so neither solver introduces
        # any rotational damping on this scene.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 5
        self.solver_iterations = 8
        self._frame: int = 0

        # Scene-wide geometric scale. ``1.0`` matches the M20 bolt
        # assembly from the IsaacGymEnvs asset pack verbatim. Setting
        # this to e.g. ``10.0`` produces an oversized nut/bolt pair in
        # the same physical layout -- useful for exercising the solver
        # at non-default length scales (contact thicknesses, SDF
        # narrow band, picking AABB, camera distance all track).
        # Matches the ``scene_scale`` knob in
        # :mod:`newton.examples.contacts.example_nut_bolt_sdf`.
        self.scene_scale = float(getattr(args, "scene_scale", 1.0))

        # CLI-controlled scene flags.
        self.bolt_is_static: bool = bool(getattr(args, "bolt_static", BOLT_IS_STATIC_DEFAULT))

        # 2D grid layout. ``grid_dims`` is the number of nut/bolt
        # pairs along (x, y); ``grid_spacing`` is centre-to-centre
        # spacing at ``scene_scale = 1.0`` (it tracks ``scene_scale``
        # below so the pairs stay non-overlapping at any length scale).
        # ``multi_scene`` controls whether the pairs share a single
        # PhoenX world (default, ``single``) or each live in their own
        # replicated world (``multi``).
        grid_x = int(getattr(args, "grid_dims_x", GRID_DIMS_DEFAULT[0]))
        grid_y = int(getattr(args, "grid_dims_y", GRID_DIMS_DEFAULT[1]))
        if grid_x < 1 or grid_y < 1:
            raise ValueError(f"grid_dims must be >= 1 (got {(grid_x, grid_y)})")
        self.grid_dims: tuple[int, int] = (grid_x, grid_y)
        self.grid_spacing: tuple[float, float] = (
            float(getattr(args, "grid_spacing_x", GRID_SPACING_DEFAULT[0])),
            float(getattr(args, "grid_spacing_y", GRID_SPACING_DEFAULT[1])),
        )
        self.multi_scene: bool = bool(getattr(args, "multi_scene", False))

        self.viewer = viewer
        self.device = wp.get_device()
        if not self.device.is_cuda:
            raise RuntimeError("example_nut_bolt requires CUDA (SDF narrow phase is CUDA-only).")

        # ---- Fetch the nut/bolt meshes --------------------------------
        print("Downloading nut/bolt assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        print(f"Assets downloaded to: {asset_path}")

        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")
        # SDF / gap stay in mesh (unscaled) units; the shape scale on
        # ``add_shape_mesh`` applies uniformly at contact-query time.
        bolt_mesh, bolt_center = _load_mesh_with_sdf(bolt_file, gap=SHAPE_CFG.gap)
        nut_mesh, nut_center = _load_mesh_with_sdf(nut_file, gap=SHAPE_CFG.gap)
        shape_cfg = SHAPE_CFG

        # ---- Build the Newton scene: NxM grid of (bolt + nut) pairs ----
        # ``scaled_bolt_center`` / ``scaled_nut_center`` shift the SDF
        # mesh back to where the asset author had it after the runtime
        # shape scale is applied (the SDF was baked around the local
        # origin, not the original AABB centre).
        scaled_bolt_center = wp.vec3(
            bolt_center[0] * self.scene_scale,
            bolt_center[1] * self.scene_scale,
            bolt_center[2] * self.scene_scale,
        )
        scaled_nut_center = wp.vec3(
            nut_center[0] * self.scene_scale,
            nut_center[1] * self.scene_scale,
            nut_center[2] * self.scene_scale,
        )

        nx, ny = self.grid_dims
        spacing_x = self.grid_spacing[0] * self.scene_scale
        spacing_y = self.grid_spacing[1] * self.scene_scale
        cell_centres_xy: list[tuple[float, float]] = []
        for j in range(ny):
            for i in range(nx):
                cx = (i - 0.5 * (nx - 1)) * spacing_x
                cy = (j - 0.5 * (ny - 1)) * spacing_y
                cell_centres_xy.append((cx, cy))
        self._cell_centres_xy = cell_centres_xy

        def _add_pair(builder: newton.ModelBuilder, cx: float, cy: float) -> tuple[int, int]:
            """Add one (bolt, nut) pair to ``builder`` centred at (cx, cy)
            and return the freshly-allocated (bolt_body_id, nut_body_id).
            """
            # Static bolt. Mesh stays in mesh-local units; the shape
            # scale below scales it at runtime, and the shape xform
            # places the scaled mesh back where the asset author put it.
            bolt_id = builder.add_body(
                xform=wp.transform(wp.vec3(cx, cy, 0.0), wp.quat_identity()),
                label="bolt",
            )
            builder.add_shape_mesh(
                bolt_id,
                mesh=bolt_mesh,
                xform=wp.transform(scaled_bolt_center, wp.quat_identity()),
                scale=(self.scene_scale, self.scene_scale, self.scene_scale),
                cfg=shape_cfg,
            )

            # Dynamic nut, 4 cm above the bolt head with a pi/8 rotation
            # about +Z so its threads are phase-offset from the bolt's
            # (without this the two SDFs line up perfectly and the nut
            # slides straight down a single groove).
            nut_id = builder.add_body(
                xform=wp.transform(
                    wp.vec3(cx, cy, 0.041 * self.scene_scale),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 8),
                ),
                label="nut",
            )
            builder.add_shape_mesh(
                nut_id,
                mesh=nut_mesh,
                xform=wp.transform(scaled_nut_center, wp.quat_identity()),
                scale=(self.scene_scale, self.scene_scale, self.scene_scale),
                cfg=shape_cfg,
            )
            return bolt_id, nut_id

        mb = newton.ModelBuilder()
        mb.default_shape_cfg.gap = shape_cfg.gap

        # Ground plane below the bolt base. Catches a free-falling bolt
        # when :data:`BOLT_IS_STATIC` is ``False`` and provides a
        # consistent reference for the picking ray. The height tracks
        # ``scene_scale`` so the plane stays below the assembly at any
        # length scale.
        mb.add_ground_plane(height=0.0 * self.scene_scale)

        self._bolt_bodies: list[int] = []
        self._nut_bodies: list[int] = []

        if self.multi_scene:
            # One world per nut/bolt pair. The sub-builder holds the
            # canonical pair at the origin; ``replicate`` clones it
            # ``nx * ny`` times with the grid spacing as world offsets.
            # The grid is centred via a final shift applied through
            # ``compute_world_offsets``: ``replicate`` itself spaces
            # worlds along (x, y) using a regular grid, so we just need
            # to pass the per-axis spacing and let it tile.
            pair_builder = newton.ModelBuilder()
            pair_builder.default_shape_cfg.gap = shape_cfg.gap
            _add_pair(pair_builder, 0.0, 0.0)
            mb.replicate(
                pair_builder,
                world_count=nx * ny,
                spacing=(spacing_x, spacing_y, 0.0),
            )
            # ``replicate`` deterministically appends two bodies per
            # world after the ground plane (which lives in the parent
            # builder, body=-1). Worlds are visited in the order
            # ``compute_world_offsets`` returns; for our 2D grid the
            # exact ordering doesn't matter for correctness because the
            # subsequent setup loops over the same range.
            for cell_index in range(nx * ny):
                self._bolt_bodies.append(2 * cell_index)
                self._nut_bodies.append(2 * cell_index + 1)
        else:
            # Single world, single PhoenX scene: every pair lives in
            # the same ``ModelBuilder`` and is laid out manually on the
            # grid.
            for cx, cy in cell_centres_xy:
                bolt_id, nut_id = _add_pair(mb, cx, cy)
                self._bolt_bodies.append(bolt_id)
                self._nut_bodies.append(nut_id)

        # Back-compat aliases -- single-pair callers / tests can still
        # reach the first pair without knowing about the grid.
        self._bolt_body = self._bolt_bodies[0]
        self._nut_body = self._nut_bodies[0]

        # We drive the broad phase below with SAP, so skip finalize's
        # O(N^2) explicit shape-contact-pair list -- it's unused with
        # SAP and would balloon to tens of GB on a 30x30 grid.
        num_pairs_total = len(self._bolt_bodies)
        self.model = mb.finalize(skip_shape_contact_pairs=True)

        # Pin every bolt (default off): zero its inverse mass + inertia
        # so :func:`init_phoenx_bodies_kernel` marks it
        # :data:`MOTION_STATIC`. The nut's density-derived mass is
        # unaffected because the mesh inertia computation already ran
        # during ``finalize``. When :data:`bolt_is_static` is
        # ``False``, skip this step and let the bolt fall under gravity
        # like the nut.
        if self.bolt_is_static:
            body_inv_mass_np = self.model.body_inv_mass.numpy()
            body_inv_inertia_np = self.model.body_inv_inertia.numpy()
            for bolt_id in self._bolt_bodies:
                body_inv_mass_np[bolt_id] = 0.0
                body_inv_inertia_np[bolt_id] = np.zeros((3, 3), dtype=np.float32)
            self.model.body_inv_mass.assign(wp.array(body_inv_mass_np, dtype=wp.float32))
            self.model.body_inv_inertia.assign(wp.array(body_inv_inertia_np, dtype=wp.mat33))

        scene_mode = "multi" if self.multi_scene else "single"
        print(
            f"[PhoenX Nut-Bolt] bodies={self.model.body_count} "
            f"shapes={self.model.shape_count} "
            f"grid={nx}x{ny} ({len(self._bolt_bodies)} pairs) "
            f"scene_mode={scene_mode}"
        )

        # ---- Collision pipeline ---------------------------------------
        # SAP broad phase + scene-tight budgets. Each cell has at most
        # three active mesh pairs -- bolt-nut, bolt-ground, nut-ground.
        # SDF narrow phase produces ~40 contacts/pair worst case
        # (helical thread) + ~4 for the plane pairs => ~48 raw contacts
        # per cell. 96/cell leaves 2x headroom for transient jitter
        # without bloating downstream PhoenX allocations: the contact
        # column container, partitioner adjacency CSR, and PGS
        # persistent-grid launch dim are all sized from
        # ``rigid_contact_max`` for per-contact state and from a
        # separate contact-column cap for per-pair solver state, so an
        # over-budget contact count does not force per-column kernels to
        # scan empty manifold slots. 16 broad-phase pairs/cell covers
        # the 3 real pairs plus SAP overlap chaff with comfortable
        # slack. The ``+ 64`` / ``+ 8`` floors keep the 1x1 case above
        # global minima.
        # 96/cell turned out to overflow on settle frames (~97
        # contacts/pair observed at 20x20, "Contact buffer overflowed
        # 38689 > 38464"); 128/cell still tripped the warning on the
        # 50x50 scene's settle (peaks of 129-130 contacts/pair); 160/cell
        # still tripped on the 80x80 scene's settle (peaks of ~160.13
        # contacts/pair, "Contact buffer overflowed 1024850 > 1024064").
        # 192/cell gives ~20% headroom over the measured 80x80 peak
        # without bloating downstream PhoenX scratch.
        rigid_contact_max_estimate = 192 * num_pairs_total + 64
        shape_pairs_max_estimate = 16 * num_pairs_total + 8
        # ``max_triangle_pairs`` sizes the SDF narrow phase's GLOBAL
        # contact-reducer buffer (``GlobalContactReducer.capacity``).
        # The SDF kernel pushes ~6K edge-test attempts per mesh-mesh
        # pair into this buffer via ``atomic_add``; at 20x20 = 400
        # pairs that totals ~2.4M attempts per frame and at 50x50 it
        # passes 8M. ``GlobalContactReducer`` auto-falls back from
        # deterministic to fast packing if this exceeds the
        # deterministic-packing ceiling (~4M, see ``CONTACT_ID_BITS``);
        # we size unconditionally to the measured workload so contacts
        # never get silently dropped on overflow.
        max_triangle_pairs_estimate = 8 * 1024 * num_pairs_total + 64 * 1024
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=_CONTACT_MATCHING,
            broad_phase="sap",
            rigid_contact_max=rigid_contact_max_estimate,
            shape_pairs_max=shape_pairs_max_estimate,
            max_triangle_pairs=max_triangle_pairs_estimate,
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])
        print(
            f"[PhoenX Nut-Bolt] contact_budget: "
            f"rigid_contact_max={rigid_contact_max} "
            f"shape_pairs_max={int(self.collision_pipeline.shape_pairs_max)} "
            f"contact_column_max={int(self.collision_pipeline.shape_pairs_max)} "
            f"broad_phase=sap"
        )

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.body_q.assign(self.state.body_q)

        # ---- PhoenX body container (slot 0 = static world anchor) ----
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        # Seed every slot's orientation to identity so the rotation-
        # to-matrix call in :func:`_phoenx_update_inertia_and_clear_forces_kernel`
        # doesn't blow up on the zero-quaternion default.
        bodies.orientation.assign(np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32))
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_q,
                self.state.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
            ],
            outputs=[
                bodies.position,
                bodies.orientation,
                bodies.velocity,
                bodies.angular_velocity,
                bodies.inverse_mass,
                bodies.inverse_inertia,
                bodies.inverse_inertia_world,
                bodies.motion_type,
                bodies.body_com,
            ],
            device=self.device,
        )
        self.bodies = bodies

        # ---- Joint-only constraint container ---------------------------
        # Contact column storage lives in :class:`ContactColumnContainer`;
        # the joint-side constraint container only needs a 1-row
        # placeholder in this contact-only scene.
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            device=self.device,
        )

        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        # ---- Solver ---------------------------------------------------
        # ``step_layout`` picks the PhoenX kernel shape:
        # * single-scene mode -> one big PhoenX world with many bodies /
        #   contacts -> ``"single_world"`` (persistent-grid kernels
        #   with ``_SINGLEWORLD_BLOCK_DIM = 32``). The default
        #   ``"multi_world"`` fast-tail launches only one warp per world,
        #   so a single world with 100s of contacts would be solved by
        #   32 threads -- catastrophically serialised.
        # * multi-scene mode -> ``nx*ny`` PhoenX worlds, each tiny ->
        #   ``"multi_world"`` fast-tail amortises launch overhead across
        #   worlds and is the correct fit.
        #
        # Mass splitting (Tonge) is **off** for this scene. The
        # nut-bolt interaction graph is trivially colourable: every
        # contact column touches at most 3 bodies (bolt + nut +
        # ground), columns from different cells are spatially disjoint
        # so they share no body, and any greedy colouring converges in
        # <= 3 colours regardless of grid size. Mass splitting was
        # designed for Kapla-style stacks where a single brick is a
        # node in dozens of columns and the chromatic number explodes;
        # turning it on here costs an interaction-graph rebuild +
        # broadcast / average / writeback per substep that buys
        # nothing, and worse, its overflow-bucket Jacobi step adds
        # slack to the dense bolt-nut SDF manifold that the PGS can't
        # damp out at grids >= ~20x20 -- which is the "explosion"
        # symptom we used to see.
        #
        # ``enable_body_pair_grouping`` collapses every (bolt, nut)
        # shape pair into a single contact column even when the SDF
        # narrow phase emits multiple manifold rows for the same body
        # pair -- typical here on the helical thread. Fewer columns =
        # fewer partitioner adjacency edges = fewer PGS iterations per
        # sweep at identical physics.
        #
        # ``max_thread_blocks=256`` (kapla's choice) caps the
        # persistent single-world PGS grid; without it the grid grows
        # with ``_constraint_capacity`` and a 400-pair scene oversubs
        # the SM occupancy faster than the megakernels can amortise
        # the launch.
        step_layout = "multi_world" if self.multi_scene else "single_world"
        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=1,
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=rigid_contact_max,
            max_contact_columns=int(self.collision_pipeline.shape_pairs_max),
            default_friction=SHAPE_CFG.mu,
            num_worlds=(nx * ny) if self.multi_scene else 1,
            step_layout=step_layout,
            mass_splitting=False,
            enable_body_pair_grouping=True,
            partitioner_algorithm="greedy",
            max_thread_blocks=256 if not self.multi_scene else None,
            device=self.device,
        )

        # Disable the single-block fused-tail kernel for this scene.
        # With ``enable_body_pair_grouping=True`` the partitioner emits
        # only a handful of colours of ~``num_pairs`` elements each
        # (e.g. 225 at a 15x15 grid). The default
        # ``FUSE_TAIL_MAX_COLOR_SIZE=256`` means colours of size 225
        # fall below the threshold and get drained by a SINGLE 1-block
        # kernel -- pinning ~65% of solver time to one SM. Setting the
        # threshold to 0 forces every colour through the persistent
        # multi-block grid instead. Measured 2.4x step-time win at a
        # 15x15 grid (35 ms -> 14 ms); 18x18 unchanged because its
        # colours are already above 256.
        if not self.multi_scene:
            self.world._fuse_threshold = 0

        # ---- Per-shape materials (COMBINE_MIN, see BOLT_GROUND_FRICTION) ----
        materials = material_table_from_list(
            [
                Material(),  # index 0: reserved default
                Material(  # index 1: bolt
                    static_friction=BOLT_GROUND_FRICTION,
                    dynamic_friction=BOLT_GROUND_FRICTION,
                    friction_combine_mode=COMBINE_MIN,
                ),
                Material(  # index 2: nut
                    static_friction=NUT_FRICTION,
                    dynamic_friction=NUT_FRICTION,
                    friction_combine_mode=COMBINE_MIN,
                ),
                Material(  # index 3: ground
                    static_friction=BOLT_GROUND_FRICTION,
                    dynamic_friction=BOLT_GROUND_FRICTION,
                    friction_combine_mode=COMBINE_MIN,
                ),
            ],
            device=self.device,
        )
        # Newton shape ordering for an N-pair grid: 0 = ground, then
        # per pair (bolt mesh, nut mesh) in insertion order. Each pair
        # reuses the same bolt / nut material indices.
        num_pairs = len(self._bolt_bodies)
        shape_material_np = np.empty(1 + 2 * num_pairs, dtype=np.int32)
        shape_material_np[0] = 3  # ground
        shape_material_np[1::2] = 1  # bolt meshes
        shape_material_np[2::2] = 2  # nut meshes
        assert shape_material_np.shape[0] == int(self.model.shape_count), (
            f"shape_material array size {shape_material_np.shape[0]} "
            f"does not match model shape_count {int(self.model.shape_count)} "
            f"(grid={self.grid_dims}, multi_scene={self.multi_scene})"
        )
        shape_material_idx = wp.array(
            shape_material_np,
            dtype=wp.int32,
            device=self.device,
        )
        self.world.set_materials(materials, shape_material_idx)

        # ---- Viewer ---------------------------------------------------
        self._xforms = wp.zeros(num_phoenx_bodies, dtype=wp.transform, device=self.device)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(
                0.25 * self.scene_scale,
                -0.25 * self.scene_scale,
                0.15 * self.scene_scale,
            ),
            pitch=-25.0,
            yaw=135.0,
        )

        # ---- Picking --------------------------------------------------
        # Half-extents for the ray cast; slot 0 (world anchor) stays
        # at zero so rays ignore it. The bolt and nut use loose AABB
        # half-extents so the picking ray latches onto either body in
        # any cell of the grid. The AABBs scale with the geometry so
        # the ray still hits at non-default scene scales.
        bolt_half_extents = (
            0.02 * self.scene_scale,
            0.02 * self.scene_scale,
            0.04 * self.scene_scale,
        )
        nut_half_extents = (
            0.025 * self.scene_scale,
            0.025 * self.scene_scale,
            0.015 * self.scene_scale,
        )
        half_extents_np = np.zeros((num_phoenx_bodies, 3), dtype=np.float32)
        for bolt_id in self._bolt_bodies:
            half_extents_np[bolt_id + 1] = bolt_half_extents
        for nut_id in self._nut_bodies:
            half_extents_np[nut_id + 1] = nut_half_extents
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # Initial per-pair nut XY for :meth:`test_final`. Reading
        # straight from ``body_q`` works for both single-scene and
        # multi-scene layouts (the latter uses
        # :func:`compute_world_offsets` which doesn't necessarily match
        # the row-major ``cell_centres_xy`` ordering when nx != ny).
        body_q_np = self.model.body_q.numpy()
        self._nut_initial_xys: list[tuple[float, float]] = [
            (float(body_q_np[nut_id][0]), float(body_q_np[nut_id][1])) for nut_id in self._nut_bodies
        ]
        self._bolt_initial_xys: list[tuple[float, float]] = [
            (float(body_q_np[bolt_id][0]), float(body_q_np[bolt_id][1])) for bolt_id in self._bolt_bodies
        ]
        self._nut_initial_xy = self._nut_initial_xys[0]

        self.graph = None
        self.capture()

    def capture(self) -> None:
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        self._sync_newton_to_phoenx()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        # Picking force is accumulated into bodies.force once per
        # frame -- :class:`PhoenXWorld`'s per-substep force kernel
        # consumes it and the post-step ``_clear_forces`` zeros it
        # for the next frame.
        self.picking.apply_force()
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
        )
        self._sync_phoenx_to_newton()

    def _sync_newton_to_phoenx(self) -> None:
        n = self.model.body_count
        wp.launch(
            newton_to_phoenx_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
            outputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_phoenx_to_newton(self) -> None:
        n = self.model.body_count
        wp.launch(
            phoenx_to_newton_kernel,
            dim=n,
            inputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._frame += 1
        # Hard frame cap. The viewer's ``is_running()`` doesn't always
        # honour ``--num-frames`` in interactive runs (the GL viewer
        # keeps the window open until the user closes it), which makes
        # bench / nsys profiling awkward. Override ``is_running`` to
        # return ``False`` after :data:`MAX_FRAMES` so the main loop in
        # :func:`newton.examples.run` exits cleanly (and still calls
        # ``test_final`` when ``--test`` is set).
        if self._frame >= MAX_FRAMES:
            self.viewer.is_running = lambda: False

    def render(self) -> None:
        wp.launch(
            pack_body_xforms_kernel,
            dim=self.world.num_bodies,
            inputs=[self.bodies, self._xforms],
            device=self.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """After the settle run every nut (and, when free, every bolt)
        in the grid must still be near its initial XY axis (no off-axis
        fly-off) and have finite state. The nut may have rotated
        significantly (threading) so we don't assert on orientation.
        With ``--no-bolt-static`` the bolts are also checked -- a bolt
        that had slid out from under the nut would show large XY drift.
        """
        positions = self.bodies.position.numpy()
        velocities = self.bodies.velocity.numpy()
        max_drift = 0.1 * self.scene_scale
        for cell_index, (bolt_id, nut_id) in enumerate(zip(self._bolt_bodies, self._nut_bodies, strict=True)):
            nut_slot = nut_id + 1
            bolt_slot = bolt_id + 1
            nut_pos = positions[nut_slot]
            nut_vel = velocities[nut_slot]
            bolt_pos = positions[bolt_slot]
            bolt_vel = velocities[bolt_slot]
            assert np.isfinite(nut_pos).all(), f"cell {cell_index}: nut position non-finite ({nut_pos})"
            assert np.isfinite(nut_vel).all(), f"cell {cell_index}: nut velocity non-finite ({nut_vel})"
            assert np.isfinite(bolt_pos).all(), f"cell {cell_index}: bolt position non-finite ({bolt_pos})"
            assert np.isfinite(bolt_vel).all(), f"cell {cell_index}: bolt velocity non-finite ({bolt_vel})"
            nut_xy_dist = float(
                np.linalg.norm(
                    np.asarray(nut_pos[:2], dtype=np.float32)
                    - np.asarray(self._nut_initial_xys[cell_index], dtype=np.float32)
                )
            )
            assert nut_xy_dist < max_drift, (
                f"cell {cell_index}: nut flew off-axis: xy_dist={nut_xy_dist:.4f} m "
                f"(max={max_drift:.4f} m at scene_scale={self.scene_scale}; "
                f"pos={tuple(float(x) for x in nut_pos)})"
            )
            if not self.bolt_is_static:
                bolt_xy_dist = float(
                    np.linalg.norm(
                        np.asarray(bolt_pos[:2], dtype=np.float32)
                        - np.asarray(self._bolt_initial_xys[cell_index], dtype=np.float32)
                    )
                )
                assert bolt_xy_dist < max_drift, (
                    f"cell {cell_index}: bolt slid off the ground: xy_dist={bolt_xy_dist:.4f} m "
                    f"(max={max_drift:.4f} m at scene_scale={self.scene_scale}; "
                    f"pos={tuple(float(x) for x in bolt_pos)})"
                )

    @staticmethod
    def create_parser():
        import argparse  # noqa: PLC0415

        parser = newton.examples.create_parser()
        parser.add_argument(
            "--scene-scale",
            type=float,
            default=1.0,
            help=(
                "Uniform geometric scale applied to the nut/bolt assembly "
                "(positions, mesh scale, SDF narrow band, contact gap, "
                "picking AABB, and camera). 1.0 matches the original "
                "M20 asset size; larger / smaller values produce a "
                "geometrically similar scene at a different length scale."
            ),
        )
        parser.add_argument(
            "--bolt-static",
            action=argparse.BooleanOptionalAction,
            default=BOLT_IS_STATIC_DEFAULT,
            help=(
                "Pin the bolt to the world (zero inv mass + inertia). "
                "When ``--no-bolt-static`` the bolt is left dynamic and "
                "drops onto the ground plane before the nut threads on; "
                "exercises the contact path with neither side anchored."
            ),
        )
        parser.add_argument(
            "--grid-dims-x",
            type=int,
            default=GRID_DIMS_DEFAULT[0],
            help=(
                "Number of nut/bolt pairs along the X axis of the grid. "
                "The default (1) reproduces the original single-pair scene."
            ),
        )
        parser.add_argument(
            "--grid-dims-y",
            type=int,
            default=GRID_DIMS_DEFAULT[1],
            help=(
                "Number of nut/bolt pairs along the Y axis of the grid. "
                "The default (1) reproduces the original single-pair scene."
            ),
        )
        parser.add_argument(
            "--grid-spacing-x",
            type=float,
            default=GRID_SPACING_DEFAULT[0],
            help=("Centre-to-centre spacing [m] along X at scene_scale=1.0. Multiplied by --scene-scale at runtime."),
        )
        parser.add_argument(
            "--grid-spacing-y",
            type=float,
            default=GRID_SPACING_DEFAULT[1],
            help=("Centre-to-centre spacing [m] along Y at scene_scale=1.0. Multiplied by --scene-scale at runtime."),
        )
        parser.add_argument(
            "--multi-scene",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Build the grid as one Newton world per pair "
                "(``--multi-scene`` -- uses ``ModelBuilder.replicate``) "
                "instead of a single shared world containing every pair "
                "(``--no-multi-scene``, the default, a.k.a. single-scene "
                "mode). Both produce the same physics; multi-scene mode "
                "matches the parallel-environment layout used by other "
                "Newton examples."
            ),
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    # Start paused for interactive viewers so the user can see the
    # initial pose before stepping. ``--test`` (and the headless null
    # viewer) override is_paused() to always step, so this is a no-op
    # in CI.

    newton.examples.run(example, args)
