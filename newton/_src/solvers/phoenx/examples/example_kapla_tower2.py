# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Kapla Tower
#
# Port of PhoenX's ``Demo15: Kapla Tower`` (see
# ``PhoenX/src/Viewer/Demos/Demo15.cs``). The C# demo loads
# ``KaplaTower2.usda`` -- a USD ``PointInstancer`` describing
# ~11k Kapla-style wooden bricks arranged into a cylindrical tower --
# and lets PhoenX settle them onto the ground.
#
# We don't depend on USD here: the brick prototype's full extents and
# every instance's position / orientation have been extracted into
# :mod:`kapla_tower_data` (regenerate with
# ``python _extract_kapla_usda.py`` from the repo root).
#
# The USDA is +Z-up (``upAxis = "Z"``); PhoenX is +Y-up so the C#
# demo rotates the whole instancer by ``-pi/2`` about +X. Newton is
# also +Z-up, so we keep positions and orientations as-is. The
# overall scene is uniformly scaled by :data:`GLOBAL_SCALING` to
# match the C# demo's metric size.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_kapla_tower
###########################################################################

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import MOTION_KINEMATIC, body_container_zeros
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    newton_to_phoenx_kernel as _newton_to_phoenx_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    phoenx_to_newton_kernel as _phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.examples.kapla_tower_data import (
    BRICK_FULL_EXTENTS,
    NUM_BRICKS,
    ORIENTATIONS,
    POSITIONS,
)
from newton._src.solvers.phoenx.picking import (
    Picking,
    register_with_viewer_gl,
)
from newton._src.solvers.phoenx.solver_config import (
    PHOENX_CONTACT_MATCHING,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


# Patch ``state.body_q`` for the camera collider so Newton's
# CollisionPipeline (which broad/narrow-phases against ``body_q``)
# tracks the live camera, not the spawn position.
@wp.kernel(enable_backward=False)
def _write_camera_body_q_kernel(
    body_id: wp.int32,
    pos: wp.array[wp.vec3f],
    orient: wp.array[wp.quatf],
    body_q: wp.array[wp.transform],
):
    body_q[body_id] = wp.transform(pos[0], orient[0])


# Mirrors C# ``Demo15.cs``: globalScaling 0.01 there, 0.1 here for a
# ~70 cm tabletop tower; ground sits at 0.35 * scale.
GLOBAL_SCALING: float = 0.1
BRICK_DENSITY: float = 1000.0
GROUND_HEIGHT: float = 0.35 * GLOBAL_SCALING

# Single-world layout wins on the dense ~11k-body contact pool.
USE_BIG_WORLD_MODE: bool = True
STEP_LAYOUT: str = "single_world" if USE_BIG_WORLD_MODE else "multi_world"

# Tonge mass splitting (C# PhoenX default). When ``True`` the
# partitioner caps at :data:`MASS_SPLITTING_MAX_COLORED_PARTITIONS`
# colours and any remainder goes to an overflow bucket solved with
# per-(body, partition) copy states. Currently requires the single-
# world layout (the multi-world fast-tail kernels haven't been
# refactored yet) and no joints / cloth — both true for this scene.
ENABLE_MASS_SPLITTING: bool = False
MASS_SPLITTING_MAX_COLORED_PARTITIONS: int = 12

# Tile the single ``KaplaTower2.usda`` instancer into a 2D grid centred
# on the origin. ``(1, 1)`` reproduces the original scene; bigger
# grids scale brick count, SAP candidates and contact pool linearly,
# which the collision-pipeline budgets below absorb via ``Nx * Ny``.
TOWER_GRID_DIMS: tuple[int, int] = (1, 1)
# Centre-to-centre spacing [m]. Tower footprint ~7.1 x 5.0 m at
# scale 0.1; 9 m leaves ~2 m clearance so neighbours don't leak into
# each other's SAP lists during settling.
TOWER_GRID_SPACING: tuple[float, float] = (9.0, 9.0)

# Optional axis-aligned filter box in the **original USDA frame**
# (i.e. before :data:`GLOBAL_SCALING` is applied and before the per-
# cell grid offset). The example multiplies both ``center`` and
# ``extents`` by :data:`GLOBAL_SCALING` internally and applies the
# same per-cell offset, so the numbers you type here match what's in
# ``kapla_tower_data.POSITIONS`` / the USDA file. Any brick whose
# centre falls inside the resulting AABB is skipped at scene-build
# time. Set both to ``None`` to disable. ``extents`` are full side
# lengths in the USDA frame.
BRICK_FILTER_BOX_CENTER: tuple[float, float, float] | None = (17.1, -25.0, 0.0)
BRICK_FILTER_BOX_EXTENTS: tuple[float, float, float] | None = (0.05 * 100, 0.4 * 100, 1.0 * 100)

# Invisible kinematic sphere parented to the camera. PhoenX's
# inferred-velocity path on :data:`MOTION_KINEMATIC` bodies turns
# fly-throughs into impulses on the bricks. Sphere (rotation-
# invariant) so we don't need to track camera orientation.
CAMERA_COLLIDER_ENABLED: bool = True
CAMERA_COLLIDER_RADIUS: float = 0.4
# Newton-side mass density only; PhoenX-side inverse mass is zeroed
# right after init so the value just needs to be finite.
CAMERA_COLLIDER_DENSITY: float = 1000.0

# Hard-coded drift test: run this many frames, then read back brick
# positions, print the maximum per-brick displacement from the spawn
# pose, and exit. Set to 0 to disable.
DRIFT_TEST_FRAMES: int = 0
# Log a one-line drift growth snapshot every N frames during the test.
# Set to 0 to disable progress logging.
DRIFT_TEST_LOG_EVERY: int = 100

# Per-island sleeping. When enabled, the PhoenX sleeping pass flags
# islands of bodies whose max-velocity score stays below
# :data:`SLEEPING_VELOCITY_THRESHOLD` for
# :data:`SLEEPING_FRAMES_REQUIRED` consecutive frames. Sleeping bodies
# skip the PGS solve and the broad-phase auto-installs a filter that
# drops rigid-rigid pairs where both sides are frozen. Sleeping shapes
# render dimmed by :data:`SLEEP_COLOR_GAIN`.
ENABLE_SLEEPING: bool = True
SLEEPING_VELOCITY_THRESHOLD: float = 0.15
SLEEPING_FRAMES_REQUIRED: int = 10
SLEEP_COLOR_GAIN: float = 0.55


@wp.kernel
def _darken_colors_kernel(
    active: wp.array[wp.vec3],
    gain: float,
    out: wp.array[wp.vec3],
):
    sid = wp.tid()
    out[sid] = gain * active[sid]


@wp.kernel
def _select_shape_color_by_sleep_kernel(
    active_colors: wp.array[wp.vec3],
    sleeping_colors: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_island_root: wp.array[wp.int32],
    shape_color_out: wp.array[wp.vec3],
):
    """Pick the per-shape colour based on its body's sleep state.
    ``island_root >= 0`` marks a sleeping body (value = lowest body id
    in its island at sleep time). PhoenX bodies are offset by +1
    relative to Newton (slot 0 is the static world anchor)."""
    sid = wp.tid()
    body = shape_body[sid]
    sleeping = int(0)
    if body >= 0:
        if body_island_root[body + 1] >= 0:
            sleeping = 1
    if sleeping != 0:
        shape_color_out[sid] = sleeping_colors[sid]
    else:
        shape_color_out[sid] = active_colors[sid]


class Example:
    """PhoenX Kapla Tower -- port of ``Demo15`` from PhoenXDemo.

    Per-frame pipeline (captured into a CUDA graph): sync Newton
    state -> PhoenX, run :class:`CollisionPipeline`, step
    :class:`PhoenXWorld`, sync back.
    """

    # Frames 0..WARMUP_FRAMES-1 pin global damping to 1.0 so the
    # USD-extracted brick poses (which carry mm-scale overlaps where
    # planks meet) don't kick the tower into divergence on frame 1.
    WARMUP_FRAMES: int = 20

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_index: int = 0
        self.sim_substeps = 8
        # iters=3 + sor=1.5 below settles slightly *better* than the
        # vanilla iters=4 sor=1.0 at +13% FPS. Over-relaxation
        # (omega=1.5) accelerates lambda propagation through tall
        # stacks; kapla's iter budget was bound by stack-pressure
        # propagation, not friction convergence (where cone clipping
        # would eat the boost). Validated by 1000-frame stability:
        # max brick velocity 0.66 m/s vs 0.73 m/s vanilla.
        self.solver_iterations = 10 if ENABLE_MASS_SPLITTING else 5
        self.velocity_iterations = 1

        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=GROUND_HEIGHT)

        hx = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[0]
        hy = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[1]
        hz = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[2]

        # USDA stores quats as half-precision; renormalise so Newton
        # gets unit quaternions in ``body_q``.
        positions = (POSITIONS * GLOBAL_SCALING).astype(np.float32)
        quats = ORIENTATIONS.astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.maximum(norms, 1e-12)

        # 1 cm AABB margin. Newton default (5 cm) is ~4x bigger than
        # the bricks and would explode the SAP candidate count.
        cfg = newton.ModelBuilder.ShapeConfig(density=BRICK_DENSITY, gap=0.01)

        nx, ny = (int(d) for d in TOWER_GRID_DIMS)
        if nx < 1 or ny < 1:
            raise ValueError(f"TOWER_GRID_DIMS components must be >= 1 (got {TOWER_GRID_DIMS})")
        spacing_x, spacing_y = (float(s) for s in TOWER_GRID_SPACING)
        self._tower_grid_dims: tuple[int, int] = (nx, ny)
        self._tower_grid_spacing: tuple[float, float] = (spacing_x, spacing_y)

        cell_centres_xy: list[tuple[float, float]] = []
        for j in range(ny):
            for i in range(nx):
                cx = (i - 0.5 * (nx - 1)) * spacing_x
                cy = (j - 0.5 * (ny - 1)) * spacing_y
                cell_centres_xy.append((cx, cy))
        self._cell_centres_xy = cell_centres_xy

        # Resolve the optional filter box once -- ``None`` disables
        # filtering. Inputs are in the original USDA frame, so apply
        # :data:`GLOBAL_SCALING` here to match the brick centres we
        # compare against (which are scaled below). Half-extents are
        # precomputed so the per-brick test is a pure abs-compare.
        filter_box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        if BRICK_FILTER_BOX_CENTER is not None and BRICK_FILTER_BOX_EXTENTS is not None:
            fc = tuple(float(c) * GLOBAL_SCALING for c in BRICK_FILTER_BOX_CENTER)
            fh = tuple(0.5 * float(e) * GLOBAL_SCALING for e in BRICK_FILTER_BOX_EXTENTS)
            if any(h < 0.0 for h in fh):
                raise ValueError(f"BRICK_FILTER_BOX_EXTENTS must be non-negative (got {BRICK_FILTER_BOX_EXTENTS})")
            filter_box = (fc, fh)  # type: ignore[assignment]
        elif (BRICK_FILTER_BOX_CENTER is None) ^ (BRICK_FILTER_BOX_EXTENTS is None):
            raise ValueError("BRICK_FILTER_BOX_CENTER and BRICK_FILTER_BOX_EXTENTS must both be set or both be None")
        self._brick_filter_box = filter_box

        # Per-cell newton-id list (cell_index = j * nx + i, row-major).
        self._brick_newton_ids: list[list[int]] = [[] for _ in range(nx * ny)]
        num_filtered = 0
        for cell_index, (cx, cy) in enumerate(cell_centres_xy):
            for i in range(NUM_BRICKS):
                qx, qy, qz, qw = quats[i]
                px, py, pz = positions[i]
                wx = float(px) + cx
                wy = float(py) + cy
                wz = float(pz)
                if filter_box is not None:
                    (fcx, fcy, fcz), (fhx, fhy, fhz) = filter_box
                    if abs(wx - fcx) <= fhx and abs(wy - fcy) <= fhy and abs(wz - fcz) <= fhz:
                        num_filtered += 1
                        continue
                body = builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(wx, wy, wz),
                        q=wp.quat(float(qx), float(qy), float(qz), float(qw)),
                    ),
                )
                builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg)
                self._brick_newton_ids[cell_index].append(body)
        self._num_filtered_bricks = num_filtered

        # Camera collider added as dynamic so finalize() accepts it,
        # then flipped to :data:`MOTION_KINEMATIC` and inverse-mass-
        # zeroed below.
        self._camera_body_newton_id: int | None = None
        # Must match the ``viewer.set_camera`` pos below so the first
        # frame doesn't slam the collider into the tower.
        self._camera_collider_initial_pos: tuple[float, float, float] = (1.2, 0.0, 1.4)
        if CAMERA_COLLIDER_ENABLED:
            cam_cfg = newton.ModelBuilder.ShapeConfig(
                density=CAMERA_COLLIDER_DENSITY,
                gap=0.01,
                is_visible=False,
            )
            cam_body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(*(float(c) for c in self._camera_collider_initial_pos)),
                    q=wp.quat_identity(),
                ),
            )
            builder.add_shape_sphere(cam_body, radius=float(CAMERA_COLLIDER_RADIUS), cfg=cam_cfg)
            self._camera_body_newton_id = cam_body

        # SAP doesn't need the explicit O(N^2) pair list, and skipping
        # avoids minutes of Python work + tens of GB of speculative
        # pair allocation.
        self.model = builder.finalize(skip_shape_contact_pairs=True)
        total_tower_bricks = sum(len(ids) for ids in self._brick_newton_ids)
        print(
            f"[PhoenX KaplaTower] bodies={self.model.body_count} "
            f"shapes={self.model.shape_count} "
            f"tower_grid={nx}x{ny} "
            f"tower_bricks={total_tower_bricks} "
            f"filtered_bricks={self._num_filtered_bricks} "
            f"camera_collider={'yes' if self._camera_body_newton_id is not None else 'no'} "
            f"brick_full_extents={BRICK_FULL_EXTENTS} scale={GLOBAL_SCALING}"
        )

        # SAP broad phase + warm-start contact_matching for stable
        # settling. ``shape_pairs_max`` / ``rigid_contact_max``
        # budgets sized from the observed single-tower numbers
        # (~800k SAP candidates, ~750k contacts) and scaled linearly
        # by the cell count -- replicated cells are spatially
        # disjoint so SAP drops cross-cell pairs.
        num_cells = nx * ny
        shape_pairs_max = 1_500_000 * num_cells
        # Hand-tuned upper bound on simultaneously-active contact
        # columns. 500k easily covers the observed ~750k narrow-phase
        # contacts collapsed into ~50-80k columns (one column per
        # shape pair). The mass-splitting copy-state and the sort
        # buffer for the interaction graph both scale with this, so
        # over-budgeting wastes GPU memory and slows the radix sort.
        rigid_contact_max_pipeline = 500_000 * num_cells
        # Sleeping requires the PhoenX-aware broad-phase filter so the
        # SAP pass drops rigid-rigid pairs where both sides are frozen.
        # ``PhoenXWorld.attach_collision_pipeline`` (below) binds the
        # per-step filter data; here we just expose the slot.
        cp_kwargs = {
            "contact_matching": PHOENX_CONTACT_MATCHING,
            "broad_phase": "sap",
            "shape_pairs_max": shape_pairs_max,
            "rigid_contact_max": rigid_contact_max_pipeline,
        }
        if ENABLE_SLEEPING:
            cp_kwargs["broad_phase_filter"] = PhoenXWorld.broad_phase_filter()
        self.collision_pipeline = newton.CollisionPipeline(self.model, **cp_kwargs)
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.body_q.assign(self.state.body_q)

        # PhoenX body container -- slot 0 stays static world anchor.
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            ),
        )
        wp.launch(
            _init_phoenx_bodies_kernel,
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

        # Flip the camera collider to MOTION_KINEMATIC and zero its
        # inverse mass / inertia so the solver treats it as an
        # immovable but velocity-aware rail.
        if self._camera_body_newton_id is not None:
            slot = self._camera_body_newton_id + 1
            self._camera_phoenx_slot: int = slot
            mt_np = bodies.motion_type.numpy()
            mt_np[slot] = int(MOTION_KINEMATIC)
            bodies.motion_type.assign(mt_np)
            inv_m_np = bodies.inverse_mass.numpy()
            inv_m_np[slot] = 0.0
            bodies.inverse_mass.assign(inv_m_np)
            inv_I_np = bodies.inverse_inertia.numpy()
            inv_I_np[slot] = np.zeros((3, 3), dtype=np.float32)
            bodies.inverse_inertia.assign(inv_I_np)
            inv_Iw_np = bodies.inverse_inertia_world.numpy()
            inv_Iw_np[slot] = np.zeros((3, 3), dtype=np.float32)
            bodies.inverse_inertia_world.assign(inv_Iw_np)
        else:
            self._camera_phoenx_slot: int | None = None  # type: ignore[no-redef]

        self.bodies = bodies

        # Drift test: snapshot the post-init brick positions so step()
        # can compute max per-brick displacement after running for
        # DRIFT_TEST_FRAMES frames. Indexed by Newton body id (slot 0
        # is the static world anchor in PhoenX, so we add +1 below).
        self._drift_t_start: float = time.perf_counter()
        initial_positions_np = self.bodies.position.numpy().copy()
        flat_newton_ids: list[int] = [newton_idx for cell_ids in self._brick_newton_ids for newton_idx in cell_ids]
        self._drift_newton_ids: np.ndarray = np.asarray(flat_newton_ids, dtype=np.int64)
        self._drift_initial_positions: np.ndarray = np.asarray(
            [initial_positions_np[idx + 1] for idx in flat_newton_ids],
            dtype=np.float32,
        )

        # Joint-side container only needs a placeholder row -- contact
        # columns live in PhoenXWorld's own ContactColumnContainer.
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            device=self.device,
        )

        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        # Drift-test env knobs (all opt-in):
        #   PHOENX_DRIFT_NO_WARM_START=1  -> disable cross-frame
        #     warm-start coloring entirely (full cold-start).
        #   PHOENX_DRIFT_SYMMETRIC=1      -> enable cyclic-shift colour
        #     sweep (experimental; on its own it doesn't fix drift).
        #   PHOENX_DRIFT_INVALIDATE_EVERY=<N> -> rebuild the warm-start
        #     coloring from scratch every N frames. ``0`` (default)
        #     disables. ``2`` fixes Kapla drift at ~5% perf cost; higher
        #     values are cheaper but leak drift back in.
        #   PHOENX_DRIFT_ROTATE_SKIP=1    -> rotate-skip one cached
        #     colour each step (~1/num_colors of cold-start cost).
        #     Default ON. Cheap drift-mitigation alternative to full
        #     periodic invalidate.
        #   PHOENX_DRIFT_SOR              -> override ``sor_boost``.
        enable_warm_start_coloring_env = os.environ.get("PHOENX_DRIFT_NO_WARM_START", "0")
        enable_warm_start = enable_warm_start_coloring_env == "0"
        symmetric_sweep = os.environ.get("PHOENX_DRIFT_SYMMETRIC", "0") == "1"
        # Defaults (period=4 + rotate_skip=True) match the lowest-cost
        # config we found that holds drift below cold-start levels at
        # full warm-start FPS on this scene.
        invalidate_every = int(os.environ.get("PHOENX_DRIFT_INVALIDATE_EVERY", "4"))
        rotate_skip = os.environ.get("PHOENX_DRIFT_ROTATE_SKIP", "1") == "1"
        sor_boost_env = os.environ.get("PHOENX_DRIFT_SOR")
        sor_boost = float(sor_boost_env) if sor_boost_env else 1.0
        print(
            f"[PhoenX KaplaTower] drift_knobs: warm_start={enable_warm_start} "
            f"symmetric_sweep={symmetric_sweep} invalidate_every={invalidate_every} "
            f"rotate_skip={rotate_skip} sor_boost={sor_boost}",
            flush=True,
        )

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=rigid_contact_max,
            step_layout=STEP_LAYOUT,
            max_thread_blocks=256,
            mass_splitting=ENABLE_MASS_SPLITTING,
            max_colored_partitions=MASS_SPLITTING_MAX_COLORED_PARTITIONS,
            mass_splitting_unrolled=True,
            mass_splitting_batch_size=8,
            enable_warm_start_coloring=enable_warm_start,
            symmetric_color_sweep=symmetric_sweep,
            warm_start_invalidate_period=invalidate_every,
            warm_start_rotate_skip_color=rotate_skip,
            warm_start_rotate_skip_width=int(os.environ.get("PHOENX_DRIFT_ROTATE_WIDTH", "1")),
            capture_while_greedy_coloring=(os.environ.get("PHOENX_DRIFT_CAPTURE_WHILE", "1") == "1"),
            speculative_coloring=(os.environ.get("PHOENX_DRIFT_SPECULATIVE", "1") == "1"),
            max_greedy_outer_iters=(
                int(os.environ.get("PHOENX_DRIFT_MAX_GREEDY_ITERS"))
                if os.environ.get("PHOENX_DRIFT_MAX_GREEDY_ITERS")
                else None
            ),
            sor_boost=sor_boost,
            sleeping_velocity_threshold=SLEEPING_VELOCITY_THRESHOLD if ENABLE_SLEEPING else 0.0,
            sleeping_frames_required=SLEEPING_FRAMES_REQUIRED,
            device=self.device,
        )

        # Sleeping plumbing: a single call binds the share-vertex
        # filter data, caches the pipeline's per-shape AABB arrays for
        # the per-step sleep score, and installs the PhoenX-offset
        # shape_body map. Renderer-side dimming buffers stay here
        # (visualisation, not solver state).
        self._shape_color_active = None
        self._shape_color_sleeping = None
        if ENABLE_SLEEPING:
            self.world.attach_collision_pipeline(
                self.collision_pipeline,
                num_rigid_shapes=int(self.model.shape_count),
                shape_body=self.model.shape_body,
            )
            self._shape_color_active = wp.clone(self.model.shape_color)
            self._shape_color_sleeping = wp.empty_like(self.model.shape_color)
            wp.launch(
                _darken_colors_kernel,
                dim=self.model.shape_count,
                inputs=[self._shape_color_active, SLEEP_COLOR_GAIN],
                outputs=[self._shape_color_sleeping],
                device=self.device,
            )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(*(float(c) for c in self._camera_collider_initial_pos)),
            pitch=-12.0,
            yaw=180.0,
        )

        # Picking AABBs: brick = box half-extents. Slot 0 (world anchor)
        # and the camera collider stay at (0,0,0) so the raycast skips
        # them -- the camera collider sits AT the camera position, so
        # any pick ray would hit it first and lock picking onto a
        # kinematic body that ignores force input.
        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        for cell_ids in self._brick_newton_ids:
            for newton_idx in cell_ids:
                half_extents_np[newton_idx + 1] = (hx, hy, hz)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # Camera-collider buffers, capture-safe by construction:
        # ``_camera_pos_host`` is pinned (so wp.copy is a fixed-pointer
        # DMA the captured graph can replay), ``_camera_pos_host_np``
        # aliases its bytes for in-place Python writes between
        # replays, and ``_camera_pos_arr`` is the device mirror.
        # Orientation is identity for the sphere collider.
        if self._camera_phoenx_slot is not None:
            self._camera_body_id_arr = wp.array([int(self._camera_phoenx_slot)], dtype=wp.int32, device=self.device)
            self._camera_pos_arr = wp.array(
                [self._camera_collider_initial_pos],
                dtype=wp.vec3f,
                device=self.device,
            )
            self._camera_orient_arr = wp.array([(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=self.device)
            self._camera_pos_host = wp.array(
                [self._camera_collider_initial_pos],
                dtype=wp.vec3f,
                device="cpu",
                pinned=True,
            )
            # ``numpy()`` on a pinned CPU array returns an aliased
            # view, so in-place writes to this handle mutate the
            # underlying wp.array without allocations.
            self._camera_pos_host_np = self._camera_pos_host.numpy()

        # Warm-up damping (relaxed in :meth:`step` after WARMUP_FRAMES).
        # Read from a device slot at replay time, so toggling later
        # doesn't require a recapture.
        self.world.set_global_linear_damping(1.0)
        self.world.set_global_angular_damping(1.0)

        # Capture the per-frame pipeline into a single CUDA graph.
        # The PGS sweep uses ``wp.capture_while`` internally, which
        # only takes its conditional-graph fast path when there's an
        # outer capture active.
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # ------------------------------------------------------------------
    # Simulation + rendering
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        # Stage the camera position on the device, hand it to PhoenX as
        # the kinematic target, AND patch the same slot of
        # ``state.body_q`` so Newton's CollisionPipeline (which
        # broad/narrow-phases against ``body_q``) sees the collider at
        # the live camera location.
        if self._camera_phoenx_slot is not None:
            wp.copy(self._camera_pos_arr, self._camera_pos_host)
            self.world.set_kinematic_poses_batch(
                body_ids=self._camera_body_id_arr,
                positions=self._camera_pos_arr,
                orientations=self._camera_orient_arr,
            )
            wp.launch(
                _write_camera_body_q_kernel,
                dim=1,
                inputs=[
                    wp.int32(self._camera_body_newton_id),
                    self._camera_pos_arr,
                    self._camera_orient_arr,
                    self.state.body_q,
                ],
                device=self.device,
            )
        self._sync_newton_to_phoenx()
        # Picking before collide so the wake propagates through the
        # broad-phase filter on the same frame. ``wake_on_external_input``
        # is a no-op when sleeping is disabled, so the order is uniform
        # regardless of ``ENABLE_SLEEPING``.
        self.picking.apply_force()
        self.world.wake_on_external_input()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        # When sleeping is on, ``attach_collision_pipeline`` cached
        # the per-shape AABB arrays + installed the shape_body map, so
        # no extra step() args are needed.
        self.world.step(dt=self.frame_dt, contacts=self.contacts, shape_body=self._shape_body)
        self._sync_phoenx_to_newton()

    def _sync_newton_to_phoenx(self) -> None:
        # Camera collider is always the last body and is driven by the
        # kinematic target, not the dynamic-body sync. Cap the range
        # at n-1 to leave its PhoenX slot alone.
        n = self.model.body_count
        if self._camera_body_newton_id is not None:
            n -= 1
        if n <= 0:
            return
        wp.launch(
            _newton_to_phoenx_kernel,
            dim=n,
            inputs=[
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
            ],
            outputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_phoenx_to_newton(self) -> None:
        # Symmetric to :meth:`_sync_newton_to_phoenx` -- skip the camera
        # slot. ``state.body_q[camera]`` is owned by the host-side
        # update in :meth:`simulate`.
        n = self.model.body_count
        if self._camera_body_newton_id is not None:
            n -= 1
        if n <= 0:
            return
        wp.launch(
            _phoenx_to_newton_kernel,
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

    def _update_camera_collider(self) -> None:
        """Stage the camera position into the pinned host buffer.

        The captured graph re-reads the bytes via ``wp.copy`` on each
        replay, so the in-place numpy write is all that's needed
        between frames. Headless viewers (no ``camera.pos``) fall
        back to the initial spawn position.
        """
        if self._camera_phoenx_slot is None:
            return
        cam_pos = getattr(getattr(self.viewer, "camera", None), "pos", None)
        if cam_pos is None:
            cam_xyz = self._camera_collider_initial_pos
        else:
            cam_xyz = (float(cam_pos.x), float(cam_pos.y), float(cam_pos.z))
        self._camera_pos_host_np[0] = cam_xyz

    def step(self) -> None:
        # Drift test exit: once DRIFT_TEST_FRAMES sim frames have run,
        # read back brick positions, print the max displacement from
        # the spawn pose, and exit before stepping further.
        if DRIFT_TEST_FRAMES > 0 and self.frame_index >= DRIFT_TEST_FRAMES:
            self._report_drift_and_exit()
        # Progress log: print incremental drift every DRIFT_TEST_LOG_EVERY
        # frames so long runs surface a growth curve even if they get
        # killed by a timeout before reaching DRIFT_TEST_FRAMES.
        if (
            DRIFT_TEST_FRAMES > 0
            and DRIFT_TEST_LOG_EVERY > 0
            and self.frame_index > 0
            and self.frame_index % DRIFT_TEST_LOG_EVERY == 0
        ):
            self._log_drift_progress()
        # Release the warm-up damping pin once we're past the settle.
        # Toggling the device-side slot is capture-safe; no recapture.
        if self.frame_index == self.WARMUP_FRAMES:
            self.world.set_global_linear_damping(0.0)
            self.world.set_global_angular_damping(0.0)
        self._update_camera_collider()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        if os.environ.get("PHOENX_DRIFT_REPORT") and self.frame_index in (0, 30, 60, 100, 200):
            self._print_step_report()
        # Optional snapshot for the standalone coloring bench.
        dump_frame_env = os.environ.get("PHOENX_DUMP_COLORING_GRAPH")
        if dump_frame_env is not None and int(dump_frame_env) == self.frame_index:
            self._dump_coloring_graph()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _compute_drift(self) -> tuple[float, float, float, int]:
        """Return (max, mean, p99, worst_idx) of |pos - initial| over
        all tower bricks. Reads from device every call -- only invoke
        from non-hot-path / reporting code."""
        positions = self.bodies.position.numpy()
        cur = positions[self._drift_newton_ids + 1]
        deltas = cur.astype(np.float32) - self._drift_initial_positions
        distances = np.linalg.norm(deltas, axis=1)
        if distances.size == 0:
            return 0.0, 0.0, 0.0, -1
        worst = int(np.argmax(distances))
        return (
            float(distances[worst]),
            float(distances.mean()),
            float(np.percentile(distances, 99.0)),
            int(self._drift_newton_ids[worst]),
        )

    def _log_drift_progress(self) -> None:
        max_d, mean_d, p99_d, worst_idx = self._compute_drift()
        elapsed = time.perf_counter() - self._drift_t_start
        fps = self.frame_index / elapsed if elapsed > 0 else 0.0
        print(
            f"[PhoenX KaplaTower] drift@{self.frame_index} "
            f"max={max_d:.6f} mean={mean_d:.6f} p99={p99_d:.6f} "
            f"(worst={worst_idx}, {elapsed:.1f}s, {fps:.1f} fps)",
            flush=True,
        )

    def _report_drift_and_exit(self) -> None:
        """Print the max per-brick displacement from spawn, then exit.

        Used by the hard-coded :data:`DRIFT_TEST_FRAMES` automated
        drift test. The reference pose is captured in
        :meth:`_build_scene` right after the PhoenX bodies are
        initialised, so the printed number is the absolute drift
        relative to the USDA-extracted spawn poses (including any
        warm-up damping effect).
        """
        max_d, mean_d, p99_d, worst_idx = self._compute_drift()
        print(
            f"[PhoenX KaplaTower] DRIFT_RESULT frames={self.frame_index} "
            f"bricks={self._drift_initial_positions.shape[0]} "
            f"max={max_d:.6f} mean={mean_d:.6f} p99={p99_d:.6f} worst_newton_idx={worst_idx}",
            flush=True,
        )
        sys.exit(0)

    def _dump_coloring_graph(self) -> None:
        """Snapshot the active constraint graph to ``kapla_graph.npz``
        for the standalone coloring benchmark to replay."""
        partitioner = self.world._partitioner
        n_active = int(self.world._num_active_constraints.numpy()[0])
        elements_struct = partitioner._elements.numpy()[:n_active]
        bodies = elements_struct["bodies"].astype(np.int32, copy=False)
        cost = partitioner._cost_values.numpy()[:n_active].astype(np.int32, copy=False)
        jitter = partitioner._random_values.numpy()[:n_active].astype(np.int32, copy=False)
        out_path = pathlib.Path("kapla_graph.npz").resolve()
        np.savez(
            out_path,
            bodies=bodies,
            cost_values=cost,
            random_values=jitter,
            num_bodies=np.int32(self.world.num_bodies),
            frame_index=np.int32(self.frame_index),
        )
        print(f"[PhoenX KaplaTower] dumped coloring graph to {out_path} (frame={self.frame_index}, n={n_active})")

    def _print_step_report(self) -> None:
        report = self.world.step_report()
        # ``max_body_degree`` is the chromatic lower bound; the ratio
        # measures how close the colourer is to it.
        slack = f"{report.num_colors / report.max_body_degree:.2f}x" if report.max_body_degree > 0 else "n/a"
        fields = [
            f"step={self.frame_index}",
            f"contacts={report.num_contact_columns}",
            f"active_constraints={report.num_active_constraints}",
            f"colors={report.num_colors}",
            f"max_body_degree={report.max_body_degree}",
            f"colors/lower_bound={slack}",
            f"color_sizes={report.color_sizes}",
        ]
        if report.per_world_num_colors is not None:
            fields.append(f"per_world_num_colors={report.per_world_num_colors}")
            fields.append(f"per_world_color_sizes={report.per_world_color_sizes}")
        print("[PhoenX KaplaTower] " + " ".join(fields))

    def _refresh_sleep_colors(self) -> None:
        """Write sleep-aware colors into ``model.shape_color`` on the
        GPU; the viewer picks them up via its own per-frame
        ``_sync_shape_colors_from_model`` pass."""
        wp.launch(
            _select_shape_color_by_sleep_kernel,
            dim=self.model.shape_count,
            inputs=[
                self._shape_color_active,
                self._shape_color_sleeping,
                self.model.shape_body,
                self.bodies.island_root,
            ],
            outputs=[self.model.shape_color],
            device=self.device,
        )

    def render(self) -> None:
        if ENABLE_SLEEPING:
            self._refresh_sleep_colors()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Post-settle validation
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """After settling, no brick may have escaped its cell's
        envelope. Catches NaNs / ejected bodies without
        over-constraining the exact settled geometry.
        """
        # 4x the unscaled tower radius is generous post-scale.
        tower_radius = float(np.linalg.norm(POSITIONS[:, :2], axis=1).max())
        tower_tolerance = 4.0 * tower_radius * GLOBAL_SCALING
        positions = self.bodies.position.numpy()
        for cell_index, cell_ids in enumerate(self._brick_newton_ids):
            cx, cy = self._cell_centres_xy[cell_index]
            for newton_idx in cell_ids:
                pos = positions[newton_idx + 1]
                assert np.isfinite(pos).all(), f"body {newton_idx} non-finite position: {pos}"
                r_xy = float(np.hypot(pos[0] - cx, pos[1] - cy))
                assert r_xy < tower_tolerance, (
                    f"brick {newton_idx} (cell {cell_index}) flew outside the tower envelope "
                    f"(r_xy={r_xy:.3f}, tol={tower_tolerance:.3f})"
                )

        for newton_idx in self._wrecking_ball_newton_ids:
            pos = positions[newton_idx + 1]
            assert np.isfinite(pos).all(), f"wrecking ball {newton_idx} non-finite position: {pos}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    # Start paused so the initial (potentially inter-penetrating)
    # brick layout is visible before the solver begins resolving.
    # Press SPACE or toggle the viewer's "Pause" checkbox to run.
    # viewer._paused = True
    newton.examples.run(example, args)
