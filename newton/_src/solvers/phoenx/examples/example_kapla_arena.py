# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Kapla Arena
#
# Port of PhoenX's ``Demo16: Kapla Arena`` (see
# ``PhoenX/src/Viewer/Demos/Demo16.cs``). The C# demo loads
# ``KaplaArena.usda`` -- a USD ``PointInstancer`` describing
# ~13k Kapla-style wooden bricks tiled into a flat circular arena --
# and lets PhoenX settle them onto the ground.
#
# We don't depend on USD here: the brick prototype's full extents and
# every instance's position / orientation have been extracted into
# :mod:`kapla_arena_data` (regenerate with
# ``python _extract_kapla_usda.py`` from the repo root).
#
# The USDA is +Z-up (``upAxis = "Z"``); PhoenX is +Y-up so the C#
# demo rotates the whole instancer by ``-pi/2`` about +X. Newton is
# also +Z-up, so we keep positions and orientations as-is. The
# overall scene is uniformly scaled by :data:`GLOBAL_SCALING` to
# match the C# demo's metric size.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_kapla_arena
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    newton_to_phoenx_kernel as _newton_to_phoenx_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    phoenx_to_newton_kernel as _phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.examples.kapla_arena_data import (
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

# ---- Geometry ------------------------------------------------------------
# Matches the C# ``float globalScaling = 0.1f`` in ``Demo16.cs``. The
# raw USD positions span ~80 m in the horizontal plane and ~3 m in
# height; multiplying by 0.1 yields an arena ~8 m across.
GLOBAL_SCALING: float = 0.1

# Brick density. Same rationale as :mod:`example_kapla_tower`: the
# absolute mass only sets the time scale, what matters is uniform
# density across all bricks.
BRICK_DENSITY: float = 1000.0

# C# uses ``ResetScene(true, 0.12f * globalScaling)``: the second arg
# is a Y-offset for the ground top surface.
GROUND_HEIGHT: float = 0.12 * GLOBAL_SCALING

# ---- Step layout toggle ---------------------------------------------------
# ~13k bodies in a single world: same big-world regime as
# :mod:`example_kapla_tower`.
USE_BIG_WORLD_MODE: bool = True
STEP_LAYOUT: str = "single_world" if USE_BIG_WORLD_MODE else "multi_world"


class Example:
    """PhoenX Kapla Arena -- port of ``Demo16`` from PhoenXDemo.

    Same per-frame pipeline as :class:`example_kapla_tower.Example`;
    differences are limited to the embedded brick data, the global
    scale, the ground height, and the substep / iteration counts.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # First ``WARMUP_FRAMES`` frames pin global linear + angular
        # damping at 1.0 to zero out velocities every substep -- the
        # USDA's Kapla brick poses have small overlaps where bricks meet
        # at corners, and without the warm-up the pent-up impulses lift
        # the arena off the ground in frame 1 and cascade into a
        # divergence. Damping is released to 0 after the warm-up.
        self.WARMUP_FRAMES: int = 20
        self.frame_index: int = 0

        # Frame pacing and solver settings match ``Demo16.cs`` exactly:
        # ``world.StepDt = 1 / 120``, ``NumberSubsteps = 4``,
        # ``SolverIterations = 4``. ``SolverVelocityIterations`` is
        # left at PhoenX's default (0) -- the arena branch of Demo16
        # doesn't touch it.
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.solver_iterations = 4
        self.velocity_iterations = 0

        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        """Build the Newton model from the embedded Kapla brick data,
        then wire up the PhoenX body / constraint containers and the
        :class:`PhoenXWorld` driver.

        Brick instance positions / orientations are loaded from
        :mod:`kapla_arena_data`, which was extracted once from
        ``KaplaArena.usda``.
        """
        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=GROUND_HEIGHT)

        hx = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[0]
        hy = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[1]
        hz = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[2]

        # The arena's source orientations come from a ``quatf[]``
        # array (single-precision) so they're already very close to
        # unit, but we still renormalise for safety -- Newton's
        # ``transform`` rotation pipeline assumes unit quaternions.
        positions = (POSITIONS * GLOBAL_SCALING).astype(np.float32)
        quats = ORIENTATIONS.astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.maximum(norms, 1e-12)

        # ``gap`` is the contact-margin half-width applied to the
        # shape's AABB during broad-phase culling. Newton's default
        # (5 cm) is meant for human-scale bodies; for the arena's
        # 5 cm half-extent planks it would inflate every AABB by
        # ~2x and bloat the SAP candidate-pair count. 1 cm is a
        # sensible margin at this scale.
        cfg = newton.ModelBuilder.ShapeConfig(density=BRICK_DENSITY, gap=0.01)
        self._brick_newton_ids: list[int] = []
        for i in range(NUM_BRICKS):
            qx, qy, qz, qw = quats[i]
            px, py, pz = positions[i]
            body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(px), float(py), float(pz)),
                    q=wp.quat(float(qx), float(qy), float(qz), float(qw)),
                ),
            )
            builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg)
            self._brick_newton_ids.append(body)

        # See :mod:`example_kapla_tower` for the rationale behind
        # ``skip_shape_contact_pairs=True``: SAP doesn't consume
        # the explicit pair list, and leaving it in place would
        # push PhoenX's contact-buffer heuristic into a
        # multi-gigabyte allocation request.
        self.model = builder.finalize(skip_shape_contact_pairs=True)
        print(
            f"[PhoenX KaplaArena] bodies={self.model.body_count} "
            f"shapes={self.model.shape_count} "
            f"brick_full_extents={BRICK_FULL_EXTENTS} scale={GLOBAL_SCALING}"
        )

        # Collision pipeline. ``broad_phase="sap"`` for the same
        # reason as :mod:`example_kapla_tower`: the ``O(N^2)``
        # default would emit far more candidates than Newton's
        # narrow phase can absorb on 13k bodies. The small
        # per-shape ``gap`` above keeps SAP's pair count tight.
        # ``shape_pairs_max`` overrides the worst-case ``N*(N-1)/2``
        # bound (~88M pairs for 13k bricks) which would otherwise
        # blow ~12 GB on contact-sorter / narrow-phase scratch.
        # Arena's actual SAP peak is ~120k pairs; ~2.5x that
        # gives plenty of room for transient settling jitter.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=PHOENX_CONTACT_MATCHING,
            broad_phase="sap",
            shape_pairs_max=300_000,
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.body_q.assign(self.state.body_q)

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
        self.bodies = bodies

        # Contact columns live in a dedicated ``ContactColumnContainer``
        # inside :class:`PhoenXWorld`; the joint-side constraint
        # container only needs a 1-row placeholder in this
        # contact-only scene.
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            device=self.device,
        )

        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=rigid_contact_max,
            step_layout=STEP_LAYOUT,
            max_thread_blocks=32,
            device=self.device,
        )

        # Camera -- arena spans ~8 m across at hip height. Sit the
        # camera back ~12 m and a few meters up so the ring is in
        # frame.
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(12.0, 0.0, 4.0),
            pitch=-15.0,
            yaw=180.0,
        )

        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        for newton_idx in self._brick_newton_ids:
            half_extents_np[newton_idx + 1] = (hx, hy, hz)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # Pin global damping to 1.0 for the warm-up window so the
        # captured graph zeroes velocities every substep.
        self.world.set_global_linear_damping(1.0)
        self.world.set_global_angular_damping(1.0)

        # CUDA graph capture for the per-frame step pipeline. Falls
        # back to direct :meth:`simulate` on CPU or when capture is
        # disabled. PhoenX's single-world PGS sweeps use
        # ``wp.capture_while`` internally, which only takes the
        # conditional-graph fast path when an outer capture is
        # active -- otherwise every colour launch does a D2H sync
        # on the colour cursor. Capturing here is what keeps those
        # out of the steady-state profile.
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # ------------------------------------------------------------------
    # Simulation + rendering
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        self._sync_newton_to_phoenx()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
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
        n = self.model.body_count
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

    def step(self) -> None:
        # End of warm-up: release the damping pin so the live demo runs
        # without artificial energy loss. The captured graph reads
        # ``self.world._global_damping`` from a device slot, so this
        # host-side write takes effect on the next ``capture_launch``
        # without re-capture.
        if self.frame_index == self.WARMUP_FRAMES:
            self.world.set_global_linear_damping(0.0)
            self.world.set_global_angular_damping(0.0)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Post-settle validation
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """After settling, no brick may have escaped a generous
        envelope around the original arena footprint."""
        radius = float(np.linalg.norm(POSITIONS[:, :2], axis=1).max())
        tolerance = 4.0 * radius * GLOBAL_SCALING
        positions = self.bodies.position.numpy()
        for newton_idx in self._brick_newton_ids:
            pos = positions[newton_idx + 1]
            assert np.isfinite(pos).all(), f"body {newton_idx} non-finite position: {pos}"
            r_xy = float(np.hypot(pos[0], pos[1]))
            assert r_xy < tolerance, (
                f"brick {newton_idx} flew outside the arena envelope (r_xy={r_xy:.3f}, tol={tolerance:.3f})"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
