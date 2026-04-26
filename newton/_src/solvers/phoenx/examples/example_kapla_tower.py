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

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.solver_config import (
    PHOENX_CONTACT_MATCHING,
)
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
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

# ---- Geometry ------------------------------------------------------------
# Matches the C# ``float globalScaling = 0.01f`` in ``Demo15.cs``. The
# raw USD positions span ~70 m in the horizontal plane and 28 m in
# height; multiplying by 0.01 yields a ~70 cm tall tabletop tower.
GLOBAL_SCALING: float = 0.1

# Brick density. The C# scene relies on PhoenX's default unit-density
# brick so the absolute mass only sets the time scale; what matters
# for stability is uniform density across all bricks. 1000 kg/m^3 is
# Newton's standard and gives each brick a sensible 3.2 g.
BRICK_DENSITY: float = 1000.0

# C# uses ``ResetScene(true, 0.35f * globalScaling)``: the second arg
# is a Y-offset for the ground top surface. Newton applies it as
# ``add_ground_plane(height=...)``.
GROUND_HEIGHT: float = 0.35 * GLOBAL_SCALING

# ---- Step layout toggle ---------------------------------------------------
# Kapla Tower is ~11k bodies in a single world, dominated by one
# huge contact pool -- the same regime where ``single_world`` wins
# in :mod:`example_tower`.
USE_BIG_WORLD_MODE: bool = True
STEP_LAYOUT: str = "single_world" if USE_BIG_WORLD_MODE else "multi_world"


class Example:
    """PhoenX Kapla Tower -- port of ``Demo15`` from PhoenXDemo.

    Pipeline per frame mirrors :class:`example_tower.Example`:
        1. Sync Newton state -> PhoenX body container.
        2. Run Newton's :class:`CollisionPipeline` to produce the
           ``Contacts`` buffer.
        3. Call :meth:`PhoenXWorld.step` to advance the solver.
        4. Sync PhoenX body container -> Newton state.

    The whole per-frame pipeline is captured into a single CUDA
    graph at construction time and replayed via
    :func:`wp.capture_launch` every frame.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # Frame pacing and solver settings match ``Demo15.cs`` exactly:
        # ``world.StepDt = 1 / 120``, ``NumberSubsteps = 15``,
        # ``SolverIterations = 3``, ``SolverVelocityIterations = 0``.
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 15
        self.solver_iterations = 3
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
        :mod:`kapla_tower_data`, which was extracted once from
        ``KaplaTower2.usda``.
        """
        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=GROUND_HEIGHT)

        # Brick half-extents after the global scale.
        hx = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[0]
        hy = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[1]
        hz = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[2]

        # Renormalise the per-instance quaternions: the source USDA
        # uses ``quath`` (half-precision) for tower orientations, so
        # the embedded floats are off-unit by O(1e-3). Newton expects
        # unit quaternions in ``body_q``.
        positions = (POSITIONS * GLOBAL_SCALING).astype(np.float32)
        quats = ORIENTATIONS.astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.maximum(norms, 1e-12)

        # ``gap`` is the contact-margin half-width applied to the
        # shape's AABB during broad-phase culling. The Newton default
        # (5 cm) was tuned for human-scale bodies and is ~4x larger
        # than each brick itself -- it pads the AABB by ~10x and
        # explodes the SAP candidate-pair count into the millions.
        # 1 cm is a comfortable margin for these 1.3 cm half-extent
        # planks.
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

        # ``skip_shape_contact_pairs=True``: avoid the ``O(N^2)``
        # Python pair-list builder in ``finalize``. We use SAP for
        # the broad phase below, which doesn't consume the
        # explicit list. Without this flag, finalize takes minutes
        # on 11k bodies *and* PhoenX's contact-buffer heuristic
        # reads ``shape_contact_pair_count`` (~64M) and tries to
        # allocate tens of GB of GPU memory.
        self.model = builder.finalize(skip_shape_contact_pairs=True)
        print(
            f"[PhoenX KaplaTower] bodies={self.model.body_count} "
            f"shapes={self.model.shape_count} "
            f"brick_full_extents={BRICK_FULL_EXTENTS} scale={GLOBAL_SCALING}"
        )

        # Collision pipeline.
        #
        # * ``contact_matching`` -- enabled so the PhoenX solver's
        #   warm-start gather can find last frame's impulses for
        #   persistent contacts (essential for the tower to settle).
        # * ``broad_phase="sap"`` -- the default ``"nxn"`` tests
        #   all ``O(N^2)`` shape pairs, which on 11k bricks is 60M+
        #   candidates. SAP only emits real overlap candidates,
        #   which combined with the small per-shape ``gap`` above
        #   keeps the candidate count proportional to the actual
        #   number of touching brick pairs.
        # * ``rigid_contact_max`` -- the default heuristic
        #   underestimates the kapla tower's contact pool. Frame 0
        #   sees ~647k contacts; once the tower starts micro-shaking
        #   transient inter-penetrations push that closer to ~750k.
        #   Allocate ~20% more so we stay stable across the whole
        #   settling phase.
        # * ``shape_pairs_max`` -- without this knob the SAP path
        #   sizes its broad-phase / narrow-phase scratch and
        #   contact-sorter buffers off the worst-case ``N*(N-1)/2``
        #   bound (~64M pairs for 11k bricks), which pushes peak
        #   GPU memory past 12 GB. The tower's actual SAP candidate
        #   count is ~800k; budget ~1.9x that for transient
        #   compression so the buffer never overflows during
        #   settling.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=PHOENX_CONTACT_MATCHING,
            broad_phase="sap",
            shape_pairs_max=1_500_000,
            rigid_contact_max=900_000,
        )
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
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(
                    np.float32
                ),
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
        self._shape_body = wp.array(
            shape_body_phoenx, dtype=wp.int32, device=self.device
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
            device=self.device,
        )

        # Camera -- the tower spans roughly 0.7 m horizontal x 0.3 m
        # tall after ``GLOBAL_SCALING``. Pull back a little under 1 m
        # so the whole structure is in frame.
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(1.2, 0.0, 0.4),
            pitch=-12.0,
            yaw=180.0,
        )

        # Picking: every brick gets the same OBB half-extents.
        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        for newton_idx in self._brick_newton_ids:
            half_extents_np[newton_idx + 1] = (hx, hy, hz)
        self._half_extents = wp.array(
            half_extents_np, dtype=wp.vec3f, device=self.device
        )
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

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
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

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
        envelope around the original tower footprint.

        A solver blow-up (NaNs, ejected bodies) scatters bricks well
        beyond the original radius; this catches that without
        over-constraining the exact settled geometry.
        """
        # Original positions span ~35 in the un-scaled USD frame;
        # 4x is a generous envelope after the global scale.
        radius = float(np.linalg.norm(POSITIONS[:, :2], axis=1).max())
        tolerance = 4.0 * radius * GLOBAL_SCALING
        positions = self.bodies.position.numpy()
        for newton_idx in self._brick_newton_ids:
            pos = positions[newton_idx + 1]
            assert np.isfinite(pos).all(), (
                f"body {newton_idx} non-finite position: {pos}"
            )
            r_xy = float(np.hypot(pos[0], pos[1]))
            assert r_xy < tolerance, (
                f"brick {newton_idx} flew outside the tower envelope "
                f"(r_xy={r_xy:.3f}, tol={tolerance:.3f})"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    # Start paused so the initial (potentially inter-penetrating)
    # brick layout is visible before the solver begins resolving.
    # Press SPACE or toggle the viewer's "Pause" checkbox to run.
    viewer._paused = True
    newton.examples.run(example, args)
