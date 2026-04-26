# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Tower of Jitter
#
# First example for :mod:`solver_phoenx`. Replicates PhoenX's
# ``Demo02: Tower of Jitter`` (see
# ``PhoenX/src/Viewer/Demos/Demo02.cs`` and ``Common.BuildTower`` in
# ``PhoenX/src/Viewer/Demos/Common.cs``): a 40-layer circular stack of
# 32 thin planks per layer, each layer rotated half a per-plank angle
# relative to the one below so the seams alternate.
#
# Uses :class:`PhoenXWorld` for the solve and Newton's
# :class:`CollisionPipeline` for contact detection -- the PhoenX solver
# itself doesn't run broad / narrow phases.
#
# PhoenX uses +Y-up; Newton uses +Z-up, so the tower's height axis is
# swapped from the C# source: Jitter Y -> Newton Z, and the plank's
# thin (radial) dimension lands on Newton's Y instead of Z.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_tower
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import (
    body_container_zeros,
)
from newton._src.solvers.phoenx.picking import (
    Picking,
    register_with_viewer_gl,
)
from newton._src.solvers.phoenx.solver_config import (
    PHOENX_CONTACT_MATCHING,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

# ---- Step layout toggle ----
# Flip this to switch between the two PhoenX dispatch strategies:
#   ``False`` (default) -> ``"multi_world"``: per-world fast-tail
#       kernels (one warp per world). Best when you replicate this
#       tower across many sub-worlds.
#   ``True``           -> ``"single_world"``: per-colour grid
#       launches via ``wp.capture_while``. Wins on one big world
#       with thousands of contacts -- which is exactly what this
#       40-layer tower is.
USE_BIG_WORLD_MODE: bool = True
STEP_LAYOUT: str = "single_world" if USE_BIG_WORLD_MODE else "multi_world"

# ---- Tower geometry (matches ``Common.BuildTower``) ----
# The C# source scales every dimension by ``scaling = 0.1 / 3``. Here
# we unscale to Newton's natural SI units (PhoenX's world happens to
# be metric too -- the scaling in C# just produces small numbers) so
# the solver sees an ordinary-sized tower.
#
# C# plank: ``BoxShapeS(scaling * vec3(3, 1, 0.2), blockMass)`` -- the
# constructor takes *full* extents, and the vertical dimension is the
# middle component (Jitter is +Y-up). On Newton +Z-up the middle
# component becomes the long radial length Z, and the third (Jitter's
# radial thickness) becomes Newton's Y.
#
# We keep the C# shape ratio but drop the explicit scaling: Newton's
# half-extents are already in meters, and the resulting tower is about
# 20 m tall and 19.5 m in radius.
TOWER_HEIGHT_LAYERS = 40
BOXES_PER_RING = 32
# Half-extents, +Z-up remap of the C# (3, 1, 0.2) full extents.
PLANK_HX = 1.5  # tangential (wraps the ring)
PLANK_HY = 0.1  # radial wall thickness (was Jitter Z)
PLANK_HZ = 0.5  # vertical height (was Jitter Y)
# Ring radius matches ``19.5f`` from the C# local offset.
RING_RADIUS = 19.5
# ``2 * PI / 64`` per C#. The full per-plank step is twice this.
HALF_ROTATION_STEP = 2.0 * math.pi / 64.0
FULL_ROTATION_STEP = 2.0 * HALF_ROTATION_STEP

# Plank density -- the C# ``blockMass = 1.0f`` at plank volume
# ``scaling^3 * 0.6 ≈ 2.2e-5`` implies a density of ~45 000 kg/m^3.
# Newton uses +Z-up with dimensions in meters and no scaling, so a
# plank's volume is 0.6 m^3; we go with the common-rock density
# 1000 kg/m^3 which gives a 600 kg plank. The absolute mass only
# sets time scale; what matters for stability is uniform density
# across all planks.
PLANK_DENSITY = 1000.0


# State-mirroring kernels shared with other PhoenX examples.
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    newton_to_phoenx_kernel as _newton_to_phoenx_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    phoenx_to_newton_kernel as _phoenx_to_newton_kernel,
)


class Example:
    """PhoenX Tower of Jitter -- first example for :mod:`solver_phoenx`.

    Pipeline per frame:
        1. Sync Newton state -> PhoenX body container.
        2. Run Newton's :class:`CollisionPipeline` to produce the
           ``Contacts`` buffer.
        3. Call :meth:`PhoenXWorld.step` to advance the solver.
        4. Sync PhoenX body container -> Newton state (for rendering /
           downstream checks).

    The contacts-only solver never runs its own joint dispatcher, so
    this example wires zero joints -- every body is a free-falling
    plank that only interacts through contact with its neighbours and
    the ground.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # Frame pacing. The C# demo uses 120 Hz with 20 substeps; we
        # keep the same substep_dt (1/2400 s) but render at 60 Hz so
        # the viewer runs at the usual cadence.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 20
        self.solver_iterations = 3

        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        """Build the Newton model, the PhoenX body/constraint containers,
        and the :class:`PhoenXWorld` driver.

        ``ModelBuilder`` populates the Newton side (ground plane + the
        tower of planks); :func:`_init_phoenx_bodies_kernel` then
        mirrors the resulting body state into a fresh PhoenX
        :class:`BodyContainer`. A contacts-only
        :class:`ConstraintContainer` of width :data:`CONTACT_DWORDS`
        rounds out the solver state.
        """
        builder = newton.ModelBuilder()

        # Ground plane -- ingests as a Newton ``plane`` shape attached
        # to the world body (shape_body == -1).
        builder.add_ground_plane()

        # Tower of planks. Orientation accumulates a per-plank rotation
        # step about +Z, plus a half-step between rings (matches the
        # C# ``orientation *= halfRotationStep`` inside the layer loop).
        self._plank_newton_ids: list[int] = []
        orientation_rad = 0.0
        for e in range(TOWER_HEIGHT_LAYERS):
            orientation_rad += HALF_ROTATION_STEP
            for _ in range(BOXES_PER_RING):
                cos_o = math.cos(orientation_rad)
                sin_o = math.sin(orientation_rad)
                # C# local offset: (0, 0.5 + e, RING_RADIUS) with Y up.
                # On +Z-up: Y_Jitter (height) -> Z_Newton,
                # Z_Jitter (radius) -> Y_Newton.
                local_x = 0.0
                local_y = RING_RADIUS
                local_z = 0.5 + e
                world_x = cos_o * local_x - sin_o * local_y
                world_y = sin_o * local_x + cos_o * local_y
                world_z = local_z
                quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), orientation_rad)
                body = builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(float(world_x), float(world_y), float(world_z)),
                        q=quat,
                    ),
                )
                # Let the shape's density compute both the mass and
                # the inertia tensor. Passing ``mass=...`` on
                # ``add_body`` *without* overriding shape density
                # produces a consistent (mass, inertia) pair; passing
                # ``density=0`` on the ShapeConfig zeros the inertia
                # contribution and Newton's inertia validator falls
                # back to a default that does not match the shape,
                # which is how the tower used to sink through the
                # ground plane (contacts fire but the normal row
                # can't hold the weight).
                builder.add_shape_box(
                    body,
                    hx=PLANK_HX,
                    hy=PLANK_HY,
                    hz=PLANK_HZ,
                    cfg=newton.ModelBuilder.ShapeConfig(density=PLANK_DENSITY),
                )
                self._plank_newton_ids.append(body)
                orientation_rad += FULL_ROTATION_STEP

        # Finalise the Newton side.
        # ``skip_shape_contact_pairs=True``: the precomputed shape-pair
        # list is only used by the ``"explicit"`` broad phase. Building
        # it is an O(N^2) Python loop, which is fine for the default
        # tower (~1280 planks) but explodes once it's tiled into a
        # grid; safe under SAP/NXN.
        self.model = builder.finalize(skip_shape_contact_pairs=True)
        print(f"[PhoenX Tower] bodies={self.model.body_count} shapes={self.model.shape_count}")

        # Collision pipeline -- contact matching must be enabled for
        # the PhoenX solver's warm-start gather to find last frame's
        # impulses for persistent contacts.
        # SAP broad phase: the all-pairs (NXN) default allocates an
        # ``N*(N-1)/2`` candidate-pair buffer per world, which is many
        # GB once the tower is replicated into a grid. SAP reports only
        # actual overlapping AABBs so ``shape_pairs_max`` shrinks to a
        # small constant.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching=PHOENX_CONTACT_MATCHING, broad_phase="sap"
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        # Newton state; seeded via FK from the queued poses.
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        # Mirror the FK-derived poses back into ``model.body_q`` so
        # the :func:`_init_phoenx_bodies_kernel` seed reads the
        # final transforms.
        self.model.body_q.assign(self.state.body_q)

        # Build the PhoenX body container. Slot 0 stays as the static
        # world anchor (zero mass / inertia, MOTION_STATIC by default);
        # Newton bodies occupy slots ``[1, body_count + 1)``.
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        # Seed every slot's orientation to identity. ``body_container_zeros``
        # leaves it at ``(0, 0, 0, 0)`` (a non-unit quaternion that the
        # rotation-to-matrix call in ``_update_inertia`` would blow up on).
        # Slots ``[1, body_count + 1)`` are immediately overwritten below
        # from ``model.body_q``; slot 0 (static world anchor) keeps the
        # identity we just wrote.
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

        # Joint-only :class:`ConstraintContainer` (the contact-column
        # storage moved to :class:`ContactColumnContainer` long ago,
        # so the constraint container only needs a 1-row placeholder
        # in this contact-only scene).
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            device=self.device,
        )

        # Shape -> PhoenX body map. Newton's ``shape_body`` stores
        # Newton body indices (or -1 for the world anchor); shift by
        # +1 to match PhoenX's slot-0-is-world layout.
        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        # Build the solver.
        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=0,  # PhoenX's default
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=rigid_contact_max,
            step_layout=STEP_LAYOUT,
            device=self.device,
        )

        # Viewer set-up: single fixed camera outside the tower aiming
        # roughly at mid-height.
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(55.0, 0.0, 22.0),
            pitch=-10.0,
            yaw=180.0,
        )

        # Picking: register one half-extent triple per PhoenX body.
        # Slot 0 is the static world anchor -- leave it at zero so
        # right-click rays skip it. Every plank gets the same
        # ``(PLANK_HX, PLANK_HY, PLANK_HZ)`` half-extents (picking
        # does body-local OBB raycast, so the per-body orientation
        # already accounts for each plank's ring-tangent heading).
        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        for newton_idx in self._plank_newton_ids:
            half_extents_np[newton_idx + 1] = (PLANK_HX, PLANK_HY, PLANK_HZ)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # CUDA graph capture for the per-frame step pipeline. Falls
        # back to direct :meth:`simulate` on CPU or when capture is
        # disabled.
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
        # Picking PD force is accumulated into ``bodies.force`` once
        # per frame, before :meth:`PhoenXWorld.step` -- the solver's
        # per-substep ``_phoenx_apply_external_forces_kernel`` picks
        # it up each substep, and the tail ``_clear_forces`` zeroes
        # it for the next frame. Calling this unconditionally keeps
        # the simulate graph capture invariant (the kernel gates on
        # ``pick_body[0] < 0`` internally).
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
        """After settling, every plank must still lie within a
        generous envelope around the original tower footprint.

        A solver blow-up (NaNs, ejected bodies) scatters planks well
        beyond ``RING_RADIUS * 3``, which this check catches without
        over-constraining the exact settled geometry.
        """
        tolerance = RING_RADIUS * 3.0
        positions = self.bodies.position.numpy()
        for newton_idx in self._plank_newton_ids:
            # PhoenX slot = Newton index + 1 (world body at slot 0).
            pos = positions[newton_idx + 1]
            assert np.isfinite(pos).all(), f"body {newton_idx} non-finite position"
            r_xy = float(math.hypot(pos[0], pos[1]))
            assert r_xy < tolerance, (
                f"plank {newton_idx} flew outside the tower envelope (r_xy={r_xy:.2f}, tol={tolerance:.2f})"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
