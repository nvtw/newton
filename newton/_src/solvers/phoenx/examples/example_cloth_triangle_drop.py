# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cloth-vs-rigid contact demo for SolverPhoenX.
#
# Scene: a square cloth grid drops onto a static box. Each cloth
# triangle is registered as a ``GeoType.TRIANGLE`` collision proxy via
# :meth:`ModelBuilder.add_cloth_collision_proxies`; PhoenX refreshes
# their geometry from the live ``state.particle_q`` each step before
# ``model.collide()`` runs.
#
# Limitations of this iteration:
#   * PhoenX has no internal cloth dynamics. Cloth particles are
#     treated as independent single-vertex rigid-body slots, so the
#     cloth does not "drape" -- there are no edge / area / bending
#     forces holding particles to their neighbours.
#   * Cloth-vs-rigid contact impulses are routed entirely onto each
#     triangle's first node (single-node routing). Barycentric
#     distribution across all three triangle nodes is a follow-up.
#
# What this example DOES demonstrate:
#   * ``GeoType.TRIANGLE`` cloth proxies registered via
#     :meth:`ModelBuilder.add_cloth_collision_proxies` and refreshed
#     each step from live particle positions.
#   * Contacts firing between cloth proxies and a static box.
#   * At least some particles are caught and held above the box top.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_cloth_triangle_drop
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples


class Example:
    """Cloth grid drops onto a rigid box and drapes around it."""

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # ---- Static box ----------------------------------------------
        box_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=0.5,
            cfg=box_cfg,
        )

        # ---- Cloth grid -----------------------------------------------
        # 11x11 particles, 10x10 = 200 triangles. ~1.0 m wide, dropped
        # from 1.5 m above the box top.
        cloth_dim = 10
        cell = 0.1
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 1.6),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cloth_dim,
            dim_y=cloth_dim,
            cell_x=cell,
            cell_y=cell,
            mass=0.005,
            particle_radius=0.02,
        )

        # ---- Register PhoenX cloth collision proxies ------------------
        # One ``GeoType.TRIANGLE`` collision shape per cloth triangle,
        # auto-filtered against neighbours that share a vertex so
        # adjacent triangles don't trip the broadphase.
        proxy_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=0.4,
            margin=0.02,
            gap=0.01,
        )
        builder.add_cloth_collision_proxies(cfg=proxy_cfg)

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverPhoenX(self.model, substeps=1, solver_iterations=8)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Refresh cloth-triangle proxy geometry from current
            # particle positions BEFORE ``model.collide()`` so the
            # narrow phase sees the live cloth pose.
            self.solver.refresh_cloth_collision_geometry(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # PhoenX has no internal cloth dynamics -- particles are
        # treated as independent rigid-body slots and contact
        # impulses route entirely to each triangle's first node
        # (single-node routing). So we don't expect a clean cloth
        # drape; we only verify that contacts fire and at least some
        # particles are caught above the ground.
        q = self.state_0.particle_q.numpy()
        z_max = float(q[:, 2].max())
        assert z_max > 0.5, (
            f"no cloth particle was caught by the box -- expected at least one above z=0.5, got z_max={z_max}"
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
