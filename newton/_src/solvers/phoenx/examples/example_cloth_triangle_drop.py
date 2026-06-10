# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cloth-vs-rigid contact demo for SolverPhoenX.
#
# Scene: a square cloth grid drops onto a static box. SolverPhoenX builds
# cloth triangle and bending constraints directly from the finalized Model,
# and wires a deformable-aware collision pipeline behind model.collide().
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
        q = self.state_0.particle_q.numpy()
        assert float(q[:, 2].min()) > -0.1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
