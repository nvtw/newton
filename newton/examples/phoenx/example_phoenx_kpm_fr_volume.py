# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Real-time KPM-FR flow around a sphere with OptiX volume rendering."""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import KPMFR3D, KPMFR3DConfig, rasterize_obstacles


class Example:
    def __init__(self, viewer, options):
        from newton._src.solvers.phoenx.examples.example_fluid_kpm_fr import (  # noqa: PLC0415
            bind_volume,
            initialize_flow,
            make_volume,
            step_flow,
            update_volume,
        )

        self.viewer = viewer
        self.sim_time = 0.0
        self.coherent = int(options.wake_style == "coherent")
        self._step_flow = step_flow
        self._update_volume = update_volume

        elements = 8 if options.test else options.elements
        order = 3 if options.test else options.order
        resolution = (elements, elements // 2, elements // 2)
        self.solver = KPMFR3D(
            KPMFR3DConfig(
                resolution,
                size=(6.0, 3.0, 3.0),
                order=order,
                reference_velocity=0.28,
                reynolds=5_000.0 if self.coherent else 100_000.0,
                cfl=0.45,
            ),
            device=wp.get_device(),
        )
        self.points = initialize_flow(self.solver)

        builder = newton.ModelBuilder()
        builder.add_shape_sphere(body=-1, radius=0.34, color=(0.50, 0.54, 0.60))
        builder.add_ground_plane(height=-0.70, color=(0.45, 0.45, 0.45))
        self.model = builder.finalize(device=self.solver.device)
        self.state = self.model.state()
        rasterize_obstacles(
            self.solver,
            self.model,
            self.state,
            origin=tuple(-0.5 * np.asarray(self.solver.config.size)),
        )

        warmup_steps = 2 if options.test else options.warmup_steps
        for _ in range(warmup_steps):
            self.sim_time = self._step_flow(self.solver, self.points, self.sim_time, self.coherent)

        self.volume_data = None
        if hasattr(viewer, "set_volume") and not options.test:
            world_transform = np.array(
                ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                dtype=np.float32,
            )
            samples = options.volume_samples_per_cell or order
            self.volume_data = make_volume(self.solver, samples, world_transform)

        viewer.set_model(self.model)
        if hasattr(viewer, "camera"):
            viewer.set_camera(pos=wp.vec3(0.75, -3.65, 0.30), pitch=-4.7, yaw=90.0)
        if self.volume_data is not None:
            bounds = (np.array((-0.80, -0.95, -0.95)), np.array((2.98, 0.95, 0.95)))
            matrix = world_transform[:3, :3]
            corners = (
                np.array(np.meshgrid(*zip(*bounds, strict=True), indexing="ij"), dtype=np.float32).reshape(3, -1).T
            )
            corners = corners @ matrix.T
            bind_volume(viewer, self.volume_data, bounds=(corners.min(0), corners.max(0)))

    def step(self):
        self.sim_time = self._step_flow(self.solver, self.points, self.sim_time, self.coherent)
        if self.volume_data is not None:
            self._update_volume(self.solver, self.points, self.volume_data)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self):
        state = self.solver.state.numpy()
        obstacle = self.solver.volume_fraction.numpy()
        if not np.isfinite(state).all():
            raise ValueError("KPM-FR state contains non-finite values")
        if not np.any(obstacle > 0.5):
            raise ValueError("sphere and ground were not rasterized into the fluid")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--elements", type=int, default=64)
        parser.add_argument("--order", type=int, choices=range(3, 7), default=6)
        parser.add_argument("--warmup-steps", type=int, default=2226)
        parser.add_argument("--volume-samples-per-cell", type=int, default=0)
        parser.add_argument("--wake-style", choices=("turbulent", "coherent"), default="turbulent")
        parser.set_defaults(viewer="optix", optix_dlss_quality="quality", optix_max_bounces=2)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
