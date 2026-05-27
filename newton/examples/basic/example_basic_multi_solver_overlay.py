# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Multi-Solver Overlay
#
# Demonstrates the viewer "layer" system by running three simulations of
# the same quadruple-pendulum chain in a single viewer window, each driven
# by a different solver (XPBD, Featherstone, MuJoCo Warp). By default the
# layers overlay exactly so the per-solver divergence is obvious; bumping
# the ``spacing`` constant lays them out side-by-side along the world
# X-axis via ``viewer.set_layer_transform``. Layers can also be toggled
# independently via the "Layers" group in the viewer sidebar.
#
# Layers are created via ``viewer.activate(layer_id)``. Each layer owns its
# own model, state, and shape batches; the viewer accumulates the render
# objects from all layers and draws them together every frame. Each layer
# also captures its own CUDA graph so per-solver stepping stays fast.
# The per-layer transform applied via ``set_layer_transform`` is a pure
# render-time displacement: it shifts every drawn object in the layer
# without changing the underlying physics state, so passing
# ``wp.transform_identity()`` (or a zero-translation transform) makes the
# layers overlay exactly.
#
# Works with every viewer backend that supports layers: ``ViewerGL`` (the
# default), ``ViewerRTX`` (ray-traced; install with ``pip install -e
# .[rtx]``), and ``ViewerNull`` for headless testing.
#
# Commands:
#   python -m newton.examples basic_multi_solver_overlay
#   python -m newton.examples basic_multi_solver_overlay --viewer rtx
#
###########################################################################

import warnings

import warp as wp

import newton
import newton.examples


def _build_pendulum_model(num_links: int = 4) -> tuple[newton.Model, list[int]]:
    """Build an ``num_links``-link pendulum chain hanging from the world.

    Returns ``(model, link_shape_indices)``. ``link_shape_indices`` lists
    the shape indices for the pendulum links only (excluding the ground
    plane) so the caller can tint just the moving geometry per layer
    while keeping the shared ground plane neutral.
    """
    builder = newton.ModelBuilder()

    hx = 1.0
    hy = 0.1
    hz = 0.1

    # Create one rigid link per chain segment with a single box shape each.
    links: list[int] = []
    link_shapes: list[int] = []
    for _ in range(num_links):
        link = builder.add_link()
        links.append(link)
        link_shapes.append(builder.add_shape_box(link, hx=hx, hy=hy, hz=hz))

    # Anchor joint: rotate around the world's Z-axis so the chain hangs
    # sideways from the viewer's default camera.
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
    joints: list[int] = []
    joints.append(
        builder.add_joint_revolute(
            parent=-1,
            child=links[0],
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
    )
    # Subsequent joints connect each link's +X end to the next link's -X end.
    for i in range(1, num_links):
        joints.append(
            builder.add_joint_revolute(
                parent=links[i - 1],
                child=links[i],
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            )
        )

    builder.add_articulation(joints, label="pendulum")
    builder.add_ground_plane(height=-3.0)
    return builder.finalize(), link_shapes


class _SolverLayer:
    """Bundle of (layer_id, model, solver, states, contacts) tracked together.

    Each layer captures its own CUDA graph (when running on a CUDA device)
    so the per-frame ``simulate`` step is a single graph replay per
    solver. Solvers that don't support capture (some MuJoCo configurations)
    transparently fall back to eager execution while the others keep their
    graphs.
    """

    def __init__(
        self,
        layer_id: str,
        color: tuple[float, float, float],
        solver_factory,
        viewer,
        sim_substeps: int,
        sim_dt: float,
    ):
        self.layer_id = layer_id
        self.color = color
        self.model, link_shape_indices = _build_pendulum_model()
        # Tint only the pendulum link shapes so each layer is visually
        # distinct, leaving the ground plane its default checkerboard.
        if self.model.shape_color is not None:
            color_vec = wp.vec3(*color)
            for s in link_shape_indices:
                self.model.shape_color[s : s + 1].fill_(color_vec)
        self.solver = solver_factory(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()

        self._viewer = viewer
        self._sim_substeps = sim_substeps
        self._sim_dt = sim_dt
        self.graph: wp.Graph | None = self._capture()

    def _simulate(self) -> None:
        for _ in range(self._sim_substeps):
            self.state_0.clear_forces()
            self._viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self._sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self) -> wp.Graph | None:
        if not wp.get_device().is_cuda:
            return None
        try:
            with wp.ScopedCapture() as capture:
                self._simulate()
            return capture.graph
        except Exception as exc:
            warnings.warn(
                f"CUDA graph capture failed for layer '{self.layer_id}': {exc}",
                stacklevel=2,
            )
            return None

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # The RTX backend defaults to a flat dome light and a near-black
        # floor, which makes the pendulum colors hard to read. Swap in the
        # studio lighting rig, a brighter slightly-glossy floor, and a
        # near-white sky/dome color so the visible background reads as
        # overcast-daylight rather than the studio preset's cool blue.
        # No-ops for any other backend.
        rtx_viewer = getattr(newton.viewer, "ViewerRTX", None)
        if rtx_viewer is not None and isinstance(viewer, rtx_viewer):
            # Studio rig + a near-white dome and an anthrazit floor. The
            # dome intensity is a compromise: pushing it higher (~2200)
            # gets the sky to pure white but the strong IBL bounce washes
            # the floor to near-white too; ~1200 keeps the sky clearly
            # bright (~232) while letting the dark albedo read as the
            # intended anthrazit gray (~173 near the camera, fading to a
            # specular sky-reflection ~199 toward the horizon — the
            # "diffuse mirror" look). Rigid-body shapes get a matte
            # plastic BRDF so the link colors don't read as polished.
            viewer.set_environment("studio")
            viewer.set_sky_color(color=(1.0, 1.0, 1.0), intensity=1800.0)
            # Dark, matte checker floor: low base diffuse + high roughness
            # so the bright dome doesn't lay a foggy specular sheen on the
            # floor. ``_generate_floor_textures`` produces a checker with
            # the light squares only ~2x the base luminance so they don't
            # become a secondary sky source under IBL.
            viewer.set_ground_material(diffuse=(0.02, 0.02, 0.025), roughness=0.85)
            viewer.set_body_material(roughness=0.9, metallic=0.0)

        # Three layers driven by different solvers. The colors form a
        # saturated red/blue/green triad spaced ~120° apart on the hue
        # wheel — bold, mutually distinguishable, no pink/magenta in the
        # red (the green channel is held low and blue lower) and no
        # earthy/dirt tones in the green (the red channel is held low so
        # it doesn't drift toward olive). All three are highly chromatic
        # against the anthrazit floor. Layers can be toggled independently
        # via the "Layers" sidebar; each captures its own CUDA graph on
        # construction (where supported).
        specs = [
            ("XPBD", (1.00, 0.22, 0.08), newton.solvers.SolverXPBD),               # vermillion red
            ("Featherstone", (0.05, 0.40, 1.00), newton.solvers.SolverFeatherstone),  # azure blue
            ("MuJoCo Warp", (0.05, 0.82, 0.32), newton.solvers.SolverMuJoCo),         # emerald green
        ]
        self.layers: list[_SolverLayer] = [
            _SolverLayer(name, color, factory, self.viewer, self.sim_substeps, self.sim_dt)
            for name, color, factory in specs
        ]

        # Bind each layer to the viewer. Activating then calling ``set_model``
        # is all the per-solver wiring needed — the viewer auto-prefixes object
        # names so all three pendulums coexist in the same scene.
        # ``set_layer_transform`` positions each layer relative to the others.
        # The default ``spacing = 0`` overlays them so per-solver divergence
        # is obvious; raise ``spacing`` (each link is 2 m long, so ~9 m gives
        # a comfortable gap between neighboring swing planes) to lay them
        # out side-by-side along the world X-axis instead.
        spacing = 0.001
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            self.viewer.activate(layer.layer_id)
            self.viewer.set_model(layer.model)
            shift = (i - 0.5 * (n - 1)) * spacing
            self.viewer.set_layer_transform(
                layer.layer_id,
                wp.transform(wp.vec3(shift, 0.0, 0.0), wp.quat_identity()),
            )

    def step(self):
        for layer in self.layers:
            layer.step()
        self.sim_time += self.frame_dt

    def test_final(self):
        # The anchor is at z=5 and the chain has four 2 m links, so any
        # body must stay within the swept envelope and exactly in the
        # x=0 plane (the joints only rotate about y).
        for layer in self.layers:
            num_links = layer.model.body_count
            newton.examples.test_body_state(
                layer.model,
                layer.state_0,
                f"{layer.layer_id} pendulum links in correct area",
                lambda q, qd: abs(q[0]) < 1e-3 and -4.0 < q[2] < 6.0,
                list(range(num_links)),
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        for layer in self.layers:
            self.viewer.activate(layer.layer_id)
            self.viewer.log_state(layer.state_0)
            self.viewer.log_contacts(layer.contacts, layer.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
