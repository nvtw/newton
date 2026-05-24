# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX soft-hexahedron pinned-face demo.

Minimal scene exercising :mod:`constraint_soft_hexahedron`. A single
8-node trilinear hex (one constraint, eight particles) hangs from a
pinned face under gravity. No collisions, no rigid bodies -- the test
case for the hex co-rotational ARAP energy in isolation.

Geometry (rest pose, axis-aligned cube of side ``cube_size`` centered
at the origin at altitude ``base_height``):

    corner 0: (-, -, -)
    corner 1: (+, -, -)
    corner 2: (+, +, -)
    corner 3: (-, +, -)
    corner 4: (-, -, +)   <-- top face (pinned by default)
    corner 5: (+, -, +)   <--
    corner 6: (+, +, +)   <--
    corner 7: (-, +, +)   <--

The default pins the 4 top-face corners and lets the bottom face hang
under gravity. Pure ARAP only penalises strain ``||F - R||``, not
absolute displacement, so single-corner pinning is degenerate (the 5
remaining rigid-motion DOFs accumulate gravity-induced drift each
substep). Face pinning is a well-conditioned demo across the full
``youngs_modulus`` range; it converges to a small elastic stretch
instead of drifting.

Pass ``--pin-mode corner`` to switch to the single-corner pin pattern
for diagnostics; expect dramatic deformation that increases with
``youngs_modulus`` for the reasons above.

Run::

    python -m newton._src.solvers.phoenx.examples.example_soft_hex_pinned
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    soft_tet_lame_from_youngs_poisson,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

#: Standard 8-corner sign triplets matching the constraint's canonical
#: isoparametric ordering. Used to lay out rest particle positions in
#: world space from the cube center + half-extent.
_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=np.float32,
)


def _build_hex_corners(
    center: np.ndarray,
    half_extent: float,
) -> np.ndarray:
    """Return ``[8, 3]`` rest corner positions for an axis-aligned cube."""
    return center[None, :] + half_extent * _CORNER_SIGNS


class Example:
    """Hex constraint pinned-corner scene.

    Args:
        viewer: A Newton :class:`ViewerBase` (typically
            :class:`ViewerNull` for headless / benchmark runs;
            :class:`ViewerGL` for visualization).
        args: ``newton.examples.init`` argument namespace (unused; kept
            for compatibility with the example harness).
        cube_size: Edge length of the rest cube [m].
        base_height: z coordinate of the cube center at rest [m].
        density: Particle density [kg / m^3]. Each corner gets mass =
            ``density * cube_volume / 8``.
        youngs_modulus: Stiffness [Pa]. Drives ``alpha_mu = 1 / k_mu``.
        poisson_ratio: Poisson ratio in ``(-1, 0.5)``. Ignored by the
            pure-ARAP variant beyond converting (E, nu) -> mu.
        beta_mu: Macklin XPBD damping [1/s].
        pin_mode: ``"top_face"`` (default; pins corners 4..7) or
            ``"corner"`` (pins only :attr:`pin_corner_index`). Single-
            corner pinning is under-constrained for pure ARAP and is
            for diagnostics only -- see this module's docstring.
        pin_corner_index: Which corner (0..7) to pin in
            ``pin_mode="corner"``. Ignored otherwise.
    """

    def __init__(
        self,
        viewer,
        args,
        *,
        cube_size: float = 0.5,
        base_height: float = 1.5,
        density: float = 500.0,
        youngs_modulus: float = 5.0e6,
        poisson_ratio: float = 0.3,
        beta_mu: float = 5.0,
        pin_mode: str = "top_face",
        pin_corner_index: int = 0,
    ):
        if pin_mode not in ("top_face", "corner"):
            raise ValueError(f"pin_mode must be 'top_face' or 'corner' (got {pin_mode!r})")
        if not 0 <= int(pin_corner_index) < 8:
            raise ValueError(f"pin_corner_index must be in [0, 8) (got {pin_corner_index})")

        self.viewer = viewer
        self.device = wp.get_device()
        self.cube_size = float(cube_size)
        self.base_height = float(base_height)
        self.density = float(density)
        self.pin_mode = pin_mode
        self.pin_corner_index = int(pin_corner_index)

        # PhoenX schedule. The hex is a single XPBD row per sweep; 5
        # substeps + 8 inner iters is plenty for the 1-element scene.
        self.sim_substeps = 5
        self.solver_iterations = 8
        self.velocity_iterations = 0
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # Rest positions.
        rest_corners = _build_hex_corners(
            center=np.array([0.0, 0.0, self.base_height], dtype=np.float32),
            half_extent=0.5 * self.cube_size,
        )
        # Particle masses: total = density * cube_volume; lump to
        # corners. Pinned corners get inv_mass=0.
        cube_volume = self.cube_size**3
        total_mass = self.density * cube_volume
        corner_mass = total_mass / 8.0
        inv_mass = np.full(8, 1.0 / corner_mass, dtype=np.float32)
        if self.pin_mode == "top_face":
            # Top face corners (ζ=+1): 4, 5, 6, 7.
            self._pinned_indices = np.array([4, 5, 6, 7], dtype=np.int32)
        else:
            self._pinned_indices = np.array([self.pin_corner_index], dtype=np.int32)
        inv_mass[self._pinned_indices] = 0.0
        # Lame parameters from (E, nu).
        self.k_mu, self.k_lambda = soft_tet_lame_from_youngs_poisson(
            youngs_modulus=float(youngs_modulus), poisson_ratio=float(poisson_ratio)
        )

        # PhoenX world: 1 hex, 8 particles, no rigid bodies (a single
        # "world anchor" slot is required by BodyContainer; we never
        # touch it from constraints).
        num_phoenx_bodies = 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        bodies.orientation.assign(
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            )
        )
        self.bodies = bodies

        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            num_soft_hexahedra=1,
            device=self.device,
        )
        self.world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            num_joints=0,
            num_particles=8,
            num_soft_hexahedra=1,
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            rigid_contact_max=0,
            step_layout="single_world",
            mass_splitting=False,
            partitioner_algorithm="greedy",
            device=self.device,
        )
        # Gravity. Anti-z is the convention across the soft-body examples.
        self.world.gravity.assign(np.array([[0.0, 0.0, -9.81]], dtype=np.float32))

        # Populate the hex.
        hex_indices = wp.array(np.arange(8, dtype=np.int32).reshape(1, 8), dtype=wp.int32, device=self.device)
        particle_q = wp.array(rest_corners, dtype=wp.vec3f, device=self.device)
        particle_qd = wp.zeros(8, dtype=wp.vec3f, device=self.device)
        particle_inv_mass = wp.array(inv_mass, dtype=wp.float32, device=self.device)
        hex_materials = wp.array(
            np.array([[self.k_mu, beta_mu]], dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self.world.populate_soft_hexahedra_from_arrays(
            hex_indices=hex_indices,
            particle_q=particle_q,
            hex_materials=hex_materials,
            particle_qd=particle_qd,
            particle_inv_mass=particle_inv_mass,
        )
        self._rest_corners = rest_corners.copy()
        self._pinned_rest = rest_corners[self._pinned_indices].copy()

        # Viewer camera: orbit the cube center from a distance scaled
        # by cube size so the hex stays in frame.
        cam_distance = max(2.0, 4.0 * self.cube_size)
        self.viewer.set_camera(
            pos=wp.vec3(cam_distance, 0.0, self.base_height),
            pitch=-10.0,
            yaw=180.0,
        )

        # Pre-allocate render buffers so :meth:`render` is allocation-free.
        # ``ViewerGL._update_vbo`` expects per-element ``wp.array[vec3]``
        # color buffers; a tuple in ``log_points(colors=...)`` crashes
        # on ``colors.numpy()`` because the GL path doesn't expand scalar
        # colors. The 12 cube edges are stable across runs.
        _edges_np = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # bottom face
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # top face
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # vertical edges
            ],
            dtype=np.int32,
        )
        self._edge_start_ids = _edges_np[:, 0]
        self._edge_end_ids = _edges_np[:, 1]
        # Pinned corners are dimmed so the viewer makes the boundary
        # condition visually obvious.
        point_colors_np = np.tile(np.array([0.8, 0.3, 0.3], dtype=np.float32), (8, 1))
        point_colors_np[self._pinned_indices] = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        self._point_color = wp.array(point_colors_np, dtype=wp.vec3f, device=self.device)
        self._line_color = wp.array(
            np.tile(np.array([0.9, 0.7, 0.2], dtype=np.float32), (_edges_np.shape[0], 1)),
            dtype=wp.vec3f,
            device=self.device,
        )

        self._frame_index = 0
        self._capture()

    def _capture(self) -> None:
        """Capture the per-frame substep + collision (no-op) graph.

        Mirrors ``example_soft_body_drop``'s pattern: one warm-up step
        outside the capture so JIT compile lands before recording, then
        a single :meth:`world.step` call captured for replay.
        """
        if self.device.is_cuda:
            self._simulate_one_frame()  # warm-up
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate_one_frame(self) -> None:
        # No collisions / no rigid sync; the hex constraint runs in
        # isolation.
        self.world.step(self.frame_dt)

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt
        self._frame_index += 1

    def render(self) -> None:
        """Push particle positions into the viewer as a point cloud + an
        edge-line overlay of the 12 cube edges.

        ``ViewerNull`` discards both. ``ViewerGL`` renders the points
        and the wire-frame outline so the hex deformation is visible.
        """
        positions = self.world.particles.position.numpy()
        starts = positions[self._edge_start_ids]
        ends = positions[self._edge_end_ids]
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_points(
            "hex_corners",
            wp.array(positions, dtype=wp.vec3f),
            radii=0.02 * self.cube_size,
            colors=self._point_color,
        )
        self.viewer.log_lines(
            "hex_edges",
            wp.array(starts, dtype=wp.vec3f),
            wp.array(ends, dtype=wp.vec3f),
            colors=self._line_color,
            width=0.01 * self.cube_size,
        )
        self.viewer.end_frame()

    # --- Test hooks (called by Newton's example harness) -------------

    def test_post_step(self) -> None:
        """Per-step invariants: no NaN, pinned corners stay put."""
        positions = self.world.particles.position.numpy()
        if not np.isfinite(positions).all():
            raise AssertionError(f"non-finite particle position at frame {self._frame_index}: {positions}")
        pinned_now = positions[self._pinned_indices]
        delta = float(np.max(np.linalg.norm(pinned_now - self._pinned_rest, axis=1)))
        if delta > 1e-4:
            raise AssertionError(f"pinned corners drifted up to {delta:.6f} m at frame {self._frame_index}")

    def test_final(self) -> None:
        """End-of-run check: positions stay finite and pinned corners
        are unchanged. With the default ``top_face`` pin pattern the
        free bottom-face corners settle into an elastic hang below
        the pinned top face."""
        positions = self.world.particles.position.numpy()
        assert np.isfinite(positions).all(), "non-finite particle positions"
        pinned_now = positions[self._pinned_indices]
        max_pin_drift = float(np.max(np.linalg.norm(pinned_now - self._pinned_rest, axis=1)))
        assert max_pin_drift < 1e-4, f"pinned corner drift {max_pin_drift:.6f} m"
        if self.pin_mode == "top_face":
            # Free bottom-face corners hang below the pinned top face.
            free_z = positions[:4, 2].mean()
            pinned_z = pinned_now[:, 2].mean()
            assert free_z < pinned_z, (
                f"bottom face didn't hang below pinned top (free_z={free_z:.3f}, pinned_z={pinned_z:.3f})"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--cube-size", type=float, default=0.5)
    parser.add_argument("--base-height", type=float, default=1.5)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--youngs-modulus", type=float, default=5.0e6)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--beta-mu", type=float, default=5.0)
    parser.add_argument(
        "--pin-mode",
        choices=("top_face", "corner"),
        default="top_face",
        help=(
            "How to anchor the hex. 'top_face' (default) pins the 4 top-face "
            "corners and lets the bottom face hang under gravity -- a "
            "well-conditioned pure-stretch demo across the full stiffness "
            "range. 'corner' pins only a single corner (chosen with "
            "--pin-corner-index); pure ARAP is under-constrained for this "
            "case and the hex will drift / over-deform even at high stiffness "
            "since rotational DOFs around the pin go un-penalised. Use only "
            "for diagnostics."
        ),
    )
    parser.add_argument(
        "--pin-corner-index",
        type=int,
        default=0,
        help="Corner index (0..7) to pin when --pin-mode=corner.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        args,
        cube_size=args.cube_size,
        base_height=args.base_height,
        density=args.density,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
        beta_mu=args.beta_mu,
        pin_mode=args.pin_mode,
        pin_corner_index=args.pin_corner_index,
    )
    newton.examples.run(example, args)
