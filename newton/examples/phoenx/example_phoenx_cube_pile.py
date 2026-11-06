# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pour cubes onto a pile with runtime body creation and removal in PhoenX.

Command: python -m newton.examples phoenx_cube_pile

A spawner grid above the ground creates about 1,000 cubes per simulated second from
device code. A spot only spawns once its previous cube has fallen clear, so new cubes
never overlap. Once 10,000 cubes are alive, every new cube removes the oldest one, which
keeps the pile churning at a steady size. Spawning, removal, collision, and stepping all
run inside one CUDA graph without host synchronization.

Success criteria: in test mode, cube states stay finite, no cube sinks through the ground,
freshly spawned cubes never overlap, and the live count tracks the spawn rate up to the
10,000-cube cap.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverPhoenX, phoenx

HALF = 0.05  # cube half-extent [m]
SPAWN_HEIGHT = 3.5  # [m]
SPAWN_SPEED = 2.0  # initial downward speed [m/s]
SPOT_SPACING = 0.25  # [m], more than a cube's diagonal (0.173 m)
SPAWN_RADIUS = 1.6  # [m]
CLEARANCE = 0.25  # a spot spawns again once its last cube is this far below [m]
# Saturated display colors, like the OptiX viewer's default palette for model shapes.
PALETTE = (
    (0.86, 0.06, 0.02),
    (0.94, 0.42, 0.01),
    (0.78, 0.72, 0.01),
    (0.46, 0.72, 0.01),
    (0.03, 0.60, 0.12),
    (0.00, 0.58, 0.38),
    (0.00, 0.62, 0.68),
    (0.00, 0.53, 0.88),
    (0.06, 0.30, 0.88),
    (0.22, 0.12, 0.78),
    (0.42, 0.10, 0.76),
    (0.58, 0.08, 0.68),
)


@wp.kernel
def _spawn_kernel(
    dyn: phoenx.DynamicsData,
    body_q: wp.array[wp.transform],
    spots: wp.array[wp.vec3],
    spot_body: wp.array[wp.int32],
    budget: int,
    counters: wp.array[wp.int32],
    ring: wp.array[wp.int32],
):
    """Spawn up to ``budget`` cubes this frame at spots whose last cube has fallen clear.

    ``counters`` holds ``[frame, spawned this frame, total spawned, total removed]``.
    """
    tid = wp.tid()
    n = spots.shape[0]
    frame = counters[0]
    spot = (tid + frame * 37) % n  # rotate which spots claim the budget first
    last = spot_body[spot]
    if last >= 0 and phoenx.is_body_active(dyn, last):
        if wp.transform_get_translation(body_q[last])[2] > SPAWN_HEIGHT - CLEARANCE:
            return
    if wp.atomic_add(counters, 1, 1) >= budget:
        return
    rng = wp.rand_init(1234, frame * n + spot)
    jitter = wp.vec3(wp.randf(rng, -0.02, 0.02), wp.randf(rng, -0.02, 0.02), 0.0)
    axis = wp.normalize(wp.vec3(wp.randf(rng, -1.0, 1.0), wp.randf(rng, -1.0, 1.0), wp.randf(rng, -1.0, 1.0)))
    body_desc = phoenx.BodyDesc()
    body_desc.pose = wp.transform(spots[spot] + jitter, wp.quat_from_axis_angle(axis, wp.randf(rng, 0.0, 3.14)))
    body_desc.velocity = wp.spatial_vector(0.0, 0.0, -SPAWN_SPEED, 0.0, 0.0, 0.0)
    body_desc.mass = 1.0
    moment = 2.0 / 3.0 * HALF * HALF
    body_desc.inertia = wp.mat33(moment, 0.0, 0.0, 0.0, moment, 0.0, 0.0, 0.0, moment)
    body = phoenx.add_body(dyn, body_desc)
    if body < 0:
        return
    shape = phoenx.ShapeDesc()
    shape.type = int(newton.GeoType.BOX)
    shape.transform = wp.transform_identity()
    shape.scale = wp.vec3(HALF)
    shape.gap = 0.01
    shape.mu = 0.6
    shape.collision_group = 1
    shape.flags = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape.prototype = -1
    phoenx.add_shape(dyn, body, shape)
    spot_body[spot] = body
    ring[wp.atomic_add(counters, 2, 1) % ring.shape[0]] = body


@wp.kernel
def _retire_kernel(dyn: phoenx.DynamicsData, target: int, counters: wp.array[wp.int32], ring: wp.array[wp.int32]):
    """Remove the oldest cubes beyond ``target`` and advance the frame."""
    while counters[2] - counters[3] > target:
        phoenx.remove_body(dyn, ring[counters[3] % ring.shape[0]])
        counters[3] = counters[3] + 1
    counters[0] = counters[0] + 1
    counters[1] = 0


@wp.kernel
def _cube_xforms_kernel(
    dyn: phoenx.DynamicsData, body_q: wp.array[wp.transform], body_start: int, xforms: wp.array[wp.transform]
):
    """Instance transforms of every reserved cube; unused rows are parked below the ground."""
    slot = wp.tid()
    body = body_start + slot
    if phoenx.is_body_active(dyn, body):
        xforms[slot] = body_q[body]
    else:
        xforms[slot] = wp.transform(wp.vec3(0.0, 0.0, -1000.0), wp.quat_identity())


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.target = args.max_cubes
        self.budget = max(1, round(args.spawn_rate * self.frame_dt))
        # Live cubes plus one frame of pending and one of quarantined removals.
        capacity = self.target + 4 * self.budget

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.reserve_dynamic_capacity(bodies=capacity, shapes=capacity)
        self.model = builder.finalize()
        self.body_start = self.model._dynamic_capacity.body_start
        self.solver = SolverPhoenX(
            self.model,
            step_layout="single_world",
            substeps=args.substeps,
            solver_iterations=args.iterations,
        )
        self.dyn = self.solver.dynamics.data
        self.pipeline = self.model._collision_pipeline
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()

        coords = np.arange(-SPAWN_RADIUS, SPAWN_RADIUS + 1.0e-6, SPOT_SPACING)
        spots = [(x, y, SPAWN_HEIGHT) for x in coords for y in coords if x * x + y * y <= SPAWN_RADIUS**2]
        self.spots = wp.array(spots, dtype=wp.vec3)
        self.spot_body = wp.full(len(spots), -1, dtype=wp.int32)
        self.counters = wp.zeros(4, dtype=wp.int32)
        self.ring = wp.zeros(capacity, dtype=wp.int32)
        self.xforms = wp.zeros(capacity, dtype=wp.transform)
        rng = np.random.default_rng(7)
        self.colors = wp.array(np.asarray(PALETTE)[rng.integers(0, len(PALETTE), capacity)], dtype=wp.vec3)

        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(6.5, -6.5, 3.5), pitch=-20.0, yaw=135.0)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.launch(
            _spawn_kernel,
            dim=self.spots.shape[0],
            inputs=[self.dyn, self.state_0.body_q, self.spots, self.spot_body, self.budget, self.counters, self.ring],
        )
        wp.launch(_retire_kernel, dim=1, inputs=[self.dyn, self.target, self.counters, self.ring])
        self.pipeline.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
        # Copy back instead of swapping so the captured graph always reads state_0.
        wp.copy(self.state_0.body_q, self.state_1.body_q)
        wp.copy(self.state_0.body_qd, self.state_1.body_qd)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def _update_xforms(self):
        wp.launch(
            _cube_xforms_kernel,
            dim=self.xforms.shape[0],
            inputs=[self.dyn, self.state_0.body_q, self.body_start, self.xforms],
        )

    def render(self):
        self._update_xforms()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_shapes("/cubes", newton.GeoType.BOX, (HALF, HALF, HALF), self.xforms, self.colors)
        self.viewer.end_frame()

    def live_cubes(self) -> int:
        counters = self.counters.numpy()
        return int(counters[2] - counters[3])

    def test_post_step(self):
        """Every half second, check finite state, ground support, and non-overlapping spawns."""
        if round(self.sim_time / self.frame_dt) % 30 != 0:
            return
        self._update_xforms()
        xforms = self.xforms.numpy()
        positions = xforms[xforms[:, 2] > -100.0, :3]
        assert np.isfinite(positions).all(), "nonfinite cube pose"
        assert positions[:, 2].min() > HALF - 0.03, f"cube sank through the ground: {positions[:, 2].min():.4f} m"
        fresh = positions[positions[:, 2] > SPAWN_HEIGHT - 0.5]
        if len(fresh) > 1:
            gaps = np.linalg.norm(fresh[:, None, :] - fresh[None, :, :], axis=-1) + np.eye(len(fresh)) * 1.0e3
            assert gaps.min() > 2.0 * np.sqrt(3.0) * HALF, f"spawned cubes overlap: {gaps.min():.4f} m"

    def test_final(self):
        """The live count follows the spawn rate up to the cap."""
        frames = round(self.sim_time / self.frame_dt)
        expected = min(self.target, frames * self.budget)
        live = self.live_cubes()
        print(f"PhoenX cube pile: {live} live cubes after {frames} frames")
        assert live >= 0.9 * expected, f"spawning fell behind: {live} of {expected} cubes"
        assert live <= self.target
        assert int(self.solver.dynamics.error_count.numpy()[0]) == 0

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # Path-traced rendering, like the Kapla tower, which also suits video recording.
        # Reserved shape rows count as viewer instances too.
        parser.set_defaults(viewer="optix", optix_max_instances=24576)
        parser.add_argument("--max-cubes", type=int, default=10000, help="Live cubes kept once the pile is full.")
        parser.add_argument("--spawn-rate", type=float, default=1000.0, help="Cubes spawned per simulated second.")
        parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per frame.")
        parser.add_argument("--iterations", type=int, default=4, help="Solver iterations per substep.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
