# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Break and rebuild a hinged plank bridge with runtime joints in PhoenX.

Command: python -m newton.examples phoenx_bridge

A bridge of hinged planks hangs across a gorge. Boulders and crates rain onto it until a
hinge's reaction force exceeds its strength; that hinge is removed, the deck tears, and
loose planks and loads that fall into the gorge despawn. A few seconds after the first
break the remaining planks are cleared and the whole bridge is rebuilt. Bodies, shapes,
and hinges are created and removed by device kernels, and breaking reads each hinge's
reaction force, all inside one CUDA graph.

Success criteria: in test mode, body states stay finite, intact hinges stay closed, no
runtime request fails, and the bridge breaks and is rebuilt at least once.
"""

import itertools

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.dynamic import (
    BodyDesc,
    DynamicsData,
    ShapeDesc,
    joint_desc,
    phoenx_add_body,
    phoenx_add_joint,
    phoenx_add_shape,
    phoenx_body_is_active,
    phoenx_joint_wrench,
    phoenx_remove_body,
    phoenx_remove_joint,
)
from newton.solvers import SolverPhoenX

GAP = 14.0  # gorge width [m]
PLANKS = 32
PITCH = 0.5  # hinge spacing along the deck [m]
PLANK_HX, PLANK_HY, PLANK_HZ = 0.5 * PITCH - 0.04, 1.0, 0.05  # half extents, shortened so neighbors never touch [m]
PLANK_MASS = 200.0  # [kg]
LOADS = 48  # most boulders and crates alive at once
SPAWN_EVERY = 36  # frames between loads
DROP_HEIGHT = 7.0  # [m]
KILL_DEPTH = -8.0  # bodies below this height despawn [m]
REBUILD_DELAY = 300  # frames from the first break to clearing the deck
CLEAR_FRAMES = 90  # frames the gorge stays empty before the rebuild
SMOOTHING = 0.05  # per-frame weight of the newest hinge force (about 20 frames of memory)
_RESERVED = PLANKS + LOADS

# Bridge phases.
BUILD, INTACT, FAILING, CLEARING = 0, 1, 2, 3
_BUILD = wp.constant(BUILD)
_INTACT = wp.constant(INTACT)
_FAILING = wp.constant(FAILING)
_CLEARING = wp.constant(CLEARING)


def deck_points(gap: float = GAP, planks: int = PLANKS, pitch: float = PITCH) -> np.ndarray:
    """Hinge points of a hanging deck: a parabola of the given chord spacing between the cliff edges."""

    def march(sag):
        z = lambda x: 4.0 * sag * ((x / gap) ** 2 - 0.25)  # noqa: E731
        points = [(-0.5 * gap, 0.0)]
        for _ in range(planks):
            x0, z0 = points[-1]
            lo, hi = x0, x0 + pitch
            for _ in range(60):  # next point exactly one pitch away along the curve
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if np.hypot(mid - x0, z(mid) - z0) < pitch else (lo, mid)
            points.append((lo, z(lo)))
        return np.asarray(points)

    lo, hi = 0.0, 0.5 * planks * pitch
    for _ in range(60):  # the sag at which the deck ends exactly on the far edge
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if march(mid)[-1, 0] > 0.5 * gap else (lo, mid)
    points = march(lo)
    return np.stack([points[:, 0], np.zeros(len(points)), points[:, 1]], axis=1)


@wp.kernel
def _director_kernel(
    dyn: DynamicsData,
    state: wp.array[wp.int32],
    planks: wp.array[wp.int32],
    hinges: wp.array[wp.int32],
    loads: wp.array[wp.int32],
    load_size: wp.array[wp.vec3],
):
    """Advance the bridge phase and drop the next load. ``state`` is ``[phase, phase frame, frame, breaks, rebuilds]``."""
    frame = state[2] + 1
    state[2] = frame
    phase = state[0]
    if phase == _FAILING and frame - state[1] > REBUILD_DELAY:
        for i in range(PLANKS):
            if planks[i] >= 0:
                phoenx_remove_body(dyn, planks[i])  # also removes its hinges
                planks[i] = -1
        for k in range(PLANKS + 1):
            hinges[k] = -1
        state[0] = _CLEARING
        state[1] = frame
    elif phase == _CLEARING and frame - state[1] > CLEAR_FRAMES:
        state[0] = _BUILD
        state[4] = state[4] + 1
    if (phase == _INTACT or phase == _FAILING) and frame % SPAWN_EVERY == 0:
        slot = (frame // SPAWN_EVERY) % LOADS
        if loads[slot] >= 0 and phoenx_body_is_active(dyn, loads[slot]):
            return  # the oldest load is still around
        rng = wp.rand_init(2024, frame)
        boulder = wp.randf(rng) < 0.6
        size = wp.vec3(0.0)
        desc = BodyDesc()
        desc.pose = wp.transform(
            wp.vec3(wp.randf(rng, -0.4 * GAP, 0.4 * GAP), wp.randf(rng, -0.5, 0.5), DROP_HEIGHT),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.randf(rng, 0.0, 3.14)),
        )
        desc.velocity = wp.spatial_vector(0.0, 0.0, -2.0, 0.0, 0.0, 0.0)
        shape = ShapeDesc()
        shape.transform = wp.transform_identity()
        shape.gap = 0.05
        shape.mu = 0.7
        shape.collision_group = 1  # loads collide with each other and with the planks
        shape.flags = int(newton.ShapeFlags.COLLIDE_SHAPES)
        shape.prototype = -1
        if boulder:
            r = wp.randf(rng, 0.3, 0.55)
            size = wp.vec3(r)
            desc.mass = 1500.0 * 4.0 / 3.0 * wp.pi * r * r * r
            desc.inertia = wp.identity(3, dtype=float) * (0.4 * desc.mass * r * r)
            shape.type = int(newton.GeoType.SPHERE)
        else:
            h = wp.randf(rng, 0.3, 0.45)
            size = wp.vec3(h)
            desc.mass = 600.0 * 8.0 * h * h * h
            desc.inertia = wp.identity(3, dtype=float) * (2.0 / 3.0 * desc.mass * h * h)
            shape.type = int(newton.GeoType.BOX)
        shape.scale = size
        body = phoenx_add_body(dyn, desc)
        if body >= 0:
            phoenx_add_shape(dyn, body, shape)
        loads[slot] = body
        load_size[slot] = wp.where(boulder, size, -size)  # negative sizes mark crates


@wp.kernel
def _build_planks_kernel(
    dyn: DynamicsData, state: wp.array[wp.int32], poses: wp.array[wp.transform], planks: wp.array[wp.int32]
):
    i = wp.tid()
    if state[0] != _BUILD:
        return
    desc = BodyDesc()
    desc.pose = poses[i]
    desc.mass = PLANK_MASS
    hx, hy, hz = PLANK_HX, PLANK_HY, PLANK_HZ
    desc.inertia = wp.mat33(hy * hy + hz * hz, 0.0, 0.0, 0.0, hx * hx + hz * hz, 0.0, 0.0, 0.0, hx * hx + hy * hy)
    desc.inertia = desc.inertia * (PLANK_MASS / 3.0)
    body = phoenx_add_body(dyn, desc)
    planks[i] = body
    shape = ShapeDesc()
    shape.type = int(newton.GeoType.BOX)
    shape.transform = wp.transform_identity()
    shape.scale = wp.vec3(hx, hy, hz)
    shape.gap = 0.05
    shape.mu = 0.9
    shape.collision_group = -1  # planks never collide with each other
    shape.flags = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape.prototype = -1
    phoenx_add_shape(dyn, body, shape)


@wp.kernel
def _build_hinges_kernel(
    dyn: DynamicsData,
    state: wp.array[wp.int32],
    hinge_parent: wp.array[wp.int32],
    hinge_child: wp.array[wp.int32],
    parent_xform: wp.array[wp.transform],
    child_xform: wp.array[wp.transform],
    planks: wp.array[wp.int32],
    hinges: wp.array[wp.int32],
):
    """Hinge ``k`` joins plank ``k - 1`` to plank ``k``; the end hinges hold the deck at the cliff edges."""
    k = wp.tid()
    if state[0] != _BUILD:
        return
    parent = wp.int32(-1)
    if hinge_parent[k] >= 0:
        parent = planks[hinge_parent[k]]
    desc = joint_desc(parent, planks[hinge_child[k]], parent_xform[k], child_xform[k])
    desc.angular_axes = wp.mat33(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)  # free about the deck's y axis
    desc.angular_count = 1
    hinges[k] = phoenx_add_joint(dyn, desc)


@wp.kernel
def _finish_build_kernel(state: wp.array[wp.int32]):
    if state[0] == _BUILD:
        state[0] = _INTACT


@wp.kernel
def _break_kernel(
    dyn: DynamicsData,
    state: wp.array[wp.int32],
    strength: float,
    hinges: wp.array[wp.int32],
    load: wp.array[wp.float32],
):
    """Remove hinges whose reaction force, smoothed over a few frames, exceeds ``strength`` [N]."""
    k = wp.tid()
    hinge = hinges[k]
    if hinge < 0:
        load[k] = 0.0
        return
    force = wp.length(wp.spatial_top(phoenx_joint_wrench(dyn, hinge)))
    load[k] = load[k] + SMOOTHING * (force - load[k])  # filters solver jitter, keeps sustained loads
    if load[k] <= strength:
        return
    phoenx_remove_joint(dyn, hinge)
    hinges[k] = -1
    wp.atomic_add(state, 3, 1)
    if state[0] == _INTACT:
        state[0] = _FAILING
        state[1] = state[2]


@wp.kernel
def _despawn_kernel(
    dyn: DynamicsData,
    body_q: wp.array[wp.transform],
    planks: wp.array[wp.int32],
    hinges: wp.array[wp.int32],
    loads: wp.array[wp.int32],
):
    """Remove planks and loads that fell into the gorge."""
    i = wp.tid()
    if i < PLANKS:
        body = planks[i]
        if body >= 0 and wp.transform_get_translation(body_q[body])[2] < KILL_DEPTH:
            phoenx_remove_body(dyn, body)
            planks[i] = -1
            hinges[i] = -1  # a removed plank takes its hinges along
            hinges[i + 1] = -1
    else:
        slot = i - PLANKS
        body = loads[slot]
        if body >= 0 and phoenx_body_is_active(dyn, body):
            if wp.transform_get_translation(body_q[body])[2] < KILL_DEPTH:
                phoenx_remove_body(dyn, body)
                loads[slot] = -1


@wp.kernel
def _instances_kernel(
    dyn: DynamicsData,
    body_q: wp.array[wp.transform],
    planks: wp.array[wp.int32],
    loads: wp.array[wp.int32],
    load_size: wp.array[wp.vec3],
    plank_xforms: wp.array[wp.transform],
    boulder_xforms: wp.array[wp.transform],
    crate_xforms: wp.array[wp.transform],
    load_scales: wp.array[wp.vec3],
):
    """Instance transforms; unused rows are parked far below the gorge."""
    i = wp.tid()
    parked = wp.transform(wp.vec3(0.0, 0.0, -1000.0), wp.quat_identity())
    if i < PLANKS:
        body = planks[i]
        plank_xforms[i] = wp.where(body >= 0, body_q[wp.max(body, 0)], parked)
        return
    slot = i - PLANKS
    body = loads[slot]
    boulder_xforms[slot] = parked
    crate_xforms[slot] = parked
    size = load_size[slot]
    load_scales[slot] = wp.vec3(wp.abs(size[0]))
    if body >= 0 and phoenx_body_is_active(dyn, body):
        if size[0] > 0.0:
            boulder_xforms[slot] = body_q[body]
        else:
            crate_xforms[slot] = body_q[body]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.strength = args.strength

        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=-10.0)
        rock = newton.ModelBuilder.ShapeConfig(mu=0.9)
        for side in (-1.0, 1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(side * (0.5 * GAP + 5.0), 0.0, -5.0), wp.quat_identity()),
                hx=5.0,
                hy=6.0,
                hz=5.0,
                cfg=rock,
                color=(0.45, 0.38, 0.3),
            )
        builder.reserve_dynamic_capacity(bodies=2 * _RESERVED, shapes=2 * _RESERVED, joints=2 * (PLANKS + 1))
        self.model = builder.finalize()
        self.solver = SolverPhoenX(
            self.model,
            joint_mode="maximal_pgs",
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

        # Plank poses and hinge frames of the hanging deck.
        points = deck_points()
        rotations, poses = [], []
        for a, b in itertools.pairwise(points):
            d = b - a
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.arctan2(-d[2], d[0])))
            rotations.append(q)
            poses.append(wp.transform(wp.vec3(*(0.5 * (a + b))), q))
        end = wp.vec3(0.5 * PITCH, 0.0, 0.0)
        hinge_parent, hinge_child, parent_xform, child_xform = [], [], [], []
        for k in range(PLANKS + 1):
            child = min(k, PLANKS - 1)
            if 0 < k < PLANKS:
                hinge_parent.append(k - 1)
                parent_xform.append(wp.transform(end, wp.quat_inverse(rotations[k - 1])))
            else:  # the cliff edges
                hinge_parent.append(-1)
                parent_xform.append(wp.transform(wp.vec3(*points[k]), wp.quat_identity()))
            hinge_child.append(child)
            child_xform.append(wp.transform(-end if k < PLANKS else end, wp.quat_inverse(rotations[child])))
        self.plank_poses = wp.array(poses, dtype=wp.transform)
        self.hinge_parent = wp.array(hinge_parent, dtype=wp.int32)
        self.hinge_child = wp.array(hinge_child, dtype=wp.int32)
        self.parent_xform = wp.array(parent_xform, dtype=wp.transform)
        self.child_xform = wp.array(child_xform, dtype=wp.transform)

        self.state = wp.array([BUILD, 0, 0, 0, 0], dtype=wp.int32)
        self.planks = wp.full(PLANKS, -1, dtype=wp.int32)
        self.hinges = wp.full(PLANKS + 1, -1, dtype=wp.int32)
        self.hinge_load = wp.zeros(PLANKS + 1, dtype=wp.float32)
        self.loads = wp.full(LOADS, -1, dtype=wp.int32)
        self.load_size = wp.zeros(LOADS, dtype=wp.vec3)
        self.plank_xforms = wp.zeros(PLANKS, dtype=wp.transform)
        self.boulder_xforms = wp.zeros(LOADS, dtype=wp.transform)
        self.crate_xforms = wp.zeros(LOADS, dtype=wp.transform)
        self.load_scales = wp.zeros(LOADS, dtype=wp.vec3)
        self.plank_colors = wp.array(
            np.asarray([(0.55, 0.36, 0.18) if i % 2 else (0.48, 0.3, 0.14) for i in range(PLANKS)]), dtype=wp.vec3
        )
        self.boulder_colors = wp.full(LOADS, wp.vec3(0.42, 0.42, 0.44), dtype=wp.vec3)
        self.crate_colors = wp.full(LOADS, wp.vec3(0.9, 0.45, 0.08), dtype=wp.vec3)
        self.materials = wp.full(LOADS, wp.vec4(0.7, 0.0, 0.0, 0.0), dtype=wp.vec4)

        viewer.set_model(self.model)
        viewer.log_geo("/unit_sphere", newton.GeoType.SPHERE, (1.0,), 0.0, True, hidden=True)
        viewer.log_geo("/unit_box", newton.GeoType.BOX, (1.0, 1.0, 1.0), 0.0, True, hidden=True)
        viewer.set_camera(wp.vec3(9.0, -14.0, 5.0), pitch=-15.0, yaw=122.0)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.launch(
            _director_kernel, dim=1, inputs=[self.dyn, self.state, self.planks, self.hinges, self.loads, self.load_size]
        )
        wp.launch(_build_planks_kernel, dim=PLANKS, inputs=[self.dyn, self.state, self.plank_poses, self.planks])
        wp.launch(
            _build_hinges_kernel,
            dim=PLANKS + 1,
            inputs=[
                self.dyn,
                self.state,
                self.hinge_parent,
                self.hinge_child,
                self.parent_xform,
                self.child_xform,
                self.planks,
                self.hinges,
            ],
        )
        wp.launch(_finish_build_kernel, dim=1, inputs=[self.state])
        self.pipeline.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
        wp.copy(self.state_0.body_q, self.state_1.body_q)
        wp.copy(self.state_0.body_qd, self.state_1.body_qd)
        self.solver.dynamics.update_joint_wrenches()
        wp.launch(
            _break_kernel, dim=PLANKS + 1, inputs=[self.dyn, self.state, self.strength, self.hinges, self.hinge_load]
        )
        wp.launch(
            _despawn_kernel,
            dim=PLANKS + LOADS,
            inputs=[self.dyn, self.state_0.body_q, self.planks, self.hinges, self.loads],
        )

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def _update_instances(self):
        wp.launch(
            _instances_kernel,
            dim=PLANKS + LOADS,
            inputs=[self.dyn, self.state_0.body_q, self.planks, self.loads, self.load_size],
            outputs=[self.plank_xforms, self.boulder_xforms, self.crate_xforms, self.load_scales],
        )

    def render(self):
        self._update_instances()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_shapes(
            "/planks", newton.GeoType.BOX, (PLANK_HX, PLANK_HY, PLANK_HZ), self.plank_xforms, self.plank_colors
        )
        self.viewer.log_instances(
            "/boulders", "/unit_sphere", self.boulder_xforms, self.load_scales, self.boulder_colors, self.materials
        )
        self.viewer.log_instances(
            "/crates", "/unit_box", self.crate_xforms, self.load_scales, self.crate_colors, self.materials
        )
        self.viewer.end_frame()

    def hinge_gaps(self) -> np.ndarray:
        """Opening of every intact plank-to-plank hinge [m]."""
        q = self.state_0.body_q.numpy()
        planks, hinges = self.planks.numpy(), self.hinges.numpy()
        gaps = []
        for k in range(1, PLANKS):
            a, b = planks[k - 1], planks[k]
            if hinges[k] < 0 or a < 0 or b < 0:
                continue
            pa = wp.transform_point(wp.transform(*q[a].tolist()), wp.vec3(0.5 * PITCH, 0.0, 0.0))
            pb = wp.transform_point(wp.transform(*q[b].tolist()), wp.vec3(-0.5 * PITCH, 0.0, 0.0))
            gaps.append(float(np.linalg.norm(np.asarray(pa) - np.asarray(pb))))
        return np.asarray(gaps)

    def test_post_step(self):
        """Every second, check finite state and closed hinges."""
        if round(self.sim_time / self.frame_dt) % 60 != 0:
            return
        live = self.solver.dynamics.bodies.data.state.numpy()
        bodies = self.model._dynamic_capacity.body_start + np.flatnonzero((live == 1) | (live == 2))
        assert np.isfinite(self.state_0.body_q.numpy()[bodies]).all(), "nonfinite body pose"
        gaps = self.hinge_gaps()
        assert gaps.size == 0 or gaps.max() < 0.05, f"intact hinge opened by {gaps.max():.3f} m"

    def test_final(self):
        """The bridge broke and was rebuilt, and every runtime request succeeded."""
        _, _, frames, breaks, rebuilds = (int(v) for v in self.state.numpy())
        print(f"PhoenX bridge: {breaks} hinges broke, {rebuilds} rebuilds in {frames} frames")
        assert int(self.solver.dynamics.error_count.numpy()[0]) == 0
        assert breaks > 0 and rebuilds > 0, "the bridge never broke and rebuilt"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="optix", num_frames=1800)
        parser.add_argument("--strength", type=float, default=90.0e3, help="Hinge breaking force [N].")
        parser.add_argument("--substeps", type=int, default=8, help="Physics substeps per frame.")
        parser.add_argument("--iterations", type=int, default=4, help="Solver iterations per substep.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
