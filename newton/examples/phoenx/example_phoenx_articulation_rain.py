# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rain four kinds of articulations onto a staircase with runtime spawning in PhoenX.

Command: python -m newton.examples phoenx_articulation_rain

Ragdolls, snakes, spiders, and ropes are spawned as whole articulations from device code
and tumble down a staircase. Once the live bodies reach a budget, the oldest articulations
are removed. All four templates share one reserve of body, shape, and joint rows, so the mix
can change freely. Each spawned articulation is a joint tree whose joints PhoenX solves
exactly, and spawning, removal, collision, and stepping run inside one CUDA graph.

Success criteria: in test mode, body states stay finite, 99% of joints stay within 1 cm,
no runtime request fails, and the live bodies follow the spawn rate up to the budget.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.dynamic import (
    DynamicsData,
    phoenx_articulation_body,
    phoenx_articulation_is_active,
    phoenx_remove_articulation,
    phoenx_spawn_articulation,
)
from newton.examples.phoenx.ragdoll_pile_scene import BODIES, JOINTS
from newton.solvers import SolverPhoenX

STEPS = 8  # staircase steps
STEP_RISE, STEP_RUN = 0.35, 0.8  # [m]
SPAWN_HEIGHT = STEPS * STEP_RISE + 2.5  # [m]
PALETTE = ((0.86, 0.06, 0.02), (0.94, 0.42, 0.01), (0.03, 0.6, 0.12), (0.0, 0.53, 0.88))
X_AXIS_CAPSULE = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)


def ragdoll() -> newton.ModelBuilder:
    """The six-box figure of the ragdoll pile, with spherical pivots."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=200.0, mu=0.6)
    torso = BODIES[0].center
    ids = []
    for body in BODIES:
        ids.append(builder.add_link(xform=wp.transform(np.asarray(body.center) - torso, wp.quat_identity())))
        builder.add_shape_box(
            ids[-1], hx=body.half_extents[0], hy=body.half_extents[1], hz=body.half_extents[2], cfg=cfg
        )
    for child, anchor in JOINTS:
        builder.add_joint_ball(
            ids[0],
            ids[child],
            parent_xform=wp.transform(np.asarray(anchor) - BODIES[0].center, wp.quat_identity()),
            child_xform=wp.transform(np.asarray(anchor) - BODIES[child].center, wp.quat_identity()),
        )
    return builder


def chain(links: int, radius: float, half: float, ball: bool) -> newton.ModelBuilder:
    """Capsules along x; hinges alternate between pitch and yaw, or ball joints for a rope."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=400.0, mu=0.5)
    pitch = 2.0 * (half + radius)
    previous = -1
    for i in range(links):
        body = builder.add_link(xform=wp.transform(wp.vec3(pitch * i, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_capsule(
            body, xform=wp.transform((0.0, 0.0, 0.0), X_AXIS_CAPSULE), radius=radius, half_height=half, cfg=cfg
        )
        if previous >= 0:
            frames = {
                "parent_xform": wp.transform(wp.vec3(0.5 * pitch, 0.0, 0.0), wp.quat_identity()),
                "child_xform": wp.transform(wp.vec3(-0.5 * pitch, 0.0, 0.0), wp.quat_identity()),
            }
            if ball:
                builder.add_joint_ball(previous, body, **frames)
            else:
                axis = newton.Axis.Y if i % 2 else newton.Axis.Z
                builder.add_joint_revolute(previous, body, axis=axis, limit_lower=-0.8, limit_upper=0.8, **frames)
        previous = body
    return builder


def spider() -> newton.ModelBuilder:
    """A hub with four two-segment legs on limited hinges."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.8)
    hub = builder.add_link()
    builder.add_shape_sphere(hub, radius=0.16, cfg=cfg)
    for k in range(4):
        yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi * k)
        out = wp.quat_rotate(yaw, wp.vec3(1.0, 0.0, 0.0))
        parent, base = hub, 0.16
        for segment in range(2):
            center = out * (base + 0.17)
            body = builder.add_link(xform=wp.transform(center, yaw))
            builder.add_shape_capsule(
                body, xform=wp.transform((0.0, 0.0, 0.0), X_AXIS_CAPSULE), radius=0.045, half_height=0.12, cfg=cfg
            )
            builder.add_joint_revolute(
                parent,
                body,
                parent_xform=wp.transform(
                    out * base if segment == 0 else wp.vec3(0.17, 0.0, 0.0), yaw if segment == 0 else wp.quat_identity()
                ),
                child_xform=wp.transform(wp.vec3(-0.17, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.Y,
                limit_lower=-0.7,
                limit_upper=0.7,
            )
            parent, base = body, base + 0.34
    return builder


TEMPLATES = (ragdoll(), chain(10, 0.07, 0.09, ball=False), spider(), chain(14, 0.045, 0.07, ball=True))
TEMPLATE_BODIES = tuple(t.body_count for t in TEMPLATES)


@wp.kernel
def _spawn_kernel(
    dyn: DynamicsData,
    rate: int,
    template_bodies: wp.array[wp.int32],
    counters: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    ring_template: wp.array[wp.int32],
):
    """Spawn ``rate`` articulations at random spots above the top step.

    ``counters`` holds ``[frame, total spawned, total removed, live bodies]``.
    """
    i = wp.tid()
    frame = counters[0]
    rng = wp.rand_init(99, frame * rate + i)
    template = wp.int32(wp.randf(rng) * float(template_bodies.shape[0])) % template_bodies.shape[0]
    x = wp.randf(rng, -1.0, 1.0) - float(STEPS) * STEP_RUN * 0.5 + 0.4
    y = (wp.randf(rng) - 0.5) * 3.0 + 2.6 * (float(i) - 0.5 * float(rate - 1))
    axis = wp.normalize(wp.vec3(wp.randf(rng, -1.0, 1.0), wp.randf(rng, -1.0, 1.0), wp.randf(rng, -1.0, 1.0)))
    pose = wp.transform(wp.vec3(x, y, SPAWN_HEIGHT), wp.quat_from_axis_angle(axis, wp.randf(rng, 0.0, 6.28)))
    spin = wp.spatial_vector(
        wp.randf(rng, -0.5, 0.5), 0.0, -1.0, wp.randf(rng, -2.0, 2.0), wp.randf(rng, -2.0, 2.0), 0.0
    )
    instance = phoenx_spawn_articulation(dyn, template, pose, spin, 0)
    if instance < 0:
        return
    slot = wp.atomic_add(counters, 1, 1) % ring.shape[0]
    ring[slot] = instance
    ring_template[slot] = template
    wp.atomic_add(counters, 3, template_bodies[template])


@wp.kernel
def _retire_kernel(
    dyn: DynamicsData,
    budget: int,
    template_bodies: wp.array[wp.int32],
    counters: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    ring_template: wp.array[wp.int32],
):
    """Remove the oldest articulations while the live bodies exceed ``budget``, then advance the frame."""
    while counters[3] > budget and counters[2] < counters[1]:
        slot = counters[2] % ring.shape[0]
        phoenx_remove_articulation(dyn, ring[slot])
        counters[3] = counters[3] - template_bodies[ring_template[slot]]
        counters[2] = counters[2] + 1
    counters[0] = counters[0] + 1


@wp.kernel
def _shape_xforms_kernel(
    dyn: DynamicsData,
    body_q: wp.array[wp.transform],
    template: int,
    body: int,
    local: wp.transform,
    xforms: wp.array[wp.transform],
):
    """Instance transforms of one template shape; other instances are parked out of sight."""
    instance = wp.tid()
    xforms[instance] = wp.transform(wp.vec3(0.0, 0.0, -1000.0), wp.quat_identity())
    if phoenx_articulation_is_active(dyn, instance) and dyn.articulation_template[instance] == template:
        b = phoenx_articulation_body(dyn, instance, body)
        if b >= 0:
            xforms[instance] = body_q[b] * local


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.rate = args.rate
        self.budget = args.max_bodies
        # Live bodies plus one frame of spawns, pending removals, and quarantine of the largest kind.
        rows = self.budget + 4 * self.rate * max(TEMPLATE_BODIES)
        self.capacity = rows // min(TEMPLATE_BODIES) + 1

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        stone = newton.ModelBuilder.ShapeConfig(mu=0.7)
        for k in range(STEPS):
            top = (STEPS - k) * STEP_RISE
            x = -0.5 * STEPS * STEP_RUN + (k + 0.5) * STEP_RUN
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, 0.0, 0.5 * top), wp.quat_identity()),
                hx=0.5 * STEP_RUN,
                hy=5.0,
                hz=0.5 * top,
                cfg=stone,
                color=(0.62, 0.6, 0.56),
            )
        builder.reserve_dynamic_capacity(
            bodies=rows, shapes=rows, joints=rows, articulations=self.capacity, articulation_templates=TEMPLATES
        )
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

        self.template_bodies = wp.array(TEMPLATE_BODIES, dtype=wp.int32)
        self.counters = wp.zeros(4, dtype=wp.int32)
        self.ring = wp.zeros(self.capacity, dtype=wp.int32)
        self.ring_template = wp.zeros(self.capacity, dtype=wp.int32)

        # One render batch per template shape over all instance slots.
        self.batches = []
        for t, template in enumerate(TEMPLATES):
            color = wp.full(self.capacity, wp.vec3(*PALETTE[t]), dtype=wp.vec3)
            for s in range(template.shape_count):
                geo = int(template.shape_type[s])
                scale = tuple(float(v) for v in template.shape_scale[s])
                if geo == newton.GeoType.CAPSULE:
                    scale = scale[:2]
                elif geo == newton.GeoType.SPHERE:
                    scale = scale[:1]
                xforms = wp.zeros(self.capacity, dtype=wp.transform)
                local = wp.transform(*template.shape_transform[s])
                self.batches.append((f"/t{t}_s{s}", t, int(template.shape_body[s]), local, geo, scale, xforms, color))

        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(7.5, -8.5, 5.5), pitch=-24.0, yaw=132.0)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.launch(
            _spawn_kernel,
            dim=self.rate,
            inputs=[self.dyn, self.rate, self.template_bodies, self.counters, self.ring, self.ring_template],
        )
        wp.launch(
            _retire_kernel,
            dim=1,
            inputs=[self.dyn, self.budget, self.template_bodies, self.counters, self.ring, self.ring_template],
        )
        self.pipeline.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
        wp.copy(self.state_0.body_q, self.state_1.body_q)
        wp.copy(self.state_0.body_qd, self.state_1.body_qd)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for name, t, body, local, geo, scale, xforms, color in self.batches:
            wp.launch(
                _shape_xforms_kernel,
                dim=self.capacity,
                inputs=[self.dyn, self.state_0.body_q, t, body, local],
                outputs=[xforms],
            )
            self.viewer.log_shapes(name, geo, scale, xforms, color)
        self.viewer.end_frame()

    def joint_gaps(self) -> np.ndarray:
        """Opening of every joint of the live articulations [m]."""
        q = self.state_0.body_q.numpy()
        live = self.solver.dynamics.joints.data.state.numpy()
        start = self.model._dynamic_capacity.joint_start
        joints = start + np.flatnonzero(live == 1)
        parent, child = self.model.joint_parent.numpy()[joints], self.model.joint_child.numpy()[joints]
        xp, xc = self.model.joint_X_p.numpy()[joints], self.model.joint_X_c.numpy()[joints]
        gaps = []
        for p, c, a, b in zip(parent, child, xp, xc, strict=True):
            pa = wp.transform_point(wp.transform(*q[p].tolist()), wp.vec3(*a[:3]))
            pb = wp.transform_point(wp.transform(*q[c].tolist()), wp.vec3(*b[:3]))
            gaps.append(float(np.linalg.norm(np.asarray(pa) - np.asarray(pb))))
        return np.asarray(gaps)

    def test_post_step(self):
        """Every second, check finite state and closed joints."""
        if round(self.sim_time / self.frame_dt) % 60 != 0:
            return
        live = self.solver.dynamics.bodies.data.state.numpy()
        bodies = self.model._dynamic_capacity.body_start + np.flatnonzero(live == 1)
        assert np.isfinite(self.state_0.body_q.numpy()[bodies]).all(), "nonfinite body pose"
        gaps = self.joint_gaps()
        if gaps.size:
            # Joints deep in the pile may be crushed briefly; nearly all stay closed.
            assert np.percentile(gaps, 99) < 0.01, f"joints opened: 99th percentile {np.percentile(gaps, 99):.3f} m"
            assert gaps.max() < 0.1, f"a joint opened by {gaps.max():.3f} m"

    def test_final(self):
        """Live bodies follow the spawn rate up to the budget, and every request succeeded."""
        frames, spawned, removed, live = (int(v) for v in self.counters.numpy())
        print(f"PhoenX articulation rain: {spawned - removed} articulations with {live} bodies after {frames} frames")
        assert int(self.solver.dynamics.error_count.numpy()[0]) == 0
        assert live <= self.budget + max(TEMPLATE_BODIES)
        assert live >= min(self.budget - max(TEMPLATE_BODIES), 0.8 * frames * self.rate * min(TEMPLATE_BODIES))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="optix", optix_max_instances=65536, num_frames=600)
        parser.add_argument("--rate", type=int, default=2, help="Articulations spawned per frame.")
        parser.add_argument("--max-bodies", type=int, default=3000, help="Live bodies kept once the scene is full.")
        parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per frame.")
        parser.add_argument("--iterations", type=int, default=2, help="Solver iterations per substep.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
