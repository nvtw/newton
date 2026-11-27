# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small bottling-plant scenes stepped with the plant's own physics settings.

The bottling plant's failure modes show up after many simulated minutes of
the full factory. These builders put the plant's bottle colliders and
materials into scenes of a few hundred bodies, and :class:`PlantPhysics`
steps them with the plant's collision pipeline and solver settings, so a
regression of one mode reproduces in seconds.
"""

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.sleeping_filter import build_phoenx_sleeping_filter_data
from newton.examples.phoenx._bottling_process import GUIDE_FRICTION
from newton.examples.phoenx._bottling_scene import add_round_bottle, seal_base_ring
from newton.examples.phoenx.example_phoenx_bottling_plant import _filter_pair, _PlantFilter
from newton.solvers import SolverPhoenX
from newton.tests.test_convex_refinement_regressions import _bottle_heel_hull

#: The plant's 60 Hz frame runs two collision passes.
FRAME_DT = 1.0 / 60.0
PASSES = 2
#: Mass properties of the source bottle; its centre of mass sits low in the thick base.
BOTTLE_MASS = 0.570
BOTTLE_COM = (0.0, 0.0, 0.0834)
BOTTLE_INERTIA = (0.00184, 0.00184, 0.000261)
BOTTLES = "/Plant/Sim/Product/Bottles"

_HEEL = None


def glass(builder):
    """The plant's bottle glass."""
    return builder.ShapeConfig(density=0.0, mu=0.16, mu_static=0.2, gap=0.002)


def belt(builder):
    """The plant's belt and deck material."""
    return builder.ShapeConfig(density=0.0, mu=0.18, mu_static=0.2, gap=0.001, is_visible=False)


def guide(builder):
    """The plant's UHMW guide rails and table guards."""
    return builder.ShapeConfig(density=0.0, mu=GUIDE_FRICTION[0], mu_static=GUIDE_FRICTION[1], gap=0.001)


def add_plant_bottle(builder, xform, name, *, velocity=None):
    """Add a bottle with the plant's colliders: the sealed base-ring hull, then the round body above it."""
    global _HEEL  # noqa: PLW0603 - one hull serves every bottle.
    if _HEEL is None:
        _HEEL = seal_base_ring(_bottle_heel_hull())
    label = f"{BOTTLES}/{name}"
    body = builder.add_body(
        xform=xform,
        mass=BOTTLE_MASS,
        com=wp.vec3(*BOTTLE_COM),
        inertia=wp.mat33(np.diag(BOTTLE_INERTIA)),
        lock_inertia=True,
        label=label,
    )
    cfg = glass(builder)
    builder.add_shape_convex_hull(body, mesh=_HEEL, cfg=cfg, label=f"{label}/Collision/Hull_00/collision")
    add_round_bottle(builder, body, f"{label}/Collision", cfg=cfg)
    if velocity is not None:
        builder.body_qd[body] = wp.spatial_vector(*velocity, 0.0, 0.0, 0.0)
    return body


def strip(first, second):
    """Triangulate the strip between two polylines, as the plant's decks and guards are."""
    vertices = np.stack((first, second), axis=1).reshape(-1, 3)
    i = np.arange(0, len(vertices) - 2, 2, dtype=np.int32)
    return vertices, np.column_stack((i, i + 2, i + 1, i + 1, i + 2, i + 3)).ravel()


def add_belt(builder, belts, vertices, indices, velocity, label, cfg=None):
    """Add a static mesh whose surface moves at ``velocity`` (one vector, or one per vertex)."""
    mesh = newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=True)
    shape = builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg or belt(builder), label=label + "/collision")
    velocity = np.broadcast_to(np.asarray(velocity, dtype=np.float32), (len(vertices), 3)).copy()
    belts.append((mesh, velocity))
    return shape


def kinetic_energy(qd):
    """Translational and rotational kinetic energy [J] of plant bottles from their body velocities."""
    return 0.5 * BOTTLE_MASS * np.sum(qd[:, :3] ** 2) + 0.5 * np.sum(np.asarray(BOTTLE_INERTIA) * qd[:, 3:] ** 2)


def tilt(q):
    """Tilt angles [rad] of bodies from their poses."""
    return 2.0 * np.arcsin(np.minimum(np.linalg.norm(q[:, 3:5], axis=1), 1.0))


class PlantPhysics:
    """Step a scene as the bottling plant does, one captured 60 Hz frame at a time.

    The collision pipeline keeps the plant's sticky matching, speculative gap
    and pair filter (bottle base rings ignore other bottles); the solver runs
    three substeps of four temporal sweeps per 120 Hz collision pass, with the
    plant's contact stiffness and sleeping thresholds. ``belts`` lists
    ``(mesh, vertex_velocities)`` pairs from :func:`add_belt`.
    """

    def __init__(self, builder, belts=(), *, gravity=None, sleeping=True):
        self.device = wp.get_cuda_device()
        with wp.ScopedDevice(self.device):
            self.model = model = builder.finalize()
            if gravity is not None:
                model.gravity.assign(np.array([gravity], dtype=np.float32))
            for mesh, velocity in belts:
                mesh.mesh.velocities.assign(velocity)
            labels = model.shape_label
            product = np.zeros(model.shape_count, dtype=np.int32)
            for shape, label in enumerate(labels):
                if label.startswith(BOTTLES + "/"):
                    product[shape] = 2 if "/Collision/Hull_00/" in label else 1
            data = _PlantFilter()
            data.collision_membership = wp.zeros(model.shape_count, dtype=wp.uint64)
            data.collision_filtered = wp.zeros(model.shape_count, dtype=wp.uint64)
            data.product = wp.array(product, dtype=wp.int32)
            data.articulation = wp.zeros(model.shape_count, dtype=wp.int32)
            data.shape_body = model.shape_body
            data.held_by = wp.zeros(max(model.body_count, 1), dtype=int)
            data.sleeping = build_phoenx_sleeping_filter_data(
                sleeping_enabled=False,
                phoenx_body_offset=1,
                shape_body=None,
                body_island_root=None,
                body_motion_type=None,
                device=self.device,
            )
            self.pipeline = newton.CollisionPipeline(
                model,
                contact_matching="sticky",
                broad_phase="sap",
                speculative_contact_gap_max=0.005,
                broad_phase_filter=(_filter_pair, _PlantFilter),
                include_static_kinematic_pairs=False,
            )
            self.pipeline.set_broad_phase_filter_data(data)
            self.solver = SolverPhoenX(
                model,
                collision_pipeline=self.pipeline,
                substeps=3,
                solver_iterations=4,
                temporal_iterations=True,
                contact_hertz=80.0,
                contact_chunk_size=64,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                velocity_iterations=1,
                enable_body_pair_grouping=True,
                sleeping_velocity_threshold=0.01 if sleeping else 0.0,
                sleeping_frames_required=1,
                sleeping_motion_threshold=0.0002 if sleeping else 0.0,
            )
            if self.solver._sleeping_filter_data is not None:
                data.sleeping = self.solver._sleeping_filter_data
            self.pipeline.set_broad_phase_filter_data(data)
            self.filter_data = data
            self.state = model.state()
            self.control = model.control()
            self.contacts = self.pipeline.contacts()
            with wp.ScopedCapture() as capture:
                self._frame()
            self.graph = capture.graph

    def _frame(self):
        self.state.clear_forces()
        self.solver.wake_on_external_input(self.state)
        dt = FRAME_DT / PASSES
        for _ in range(PASSES):
            self.pipeline.collide(self.state, self.contacts, dt=dt)
            self.solver.step(self.state, self.state, self.control, self.contacts, dt)

    def run(self, frames, every=0, sample=None):
        """Replay ``frames`` frames, calling ``sample(frame, body_q, body_qd)`` every ``every`` frames."""
        with wp.ScopedDevice(self.device):
            for frame in range(1, frames + 1):
                wp.capture_launch(self.graph)
                if every and sample is not None and frame % every == 0:
                    sample(frame, self.state.body_q.numpy(), self.state.body_qd.numpy())
            return self.state.body_q.numpy(), self.state.body_qd.numpy()
