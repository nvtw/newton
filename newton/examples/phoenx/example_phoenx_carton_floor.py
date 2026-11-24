# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect a factory bottle balanced across the edge of a carton bottom flap.

Three setups share one dynamic pallet. On the left, B01642 stands on four
floor panels. The middle carton reproduces PD1_E06 with twelve bottles. The
right carton removes its two raised minor bottom flaps. Both cartons have
transparent walls, partitions and lids. The orange bottle has the same local
pose in each setup. Body poses come from step 872 of the mature pallet replay.
The two cartons share inertial properties to isolate the support geometry.
Success requires both cartons to retain their bottles and stay on the pallet.

Run commands:
    python -m newton.examples phoenx_carton_floor --paused
    python -m newton.examples phoenx_carton_floor --substeps 8 --iterations 3
    python -m newton.examples phoenx_carton_floor --gap-mm 1.0 --paused
    python -m newton.examples phoenx_carton_floor --viewer null --test
"""

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverPhoenX


def _transform(pose):
    return wp.transform(pose[:3], pose[3:])


def _contact_material(builder, data, index, prefix="", density=0.0):
    return builder.ShapeConfig(
        density=density,
        gap=float(data[prefix + "gap"][index]),
        margin=float(data[prefix + "margin"][index]),
        **{
            name: float(data[prefix + "material_" + name][index])
            for name in ("mu", "mu_static", "ke", "kd", "kf", "restitution")
        },
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("--substeps and --iterations must be positive")
        self.substeps = args.substeps
        self.dt = self.frame_dt / self.substeps
        builder = newton.ModelBuilder()
        asset = Path(__file__).parents[1] / "assets/bottling_plant/carton_floor.npz"
        with np.load(asset) as source:
            data = dict(source)
        poses = data["pose"]
        subject = 5 + int(data["subject"])
        meshes = [newton.Mesh(data[f"v{i}"], data[f"f{i}"], compute_inertia=False) for i in range(8)]
        self.pallet = builder.add_body(
            xform=_transform(data["pallet_q"]),
            mass=float(data["pallet_mass"]),
            com=wp.vec3(data["pallet_com"]),
            inertia=wp.mat33(data["pallet_inertia"]),
            lock_inertia=True,
            label="pallet",
        )
        for i, half in enumerate(data["pallet_scale"]):
            builder.add_shape_box(
                self.pallet,
                xform=_transform(data["pallet_transform"][i]),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=_contact_material(builder, data, i, "pallet_"),
                color=(0.49, 0.31, 0.17),
            )
        # Estimate the cut-out panel mass from the carton root's mean board density.
        root_volume = 8 * np.prod(data["scale"][data["owner"] == 0], axis=1).sum()
        board_density = float(data["mass"][0] / root_volume)
        self.cartons = {}
        gap = args.gap_mm
        if gap is not None and not (np.isfinite(gap) and gap >= 0):
            raise ValueError("--gap-mm must be finite and nonnegative")
        bottle_shapes = []
        for variant, offset in (("isolated", -0.32), ("stepped", 0.0), ("flat", 0.32)):
            full = variant != "isolated"
            carton_shapes = []
            bodies = {}
            for i, pose in enumerate(poses):
                if not full and i not in (0, subject):
                    continue
                xf = wp.transform(np.add(pose[:3], (offset, 0, data["deck_height"])), pose[3:])
                add = builder.add_link if full and i < 5 else builder.add_body
                cutout = not full and i == 0
                body = add(
                    xform=xf,
                    mass=0.0 if cutout else float(data["mass"][i]),
                    com=wp.vec3(data["com"][i]),
                    inertia=wp.mat33(data["inertia"][i]),
                    lock_inertia=not cutout,
                    label=f"{variant}/{data['body_names'][i]}",
                )
                bodies[i] = body
                hulls = []
                for s in np.flatnonzero(data["owner"] == min(i, 5)):
                    name = str(data["shape_names"][s])
                    floor = name.startswith("BottomFlaps/")
                    if variant == "flat" and name.startswith("BottomFlaps/Minor_"):
                        continue
                    if not full and i == 0 and not floor:
                        continue
                    cfg = _contact_material(builder, data, s, density=board_density if cutout else 0.0)
                    if gap is not None:
                        cfg.gap = gap * 0.001
                    xf = _transform(data["transform"][s])
                    half = data["scale"][s]
                    if i >= 5:
                        part = int(name.rsplit("Hull_", maxsplit=1)[-1])
                        hulls.append(
                            builder.add_shape_convex_hull(
                                body,
                                mesh=meshes[part],
                                scale=half,
                                xform=xf,
                                cfg=cfg,
                                color=(0.92, 0.29, 0.05) if i == subject else (0.09, 0.48, 0.49),
                            )
                        )
                    else:
                        shape = builder.add_shape_box(
                            body,
                            hx=half[0],
                            hy=half[1],
                            hz=half[2],
                            xform=xf,
                            cfg=cfg,
                            opacity=1.0 if floor else 0.13,
                            label=f"{builder.body_label[body]}/{name}",
                            color=(0.77, 0.52, 0.29) if "Minor" in name else (0.58, 0.36, 0.19),
                        )
                        if full:
                            carton_shapes.append(shape)
                if hulls:
                    bottle_shapes.append(hulls)
            if full:
                self.cartons[bodies[0]] = [bodies[j] for j in range(5, 17)]
                for k, shape in enumerate(carton_shapes):
                    for other in carton_shapes[:k]:
                        builder.add_shape_collision_filter_pair(shape, other)
                joints = [builder.add_joint_free(bodies[0])]
                for i in range(1, 5):
                    relative = wp.transform_inverse(_transform(poses[0])) * _transform(poses[i])
                    joints.append(builder.add_joint_fixed(bodies[0], bodies[i], parent_xform=relative))
                builder.add_articulation(joints)
            else:
                self.floor = bodies[0]
                self.single = bodies[subject]
        # The factory excludes the grasp hull in bottle-bottle pairs.
        for i, hulls in enumerate(bottle_shapes):
            for other in bottle_shapes[:i]:
                for shape in other:
                    builder.add_shape_collision_filter_pair(hulls[0], shape)
                for shape in hulls[1:]:
                    builder.add_shape_collision_filter_pair(other[0], shape)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.pipeline = newton.CollisionPipeline(
            self.model, contact_matching="sticky", speculative_contact_gap_max=0.005
        )
        self.solver = SolverPhoenX(
            self.model,
            collision_pipeline=self.pipeline,
            step_layout="single_world",
            joint_mode="maximal_pgs",
            direct_joint_mask=np.zeros(self.model.joint_count, dtype=bool),
            substeps=1,
            solver_iterations=args.iterations,
            velocity_iterations=1,
            temporal_iterations=True,
            contact_hertz=80,
            enable_body_pair_grouping=True,
            contact_chunk_size=64,
            sleeping_velocity_threshold=0.01,
            sleeping_frames_required=30 * self.substeps,
        )
        self.state = self.model.state()
        self.contacts = self.pipeline.contacts()
        self.contact_state = newton.State()
        self.contact_state.body_q = wp.clone(self.state.body_q)
        self.contact_state.body_qd = wp.clone(self.state.body_qd)
        self.contact_observables = self.solver.observables(
            contacts=self.contacts, kinds={self.solver.ObservableKind.CONTACT_F}
        )
        self.initial_contacts_drawn = False
        self.control = self.model.control()
        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(0, -1.06, 0.79), pitch=-26, yaw=90)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._advance()
            self.graph = capture.graph

    def step(self):
        self.state.clear_forces()
        self.viewer.apply_forces(self.state)
        self.solver.wake_on_external_input(self.state)
        if self.graph is None:
            self._advance()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def _advance(self):
        dt = self.dt
        for substep in range(self.substeps):
            last = substep == self.substeps - 1
            self.pipeline.collide(self.state, self.contacts, dt=dt)
            if last:
                # PhoenX advances in place. Contacts refer to the input poses of this substep.
                wp.copy(self.contact_state.body_q, self.state.body_q)
                wp.copy(self.contact_state.body_qd, self.state.body_qd)
            self.solver.step(
                self.state,
                self.state,
                self.control,
                self.contacts,
                dt,
                observables=self.contact_observables if last else None,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        if self.sim_time == 0.0 and self.viewer.show_contacts and not self.initial_contacts_drawn:
            self.pipeline.collide(self.state, self.contacts, dt=self.dt)
            self.initial_contacts_drawn = True
        self.viewer.log_contacts(
            self.contacts, self.contact_state, observables=self.contact_observables if self.sim_time else None
        )
        self.viewer.end_frame()

    def test_final(self):
        poses = self.state.body_q.numpy()
        assert np.isfinite(poses).all()
        for body, contents in self.cartons.items():
            carton = wp.transform_inverse(_transform(poses[body]))
            for i in contents:
                p = wp.transform_point(carton, wp.vec3(poses[i, :3]))
                assert abs(p[0]) < 0.14 and abs(p[1]) < 0.105 and 0.0 < p[2] < 0.26
        floor = wp.transform_inverse(_transform(poses[self.floor]))
        x, y, z = wp.transform_point(floor, wp.vec3(poses[self.single, :3]))
        assert abs(x) < 0.14 and abs(y) < 0.105 and 0 < z < 0.1
        pallet = wp.transform_inverse(_transform(poses[self.pallet]))
        for body in (self.floor, *self.cartons):
            x, y, z = wp.transform_point(pallet, wp.vec3(poses[body, :3]))
            assert abs(x) < 0.5 and abs(y) < 0.6 and 0.15 < z < 0.17

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--gap-mm",
            type=float,
            help="Override the contact detection gap per bottle and carton shape in mm (authored default 2 mm).",
        )
        parser.add_argument("--substeps", type=int, default=2, help="Collision and solver steps per 60 Hz frame.")
        parser.add_argument("--iterations", type=int, default=12, help="Biased contact sweeps per substep.")
        parser.set_defaults(viewer="gl")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
