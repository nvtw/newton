"""Compare PhoenX ownership modes on the same Colibri scene."""

import json
from pathlib import Path

import warp as wp

import newton
from newton.examples.kamino.example_kamino_colibri import Example as KaminoExample
from newton.examples.kamino.example_kamino_colibri import apply_source_damping, build_scene


class Example(KaminoExample):
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.outer_substeps = getattr(args, "outer_substeps", 1)
        self.fix_base = getattr(args, "fix_base", False)
        self.source_damping = getattr(args, "source_damping", False)
        builder = build_scene(
            body_count=args.body_count,
            fix_base=self.fix_base,
            contact_gap=getattr(args, "contact_gap", 0.0001),
            source_contact_offsets=getattr(args, "source_contact_offsets", False),
            mesh_cylinders=getattr(args, "mesh_cylinders", False),
        )
        offset_map = getattr(args, "contact_offset_map", None)
        if offset_map is not None:
            offsets = json.loads(Path(offset_map).read_text())
            for index, label in enumerate(builder.shape_label):
                if label in offsets:
                    builder.shape_gap[index] = float(offsets[label])
        self.model = builder.finalize(skip_validation_joints=True)
        if getattr(args, "frictionless", False):
            self.model.shape_material_mu.zero_()
        newton.eval_ik(self.model, self.model, self.model.joint_q, self.model.joint_qd)
        # Preserve authored off-manifold poses in an independent simulation state.
        # Generalized-coordinate initialization must not change this initial state.
        initial_state = None
        if getattr(args, "mode", "maximal") == "maximal" and not getattr(args, "project_initial", False):
            initial_state = self.model.state()
        self.contact_capacity = 8192
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=getattr(args, "matching", "sticky"),
            rigid_contact_max=self.contact_capacity,
            speculative_contact_gap_max=getattr(args, "speculative_contact_gap_max", None),
        )
        # Keep relaxation neutral while diagnosing solver stability.
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            collision_pipeline=self.collision_pipeline,
            substeps=getattr(args, "substeps", 8),
            solver_iterations=getattr(args, "iterations", 8),
            velocity_relaxation=(
                "final_substep"
                if getattr(args, "public_solver", False) and not getattr(args, "relax_every_substep", False)
                else "each_substep"
            ),
            prepare_refresh_stride=1,
            sor_boost=1.0,
            mass_splitting=getattr(args, "mass_splitting", False),
            max_colored_partitions=getattr(args, "max_colored_partitions", 12),
            mass_splitting_batch_size=getattr(args, "mass_splitting_batch_size", 8),
            mass_splitting_color_group_size=getattr(args, "mass_splitting_color_group_size", 0),
            articulation_mode=getattr(args, "mode", "maximal"),
            joint_solver="block_pgs" if getattr(args, "public_solver", False) else "direct",
            contact_chunk_size=getattr(args, "contact_chunk_size", 0) if getattr(args, "public_solver", False) else 0,
            parallel_contact_prepare=getattr(args, "public_solver", False) and getattr(args, "parallel_prepare", False),
            step_layout=getattr(args, "layout", "single_world"),
        )
        if getattr(args, "forest", False):
            from newton._src.solvers.phoenx.articulations.reduced_forest import (
                ReducedForestContactSystem,
            )

            backend = self.solver._reduced_articulation
            if backend is None:
                raise ValueError("The forest contact prototype requires reduced articulation mode")
            backend.forest_contact_system = ReducedForestContactSystem(backend, self.contact_capacity)
        self.state_0 = initial_state if initial_state is not None else self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()
        self.initial_q = self.state_0.body_q.numpy().copy()
        self.viewer.set_model(self.model)
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(wp.vec3(0.4, -0.7, 0.35), pitch=-10, yaw=120)
        self.graph = None
        if self.model.device.is_cuda and not getattr(args, "no_graph", False):
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.outer_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts, dt=self.frame_dt / self.outer_substeps)
            self.state_0.clear_forces()
            if self.source_damping:
                apply_source_damping(self.model, self.state_0)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(
                self.state_0, self.state_0, self.control, self.contacts, self.frame_dt / self.outer_substeps
            )
