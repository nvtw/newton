# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.log_level = wp.LOG_WARNING

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_config import pr_gate_repeat

import newton.examples
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionDetector
from newton.examples.cloth.example_cloth_franka import Example as ExampleClothManipulation
from newton.examples.cloth.example_cloth_twist import Example as ExampleClothTwist
from newton.viewer import ViewerNull

DEFORMABLE_COLLISION_CASES = (
    ("dense_unfiltered", 256, 1, 0),
    ("dense_filtered", 128, 1, 2),
    ("dense_unfiltered", 16, 1024, 0),
    ("dense_filtered", 16, 128, 2),
    ("sparse", 16, 128, 2),
    ("folded", 32, 32, 2),
    ("layered", 16, 64, 2),
    ("dense_10m", 16, 11112, 1),
    ("sparse_10m", 64, 630, 1),
    ("folded_10m", 32, 2602, 1),
    ("layered_10m", 16, 5556, 1),
)

DEFORMABLE_RIGID_CASES = (
    ("sphere_dense", "sphere", 64, 128, 1, False),
    ("sphere_sparse", "sphere", 64, 128, 1, True),
    ("box_single_world", "box", 724, 1, 1, False),
    ("box_multi_shape", "box", 32, 128, 16, False),
    ("capsule", "capsule", 32, 64, 16, False),
    ("cylinder", "cylinder", 32, 64, 16, False),
    ("cone", "cone", 32, 64, 16, False),
    ("ellipsoid", "ellipsoid", 32, 64, 16, False),
    ("mesh_sdf", "mesh", 32, 64, 16, False),
    ("infinite_plane", "plane", 64, 128, 1, False),
    ("finite_plane", "finite_plane", 64, 128, 1, False),
    ("heightfield", "heightfield", 64, 128, 1, False),
    ("mixed", "mixed", 32, 64, 20, False),
    ("mixed_sparse", "mixed", 32, 64, 20, True),
    ("sphere_10m_rl", "sphere", 64, 1250, 1, False),
)


def _make_collision_grid(resolution, height):
    x, y = np.meshgrid(np.arange(resolution) * 0.01, np.arange(resolution) * 0.01)
    vertices = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height))).astype(np.float32)
    triangles = []
    for row in range(resolution - 1):
        for column in range(resolution - 1):
            lower = row * resolution + column
            triangles.extend(
                ((lower, lower + 1, lower + resolution), (lower + 1, lower + resolution + 1, lower + resolution))
            )
    return vertices, np.asarray(triangles, dtype=np.int32)


def _make_collision_world(scene, resolution):
    heights = (
        (0.0, 0.006, 0.012, 0.018)
        if scene.startswith("layered")
        else (0.0, 0.1 if scene.startswith("sparse") else 0.006)
    )
    vertices = []
    triangles = []
    vertex_offset = 0
    for layer, height in enumerate(heights):
        layer_vertices, layer_triangles = _make_collision_grid(resolution, height)
        if scene.startswith("folded") and layer == 1:
            extent = max((resolution - 1) * 0.01, 0.01)
            layer_vertices[:, 2] += 0.04 * (layer_vertices[:, 0] / extent - 0.5)
        vertices.append(layer_vertices)
        triangles.append(layer_triangles + vertex_offset)
        vertex_offset += len(layer_vertices)

    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    world.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=np.concatenate(vertices),
        indices=np.concatenate(triangles).reshape(-1),
        density=1.0,
        tri_ke=1.0,
        tri_ka=1.0,
        tri_kd=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    return world


def _make_deformable_rigid_world(kind, resolution, shape_count, sparse):
    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    extent = 2.0
    cloth_height = 4.0 if sparse else (0.02 if kind in ("plane", "finite_plane", "heightfield") else 0.45)
    world.add_cloth_grid(
        pos=wp.vec3(-0.5 * extent, -0.5 * extent, cloth_height),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=resolution,
        dim_y=resolution,
        cell_x=extent / resolution,
        cell_y=extent / resolution,
        mass=0.1,
        particle_radius=0.0,
    )

    side = int(np.ceil(np.sqrt(shape_count)))
    mesh = newton.Mesh.create_box(0.18, 0.18, 0.18) if kind in ("mesh", "mixed") else None
    heightfield = (
        newton.Heightfield(data=np.zeros((17, 17), dtype=np.float32), nrow=17, ncol=17, hx=0.5, hy=0.5)
        if kind in ("heightfield", "mixed")
        else None
    )
    shape_kinds = (
        "sphere",
        "box",
        "capsule",
        "cylinder",
        "cone",
        "ellipsoid",
        "mesh",
        "plane",
        "finite_plane",
        "heightfield",
    )
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    for shape_index in range(shape_count):
        row, column = divmod(shape_index, side)
        x = (column + 0.5) * extent / side - 0.5 * extent
        y = (row + 0.5) * extent / side - 0.5 * extent
        shape_kind = shape_kinds[shape_index % len(shape_kinds)] if kind == "mixed" else kind
        z = 0.42 if shape_kind in ("plane", "finite_plane", "heightfield") else 0.4
        xform = wp.transform(wp.vec3(x, y, z), wp.quat_identity())
        if shape_kind == "sphere":
            world.add_shape_sphere(body=-1, xform=xform, radius=0.5 / side, cfg=cfg)
        elif shape_kind == "box":
            world.add_shape_box(body=-1, xform=xform, hx=0.5 / side, hy=0.5 / side, hz=0.5, cfg=cfg)
        elif shape_kind == "capsule":
            world.add_shape_capsule(body=-1, xform=xform, radius=0.3 / side, half_height=0.3, cfg=cfg)
        elif shape_kind == "cylinder":
            world.add_shape_cylinder(body=-1, xform=xform, radius=0.4 / side, half_height=0.5, cfg=cfg)
        elif shape_kind == "cone":
            world.add_shape_cone(body=-1, xform=xform, radius=0.4 / side, half_height=0.5, cfg=cfg)
        elif shape_kind == "ellipsoid":
            world.add_shape_ellipsoid(body=-1, xform=xform, rx=0.5 / side, ry=0.35 / side, rz=0.5, cfg=cfg)
        elif shape_kind == "mesh":
            shape = world.add_shape_mesh(body=-1, xform=xform, mesh=mesh, scale=(1.0 / side,) * 3, cfg=cfg)
            world.shape_force_sdf[shape] = True
        elif shape_kind == "plane":
            world.add_shape_plane(body=-1, xform=xform, width=0.0, length=0.0, cfg=cfg)
        elif shape_kind == "finite_plane":
            world.add_shape_plane(body=-1, xform=xform, width=0.8 / side, length=0.8 / side, cfg=cfg)
        elif shape_kind == "heightfield":
            world.add_shape_heightfield(xform=xform, heightfield=heightfield, cfg=cfg)

    return world


class DeformableSelfCollision:
    """Benchmark dense self-collision in one large and many RL-style worlds."""

    params = (DEFORMABLE_COLLISION_CASES,)
    param_names = ["case"]
    repeat = pr_gate_repeat(5)
    number = 1

    def setup(self, case):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented

        scene, resolution, world_count, filter_threshold = case
        builder = newton.ModelBuilder()
        builder.replicate(_make_collision_world(scene, resolution), world_count)
        self.model = builder.finalize(device=device)
        self.detector = TriMeshCollisionDetector(
            self.model,
            init_collision_info=True,
            topological_contact_filter_threshold=filter_threshold,
            vertex_collision_buffer_pre_alloc=32,
            edge_collision_buffer_pre_alloc=64,
        )
        self.radius = 0.012
        self.launch_count = 20

        for _ in range(5):
            self._detect()
        with wp.ScopedCapture(device=device) as capture:
            self._detect()
        self.graph = capture.graph

    def _detect(self):
        self.detector.refit(self.model.particle_q)
        self.detector.vertex_triangle_collision_detection(self.radius)
        self.detector.edge_edge_collision_detection(self.radius)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_detect(self, case):
        for _ in range(self.launch_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


class DeformableRigidCollision:
    """Benchmark full-surface deformable contact against individual and mixed rigid geometry."""

    params = (DEFORMABLE_RIGID_CASES,)
    param_names = ["case"]
    repeat = pr_gate_repeat(5)
    number = 1
    timeout = 600

    def setup(self, case):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented

        _name, kind, resolution, world_count, shape_count, sparse = case
        builder = newton.ModelBuilder()
        builder.replicate(
            _make_deformable_rigid_world(kind, resolution, shape_count, sparse),
            world_count,
        )
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_gap=0.05,
            enable_rigid_soft_full_surface_contact=True,
            verify_buffers=False,
        )
        self.contacts = self.pipeline.contacts()
        self.launch_count = 20

        for _ in range(3):
            self.pipeline.collide(self.state, self.contacts)
        if kind == "mixed" and not sparse:
            contact_count = int(self.contacts.soft_contact_count.numpy()[0])
            contacted_shapes = np.unique(self.contacts.soft_contact_shape.numpy()[:contact_count])
            if len(contacted_shapes) != self.model.shape_count:
                raise RuntimeError(
                    f"mixed benchmark contacts {len(contacted_shapes)} of {self.model.shape_count} rigid shapes"
                )
        with wp.ScopedCapture(device=device) as capture:
            self.pipeline.collide(self.state, self.contacts)
        self.graph = capture.graph

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide(self, case):
        for _ in range(self.launch_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


class FastExampleClothManipulation:
    timeout = 300
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = 30
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        self.example = ExampleClothManipulation(ViewerNull(num_frames=self.num_frames), args)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, args=None)

        wp.synchronize_device()


class FastExampleClothTwist:
    repeat = pr_gate_repeat(5)
    number = 1

    def setup(self):
        self.num_frames = 100
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        self.example = ExampleClothTwist(ViewerNull(num_frames=self.num_frames), args)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, None)

        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "DeformableSelfCollision": DeformableSelfCollision,
        "DeformableRigidCollision": DeformableRigidCollision,
        "FastExampleClothManipulation": FastExampleClothManipulation,
        "FastExampleClothTwist": FastExampleClothTwist,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
