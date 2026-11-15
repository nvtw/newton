# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical sensor/drive I/O and read-only loading of source plant controllers."""

import importlib.machinery
import importlib.util
import sys

import warp as wp

import newton


class _ReadOnlyLoader(importlib.machinery.SourceFileLoader):
    def set_data(self, path, data, *, _mode=0o666):
        """Keep Python bytecode out of the downloaded asset tree."""


def _load_controller(name, path):
    """Import a source controller without writing into its asset directory."""
    spec = importlib.util.spec_from_file_location(name, path, loader=_ReadOnlyLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@wp.func
def _bottle_ray(
    pose: wp.transform,
    origin: wp.vec3,
    direction: wp.vec3,
    reach: float,
    hull_mesh: wp.array[wp.uint64],
    hull_xform: wp.array[wp.transform],
    hull_scale: wp.array[wp.vec3],
) -> float:
    inverse = wp.transform_inverse(pose)
    origin = wp.transform_point(inverse, origin)
    closest = reach + 1.0
    if wp.length(origin - wp.vec3(0.0, 0.0, 0.115)) <= 0.14 + reach:
        direction = wp.transform_vector(inverse, direction)
        for hull in range(hull_mesh.shape[0]):
            local = wp.transform_inverse(hull_xform[hull])
            query = wp.mesh_query_ray(
                hull_mesh[hull],
                wp.cw_div(wp.transform_point(local, origin), hull_scale[hull]),
                wp.cw_div(wp.transform_vector(local, direction), hull_scale[hull]),
                reach,
            )
            if query.result:
                closest = wp.min(closest, query.t)
    return closest


@wp.kernel
def _photoeyes(
    body_q: wp.array[wp.transform],
    bottles: wp.array[int],
    origins: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    distances: wp.array[float],
    hull_mesh: wp.array[wp.uint64],
    hull_xform: wp.array[wp.transform],
    hull_scale: wp.array[wp.vec3],
    stride: int,
    observation: wp.array2d[float],
):
    sensor, lane = wp.tid()
    origin, direction, distance = origins[sensor], directions[sensor], distances[sensor]
    for i in range(lane, bottles.shape[0], 32):
        if _bottle_ray(body_q[bottles[i]], origin, direction, distance, hull_mesh, hull_xform, hull_scale) <= distance:
            wp.atomic_max(observation, sensor / stride, sensor % stride, 1.0)
            break


@wp.kernel
def _actuator_observations(
    body_q: wp.array[wp.transform],
    actuator_body: wp.array2d[int],
    cell_inverse: wp.array[wp.transform],
    axes: wp.array[wp.vec3],
    actuator_zero: wp.array2d[float],
    mode: wp.array[int],
    age: wp.array[float],
    observation: wp.array2d[float],
):
    cell = wp.tid()
    for i in range(4):
        position = wp.transform_point(cell_inverse[cell], wp.transform_get_translation(body_q[actuator_body[cell, i]]))
        observation[cell, 7 + i] = wp.dot(position, axes[i]) - actuator_zero[cell, i]
    observation[cell, 11] = float(mode[cell] == 4)
    observation[cell, 12] = float(mode[cell] != 4 or age[cell] >= 0.27)


@wp.kernel
def _valves(
    actuator_dof: wp.array2d[int],
    target: wp.array2d[float],
    speeds: wp.array[float],
    joint_target_qd: wp.array[float],
):
    cell, axis = wp.tid()
    # Source pneumatic valves keep pressure on the cylinder at the stroke
    # limit. The authored force cap and joint stops determine its response.
    direction = 1.0 if target[cell, axis] > 0.0 else -1.0
    joint_target_qd[actuator_dof[cell, axis]] = direction * speeds[axis]


@wp.func
def _ray_box(origin: wp.vec3, direction: wp.vec3, lo: wp.vec3, hi: wp.vec3, distance: float) -> bool:
    near, far = float(0.0), distance
    for axis in range(3):
        if wp.abs(direction[axis]) < 1.0e-9:
            if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                return False
        else:
            a = (lo[axis] - origin[axis]) / direction[axis]
            b = (hi[axis] - origin[axis]) / direction[axis]
            near = wp.max(near, wp.min(a, b))
            far = wp.min(far, wp.max(a, b))
    return far >= near


@wp.kernel
def _case_photoeyes(
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_xform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_type: wp.array[int],
    shape_mesh: wp.array[wp.uint64],
    shapes: wp.array[int],
    bounds_lo: wp.array[wp.vec3],
    bounds_hi: wp.array[wp.vec3],
    origins: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    distances: wp.array[float],
    hits: wp.array[int],
):
    sensor, lane = wp.tid()
    for i in range(lane, shapes.shape[0], 32):
        shape = shapes[i]
        world = wp.transform_multiply(body_q[shape_body[shape]], shape_xform[shape])
        inverse = wp.transform_inverse(world)
        origin = wp.transform_point(inverse, origins[sensor])
        direction = wp.transform_vector(inverse, directions[sensor])
        if not _ray_box(origin, direction, bounds_lo[i], bounds_hi[i], distances[sensor]):
            continue
        if shape_type[shape] == int(newton.GeoType.BOX):
            wp.atomic_max(hits, sensor, 1)
            break
        query = wp.mesh_query_ray(
            shape_mesh[shape],
            wp.cw_div(origin, shape_scale[shape]),
            wp.cw_div(direction, shape_scale[shape]),
            distances[sensor],
        )
        if query.result:
            wp.atomic_max(hits, sensor, 1)
            break


@wp.kernel
def _case_drives(
    dofs: wp.array[int],
    coordinates: wp.array[int],
    position: wp.array[float],
    velocity: wp.array[float],
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    i = wp.tid()
    target_q[coordinates[i]] = position[i]
    target_qd[dofs[i]] = velocity[i]
