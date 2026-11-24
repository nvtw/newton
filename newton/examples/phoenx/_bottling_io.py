# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical sensor/drive I/O and read-only loading of source plant controllers."""

import importlib.machinery
import importlib.util
import sys
from typing import Any

import numpy as np
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


_PHOTOEYE_BATCH_SIZE = 256


@wp.kernel
def _photoeyes(
    body_q: wp.array[wp.transform],
    bottles: wp.array[int],
    bottle_count: wp.array[int],
    origins: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    distances: wp.array[float],
    hull_mesh: wp.array[wp.uint64],
    hull_xform: wp.array[wp.transform],
    hull_scale: wp.array[wp.vec3],
    stride: int,
    observation: wp.array2d[float],
):
    # Each warp tests one batch of bottles, so a sensor's scan spreads over
    # many blocks instead of serializing every bottle on 32 lanes.
    sensor, batch, lane = wp.tid()
    origin, direction, distance = origins[sensor], directions[sensor], distances[sensor]
    begin = batch * wp.static(_PHOTOEYE_BATCH_SIZE)
    end = wp.min(begin + wp.static(_PHOTOEYE_BATCH_SIZE), bottle_count[0])
    for i in range(begin + lane, end, 32):
        bottle = bottles[i]
        if _bottle_ray(body_q[bottle], origin, direction, distance, hull_mesh, hull_xform, hull_scale) <= distance:
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
    # The KR10's loaded moves (LOADED, LOADED_HOVER) lift the group out of the nest.
    lifting = mode[cell] == 4 or mode[cell] == 7
    observation[cell, 11] = float(lifting)
    observation[cell, 12] = float(not lifting or age[cell] >= 0.27)


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


_CASE_PHOTOEYE_BATCH_SIZE = 256


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
    sensor, batch, lane = wp.tid()
    begin = batch * wp.static(_CASE_PHOTOEYE_BATCH_SIZE)
    end = wp.min(begin + wp.static(_CASE_PHOTOEYE_BATCH_SIZE), shapes.shape[0])
    for i in range(begin + lane, end, 32):
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


@wp.kernel
def _gather(source: wp.array[Any], indices: wp.array[int], target: wp.array[Any]):
    i = wp.tid()
    target[i] = source[indices[i]]


class _HostLink:
    """Exchange host PLC I/O with the device without stalling the GPU.

    After each frame, the watched device arrays are copied into pinned host
    buffers behind an event. The host PLCs run on the previous frame's copy
    while the GPU computes the current frame, and their commands are queued as
    pinned uploads that the next frame consumes. Host PLCs therefore see one
    frame (one scan cycle) of additional sensor-to-actuator latency. Large
    arrays are watched as gathered subsets, scattered back on the host into
    full-size arrays whose unwatched entries stay NaN (or zero for integers).
    """

    def __init__(self, device):
        self.device = device
        self.frame = 0
        self.snapshot = -1
        self._sources = {}
        self._subsets = {}
        self._watched = None
        self._writes = {}
        self._events = tuple(wp.Event(device) for _ in range(2))

    def watch(self, name, array, indices=None):
        """Copy ``array`` to the host every frame, or only rows ``indices``."""
        if self._watched is not None:
            raise RuntimeError("Watch host I/O before the first frame")
        if indices is None:
            self._sources[name] = array
        else:
            entry = self._subsets.setdefault(name, (array, set()))
            entry[1].update(int(i) for i in np.ravel(indices))

    def _pinned(self, array):
        return tuple(wp.empty(array.shape, dtype=array.dtype, device="cpu", pinned=True) for _ in range(2))

    def gather(self):
        """Gather watched subsets on the device; graph-capturable."""
        if self._watched is None:
            self._watched = {name: (array, self._pinned(array), None) for name, array in self._sources.items()}
            for name, (array, wanted) in self._subsets.items():
                indices = np.asarray(sorted(wanted), dtype=np.int32)
                compact = wp.empty(len(indices), dtype=array.dtype, device=self.device)
                full = np.zeros(
                    array.numpy().shape, dtype=wp.dtype_to_numpy(getattr(array.dtype, "_wp_scalar_type_", array.dtype))
                )
                if np.issubdtype(full.dtype, np.floating):
                    full[:] = np.nan
                index_array = wp.array(indices, dtype=int, device=self.device)
                self._watched[name] = (compact, self._pinned(compact), (array, index_array, indices, full))
        for compact, _, subset in self._watched.values():
            if subset is not None:
                wp.launch(
                    _gather, dim=len(subset[2]), inputs=[subset[0], subset[1]], outputs=[compact], device=self.device
                )

    def publish(self):
        """Queue this frame's host copies behind the frame's device work."""
        slot = self.frame % 2
        for array, pinned, _ in self._watched.values():
            wp.copy(pinned[slot], array)
        wp.record_event(self._events[slot])
        self.frame += 1

    def latest(self):
        """Wait for, and return, the previous frame's copies."""
        slot = self.frame % 2
        wp.synchronize_event(self._events[slot])
        self.snapshot = self.frame - 2
        values = {}
        for name, (_, pinned, subset) in self._watched.items():
            values[name] = pinned[slot].numpy()
            if subset is not None:
                subset[3][subset[2]] = values[name]
                values[name] = subset[3]
        return values

    def observed(self, stamp):
        """Whether the latest copy reflects commands written at ``stamp = frame``."""
        return self.snapshot >= stamp

    def write(self, array, values):
        """Queue an upload for the next frame; call only after :meth:`latest`."""
        entry = self._writes.get(id(array))
        if entry is None:
            entry = self._writes[id(array)] = (array, self._pinned(array))
        pinned = entry[1][self.frame % 2]
        pinned.numpy()[...] = values
        wp.copy(array, pinned)
