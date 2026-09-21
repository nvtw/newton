# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Capture exact 120 Hz generation poses and contacts around a late failure."""

from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import ContactViews
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer


@wp.kernel
def _capture_before(
    counter: wp.array[wp.int32],
    q: wp.array[wp.transformf],
    qd: wp.array[wp.spatial_vectorf],
    q_history: wp.array2d[wp.transformf],
    qd_history: wp.array2d[wp.spatial_vectorf],
):
    i = wp.tid()
    slot = counter[0] % q_history.shape[0]
    q_history[slot, i] = q[i]
    qd_history[slot, i] = qd[i]


@wp.kernel
def _capture_after(
    counter: wp.array[wp.int32],
    q: wp.array[wp.transformf],
    qd: wp.array[wp.spatial_vectorf],
    q_history: wp.array2d[wp.transformf],
    qd_history: wp.array2d[wp.spatial_vectorf],
    views: ContactViews,
    cc: ContactContainer,
    step_ids: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    shapes: wp.array2d[wp.vec2i],
    point0: wp.array2d[wp.vec3f],
    point1: wp.array2d[wp.vec3f],
    normals: wp.array2d[wp.vec3f],
    margins: wp.array2d[wp.vec2f],
    impulses: wp.array2d[wp.vec3f],
):
    i = wp.tid()
    step = counter[0]
    slot = step % q_history.shape[0]
    count = wp.min(views.rigid_contact_count[0], point0.shape[1])
    if i == 0:
        step_ids[slot] = step
        counts[slot] = count
    if i < q.shape[0]:
        q_history[slot, i] = q[i]
        qd_history[slot, i] = qd[i]
    if i < count:
        shapes[slot, i] = wp.vec2i(views.rigid_contact_shape0[i], views.rigid_contact_shape1[i])
        point0[slot, i] = views.rigid_contact_point0[i]
        point1[slot, i] = views.rigid_contact_point1[i]
        normals[slot, i] = views.rigid_contact_normal[i]
        margins[slot, i] = wp.vec2f(views.rigid_contact_margin0[i], views.rigid_contact_margin1[i])
        impulses[slot, i] = wp.vec3f(cc.impulses[0, i], cc.impulses[1, i], cc.impulses[2, i])


@wp.kernel
def _advance(counter: wp.array[wp.int32]):
    counter[0] += 1


import runpy

from newton._src.solvers.phoenx.solver import SolverPhoenX

trace = {}


original_init = SolverPhoenX.__init__


class TracedExample:
    def __init__(self, model, *args, **kwargs):
        print("TRACE CONSTRUCTOR ACTIVE", flush=True)
        original_init(self, model, *args, **kwargs)
        self.solver = self
        self.model = model
        world = self.solver.world
        device = self.model.device
        slots, bodies, points = 128, self.model.body_count, world.rigid_contact_max
        buffers = {
            "pre_q": wp.zeros((slots, bodies), dtype=wp.transformf, device=device),
            "pre_qd": wp.zeros((slots, bodies), dtype=wp.spatial_vectorf, device=device),
            "post_q": wp.zeros((slots, bodies), dtype=wp.transformf, device=device),
            "post_qd": wp.zeros((slots, bodies), dtype=wp.spatial_vectorf, device=device),
            "step_ids": wp.full(slots, -1, dtype=wp.int32, device=device),
            "counts": wp.zeros(slots, dtype=wp.int32, device=device),
            "shapes": wp.zeros((slots, points), dtype=wp.vec2i, device=device),
            "point0": wp.zeros((slots, points), dtype=wp.vec3f, device=device),
            "point1": wp.zeros((slots, points), dtype=wp.vec3f, device=device),
            "normals": wp.zeros((slots, points), dtype=wp.vec3f, device=device),
            "margins": wp.zeros((slots, points), dtype=wp.vec2f, device=device),
            "impulses": wp.zeros((slots, points), dtype=wp.vec3f, device=device),
        }
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        original_step = self.solver.step

        def traced_step(state_in, state_out, control, contacts, dt):
            wp.launch(
                _capture_before,
                bodies,
                [counter, state_in.body_q, state_in.body_qd, buffers["pre_q"], buffers["pre_qd"]],
                device=device,
            )
            original_step(state_in, state_out, control, contacts, dt)
            wp.launch(
                _capture_after,
                max(bodies, points),
                [
                    counter,
                    state_out.body_q,
                    state_out.body_qd,
                    buffers["post_q"],
                    buffers["post_qd"],
                    world._contact_views,
                    world._contact_container,
                    buffers["step_ids"],
                    buffers["counts"],
                    buffers["shapes"],
                    buffers["point0"],
                    buffers["point1"],
                    buffers["normals"],
                    buffers["margins"],
                    buffers["impulses"],
                ],
                device=device,
            )
            wp.launch(_advance, 1, [counter], device=device)

        self.solver.step = traced_step
        trace.update(
            buffers=buffers, model=self.model, output=Path("/tmp/colibri_public_group4_tail.trace.npz"), example=self
        )


SolverPhoenX.__init__ = TracedExample.__init__
try:
    runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
finally:
    print("TRACE SAVED", bool(trace), flush=True)
    if trace:
        model = trace["model"]
        np.savez_compressed(
            trace["output"],
            **{k: v.numpy() for k, v in trace["buffers"].items()},
            body_labels=model.body_label,
            shape_labels=model.shape_label,
            shape_body=model.shape_body.numpy(),
            body_com=model.body_com.numpy(),
            shape_gap=model.shape_gap.numpy(),
            body_mass=model.body_mass.numpy(),
            body_inertia=model.body_inertia.numpy(),
            copy_counts=trace["example"].world._copy_state.count_per_node.numpy(),
        )
