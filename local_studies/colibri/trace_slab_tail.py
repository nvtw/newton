# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Trace generation/transport and actual copy assignment at the slab tail failure."""

import inspect

import numpy as np
import warp as wp

from local_studies.colibri import check_slab_colibri as slab
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer

slab.runner.Example = slab.SlabExample
from local_studies.colibri.trace_late_contacts import TracedExample, trace  # noqa: E402


@wp.kernel
def capture_copies(
    counter: wp.array[wp.int32],
    count: wp.array[wp.int32],
    copies: wp.array[wp.int32],
    owner: wp.array[wp.int32],
    row_slab: wp.array[wp.int32],
    cc: ContactContainer,
    copy_history: wp.array2d[wp.int32],
    slab_history: wp.array2d[wp.int32],
    mobility_history: wp.array2d[wp.vec2f],
):
    index = wp.tid()
    slot = (counter[0] - 1) % copy_history.shape[0]
    if index < copies.shape[0]:
        copy_history[slot, index] = copies[index]
    if index < count[0]:
        slab_history[slot, index] = row_slab[owner[index]]
        mobility_history[slot, index] = wp.vec2f(cc.derived[0, index], cc.derived[3, index])


class TracedSlab(TracedExample):
    test_post_step = slab.SlabExample.test_post_step

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        buffers = trace["buffers"]
        slots = 256
        for name, value in list(buffers.items()):
            shape = (slots, *value.shape[1:])
            buffers[name] = wp.zeros(shape, dtype=value.dtype, device=self.model.device)
        buffers["step_ids"].fill_(-1)
        world = self.solver.world
        points = world.rigid_contact_max
        copies = wp.zeros((slots, world.num_bodies), dtype=wp.int32, device=self.model.device)
        slabs = wp.zeros((slots, points), dtype=wp.int32, device=self.model.device)
        mobility = wp.zeros((slots, points), dtype=wp.vec2f, device=self.model.device)
        original = self.solver.step
        counter = inspect.getclosurevars(original).nonlocals["counter"]

        def step(*args, **kwargs):
            original(*args, **kwargs)
            wp.launch(
                capture_copies,
                max(points, world.num_bodies),
                [
                    counter,
                    world._contact_views.rigid_contact_count,
                    world._copy_state.count_per_node,
                    world._cid_of_contact_cur,
                    self.solver._slab_reference["row_slab"],
                    world._contact_container,
                    copies,
                    slabs,
                    mobility,
                ],
                device=self.model.device,
            )

        self.solver.step = step
        buffers.update(body_copy_counts=copies, point_slab=slabs, normal_mass_bias=mobility)


if __name__ == "__main__":
    slab.runner.install_fused = slab.install_fused
    slab.runner.Example = TracedSlab
    try:
        slab.runner.main()
    finally:
        if trace:
            model = trace["model"]
            np.savez_compressed(
                trace["output"],
                **{key: value.numpy() for key, value in trace["buffers"].items()},
                body_labels=model.body_label,
                shape_labels=model.shape_label,
                shape_body=model.shape_body.numpy(),
                body_com=model.body_com.numpy(),
                shape_gap=model.shape_gap.numpy(),
            )
