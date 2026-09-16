"""Opt-in parallel contact geometry followed by unchanged colored warm starts."""

import functools
import types

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import _make_contact_prepare_for_iteration_at
from newton._src.solvers.phoenx.constraints.constraint_container import constraint_bodies_make
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer


@functools.cache
def make_kernel(has_soft_pd):
    prepare_point = _make_contact_prepare_for_iteration_at(
        cloth_support=False,
        has_mass_splitting=False,
        has_soft_contact_pd=has_soft_pd,
        pointwise_geometry=True,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        columns: ContactColumnContainer,
        num_columns: wp.array[wp.int32],
        bodies: BodyContainer,
        particles: ParticleContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
    ):
        cid, lane = wp.tid()
        if cid >= num_columns[0]:
            return
        count = contact_get_contact_count(columns, cid)
        first = contact_get_contact_first(columns, cid)
        pair = constraint_bodies_make(contact_get_body1(columns, cid), contact_get_body2(columns, cid))
        for offset in range(lane, count, 128):
            prepare_point(
                columns,
                cid,
                first + offset,
                bodies,
                particles,
                num_bodies,
                pair,
                idt,
                cc,
                contacts,
                copy_state,
                wp.int32(0),
            )

    return kernel


def install(solver):
    w = solver.world
    if (
        w.step_layout != "single_world"
        or w.mass_splitting_enabled
        or w._colored_contact_headers
        or w._contact_patch_enabled
        or w.num_cloth_triangles
        or w.num_cloth_bending
        or w.num_soft_tetrahedra
        or w.num_soft_hexahedra
    ):
        raise ValueError("Parallel geometry prototype requires unsplit nonpatch rigid single-world.")
    original = w._singleworld_head_plus_tail_sweep
    original_prepare = w._singleworld_kernels()[0]
    cached_head, cached_tail = w._singleworld_cached_prepare_kernels()
    kernel = make_kernel(w._dispatch_specialization_flags()["has_soft_contact_pd"])

    def sweep(self, head, tail, idt, contact_container=None):
        if head is original_prepare:
            cc = self._contact_container if contact_container is None else contact_container
            wp.launch(
                kernel,
                dim=(self.max_contact_columns, 128),
                inputs=[
                    self._contact_cols,
                    self._ingest_scratch.num_contact_columns,
                    self.bodies,
                    self._particles_or_sentinel(),
                    wp.int32(self.num_bodies),
                    idt,
                    cc,
                    self._active_contact_views(),
                    self._copy_state,
                ],
                device=self.device,
            )
            return original(cached_head, cached_tail, idt, contact_container)
        return original(head, tail, idt, contact_container)

    w._singleworld_head_plus_tail_sweep = types.MethodType(sweep, w)
