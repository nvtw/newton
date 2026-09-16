# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local diagnostic for mesh candidates within the existing search envelope."""

from functools import wraps

import warp as wp

from newton._src.geometry import narrow_phase
from newton._src.geometry.contact_data import ContactData, prepare_speculative_contact
from newton._src.sim.collide import ContactWriterData, _write_contact_at_index


@wp.func
def _write_geometric_envelope_contact(contact: ContactData, writer: ContactWriterData, output_index: int):
    normal, point_a, point_b, separation = prepare_speculative_contact(contact)
    index = output_index
    if index < 0:
        if separation > contact.gap_sum:
            return
        index = wp.atomic_add(writer.contact_count, 0, 1)
    _write_contact_at_index(contact, writer, index, point_a, point_b, normal)


def install_conservative_mesh_candidates():
    """Admit both velocity signs in reduced mesh/SDF search candidates.

    The pipeline still computes its original velocity-expanded search gaps.
    Its physical shape gaps, witness distances and solver equations are untouched.
    Reduced export also uses geometric admission; primitive direct writers
    keep their original policy. This installer is diagnostic, not a public API.
    """
    original = narrow_phase.create_narrow_phase_process_mesh_mesh_contacts_kernel
    original_export = narrow_phase.create_export_reduced_contacts_kernel

    @wraps(original)
    def create_kernel(*args, **kwargs):
        if kwargs.get("reduce_contacts", False) and kwargs.get("speculative", False):
            kwargs = dict(kwargs)
            kwargs["speculative"] = False
        return original(*args, **kwargs)

    narrow_phase.create_narrow_phase_process_mesh_mesh_contacts_kernel = create_kernel

    def create_export(_writer):
        return original_export(_write_geometric_envelope_contact)

    narrow_phase.create_export_reduced_contacts_kernel = create_export

    def restore():
        narrow_phase.create_narrow_phase_process_mesh_mesh_contacts_kernel = original
        narrow_phase.create_export_reduced_contacts_kernel = original_export

    return restore
