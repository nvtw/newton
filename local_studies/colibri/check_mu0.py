# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run the common full-scene validator and retain state for exact A/B checks."""

import sys
from pathlib import Path

import numpy as np

from local_studies.colibri import check_bilateral_pgs

_instances = []
_original_example = check_bilateral_pgs.Example


class _SavedExample(_original_example):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        _instances.append(self)


if __name__ == "__main__":
    check_bilateral_pgs.Example = _SavedExample
    check_bilateral_pgs.main()
    example = _instances[0]
    destination = Path(sys.argv[sys.argv.index("--output") + 1]).with_suffix(".npz")
    count = int(example.contacts.rigid_contact_count.numpy()[0])
    world = example.solver.world
    cc = world._contact_container
    np.savez(
        destination,
        q=example.state_0.body_q.numpy(),
        qd=example.state_0.body_qd.numpy(),
        impulses=cc.impulses.numpy()[:, :count],
        references=cc.lambdas.numpy()[:, :count],
        derived=cc.derived.numpy()[:, :count],
        constraint_bodies=world._elements.numpy()["bodies"],
        active_constraint_count=world._num_active_constraints.numpy(),
        color_starts=world._partitioner.color_starts.numpy(),
        color_ids=world._partitioner.element_ids_by_color.numpy(),
        num_colors=world._partitioner.num_colors.numpy(),
        contact_column_data=world._contact_cols.data.numpy(),
        contact_column_count=world._ingest_scratch.num_contact_columns.numpy(),
        joint_count=np.asarray([world.num_joints]),
    )
