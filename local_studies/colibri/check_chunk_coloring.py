# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate actual colored endpoint exclusivity during a Colibri screen."""

import numpy as np

from local_studies.colibri import check_bilateral_pgs as runner

original = runner.Example.test_post_step


def checked(self):
    original(self)
    world = self.solver.world
    active = int(world._num_active_constraints.numpy()[0])
    partitioner = world._partitioner
    colors = int(partitioner.num_colors.numpy()[0])
    starts = partitioner.color_starts.numpy()
    ids = partitioner.element_ids_by_color.numpy()
    elements = world._elements.numpy()["bodies"]
    all_ids = []
    for color in range(min(colors, world.max_colored_partitions)):
        seen = {}
        for raw_cid in ids[starts[color] : starts[color + 1]]:
            cid = int(raw_cid)
            assert 0 <= cid < active, (color, cid, active)
            all_ids.append(cid)
            for node in {int(value) for value in elements[cid] if value >= 0}:
                assert node not in seen, f"Color {color} shares node {node}: constraints {seen.get(node)} and {cid}"
                seen[node] = cid
    assert len(all_ids) == len(set(all_ids)), "A constraint appears twice in colored partitions"
    count = int(world._ingest_scratch.num_contact_columns.numpy()[0])
    data = world._contact_cols.data.numpy()
    # Use the public kernel accessors through the same offsets they encode.
    from newton._src.solvers.phoenx.constraints.constraint_contact import _OFF_CONTACT_COUNT, _OFF_CONTACT_FIRST

    first = np.ascontiguousarray(data[_OFF_CONTACT_FIRST]).view(np.int32)
    length = np.ascontiguousarray(data[_OFF_CONTACT_COUNT]).view(np.int32)
    point_owner = world._cid_of_contact_cur.numpy()
    touched = set()
    for column in range(count):
        for point in range(int(first[column]), int(first[column] + length[column])):
            assert point not in touched, f"Point {point} is assigned to multiple columns"
            touched.add(point)
            assert point_owner[point] == world._contact_offset + column, (
                point,
                int(point_owner[point]),
                world._contact_offset + column,
            )


runner.Example.test_post_step = checked

if __name__ == "__main__":
    runner.main()
