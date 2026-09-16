"""Independent support-star coverage, ownership and overlap checks."""

import unittest

import numpy as np
import warp as wp

from local_studies.colibri.support_star_groups import regroup, topology
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactColumnContainer
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData


class TestSupportStarGroups(unittest.TestCase):
    def test_overlap_withdrawal_and_capacity_fallback(self):
        """Overlapping stars own each row once and retain original local order."""
        for device in ("cpu", "cuda:0"):
            for withdrawn in (False, True):
                for limit in (2, 64):
                    joints = [(1, 2), (1, 3), (2, 4)]
                    contact = [(1, 3 if withdrawn else 0)] * 9 + [(2, 0)] * 2 + [(3, 4)] * 2
                    edges = joints + contact
                    count = len(edges)
                    elements = wp.zeros(count, dtype=ElementInteractionData, device=device)
                    raw = elements.numpy()
                    raw["bodies"][:] = -1
                    for row, (a, b) in enumerate(edges):
                        raw["bodies"][row, :2] = [a if a else -1, b if b else -1]
                    elements.assign(raw)
                    headers = np.zeros((32, len(contact)), np.float32)
                    headers.view(np.int32)[1] = [a for a, b in contact]
                    headers.view(np.int32)[2] = [b for a, b in contact]
                    columns = ContactColumnContainer()
                    columns.data = wp.array(headers, device=device)
                    bodies = BodyContainer()
                    bodies.inverse_mass = wp.array([0, 1, 1, 1, 1], dtype=wp.float32, device=device)
                    active = wp.array([count], dtype=wp.int32, device=device)
                    data = topology.allocate(count, 5, device)
                    topology.build(data, elements, active, 4, device)
                    original_pid = data["row_partition"].numpy().copy()
                    original_color = data["row_color"].numpy().copy()
                    buffers = [wp.zeros(count + 7, dtype=wp.int32, device=device) for _ in range(5)]
                    parent, sizes, ids, stages, partitions = buffers
                    group_count = wp.zeros(1, dtype=wp.int32, device=device)
                    wp.launch(
                        regroup,
                        1,
                        [
                            elements,
                            active,
                            columns,
                            bodies,
                            len(joints),
                            len(joints),
                            4,
                            data["ids"],
                            data["starts"],
                            data["num_colors"],
                            data["row_partition"],
                            parent,
                            sizes,
                            ids,
                            stages,
                            partitions,
                            group_count,
                            limit,
                        ],
                        device=device,
                    )
                    got = ids.numpy()[:count]
                    np.testing.assert_array_equal(np.sort(got), np.arange(count))
                    pid = data["row_partition"].numpy()[:count]
                    starts = partitions.numpy()
                    offsets = stages.numpy()
                    for group in range(int(group_count.numpy()[0])):
                        ordered = []
                        for stage in range(starts[group], starts[group + 1]):
                            rows = got[offsets[stage] : offsets[stage + 1]]
                            endpoints = [b for row in rows for b in edges[row] if b]
                            self.assertEqual(len(endpoints), len(set(endpoints)))
                            self.assertTrue(np.all(pid[rows] == group))
                            ordered.extend(rows.tolist())
                        self.assertEqual(ordered, sorted(ordered, key=lambda row: (original_color[row], row)))
                    if limit == 2:
                        np.testing.assert_array_equal(pid, original_pid[:count])
                    elif not withdrawn:
                        self.assertEqual(len(set(pid[:14])), 1)
                        self.assertEqual(int(parent.numpy()[1]), int(parent.numpy()[2]))
                    else:
                        self.assertEqual(int(parent.numpy()[1]), -1)
                        self.assertEqual(pid[0], pid[2])
                        self.assertEqual(pid[0], pid[12])


if __name__ == "__main__":
    unittest.main()
