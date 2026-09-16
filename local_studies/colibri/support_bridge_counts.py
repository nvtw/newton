"""Reconstruct and verify captured rigid copy counts before a frozen bridge."""

import numpy as np
import warp as wp

from local_studies.colibri.support_star_groups import regroup, topology
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints import constraint_joint as schema
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactColumnContainer
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData


def counts(snapshot="/tmp/colibri_support_relax330.npz"):
    d = np.load(snapshot)
    joints = int(d["num_joints"][0])
    columns = int(d["column_count"][0])
    n = joints + columns
    bodies = len(d["inverse_mass"])
    a = d["joint_data"].view(np.int32)
    h = d["headers"].view(np.int32)
    ends = []
    for j in range(joints):
        ends.append(a[[int(schema._OFF_BODY1), int(schema._OFF_BODY2)], j])
    for j in range(columns):
        ends.append(h[1:3, j])
    e = wp.zeros(n, dtype=ElementInteractionData, device="cpu")
    raw = e.numpy()
    raw["bodies"][:] = -1
    for row, pair in enumerate(ends):
        raw["bodies"][row, :2] = [int(b) if b > 0 else -1 for b in pair]
    e.assign(raw)
    active = wp.array([n], dtype=wp.int32, device="cpu")
    data = topology.allocate(n, bodies, "cpu")
    topology.build(data, e, active, 4, "cpu", rigid_only=True)

    def derive(pid):
        return np.array(
            [len({int(pid[row]) for row in range(n) if b in raw["bodies"][row]}) for b in range(bodies)], np.int32
        )

    original = derive(data["row_partition"].numpy())
    np.testing.assert_array_equal(original, d["copy_count"])
    c = ContactColumnContainer()
    c.data = wp.array(d["headers"], device="cpu")
    b = BodyContainer()
    b.inverse_mass = wp.array(d["inverse_mass"], device="cpu")
    size = n + bodies + 2
    buffers = [wp.zeros(size, dtype=wp.int32, device="cpu") for _ in range(5)]
    count = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(
        regroup,
        1,
        [
            e,
            active,
            c,
            b,
            joints,
            joints,
            4,
            data["ids"],
            data["starts"],
            data["num_colors"],
            data["row_partition"],
            *buffers,
            count,
            64,
        ],
        device="cpu",
    )
    star = derive(data["row_partition"].numpy())
    print("COPY_COUNT_PROOF", original[:3], star[:3], flush=True)
    np.savez(
        "/tmp/colibri_support_bridge_counts.npz",
        original=original,
        star=star,
        row_partition=data["row_partition"].numpy(),
        elements=raw,
    )
    return star


if __name__ == "__main__":
    counts()
