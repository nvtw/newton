"""Capture the existing post-integration relaxation without altering physics."""

from unittest.mock import patch

import numpy as np

from newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio import SolverSetting, _build_scene
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np


def main():
    scene, _, _ = _build_scene(mass_ratio=400, setting=SolverSetting(8, 8), velocity_iterations=1, sor_boost=1)
    for _ in range(240):
        scene.step()
    world = scene.world
    cc = world._contact_container
    frame = [240]

    def snapshot(suffix):
        n = int(world._cc_valid_count.numpy()[0])
        np.savez(
            "/tmp/high_mass_relax_" + suffix + ".npz",
            positions=world.bodies.position.numpy(),
            velocity=world.bodies.velocity.numpy(),
            angular_velocity=world.bodies.angular_velocity.numpy(),
            inverse_mass=world.bodies.inverse_mass.numpy(),
            inverse_inertia=inertia_sym6_unpack_np(world.bodies.inverse_inertia_world.numpy()),
            impulses=cc.impulses.numpy()[:, :n],
            derived=cc.derived.numpy()[:, :n],
            lambdas=cc.lambdas.numpy()[:, :n],
            headers=world._contact_cols.data.numpy(),
            column_count=world._ingest_scratch.num_contact_columns.numpy(),
        )

    dispatcher = type(world._dispatcher)
    original = dispatcher.relax

    def wrapped(self, idt):
        capture = frame[0] == 244 and world._current_substep_index == 0
        if capture:
            snapshot("before")
        original(self, idt)
        if capture:
            snapshot("after")

    with patch.object(dispatcher, "relax", wrapped):
        for index in range(240, 245):
            frame[0] = index
            scene._simulate()
    print("Captured unchanged frame244/substep0 before and after relaxation")


if __name__ == "__main__":
    main()
