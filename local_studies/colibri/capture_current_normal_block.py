"""Read actual solve buffers at current native frame245/substep1 relaxation."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio import SolverSetting, _build_scene
from newton._src.solvers.phoenx.body import inertia_sym6_unpack_np


def main():
    scene, _, _ = _build_scene(mass_ratio=400, setting=SolverSetting(8, 8), velocity_iterations=1, sor_boost=1)
    for _ in range(240):
        scene.step()
    w = scene.world
    frame = [240]
    reports = {}

    def snapshot(stage):
        cc = w._contact_container_solve if w._colored_contact_rows else w._contact_container
        cols = w._contact_cols_packed if w._colored_contact_headers else w._contact_cols
        if w._colored_contact_headers:
            active = int(w._num_active_constraints.numpy()[0])
            ids = w._partitioner.element_ids_by_color.numpy()[:active]
            slots = np.flatnonzero(ids >= w._contact_offset)
        else:
            slots = np.arange(int(w._ingest_scratch.num_contact_columns.numpy()[0]))
        headers = cols.data.numpy()[:, slots].copy()
        h = headers.view(np.int32)
        point_ids = []
        for c in range(len(slots)):
            first, count = h[5:7, c]
            h[5, c] = len(point_ids)
            point_ids.extend(range(first, first + count))
        assert len(point_ids) == len(set(point_ids))
        data = {
            "positions": w.bodies.position.numpy(),
            "velocity": w.bodies.velocity.numpy(),
            "angular_velocity": w.bodies.angular_velocity.numpy(),
            "inverse_mass": w.bodies.inverse_mass.numpy(),
            "inverse_inertia": inertia_sym6_unpack_np(w.bodies.inverse_inertia_world.numpy()),
            "headers": headers,
            "column_count": np.array([len(slots)]),
            "source_rows": np.array(point_ids),
            "source_slots": slots,
        }
        for name in ("derived", "lambdas", "impulses"):
            data[name] = getattr(cc, name).numpy()[:, point_ids]
        defects = []
        for c in range(len(slots)):
            a, b = h[1:3, c]
            for p in range(h[5, c], h[5, c] + h[6, c]):
                defects.append(
                    data["positions"][a].astype(float)
                    + data["derived"][9:12, p]
                    - data["positions"][b].astype(float)
                    - data["derived"][12:15, p]
                )
        reports[stage] = {
            "max_common_point_error_m": float(np.max(np.linalg.norm(defects, axis=1))),
            "point_count": len(point_ids),
            "packed_rows": w._colored_contact_rows,
            "packed_headers": w._colored_contact_headers,
            "mass_splitting": w.mass_splitting_enabled,
        }
        np.savez("/tmp/high_mass_current_245_1_" + stage + ".npz", **data)

    dispatcher = type(w._dispatcher)
    original = dispatcher.relax

    def wrapped(self, idt):
        selected = frame[0] == 245 and w._current_substep_index == 1
        if selected:
            snapshot("before")
        original(self, idt)
        if selected:
            snapshot("after")

    with patch.object(dispatcher, "relax", wrapped):
        for index in range(240, 246):
            frame[0] = index
            scene._simulate()
    assert set(reports) == {"before", "after"}
    paths = ["solver_phoenx.py", "constraints/constraint_contact_cloth.py", "constraints/contact_projection.py"]
    reports["source_sha256"] = {
        p: hashlib.sha256(Path("newton/_src/solvers/phoenx", p).read_bytes()).hexdigest() for p in paths
    }
    Path("/tmp/high_mass_current_245_1_capture.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
