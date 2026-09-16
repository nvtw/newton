"""Save current public contact columns after a warmed trajectory."""

import json
import runpy

import numpy as np

import newton.examples

original_run = newton.examples.run


def save_columns(example, args):
    try:
        return original_run(example, args)
    finally:
        world = example.solver.world
        headers = world._contact_cols.data.numpy()
        impulses = world._contact_container.impulses.numpy()
        integers = headers.view(np.int32)
        rows = []
        for cid in range(int(world._ingest_scratch.num_contact_columns.numpy()[0])):
            start, count = int(integers[5, cid]), int(integers[6, cid])
            if count > 0:
                active = np.any(impulses[:, start : start + count] != 0, axis=0)
                rows.append({"cid": cid, "count": count, "nonzero": int(active.sum())})
        report = {
            "columns": len(rows),
            "all_zero_columns": sum(row["nonzero"] == 0 for row in rows),
            "points": sum(row["count"] for row in rows),
            "nonzero_points": sum(row["nonzero"] for row in rows),
            "rows": rows,
        }
        with open("/tmp/colibri_warm_column_population.json", "w") as stream:
            json.dump(report, stream, indent=2)
        np.savez("/tmp/colibri_warm_column_snapshot.npz", headers=headers, impulses=impulses)
        print("COLUMN_POPULATION", {k: v for k, v in report.items() if k != "rows"}, flush=True)


newton.examples.run = save_columns
runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
