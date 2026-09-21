import argparse

import numpy as np
import warp as wp

from local_studies.colibri.phoenx_scene import Example
from newton.viewer import ViewerNull

e = Example(
    ViewerNull(), argparse.Namespace(body_count=15, mode="maximal", layout="single_world", substeps=1, iterations=8)
)
e.graph = None
e.frame_dt = 1.0 / 1200
for frame in range(20):
    e.step()
    cc = e.solver.world._contact_container
    n = int(e.contacts.rigid_contact_count.numpy()[0])
    print("GAPS", frame, cc.derived.numpy()[15, :n].min(), cc.derived.numpy()[15, :n].max(), flush=True)
    q = e.state_0.body_q.numpy()
    v = e.state_0.body_qd.numpy()
    print(frame, "contacts", e.contacts.rigid_contact_count.numpy(), "max_speed", np.nanmax(abs(v)), flush=True)
    if not np.isfinite(q).all() or not np.isfinite(v).all() or np.nanmax(abs(v)) > 100:
        print("BODY_SPEEDS", list(zip(e.model.body_label, np.linalg.norm(v, axis=1))), flush=True)
        direct = e.solver._direct_equality_system
        print(
            "REACTIONS",
            [
                (e.model.joint_label[int(j)], float(x))
                for j, x in zip(direct.topology.row_joint, direct.accumulated_impulse.numpy())
                if abs(x) > 10
            ],
            flush=True,
        )
        response = e.solver._direct_contact_response.data
        cc = e.solver.world._contact_container
        impulses = cc.impulses.numpy()[0]
        derived = cc.derived.numpy()
        mobility = response.mobility.numpy()
        bodies0 = response.contact_body0.numpy()
        bodies1 = response.contact_body1.numpy()
        for k in np.argsort(impulses)[-10:]:
            print(
                "CONTACT",
                int(k),
                "impulse",
                float(impulses[k]),
                "bodies",
                e.model.body_label[bodies0[k] - 1] if bodies0[k] > 0 else "world",
                e.model.body_label[bodies1[k] - 1] if bodies1[k] > 0 else "world",
                "mobility",
                mobility[:, k],
                "derived",
                derived[:, k],
                flush=True,
            )
        symbolic = direct.solver.symbolic
        size = len(direct.topology.row_joint)
        matrix = np.zeros((size, size))
        values = direct.matrix.numpy()[symbolic.matrix_storage]
        matrix[symbolic.matrix_row, symbolic.matrix_column] = values
        matrix[symbolic.matrix_column, symbolic.matrix_row] = values
        print("EIGENVALUES", np.linalg.eigvalsh(matrix), flush=True)
        for obj in [e.solver._direct_equality_system, getattr(e.solver._direct_contact_response, "data", None)]:
            if obj is None:
                continue
            for name, arr in vars(obj).items():
                if isinstance(arr, wp.array) and arr.dtype == wp.float32:
                    a = arr.numpy()
                    print(
                        type(obj).__name__,
                        name,
                        a.shape,
                        "finite",
                        np.isfinite(a).all(),
                        "range",
                        np.nanmin(a) if a.size else 0,
                        np.nanmax(a) if a.size else 0,
                        flush=True,
                    )
        break
