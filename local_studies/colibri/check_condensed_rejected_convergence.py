"""Replay the rejected live contact operator with unchanged native metric PGS."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri.check_d6_frozen_rows import contact_sweep
from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def main():
    """Separate a finite sweep budget from numerical stagnation without changing physics."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="/tmp/colibri_two_body_condensed_combined_live60.rejected.npz")
    parser.add_argument("--output",default="/tmp/colibri_condensed_rejected_convergence.json")
    parser.add_argument("--max-sweeps",type=int,default=256)
    args=parser.parse_args()
    source=args.source
    z = np.load(source)
    a = {k: z[k] for k in z.files}
    c, p = a["C"], a["P"]
    g = p @ c.T
    h = c @ g
    lam = a["old"].copy()
    v = a["vbar"] + g @ lam
    arrays = [
        wp.array(x, dtype=wp.float64, device="cpu")
        for x in (c, g, h, a["rhs"] - c @ a["vbar"], a["gamma"], a["mu"], lam, v)
    ]
    evaluate, *_ = natural_map_evaluator(a["A"], a["rhs"], a["gamma"], a["mu"])
    records = []
    for sweep in range(1, args.max_sweeps+1):
        wp.launch(contact_sweep, dim=1, inputs=[*arrays, 0, len(a["mu"]), 12], device="cpu")
        if sweep in (32, 64, 128, 256, 512, 1024):
            lam, v = arrays[-2].numpy(), arrays[-1].numpy()
            r = evaluate(lam)[0].reshape(-1, 3)
            joint = np.linalg.solve(a["K"], a["targets"] - a["B"] @ a["free"] - a["B"] @ a["W"] @ c.T @ lam)
            record = {
                "sweeps": sweep,
                "residual": float(np.max(abs(r))),
                "normal": float(np.max(abs(r[:, 0]))),
                "tangent": float(np.max(abs(r[:, 1:]))),
                "joint": float(np.max(abs(a["B"] @ v + a["diagonal"] * joint - a["targets"]))),
                "response": float(np.max(abs(v - a["free"] - a["W"] @ (c.T @ lam + a["B"].T @ joint)))),
            }
            records.append(record)
            print(record, flush=True)
    Path(args.output).write_text(
        json.dumps({"source": source, "records": records}, indent=2)
    )


if __name__ == "__main__":
    main()
