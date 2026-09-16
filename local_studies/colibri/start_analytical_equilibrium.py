"""Start the two-body diagnostic at an independently certified static pose.

This changes initial conditions only. It does not pin the base, reset poses
after stepping, seed contact impulses, or change any authored physical law.
Finite normal compliance can still require a small initial settling motion.
"""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.kamino import example_kamino_colibri as source


def main():
    """Validate the force certificate and initialize the unrestrained bodies."""
    certificate_path = Path("/tmp/colibri_static_certificate_exact_flat-base_analytical_equilibrium.npz")
    with np.load(certificate_path) as certificate:
        q = certificate["q"].copy()
        contacts = certificate["C"]
        joints = certificate["B"]
        witness = certificate["extrema_witness"][0]
        count = contacts.shape[0]
        force = witness[:count].reshape(-1, 3)
        residual = contacts.T @ witness[:count] + joints.T @ witness[count:] + certificate["gravity"]
        cone_excess = np.linalg.norm(force[:, 1:], axis=1) - certificate["mu"] * force[:, 0]
        assert not certificate["is_outer_polygon"]
        assert np.max(np.abs(residual)) < 1e-12
        assert np.max(cone_excess) < 1e-12 and np.min(force[:, 0]) >= -1e-12
        assert np.linalg.norm(certificate["pivots"][0] - certificate["pivots"][1]) < 1e-12

    original = source.build_scene
    metadata = {
        "certificate": str(certificate_path),
        "force_balance_error_N": float(np.max(np.abs(residual))),
        "coulomb_cone_excess_N": float(np.max(cone_excess)),
        "scope": "Independent static initial pose only; all bodies remain dynamic; unseeded contact history",
    }

    def build_scene(*args, **kwargs):
        assert kwargs["body_count"] == 2 and not kwargs["fix_base"]
        builder = original(*args, **kwargs)
        for label, pose in zip(("FrameGround", "Frame"), q, strict=True):
            body = builder.body_label.index(label)
            assert builder.body_mass[body] > 0.0
            builder.body_q[body] = wp.transform(wp.vec3(pose[:3]), wp.quat(pose[3:]))
        metadata["initial_q_float32"] = [
            np.asarray(builder.body_q[builder.body_label.index(label)]).tolist() for label in ("FrameGround", "Frame")
        ]
        return builder

    module, *arguments = sys.argv[1:]
    sys.argv = [module, *arguments]
    source.build_scene = build_scene
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        source.build_scene = original
        if "--output" in arguments:
            Path(arguments[arguments.index("--output") + 1]).with_suffix(".equilibrium_start.json").write_text(
                json.dumps(metadata, indent=2)
            )


if __name__ == "__main__":
    main()
