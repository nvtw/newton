"""Quick reproducer for the nut-bolt explosion at scale=1."""

from __future__ import annotations

import sys
import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.examples.example_phoenx_nut_bolt import Example


def main() -> None:
    sys.argv = [
        "_debug_nutbolt",
        "--viewer",
        "null",
        "--num-frames",
        "1",
        "--scene-scale",
        "1.0",
    ]
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    ex = Example(viewer, args)

    print(f"bolt_body={ex._bolt_body}, nut_body={ex._nut_body}")
    for frame in range(300):
        ex.step()
        body_q = ex.state.body_q.numpy()
        body_qd = ex.state.body_qd.numpy()
        nut_pos = body_q[ex._nut_body, :3]
        nut_vel = body_qd[ex._nut_body, :3]
        nut_ang = body_qd[ex._nut_body, 3:]
        if frame % 20 == 0 or frame == 299:
            xy_dist = float(np.sqrt(nut_pos[0] ** 2 + nut_pos[1] ** 2))
            print(
                f"frame {frame:3d}: nut_pos={nut_pos}, "
                f"xy={xy_dist:.5f}, "
                f"|v|={np.linalg.norm(nut_vel):.3f}, "
                f"|w|={np.linalg.norm(nut_ang):.3f}"
            )
        if not np.isfinite(nut_pos).all():
            print(f"NAN at frame {frame}")
            break


if __name__ == "__main__":
    main()
