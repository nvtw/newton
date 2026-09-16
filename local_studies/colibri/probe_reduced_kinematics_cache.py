# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare published reduced frames with redundant reconstruction, eagerly."""

import argparse
import json
from pathlib import Path

import numpy as np

from newton.examples.robot.example_robot_policy import Example
from newton.viewer import ViewerNull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = Example.create_parser().parse_args(["--robot", "g1_29dof", "--solver", "phoenx"])
    example = Example(ViewerNull(), config)
    bridge = example.solver._reduced_articulation
    system = bridge.system
    original = system.update_local_kinematics
    fields = ("articulation_origin", "body_q_local", "body_q_com", "joint_anchor_local")
    summary = {name: {"different_calls": 0, "max_abs_difference": 0.0} for name in fields}
    current_calls = 0

    def inspect_cache(state):
        nonlocal current_calls
        if not bridge._kinematics_current:
            return original(state)
        before = {name: getattr(system, name).numpy() for name in fields}
        original(state)
        current_calls += 1
        for name in fields:
            after = getattr(system, name).numpy()
            if not np.array_equal(before[name], after):
                summary[name]["different_calls"] += 1
                summary[name]["max_abs_difference"] = max(
                    summary[name]["max_abs_difference"], float(np.max(np.abs(before[name] - after)))
                )
                if summary[name]["different_calls"] == 1:
                    np.savez(
                        args.output.with_name(args.output.stem + "_" + name + ".npz"),
                        before=before[name],
                        after=after,
                        joint_q=state.joint_q.numpy(),
                        body_q=state.body_q.numpy(),
                    )

    system.update_local_kinematics = inspect_cache
    example.graph = None
    for _ in range(args.frames):
        example.step()
    example.test_final()
    report = {"current_cache_checks": current_calls, "fields": summary, "frames": args.frames}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
