"""Run the existing cached G1 ONNX controller without changing solver settings."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.robot.example_robot_policy import Example
from newton.viewer import ViewerNull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-frames", type=int, default=0)
    args = parser.parse_args()
    wp.init()
    config = Example.create_parser().parse_args(["--robot", "g1_29dof", "--solver", "phoenx"])
    example = Example(ViewerNull(), config)
    history = []
    poses = []
    velocities = []
    failure = None
    for frame in range(args.frames):
        example.step()
        q = example.state_0.body_q.numpy()
        qd = example.state_0.body_qd.numpy()
        action = example._prev_act_wp.numpy()
        poses.append(q)
        velocities.append(qd)
        finite = all(np.isfinite(a).all() for a in (q, qd, action))
        history.append(
            {
                "frame": frame,
                "finite": bool(finite),
                "base_height_m": float(q[0, 2]),
                "minimum_body_origin_height_m": float(q[:, 2].min()),
                "max_angular_speed_rad_s": float(np.linalg.norm(qd[:, 3:], axis=1).max()),
                "max_action": float(np.abs(action).max()),
            }
        )
        if not finite:
            failure = f"Nonfinite state/controller output at frame {frame}"
            break
    try:
        example.test_final()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    result = {
        "status": "failed" if failure else "passed",
        "failure": failure,
        "policy": "cached mjw_g1_29DOF.onnx",
        "command": [0, 0, 0],
        "physics_seconds": len(history) * example.frame_dt * example.decimation,
        "history": history,
    }
    if args.benchmark_frames:
        for _ in range(10):
            example.step()
        wp.synchronize_device(example.model.device)
        start = time.perf_counter()
        for _ in range(args.benchmark_frames):
            example.step()
        wp.synchronize_device(example.model.device)
        elapsed = time.perf_counter() - start
        result["timing"] = {
            "frames": args.benchmark_frames,
            "seconds": elapsed,
            "ms_per_policy_step": 1000 * elapsed / args.benchmark_frames,
            "policy_steps_per_second": args.benchmark_frames / elapsed,
            "scope": "Existing Example.step including ONNX inference and all physics; checks excluded",
        }
        result["timed_final_base_height_m"] = float(example.state_0.body_q.numpy()[0, 2])
        result["timed_final_finite"] = bool(np.isfinite(example.state_0.body_q.numpy()).all())
        example.test_final()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), body_q=np.asarray(poses), body_qd=np.asarray(velocities))
    print({k: v for k, v in result.items() if k != "history"})
    print("FINAL", history[-1])
    if failure:
        raise AssertionError(failure)


if __name__ == "__main__":
    main()
