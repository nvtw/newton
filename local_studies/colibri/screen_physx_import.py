"""Time completed PhysX steps of the original Colibri USD."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import ovphysx
import ovstage
from ovphysx import PhysX, PhysXConfig, TensorType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default="/home/twidmer/Documents/colibri/Colibri.usd")
    parser.add_argument("--cache-dir", default="/tmp/colibri_physx_cooked")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", default="/tmp/colibri_physx_benchmark.json")
    args = parser.parse_args()
    physx = PhysX(config=PhysXConfig(cooked_collider_cache_dir=args.cache_dir))
    stage = ovstage.Stage("colibri-reference")
    bindings = []
    attached = False
    try:
        print("POPULATE", args.usd, flush=True)
        ovstage.population.open_usd(stage, args.usd, ordinal=1, domains=ovstage.PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        print("ATTACH", flush=True)
        physx.attach_ovstage(stage, read_ordinal=1)
        attached = True
        poses = physx.create_tensor_binding(
            prim_paths=json.loads(Path("/tmp/colibri_usd_rigid_body_paths.json").read_text()),
            tensor_type=TensorType.RIGID_BODY_POSE,
            raise_if_empty=True,
        )
        bindings.append(poses)
        velocities = physx.create_tensor_binding(
            prim_paths=poses.prim_paths, tensor_type=TensorType.RIGID_BODY_VELOCITY, raise_if_empty=True
        )
        bindings.append(velocities)
        assert poses.count == 37, f"Expected 37 rigid bodies, got {poses.count}"
        position = np.zeros(poses.shape, dtype=np.dtype(str(poses.dtype)))
        velocity = np.zeros(velocities.shape, dtype=np.dtype(str(velocities.dtype)))
        poses.read(position)
        initial = position.copy()
        properties = {}
        for name in ("MASS", "INERTIA", "COM_POSE", "CONTACT_OFFSET", "REST_OFFSET", "SHAPE_FRICTION_AND_RESTITUTION"):
            binding = physx.create_tensor_binding(
                prim_paths=poses.prim_paths, tensor_type=getattr(TensorType, "RIGID_BODY_" + name)
            )
            bindings.append(binding)
            values = np.zeros(binding.shape, dtype=np.dtype(str(binding.dtype)))
            binding.read(values)
            properties[name.lower()] = values
        print("LOADED", poses.count, "bodies; warmup", args.warmup, "frames", flush=True)
        if args.warmup:
            physx.step_n_sync(2 * args.warmup, 1.0 / 120.0)
        print("TIMING", args.frames, "frames", flush=True)
        times = []
        for frame in range(args.frames):
            start = time.perf_counter()
            physx.step_n_sync(2, 1.0 / 120.0)
            times.append(time.perf_counter() - start)
            poses.read(position)
            displacement = float(np.max(np.linalg.norm(position[:, :3] - initial[:, :3], axis=1)))
            print("DISPLACEMENT", frame + 1, displacement, flush=True)
            if displacement > 1000:
                print("FAILED_IMPORT_PHYSICS", frame + 1, flush=True)
                break
            if (frame + 1) % 60 == 0:
                print("FRAME", frame + 1, flush=True)
        poses.read(position)
        velocities.read(velocity)
        assert np.isfinite(position).all() and np.isfinite(velocity).all()
        report = {
            "solver": "PhysX",
            "package_version": ovphysx.__version__,
            "source": args.usd,
            "body_count": poses.count,
            "frames": args.frames,
            "warmup_frames": args.warmup,
            "outer_steps_per_frame": 2,
            "outer_dt": 1.0 / 120.0,
            "settings": "authored USD, including damping and scene iteration minimum",
            "mean_frame_ms": 1000 * float(np.mean(times)),
            "p95_frame_ms": 1000 * float(np.percentile(times, 95)),
            "fps": 1.0 / float(np.mean(times)),
            "note": "Completed physics steps only; excludes rendering and validation reads. GPU isolation must be recorded separately.",
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        np.savez(
            Path(args.output).with_suffix(".npz"),
            initial=initial,
            q=position,
            qd=velocity,
            paths=poses.prim_paths,
            frame_times=times,
            **properties,
        )
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for binding in reversed(bindings):
            binding.destroy()
        if attached:
            physx.detach_ovstage()
        stage.destroy()
        physx.release()


if __name__ == "__main__":
    main()
