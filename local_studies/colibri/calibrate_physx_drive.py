"""Measure source two-body motion; per-frame readbacks make this an accuracy run."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import ovphysx
import ovstage
from ovphysx import PhysX, PhysXConfig, TensorType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--output", required=True)
    parser.add_argument("--outer-dt", type=float, default=1 / 120)
    args = parser.parse_args()
    paths = ["/World/Colibri/FrameAsm/FrameGround", "/World/Colibri/FrameAsm/Frame"]
    physics = PhysX(config=PhysXConfig(cooked_collider_cache_dir="/tmp/colibri_physx_cooked"))
    stage = ovstage.Stage("colibri-two-body")
    bindings = []
    attached = False
    try:
        ovstage.population.open_usd(stage, args.usd, ordinal=1, domains=ovstage.PopulationDomain.ALL)
        stage.advance_write_floor(ordinal=1).wait()
        physics.attach_ovstage(stage, read_ordinal=1)
        attached = True
        physics.step_sync(args.outer_dt)

        def bind(kind):
            b = physics.create_tensor_binding(
                prim_paths=paths, tensor_type=getattr(TensorType, "RIGID_BODY_" + kind), raise_if_empty=True
            )
            bindings.append(b)
            return b, np.zeros(b.shape, dtype=np.dtype(str(b.dtype)))

        poses, q = bind("POSE")
        velocities, qd = bind("VELOCITY")
        assert poses.count == 2
        properties = {}
        for name in ("MASS", "INERTIA", "COM_POSE", "CONTACT_OFFSET", "REST_OFFSET", "SHAPE_FRICTION_AND_RESTITUTION"):
            b, value = bind(name)
            b.read(value)
            properties[name.lower()] = value
        qhist, vhist = [], []
        for frame in range(args.frames + 1):
            if frame:
                physics.step_sync(args.outer_dt)
            poses.read(q)
            velocities.read(qd)
            qs, vs = q.astype(float), qd.astype(float)
            qs[:, :3] *= 0.01
            vs[:, :3] *= 0.01
            assert np.isfinite(qs).all() and np.isfinite(vs).all()
            qhist.append(qs)
            vhist.append(vs)
            if frame % 600 == 0:
                print(
                    "FRAME",
                    frame,
                    "base_displacement_m",
                    np.linalg.norm(qs[0, :3] - qhist[0][0, :3]),
                    "speed",
                    np.linalg.norm(vs[0, :3]),
                    flush=True,
                )
        qhist, vhist = np.array(qhist), np.array(vhist)
        origin = qhist[:, 0, :3] - qhist[0, 0, :3]
        dots = np.abs(np.sum(qhist[:, 0, 3:] * qhist[0, 0, 3:], axis=1))
        angle = 2 * np.arccos(np.clip(dots, -1, 1))
        out = {
            "source": args.usd,
            "source_sha256": hashlib.sha256(Path(args.usd).read_bytes()).hexdigest(),
            "ovphysx_version": ovphysx.__version__,
            "frames": args.frames,
            "initial_normal_step_s": args.outer_dt,
            "duration_after_initial_step_s": args.frames * args.outer_dt,
            "outer_dt": args.outer_dt,
            "position_iterations": 30,
            "velocity_iterations": 1,
            "scope": "Accuracy trajectory with readbacks; not timing. TGS temporal budget matched, not algorithmically identical microsteps.",
            "sleep_status": "No rigid-body sleep tensor is exposed; zero speed alone is not a sleep certificate. Use explicit zero-threshold companion control.",
            "base_max_displacement_m": float(np.max(np.linalg.norm(origin, axis=1))),
            "base_final_displacement_m": float(np.linalg.norm(origin[-1])),
            "base_max_rotation_rad": float(np.max(angle)),
            "checkpoints": {},
        }
        for frame in (600, 1200, 1800, 3600):
            if frame <= args.frames:
                out["checkpoints"][str(frame * args.outer_dt)] = {
                    "base_displacement_m": float(np.linalg.norm(origin[frame])),
                    "base_speed_m_s": float(np.linalg.norm(vhist[frame, 0, :3])),
                    "base_angular_speed_rad_s": float(np.linalg.norm(vhist[frame, 0, 3:])),
                }
        np.savez_compressed(
            Path(args.output).with_suffix(".npz"), q=qhist, qd=vhist, paths=np.array(paths), **properties
        )
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2), flush=True)
    finally:
        for b in reversed(bindings):
            b.destroy()
        if attached:
            physics.detach_ovstage()
        stage.destroy()
        physics.release()


if __name__ == "__main__":
    main()
