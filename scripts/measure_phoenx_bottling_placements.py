# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure carton handoff, settling, and creep on the Berlin bottling plant's moving pallets.

Run `uv run scripts/measure_phoenx_bottling_placements.py --output placements.json` with
the plant assets installed. The default 330 simulated seconds cover both pallet layers.
The headless measurement loads the same shapes as the default OptiX scene so contact
ordering matches the rendered workload. Use `--no-visuals` to measure the smaller model.
Compare runs by station, slot and carton. The first release frame is the first frame
with an open gripper after it held that carton. The handoff interval includes the last
held frame. Distances are measured relative to the pallet for cartons and relative to
the carton for bottle centers of mass.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import newton.examples
from newton.examples.phoenx.example_phoenx_bottling_plant import Example


def _rotate(q, vectors):
    cross = 2 * np.cross(q[..., :3], vectors)
    return vectors + q[..., 3, None] * cross + np.cross(q[..., :3], cross)


def _local(pose, points):
    return _rotate(pose[3:] * [-1, -1, -1, 1], points - pose[:3])


@dataclass
class _Placement:
    station: str
    slot: int
    carton: int
    pallet: int
    bottles: np.ndarray
    release_frame: int
    samples: list = field(default_factory=list)

    def sample(self, poses, bottle_com):
        cargo = poses[self.carton]
        bottles = poses[self.bottles]
        centers = bottles[:, :3] + _rotate(bottles[:, 3:], bottle_com[self.bottles])
        self.samples.append((_local(poses[self.pallet], cargo[:3]), _local(cargo, centers)))

    def report(self, labels, frame_dt):
        cartons, bottles = (1000 * np.stack(v) for v in zip(*self.samples, strict=True))
        result = {
            "station": self.station,
            "slot": self.slot,
            "carton": labels[self.carton],
            "pallet": labels[self.pallet],
            "bottle_count": len(self.bottles),
            "release_frame": self.release_frame,
            "pallet_relative_release_height_mm": float(cartons[1, 2]),
        }
        # Index zero is the last held frame. Omit metrics for incomplete windows.
        for name, start, end in (
            ("handoff", 0, round(0.1 / frame_dt)),
            ("rest", 1 + round(1 / frame_dt), 1 + round(5 / frame_dt)),
            ("creep", 1 + round(5 / frame_dt), 1 + round(15 / frame_dt)),
        ):
            if end >= len(cartons):
                continue
            path = cartons[start : end + 1]
            offsets = bottles[start : end + 1] - bottles[start]
            result[name] = {
                "carton_vertical_range_mm": float(np.ptp(path[:, 2])),
                "carton_lateral_excursion_mm": float(np.linalg.norm(path[:, :2] - path[0, :2], axis=1).max()),
                "carton_rms_speed_mm_s": float(np.sqrt(np.square(np.diff(path, axis=0)).sum(axis=1).mean()) / frame_dt),
                "bottle_max_excursion_mm": float(np.linalg.norm(offsets, axis=2).max()) if len(self.bottles) else None,
            }
        return result


def measure(example, args):
    transport = example.controller.transport
    bottle_ids = example.bottle_bodies
    body_com = example.model.body_com.numpy().astype(np.float64)
    design = example.controller.case_modules.design.CASE
    half = np.array([design["L"], design["W"]]) * 0.5
    last_held = transport.held.numpy().copy()
    last_poses = example.state_0.body_q.numpy().astype(np.float64)
    active = []
    results = []
    tracking_frames = round(15 / example.frame_dt)
    frame = 0
    step = example.step

    def record_step():
        nonlocal frame, last_held, last_poses, active
        step()
        frame += 1
        held = transport.held.numpy()
        if active or np.any(last_held >= 0) or np.any(held >= 0):
            poses = example.state_0.body_q.numpy().astype(np.float64)
            for cell in np.flatnonzero((last_held >= 0) & (held != last_held)):
                carton = int(last_held[cell])
                station = transport.names[cell]
                offsets = last_poses[transport.pallet_bodies, :3] - transport.station_positions[cell]
                pallet = transport.pallet_bodies[np.argmin(np.linalg.norm(offsets, axis=1))]
                bottles = last_poses[bottle_ids]
                centers = bottles[:, :3] + _rotate(bottles[:, 3:], body_com[bottle_ids])
                local = _local(last_poses[carton], centers)
                inside = (np.abs(local[:, :2]) < half).all(axis=1) & (local[:, 2] > 0) & (local[:, 2] < design["H"])
                event = _Placement(
                    station, transport.palletizers[station].slot, carton, pallet, bottle_ids[inside], frame
                )
                event.sample(last_poses, body_com)
                active.append(event)
            tracking = []
            for event in active:
                event.sample(poses, body_com)
                if frame - event.release_frame >= tracking_frames:
                    results.append(event.report(example.model.body_label, example.frame_dt))
                else:
                    tracking.append(event)
            active = tracking
            last_poses = poses
        last_held = held.copy()

    example.step = record_step
    try:
        newton.examples.run(example, args)
    finally:
        example.step = step
    results.extend(event.report(example.model.body_label, example.frame_dt) for event in active)
    return sorted(results, key=lambda result: result["release_frame"])


def main():
    parser = Example.create_parser()
    parser.set_defaults(viewer="null", visuals=True, headless=True, num_frames=19_800)
    parser.add_argument("--output", type=Path, required=True)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if example.controller is None:
        parser.error("placement measurement requires the plant controller")
    events = measure(example, args)
    if not args.test:
        example.test_final()
    report = {"frame_dt": example.frame_dt, "shape_count": example.model.shape_count, "placements": events}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Measured {len(events)} vacuum releases over {example.sim_time:.2f} simulated seconds")


if __name__ == "__main__":
    main()
