"""Infer source-drive torque from independent rigid-body momentum balance."""

# ruff: noqa: TID253 -- standalone optional-dependency diagnostic.

import argparse
import ast
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="/tmp/colibri_physx_drive_calibration.npz")
args = parser.parse_args()
input_path = Path(args.input)
d = np.load(input_path)
run_metadata = json.loads(input_path.with_suffix(".json").read_text())
metadata = json.loads(Path(run_metadata["source"]).with_suffix(".fixture.json").read_text())
q = np.concatenate([np.array(metadata["initial_source_body_poses_si"])[None], d["q"]])
v = np.concatenate([np.zeros_like(d["qd"][:1]), d["qd"]])
dt = json.loads(input_path.with_suffix(".json").read_text())["outer_dt"]
r = [Rotation.from_quat(q[:, i, 3:]) for i in range(2)]
angle = (r[0].inv() * r[1]).as_rotvec()[:, 2]
axis = r[0].apply(np.tile([0, 0, 1], (len(q), 1)))
speed = np.sum(axis * (v[:, 1, 3:] - v[:, 0, 3:]), axis=1)
pivot = np.array(ast.literal_eval(metadata["joint_attributes"]["physics:localPos0"])) * 0.015
centers = [q[:, i, :3] + r[i].apply(d["com_pose"][i, :3] * 0.01) for i in range(2)]
hinges = [q[:, i, :3] + r[i].apply(pivot) for i in range(2)]
torques = []
linear_momenta = []
angular_momenta = []
kinetic = np.zeros(len(q))
for i in range(2):
    rot = r[i].as_matrix()
    inertia = np.einsum("nij,jk,nlk->nil", rot, d["inertia"][i].reshape(3, 3) * 1e-4, rot)
    angular = np.einsum("nij,nj->ni", inertia, v[:, i, 3:])
    linear = d["mass"][i] * v[:, i, :3]
    linear_momenta.append(linear)
    angular_momenta.append(angular + np.cross(centers[i], linear))
    kinetic += 0.5 * (np.sum(v[:, i, :3] * linear, axis=1) + np.sum(v[:, i, 3:] * angular, axis=1))
    arm = hinges[i] - centers[i]
    torque = np.diff(angular, axis=0) / dt - np.cross((arm[1:] + arm[:-1]) / 2, np.diff(linear, axis=0) / dt)
    torques.append((1 if i else -1) * np.sum(torque * (axis[1:] + axis[:-1]) / 2, axis=1))
torque = np.mean(torques, axis=0)
error = np.deg2rad(20) - angle
reports = []
for start in (0, 1, 2, 5):
    for end in (12, 60, len(torque)):
        # Endpoint averaging is a numerical quadrature diagnostic, not an exact
        # reconstruction of PhysX's internal TGS spring evaluations.
        a = np.c_[(error[1:] + error[:-1]) / 2, -(speed[1:] + speed[:-1]) / 2]
        fit = np.linalg.lstsq(a[start:end], torque[start:end], rcond=None)[0]
        reports.append(
            {
                "start": start,
                "end": end,
                "stiffness": float(fit[0]),
                "damping": float(fit[1]),
                "residual": float(np.max(abs(a[start:end] @ fit - torque[start:end]))),
                "condition": float(np.linalg.cond(a[start:end])),
            }
        )
out = {
    "fits": reports,
    "body_torque_disagreement": float(np.max(abs(torques[0] - torques[1]))),
    "initial_angle_deg": float(np.rad2deg(angle[0])),
    "final_angle_deg": float(np.rad2deg(angle[-1])),
    "first_torques": torque[:12].tolist(),
    "first_speeds": speed[:12].tolist(),
    "max_total_linear_momentum": float(np.max(np.linalg.norm(np.sum(linear_momenta, axis=0), axis=1))),
    "max_total_angular_momentum": float(np.max(np.linalg.norm(np.sum(angular_momenta, axis=0), axis=1))),
    "max_kinetic_energy_J": float(np.max(kinetic)),
    "max_nominal_spring_plus_kinetic_energy_change_J": float(
        np.max(kinetic + 0.5 * (100 * 1e-4 * 180 / np.pi) * (error**2 - error[0] ** 2))
    ),
    "scope": "Finite-interval momentum balance with endpoint quadrature; internal30 TGS drive evaluations unobserved.",
}
input_path.with_suffix(".analysis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
