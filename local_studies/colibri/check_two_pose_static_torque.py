"""Static Coulomb torque intervals at actual saved PhysX and PhoenX poses.

Uses authored triangle geometry against the source plane, not saved stale
contact planes. A 0.1 micrometer contact band is an explicit geometric
relaxation; outer-cone infeasibility remains a stronger negative control.
"""

import ast
import json
from pathlib import Path

import numpy as np


def constants():
    """Read authored SI geometry without initializing either physics engine."""
    tree = ast.parse(Path("newton/examples/kamino/example_kamino_colibri.py").read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return out


def geometry(data):
    """Load exactly the Python scene's meshes and 32-sided cylinder surfaces."""
    import trimesh
    from scipy.spatial.transform import Rotation

    result = []
    for body, kind, label, source, dim in data["SHAPES"]:
        if body not in ("FrameGround", "Frame") or label not in data["COLLISION_LABELS"] or "/Flower/" in label:
            continue
        if kind == "mesh":
            mesh = trimesh.load(Path("newton/examples/assets/colibri") / source, force="mesh", process=False)
            affine = np.asarray(dim)
            vertices = np.asarray(mesh.vertices) @ affine[:, :3].T + affine[:, 3]
        else:
            mesh = trimesh.creation.cylinder(radius=dim[0], height=2 * dim[1], sections=32)
            vertices = Rotation.from_quat(source[3:]).apply(mesh.vertices) + source[:3]
        result.append(
            (
                0 if body == "FrameGround" else 1,
                label,
                vertices,
                np.asarray(mesh.faces),
                0.5 * (0.5 + data["SHAPE_MATERIALS"].get(label, (1000, 0.5))[1]),
            )
        )
    return result


def audit(name, q, qd, mass, com, meshes, outer=True, band=1e-7):
    """Compute a force equilibrium interval with all actual nearby supports."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    plane = 0.002965275
    rotations = [Rotation.from_quat(row[3:]) for row in q]
    centers = np.array([q[i, :3] + rotations[i].apply(com[i]) for i in range(2)])
    localpivot = np.array([-0.091907609, -0.157658043, 0])
    pivots = np.array([q[i, :3] + rotations[i].apply(localpivot) for i in range(2)])
    pivot = pivots.mean(0)
    axis = rotations[0].apply([0, 0, 1])
    angle = (rotations[0].inv() * rotations[1]).as_rotvec()[2]
    axes = rotations[0].as_matrix().T
    b = np.zeros((6, 12))
    for j in range(3):
        e = np.eye(3)[j]
        for i, sign in ((0, -1), (1, 1)):
            b[j, 6 * i : 6 * i + 6] = sign * np.r_[e, np.cross(pivot - centers[i], e)]
    for j in range(3):
        b[j + 3, 3:6] = -axes[j]
        b[j + 3, 9:12] = axes[j]
    rows = []
    mus = []
    detail = []
    ground = []
    minimum = [np.inf, np.inf]
    for body, label, local, faces, mu in meshes:
        vertices = rotations[body].apply(local) + q[body, :3]
        gap = vertices[:, 2] - plane
        minimum[body] = min(minimum[body], float(gap.min()))
        if gap.min() > band:
            continue
        points = list(vertices[gap <= band])
        edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
        for ia, ib in edges:
            ga, gb = gap[ia] - band, gap[ib] - band
            if ga * gb < 0:
                points.append(vertices[ia] + (vertices[ib] - vertices[ia]) * (-ga) / (gb - ga))
        xy = np.unique(np.round(np.asarray(points)[:, :2], 14), axis=0)
        if len(xy) > 2:
            try:
                xy = xy[ConvexHull(xy).vertices]
            except Exception as exc:
                raise RuntimeError("Unresolved support hull") from exc
        detail.append({"label": label, "minimum_gap_m": float(gap.min()), "support_extremes": len(xy), "mu": mu})
        for x, y in xy:
            point = np.array([x, y, plane])
            c = np.zeros((3, 12))
            for j, e in enumerate(([0, 0, 1], [1, 0, 0], [0, 1, 0])):
                c[j, 6 * body : 6 * body + 6] = np.r_[e, np.cross(point - centers[body], e)]
            rows.append(c)
            mus.append(mu)
            ground.append(point)
    c = np.asarray(rows).reshape(-1, 12)
    gravity = np.zeros(12)
    gravity[2] = -mass[0]
    gravity[8] = -mass[1]
    count = len(mus)
    rays = np.zeros((3 * count, 64 * count))
    for i, mu in enumerate(mus):
        for j in range(64):
            phi = 2 * np.pi * j / 64
            rays[3 * i : 3 * i + 3, 64 * i + j] = [
                1,
                mu * np.cos(phi) / (np.cos(np.pi / 64) if outer else 1.0),
                mu * np.sin(phi) / (np.cos(np.pi / 64) if outer else 1.0),
            ]
    op = np.c_[c.T @ rays, b[:5].T, b[5]]
    scale = np.maximum(np.linalg.norm(op, axis=1), abs(gravity))
    extrema = []
    balances = []
    witnesses = []
    for sign in (1.0, -1.0):
        objective = np.zeros(op.shape[1])
        objective[-1] = sign
        result = linprog(
            objective,
            A_eq=op / scale[:, None],
            b_eq=-gravity / scale,
            bounds=[(0, None)] * (count * 64) + [(None, None)] * 6,
            method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
        )
        if result.success:
            forces = (rays @ result.x[: count * 64]).reshape(-1, 3)
            if not outer:
                assert np.max(np.linalg.norm(forces[:, 1:], axis=1) - np.asarray(mus) * forces[:, 0]) < 1e-10
            witnesses.append(np.r_[forces.ravel(), result.x[count * 64 :]])
        extrema.append(float(result.x[-1]) if result.success else result.message)
        balances.append(float(np.max(abs(op @ result.x + gravity))) if result.success else None)
    certificate = Path(
        "/tmp/colibri_static_certificate_"
        + ("analytical" if name == "Analytical spring/gravity equilibrium" else name.lower().replace(" ", "_"))
        + ".npz"
    )
    np.savez(
        certificate,
        q=q,
        mass=mass,
        com_local=com,
        com_world=centers,
        pivots=pivots,
        C=c,
        B=b,
        gravity=gravity,
        mu=np.asarray(mus),
        contact_points=np.asarray(ground),
        extrema_witness=np.asarray(witnesses),
        plane_height=plane,
        contact_band=band,
        is_outer_polygon=outer,
    )
    stiffness = 0.5729577951308232
    spring = stiffness * (0.3490658503988659 - angle)
    # qd conventions differ across the saved engines; static torque deliberately
    # evaluates zero relative velocity, so no unverified damping subtraction.
    gravity_torque = float(axis @ np.cross(centers[1] - pivot, [0, 0, -mass[1]]))
    return {
        "name": name,
        "certificate_npz": str(certificate),
        "angle_deg": float(np.rad2deg(angle)),
        "spring_at_rest_Nm": float(spring),
        "frame_gravity_torque_Nm": gravity_torque,
        "rest_torque_without_frame_support_Nm": float(spring + gravity_torque),
        "outer_static_drive_interval_Nm": extrema,
        "polygon": "outer" if outer else "inner",
        "lp_balance_error_N": balances,
        "minimum_ground_clearance_m": minimum,
        "contact_band_m": band,
        "eligible_shapes": detail,
        "com_world_m": centers.tolist(),
        "hinge_axis_world": axis.tolist(),
        "pivot_world_m": pivot.tolist(),
        "anchor_discrepancy_m": float(np.linalg.norm(pivots[0] - pivots[1])),
        "scope": "Fixed actual pose, shared midpoint hinge, original source masses/gravity; no recovery forces. Circumscribed cone and relaxed contact band give outer feasibility only; no unique equilibrium angle claim.",
    }


def analytical_equilibrium(nx, shapes, label="Analytical spring/gravity equilibrium", band=1e-7):
    """Find one free-Frame spring/gravity equilibrium and certify support."""
    from scipy.optimize import brentq
    from scipy.spatial.transform import Rotation

    q = nx["q"][:2].astype(float).copy()
    base = Rotation.from_quat(q[0, 3:])
    localpivot = np.array([-0.091907609, -0.157658043, 0])
    pivot = q[0, :3] + base.apply(localpivot)
    axis = base.apply([0, 0, 1])
    mass = nx["body_mass"][:2]
    com = nx["body_com"][:2]
    k = 0.5729577951308232
    target = 0.3490658503988659

    def torque(angle):
        child = base * Rotation.from_rotvec([0, 0, angle])
        arm = child.apply(com[1] - localpivot)
        return k * (target - angle) + axis @ np.cross(arm, [0, 0, -mass[1]])

    angle = brentq(torque, np.deg2rad(10), np.deg2rad(24), xtol=1e-14)
    child = base * Rotation.from_rotvec([0, 0, angle])
    q[1, :3] = pivot - child.apply(localpivot)
    q[1, 3:] = child.as_quat()
    result = audit(label, q, np.zeros((2, 6)), mass, com, shapes, outer=False, band=band)
    result["scalar_torque_residual_Nm"] = float(torque(angle))
    result["equilibrium_q"] = q.tolist()
    result["scope"] = (
        "One local free-Frame equilibrium at the saved base pose; inscribed friction polygon certifies support on projected near-plane contacts; not uniqueness among other contact modes"
    )
    assert abs(torque(angle)) < 1e-12
    assert result["minimum_ground_clearance_m"][1] > 0.001
    interval = result["outer_static_drive_interval_Nm"]
    assert interval[0] - 1e-9 <= result["spring_at_rest_Nm"] <= interval[1] + 1e-9
    return result


def main():
    """Compare independently computed physical support intervals at both poses."""
    shapes = geometry(constants())
    px = np.load("/tmp/colibri_physx_two_body_awake_plane1mm60.npz")
    nx = np.load("/tmp/colibri_base_frame_break_totalnormal3600.npz")
    reports = [
        audit("PhysX", px["q"][-1], px["qd"][-1], px["mass"], px["com_pose"][:, :3] * 0.01, shapes),
        audit("PhoenX", nx["q"][:2], nx["qd"][:2], nx["body_mass"][:2], nx["body_com"][:2], shapes),
    ]
    analytical = analytical_equilibrium(nx, shapes)
    reports.append(analytical)
    lifted = np.asarray(analytical["equilibrium_q"]).copy()
    lifted[:, :3] += np.array([0, 0, -analytical["minimum_ground_clearance_m"][0]])
    lifted_report = audit(
        "Exact lifted analytical pose",
        lifted,
        np.zeros((2, 6)),
        nx["body_mass"][:2],
        nx["body_com"][:2],
        shapes,
        outer=False,
        band=1e-12,
    )
    lifted_report["scope"] = (
        "Same common-pivot analytical hinge pose lifted rigidly to exact base-plane contact; 1pm numerical geometric tolerance only; infeasibility concerns this fixed base orientation"
    )
    reports.append(lifted_report)
    from scipy.spatial.transform import Rotation

    data = constants()
    source_q = np.array(data["BODY_POSES"]["FrameGround"], dtype=float)
    source_rotation = Rotation.from_quat(source_q[3:])
    up = source_rotation.apply([0, 1, 0])
    cross = np.cross(up, [0, 0, 1])
    angle = np.arctan2(np.linalg.norm(cross), up[2])
    correction = Rotation.from_rotvec(cross / np.linalg.norm(cross) * angle)
    flat_rotation = correction * source_rotation
    source_q[3:] = flat_rotation.as_quat()
    base_vertices = next(item[2] for item in shapes if item[1] == "FrameGround/Base")
    source_q[2] = 0.002965275 - flat_rotation.apply(base_vertices)[:, 2].min()
    flat_q = nx["q"][:2].copy().astype(float)
    flat_q[0] = source_q
    flat_input = {"q": flat_q, "body_mass": nx["body_mass"], "body_com": nx["body_com"]}
    flat_report = analytical_equilibrium(flat_input, shapes, label="Exact flat-base analytical equilibrium", band=1e-12)
    flat_report["source_orientation_correction_rad"] = float(angle)
    flat_report["scope"] = (
        "Analytical free-base planar-support mode: align authored bottom face to ground by a measured tiny rotation, put it exactly on plane, solve free Frame spring/gravity torque, certify inscribed Coulomb forces; no production pose modification, no uniqueness claim"
    )
    reports.append(flat_report)
    out = Path("/tmp/colibri_two_pose_static_torque.json")
    out.write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
