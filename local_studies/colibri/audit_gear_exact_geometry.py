"""CPU exact-triangle audit of failed gear witnesses; no solver changes."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.analyze_velocity4_native_tail import rotate


def inverse_point(q, point):
    """Transform world points to a saved body's local coordinates."""
    inverse = q[3:].copy()
    inverse[:3] *= -1
    return rotate(inverse, point - q[:3])


def distance(mesh, point):
    """Return nearest-triangle distance, solid-angle winding, and normal."""
    import trimesh

    triangles = mesh.triangles
    closest = trimesh.triangles.closest_point(triangles, np.broadcast_to(point, (len(triangles), 3)))
    distances = np.linalg.norm(closest - point, axis=1)
    index = int(np.argmin(distances))
    rays = triangles - point
    a, b, c = rays[:, 0], rays[:, 1], rays[:, 2]
    la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
    denominator += np.einsum("ij,ij->i", b, c) * la + np.einsum("ij,ij->i", c, a) * lb
    winding = np.arctan2(numerator, denominator).sum() / (2 * np.pi)
    signed = float(distances[index]) * (-1 if abs(winding) > 0.5 else 1)
    return {
        "signed_distance": signed,
        "winding": float(winding),
        "triangle": index,
        "nearest": closest[index].tolist(),
        "triangle_normal": mesh.face_normals[index].tolist(),
    }


def main():
    import trimesh

    prefix = "/tmp/colibri_spatial_priority_velocity4_trace18000"
    data = np.load(prefix + ".trace.npz")
    saved = np.load(prefix + ".query_29839_geometric_raw_post.npz")
    geometry = np.load("/tmp/colibri_failed_gear_source_meshes.npz")
    slot = int(np.flatnonzero(data["step_ids"] == 29839)[0])
    poses = {"pre": data["generation_q"][4 * slot].astype(float), "post": data["post_q"][4 * slot + 3].astype(float)}
    shapes = (46, 48)
    bodies = [int(data["shape_body"][shape]) for shape in shapes]
    meshes = {}
    topology = {}
    for shape in shapes:
        vertices = geometry[f"{shape}_vertices"].astype(float) * geometry[f"{shape}_scale"]
        transform = geometry[f"{shape}_transform"]
        body_vertices = transform[:3] + rotate(transform[3:], vertices)
        faces = geometry[f"{shape}_indices"].reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=body_vertices, faces=faces, process=False)
        welded = trimesh.Trimesh(vertices=body_vertices, faces=faces, process=True)
        topology[str(shape)] = {
            "watertight_after_weld": bool(welded.is_watertight),
            "winding_consistent": bool(welded.is_winding_consistent),
            "triangles": len(faces),
        }
        meshes[shape] = mesh
    motion = data["generation_qd"][4 * slot].astype(float)
    search = []
    for shape, body in zip(shapes, bodies, strict=True):
        q = poses["pre"][body]
        transform = geometry[f"{shape}_transform"]
        origin = q[:3] + rotate(q[3:], transform[:3])
        com = q[:3] + rotate(q[3:], data["body_com"][body])
        velocity = motion[body, :3] + np.cross(motion[body, 3:], origin - com)
        vertices = geometry[f"{shape}_vertices"].astype(float) * geometry[f"{shape}_scale"]
        furthest = np.maximum(np.abs(vertices.min(axis=0)), np.abs(vertices.max(axis=0)))
        radius = max(np.linalg.norm(furthest), float(geometry[f"{shape}_radius"]))
        extension = min((np.linalg.norm(velocity) + np.linalg.norm(motion[body, 3:]) * radius) / 120, 0.005)
        search.append(
            {
                "shape": shape,
                "angular_radius": float(radius),
                "origin_speed": float(np.linalg.norm(velocity)),
                "angular_speed": float(np.linalg.norm(motion[body, 3:])),
                "extension": float(extension),
                "physical_gap": float(data["shape_gap"][shape]),
                "search_gap": float(data["shape_gap"][shape] + extension),
            }
        )
    selected = np.flatnonzero(saved["post_gap"] < -0.0005)
    rows = []
    for index in selected:
        q = poses["post"]
        a, b = bodies
        p0 = q[a, :3] + rotate(q[a, 3:], saved["point0"][index])
        p1 = q[b, :3] + rotate(q[b, 3:], saved["point1"][index])
        sample = (p0 + p1) * 0.5
        source_checks = [
            distance(meshes[shape], inverse_point(q[body], sample)) for shape, body in zip(shapes, bodies, strict=True)
        ]
        source_side = int(np.argmin([abs(check["signed_distance"]) for check in source_checks]))
        source_body = bodies[source_side]
        target_body = bodies[1 - source_side]
        source_shape = shapes[source_side]
        target_shape = shapes[1 - source_side]
        source_local = inverse_point(q[source_body], sample)
        results = {}
        for name, q in poses.items():
            sample_world = q[source_body, :3] + rotate(q[source_body, 3:], source_local)
            target_local = inverse_point(q[target_body], sample_world)
            results[name] = distance(meshes[target_shape], target_local)
            if name == "pre":
                closest_world = q[target_body, :3] + rotate(q[target_body, 3:], np.asarray(results[name]["nearest"]))
                outward = sample_world - closest_world
                outward /= np.linalg.norm(outward)
                source_com = q[source_body, :3] + rotate(q[source_body, 3:], data["body_com"][source_body])
                target_com = q[target_body, :3] + rotate(q[target_body, 3:], data["body_com"][target_body])
                source_v = motion[source_body, :3] + np.cross(motion[source_body, 3:], sample_world - source_com)
                target_v = motion[target_body, :3] + np.cross(motion[target_body, 3:], closest_world - target_com)
                vn = float(np.dot(source_v - target_v, outward))
                results[name]["exact_gap_initial_derivative_m_s"] = vn
                results[name]["constant_twist_linear_prediction_m"] = results[name]["signed_distance"] + vn / 120
        rows.append(
            {
                "id": int(index),
                "stored_post_gap": float(saved["post_gap"][index]),
                "source_shape": source_shape,
                "source_surface_error": abs(source_checks[source_side]["signed_distance"]),
                "opposing_post_distance": source_checks[1 - source_side]["signed_distance"],
                "source_local": source_local.tolist(),
                "target_shape": target_shape,
                **results,
            }
        )
    report = {
        "scope": "Source side inferred from exact post midpoint surface distance; both candidates recorded by distance check.",
        "topology": topology,
        "search_envelope_cpu_reconstruction": search,
        "pair_search_gap": sum(x["search_gap"] for x in search),
        "count": len(rows),
        "rows": rows,
    }
    Path(prefix + ".exact_geometry.json").write_text(json.dumps(report, indent=2))
    deepest = min(rows, key=lambda row: row["stored_post_gap"])
    print(
        json.dumps(
            {
                "topology": topology,
                "search": search,
                "pair_search_gap": report["pair_search_gap"],
                "count": len(rows),
                "deepest": deepest,
                "max_source_error": max(row["source_surface_error"] for row in rows),
                "pre_signed_range": [
                    min(row["pre"]["signed_distance"] for row in rows),
                    max(row["pre"]["signed_distance"] for row in rows),
                ],
                "post_signed_range": [
                    min(row["post"]["signed_distance"] for row in rows),
                    max(row["post"]["signed_distance"] for row in rows),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
