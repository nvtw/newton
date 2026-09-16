"""CPU nearest-triangle and winding audit of future gear features."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from pathlib import Path
import numpy as np
import trimesh
from local_studies.colibri.audit_gear_exact_geometry import distance, inverse_point
from local_studies.colibri.analyze_velocity4_native_tail import rotate

geometry = np.load("/tmp/physical_head_gear_source_meshes.npz")
poses = np.load("/tmp/physical_head_gear_generation_pose.npz")
witnesses = np.load("/tmp/physical_head_gear_failed_raw.npz")
model = np.load("/tmp/colibri_physical_head12_batch1024_contacts.npz")
meshes = {}
topology = {}
for shape in (48, 50):
    t = geometry[f"{shape}_transform"]
    v = t[:3] + rotate(t[3:], geometry[f"{shape}_vertices"] * geometry[f"{shape}_scale"])
    f = geometry[f"{shape}_indices"].reshape(-1,3)
    meshes[shape] = trimesh.Trimesh(vertices=v, faces=f, process=False)
    welded = trimesh.Trimesh(vertices=v, faces=f, process=True)
    topology[str(shape)] = dict(watertight=bool(welded.is_watertight), consistent_winding=bool(welded.is_winding_consistent))
rows = []
for i in np.flatnonzero(witnesses["query_gap"] < -.001):
    shapes = witnesses["shapes"][i]
    bodies = model["shape_body"][shapes]
    q = poses["post_q"].astype(float)
    p = [q[b,:3] + rotate(q[b,3:], witnesses[f"point{s}"][i]) for s,b in enumerate(bodies)]
    mid = (p[0]+p[1])/2
    sides = [distance(meshes[int(shape)], inverse_point(q[b],mid)) for shape,b in zip(shapes,bodies)]
    source = int(np.argmin([abs(d["signed_distance"]) for d in sides]))
    target = 1-source
    local = inverse_point(q[bodies[source]],mid)
    record = dict(id=int(i), post_stored_gap=float(witnesses["query_gap"][i]),
                  source_shape=int(shapes[source]), target_shape=int(shapes[target]),
                  source_surface_error=abs(sides[source]["signed_distance"]))
    for name, q in (("generation",poses["pre_q"].astype(float)),("failed",poses["post_q"].astype(float))):
        point = q[bodies[source],:3] + rotate(q[bodies[source],3:],local)
        result = distance(meshes[int(shapes[target])],inverse_point(q[bodies[target]],point))
        record[name] = result
    rows.append(record)
report = dict(topology=topology, rows=rows,
              scope="Exact triangles from authored collision meshes; midpoint source side inferred by nearest surface. CUDA SDF construction bypassed only during CPU mesh extraction.")
Path("/tmp/physical_head_gear_exact_geometry.json").write_text(json.dumps(report,indent=2))
print(json.dumps(dict(count=len(rows),topology=topology,deepest=min(rows,key=lambda r:r["post_stored_gap"]),
                     generation_distance_range=[min(r["generation"]["signed_distance"] for r in rows),max(r["generation"]["signed_distance"] for r in rows)],
                     max_source_error=max(r["source_surface_error"] for r in rows)),indent=2))
