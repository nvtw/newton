# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Sample actual pre-generation SDF at the material points of later fresh contacts."""

import json

import numpy as np
import warp as wp

from newton._src.geometry.sdf_contact import sample_sdf_grad_using_mesh
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf_grad_hw
from newton.examples.kamino.example_kamino_colibri import build_scene


@wp.kernel
def query(
    target: wp.array[int],
    source: wp.array[int],
    points: wp.array[wp.vec3],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    body: wp.array[int],
    shape_pose: wp.array[wp.transform],
    scales: wp.array[wp.vec3],
    meshes: wp.array[wp.uint64],
    indices: wp.array[int],
    sdf: wp.array[TextureSDFData],
    output: wp.array2d[float],
):
    i = wp.tid()
    t, s = target[i], source[i]
    bt, bs = body[t], body[s]
    world = wp.transform_point(q[bs], points[i])
    target_pose = q[bt] * shape_pose[t]
    local = wp.cw_div(wp.transform_point(wp.transform_inverse(target_pose), world), scales[t])
    d, g = texture_sample_sdf_grad_hw(sdf[indices[t]], local)
    dm, gm = sample_sdf_grad_using_mesh(meshes[t], local, 10.0, 0)
    world_g = wp.normalize(wp.transform_vector(target_pose, wp.cw_div(g, scales[t])))
    source_v = wp.spatial_top(qd[bs]) + wp.cross(wp.spatial_bottom(qd[bs]), world - wp.transform_point(q[bs], com[bs]))
    target_v = wp.spatial_top(qd[bt]) + wp.cross(wp.spatial_bottom(qd[bt]), world - wp.transform_point(q[bt], com[bt]))
    output[i, 0] = d
    output[i, 1] = dm
    output[i, 2] = wp.dot(world_g, source_v - target_v)
    output[i, 3] = world_g[0]
    output[i, 4] = world_g[1]
    output[i, 5] = world_g[2]


archive = np.load("/tmp/colibri_public_group4_tail.trace.npz")
records = json.load(open("/tmp/colibri_public_group4_failed_pose_contacts.json"))
r = min(records[-2]["contacts"], key=lambda x: x["gap"])
builder = build_scene(
    body_count=36, fix_base=False, contact_gap=0.001, source_contact_offsets=True, mesh_cylinders=True
)
model = builder.finalize(skip_validation_joints=True)
slot = int(np.flatnonzero(archive["step_ids"] == 8147)[0])
results = []
for reverse in (False, True):
    target = np.array([r["shape_ids"][int(reverse)]], dtype=np.int32)
    source = np.array([r["shape_ids"][1 - int(reverse)]], dtype=np.int32)
    points = np.array([r["point0"] if reverse else r["point1"]], dtype=np.float32)
    out = wp.zeros((1, 6), dtype=float, device=model.device)
    wp.launch(
        query,
        1,
        [
            wp.array(target, dtype=int, device=model.device),
            wp.array(source, dtype=int, device=model.device),
            wp.array(points, dtype=wp.vec3, device=model.device),
            wp.array(archive["pre_q"][slot], dtype=wp.transform, device=model.device),
            wp.array(archive["pre_qd"][slot], dtype=wp.spatial_vector, device=model.device),
            model.body_com,
            model.shape_body,
            model.shape_transform,
            model.shape_scale,
            model.shape_source_ptr,
            model._shape_sdf_index,
            model._texture_sdf_data,
            out,
        ],
        device=model.device,
    )
    results.append(dict(target=int(target[0]), source=int(source[0]), values=out.numpy().tolist()))
Path = None
json.dump(results, open("/tmp/colibri_public_8147_pre_sdf.json", "w"), indent=2)
print(results)
