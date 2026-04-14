"""Quick determinism test — runs 50 frames and reports differences in detail."""

import numpy as np
import warp as wp

wp.init()

import newton  # noqa: E402

# Build the same scene as the 500-step test
def build_scene(device):
    builder = newton.ModelBuilder()
    builder.default_shape_ke = 1.0e4
    builder.default_shape_kd = 100.0
    builder.default_shape_kf = 100.0
    builder.default_shape_mu = 0.5

    from newton._src.geometry.terrain_generator import create_mesh_terrain

    terrain_vertices, terrain_indices = create_mesh_terrain(
        grid_size=(6, 6),
        block_size=(5.0, 5.0),
        terrain_types=["pyramid_stairs"],
        terrain_params={
            "pyramid_stairs": {"step_width": 0.4, "step_height": 0.05, "platform_width": 0.8},
        },
        seed=42,
    )
    terrain_mesh = newton.Mesh(terrain_vertices, terrain_indices)
    terrain_mesh.build_sdf(max_resolution=512)
    builder.add_shape_mesh(body=-1, mesh=terrain_mesh, xform=wp.transform(p=wp.vec3(-15.0, -15.0, -0.5)))

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_radius = 0.35
    ico_base_vertices = np.array(
        [[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
         [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
         [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]],
        dtype=np.float32,
    )
    for i in range(len(ico_base_vertices)):
        ico_base_vertices[i] = ico_base_vertices[i] / np.linalg.norm(ico_base_vertices[i]) * ico_radius
    ico_face_indices = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    ico_vertices, ico_normals, ico_indices = [], [], []
    for face_idx, face in enumerate(ico_face_indices):
        v0, v1, v2 = ico_base_vertices[face[0]], ico_base_vertices[face[1]], ico_base_vertices[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)
        ico_vertices.extend([v0, v1, v2])
        ico_normals.extend([normal, normal, normal])
        base = face_idx * 3
        ico_indices.extend([base, base + 1, base + 2])
    ico_mesh = newton.Mesh(
        np.array(ico_vertices, dtype=np.float32),
        np.array(ico_indices, dtype=np.int32),
        normals=np.array(ico_normals, dtype=np.float32),
    )

    hs = 0.3
    cube_verts = np.array(
        [[-hs, -hs, -hs], [hs, -hs, -hs], [hs, hs, -hs], [-hs, hs, -hs],
         [-hs, -hs, hs], [hs, -hs, hs], [hs, hs, hs], [-hs, hs, hs]],
        dtype=np.float32,
    )
    cube_tris = np.array(
        [0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
         2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5],
        dtype=np.int32,
    )
    cube_mesh = newton.Mesh(cube_verts, cube_tris)
    cube_mesh.build_sdf(max_resolution=64)

    shape_types = ["sphere", "box", "capsule", "mesh_cube", "cylinder", "cone", "icosahedron"]
    grid_offset = wp.vec3(-5.0, -5.0, 0.5)
    rng = np.random.default_rng(42)
    shape_index = 0
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                pos = wp.vec3(
                    float(grid_offset[0]) + ix * 1.5 + (rng.random() - 0.5) * 0.4,
                    float(grid_offset[1]) + iy * 1.5 + (rng.random() - 0.5) * 0.4,
                    float(grid_offset[2]) + iz * 1.5 + (rng.random() - 0.5) * 0.4,
                )
                shape_type = shape_types[shape_index % len(shape_types)]
                shape_index += 1
                body = builder.add_body(xform=wp.transform(p=pos, q=wp.quat_identity()))
                if shape_type == "sphere":
                    builder.add_shape_sphere(body, radius=0.3)
                elif shape_type == "box":
                    builder.add_shape_box(body, hx=0.3, hy=0.3, hz=0.3)
                elif shape_type == "capsule":
                    builder.add_shape_capsule(body, radius=0.2, half_height=0.4)
                elif shape_type == "cylinder":
                    builder.add_shape_cylinder(body, radius=0.25, half_height=0.35)
                elif shape_type == "cone":
                    builder.add_shape_cone(body, radius=0.3, half_height=0.4)
                elif shape_type == "mesh_cube":
                    builder.add_shape_mesh(body, mesh=cube_mesh)
                elif shape_type == "icosahedron":
                    builder.add_shape_convex_hull(body, mesh=ico_mesh)
                joint = builder.add_joint_free(body)
                builder.add_articulation([joint])

    return builder.finalize(device=device)


def main():
    device = "cuda:0"
    with wp.ScopedDevice(device):
        model = build_scene(device)

        pipeline_a = newton.CollisionPipeline(
            model, broad_phase="nxn", deterministic=True, reduce_contacts=True, rigid_contact_max=50000,
        )
        pipeline_b = newton.CollisionPipeline(
            model, broad_phase="nxn", deterministic=True, reduce_contacts=True, rigid_contact_max=50000,
        )
        contacts_a = pipeline_a.contacts()
        contacts_b = pipeline_b.contacts()

        solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_contact_relaxation=0.8)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        fps = 100
        sim_substeps = 10
        sim_dt = 1.0 / fps / sim_substeps

        checked_arrays = [
            "rigid_contact_shape0", "rigid_contact_shape1",
            "rigid_contact_point0", "rigid_contact_point1",
            "rigid_contact_normal",
            "rigid_contact_offset0", "rigid_contact_offset1",
            "rigid_contact_margin0", "rigid_contact_margin1",
        ]

        total_checks = 0
        failures = 0
        for _frame in range(50):
            for _sub in range(sim_substeps):
                state_0.clear_forces()
                pipeline_a.collide(state_0, contacts_a)
                pipeline_b.collide(state_0, contacts_b)

                count_a = int(contacts_a.rigid_contact_count.numpy()[0])
                count_b = int(contacts_b.rigid_contact_count.numpy()[0])
                if count_a != count_b:
                    print(f"COUNT MISMATCH at frame {_frame} sub {_sub}: {count_a} vs {count_b}")
                    failures += 1
                elif count_a > 0:
                    # Check sort keys
                    keys_a = pipeline_a._sort_key_array.numpy()[:count_a]
                    keys_b = pipeline_b._sort_key_array.numpy()[:count_a]
                    keys_match = np.array_equal(keys_a, keys_b)

                    for name in checked_arrays:
                        a = getattr(contacts_a, name).numpy()[:count_a]
                        b = getattr(contacts_b, name).numpy()[:count_a]
                        if not np.array_equal(a, b):
                            diff_mask = a != b
                            n_diff = int(np.count_nonzero(diff_mask))
                            print(f"\nDETERMINISM FAILURE: {name} at frame {_frame} sub {_sub}")
                            print(f"  {n_diff} elements differ, {count_a} contacts, keys_match={keys_match}")
                            diff_indices = np.argwhere(diff_mask)
                            for raw_idx in diff_indices[:5]:
                                tidx = tuple(raw_idx)
                                print(f"  [{tidx}]: a={a[tidx]!r}  b={b[tidx]!r}")
                            if not keys_match:
                                key_diff = np.argwhere(keys_a != keys_b)
                                print(f"  sort_key diffs at: {key_diff[:10].flatten().tolist()}")
                                for ki in key_diff[:3].flatten():
                                    print(f"    key[{ki}]: a=0x{keys_a[ki]:016x}  b=0x{keys_b[ki]:016x}")
                            # Show shape pairs for differing contacts
                            s0 = contacts_a.rigid_contact_shape0.numpy()[:count_a]
                            s1 = contacts_a.rigid_contact_shape1.numpy()[:count_a]
                            for raw_idx in diff_indices[:5]:
                                ci = raw_idx[0] if len(raw_idx) > 1 else int(raw_idx)
                                print(f"  contact[{ci}]: shapes=({s0[ci]}, {s1[ci]}), key=0x{keys_a[ci]:016x}")
                            failures += 1
                            break  # Only report first failing array per substep

                total_checks += 1
                solver.step(state_0, state_1, control, contacts_a, sim_dt)
                state_0, state_1 = state_1, state_0

        print(f"\n{'='*60}")
        print(f"Completed {total_checks} substep checks across 50 frames")
        print(f"Failures: {failures}")
        if failures == 0:
            print("ALL DETERMINISTIC!")


if __name__ == "__main__":
    main()
