"""Diagnostic script to track cube drift in PhoenX pyramid over 2000 frames."""

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.collision import PhoenXCollisionPipeline
from newton._src.solvers.phoenx.solver_phoenx import SolverState

wp.init()


def run_pyramid_drift_test():
    num_layers = 3
    h = 0.5
    spacing = 2.0 * h + 0.02

    box_positions = []
    for layer in range(num_layers):
        n = num_layers - layer
        z = layer * spacing + h
        offset = -(n - 1) * spacing * 0.5
        for row in range(n):
            for col in range(n):
                x = offset + col * spacing
                y = offset + row * spacing
                box_positions.append((x, y, z))

    num_boxes = len(box_positions)
    num_shapes = num_boxes + 1
    body_cap = num_boxes + 1
    contact_cap = max(num_boxes * 16, 512)

    device = "cpu"

    ss = SolverState(
        body_capacity=body_cap, contact_capacity=contact_cap,
        shape_count=num_shapes, device=device, default_friction=0.6,
    )
    pipeline = PhoenXCollisionPipeline(
        max_shapes=num_shapes, max_contacts=contact_cap, device=device,
    )

    h_ground = ss.add_body(position=(0, 0, 0), is_static=True)
    ground_row = int(ss.body_store.handle_to_index.numpy()[h_ground])
    ss.set_shape_body(0, h_ground)
    pipeline.add_shape_plane(body_row=ground_row)

    mass = 1.0
    inv_mass = 1.0 / mass
    inv_inertia = np.eye(3, dtype=np.float32) * (6.0 * inv_mass / (2.0 * h) ** 2)

    box_handles = []
    for i, (px, py, pz) in enumerate(box_positions):
        bh = ss.add_body(
            position=(px, py, pz), inverse_mass=inv_mass,
            inverse_inertia_local=inv_inertia,
            linear_damping=0.995, angular_damping=0.99,
        )
        shape_idx = i + 1
        ss.set_shape_body(shape_idx, bh)
        pipeline.add_shape_box(
            body_row=int(ss.body_store.handle_to_index.numpy()[bh]),
            half_extents=(h, h, h),
        )
        box_handles.append(bh)

    pipeline.finalize()

    initial_positions = np.array(box_positions, dtype=np.float32)

    dt = 1.0 / 60.0
    substeps = 8
    sub_dt = dt / substeps
    num_frames = 2000

    h2i = ss.body_store.handle_to_index.numpy()
    box_rows = [int(h2i[bh]) for bh in box_handles]

    # Track positions at intervals
    check_frames = [0, 100, 250, 500, 1000, 1500, 2000]
    snapshots = {}

    for frame in range(num_frames + 1):
        if frame in check_frames:
            wp.synchronize_device(device)
            positions = ss.body_store.column_of("position").numpy()
            velocities = ss.body_store.column_of("velocity").numpy()
            ang_vels = ss.body_store.column_of("angular_velocity").numpy()
            nc = ss.contact_store.count.numpy()[0]

            snapshot = []
            for i, row in enumerate(box_rows):
                pos = positions[row].copy()
                vel = velocities[row].copy()
                angvel = ang_vels[row].copy()
                snapshot.append((pos, vel, angvel))
            snapshots[frame] = (snapshot, nc)

        if frame >= num_frames:
            break

        ss.update_world_inertia()
        for _ in range(substeps):
            ss.warm_starter.begin_frame()
            pipeline.collide(ss)
            ss.step(sub_dt, gravity=(0, 0, -9.81), num_iterations=12)
            ss.export_impulses()

    # Report
    print(f"\nPyramid drift analysis ({num_boxes} boxes, {num_layers} layers)")
    print(f"Settings: substeps={substeps}, PGS iterations=12, friction=0.6")
    print(f"Damping: linear=0.995, angular=0.99")
    print("=" * 80)

    for frame in check_frames:
        snapshot, nc = snapshots[frame]
        print(f"\n--- Frame {frame} (contacts: {nc}) ---")

        max_z_drift = 0.0
        max_xy_drift = 0.0
        max_vel = 0.0
        max_angvel = 0.0

        for i, (pos, vel, angvel) in enumerate(snapshot):
            init = initial_positions[i]
            z_drift = pos[2] - init[2]
            xy_drift = np.sqrt((pos[0] - init[0]) ** 2 + (pos[1] - init[1]) ** 2)
            speed = np.linalg.norm(vel)
            aspeed = np.linalg.norm(angvel)

            max_z_drift = max(max_z_drift, abs(z_drift))
            max_xy_drift = max(max_xy_drift, xy_drift)
            max_vel = max(max_vel, speed)
            max_angvel = max(max_angvel, aspeed)

        print(f"  Max Z drift:       {max_z_drift:.6f} m")
        print(f"  Max XY drift:      {max_xy_drift:.6f} m")
        print(f"  Max linear speed:  {max_vel:.6f} m/s")
        print(f"  Max angular speed: {max_angvel:.6f} rad/s")

        # Print per-box details for the last snapshot
        if frame == check_frames[-1]:
            print(f"\n  Per-box details at frame {frame}:")
            for i, (pos, vel, angvel) in enumerate(snapshot):
                init = initial_positions[i]
                z_drift = pos[2] - init[2]
                xy_drift = np.sqrt((pos[0] - init[0]) ** 2 + (pos[1] - init[1]) ** 2)
                layer = 0
                count = 0
                for l in range(num_layers):
                    nl = (num_layers - l) ** 2
                    if i < count + nl:
                        layer = l
                        break
                    count += nl
                print(f"    Box {i:2d} (layer {layer}): "
                      f"pos=({pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f})  "
                      f"z_drift={z_drift:+.6f}  xy_drift={xy_drift:.6f}  "
                      f"|v|={np.linalg.norm(vel):.6f}  |w|={np.linalg.norm(angvel):.6f}")

    # Check: are accumulated impulses reasonable?
    print("\n\n--- Contact store diagnostics at final frame ---")
    cs = ss.contact_store
    nc = cs.count.numpy()[0]
    if nc > 0:
        acc_n = cs.column_of("accumulated_normal_impulse").numpy()[:nc]
        print(f"  Active contacts: {nc}")
        print(f"  Normal impulse: min={acc_n.min():.6f}, max={acc_n.max():.6f}, mean={acc_n.mean():.6f}")

        biases = cs.column_of("bias").numpy()[:nc]
        print(f"  Bias: min={biases.min():.6f}, max={biases.max():.6f}, mean={biases.mean():.6f}")

        # Check body indices in contacts
        b0 = cs.column_of("body0").numpy()[:nc]
        b1 = cs.column_of("body1").numpy()[:nc]
        print(f"  Body0 range: [{b0.min()}, {b0.max()}]")
        print(f"  Body1 range: [{b1.min()}, {b1.max()}]")

        # Contact normals
        normals = cs.column_of("normal").numpy()[:nc]
        print(f"  Normal Z component: min={normals[:, 2].min():.4f}, max={normals[:, 2].max():.4f}")

    # Partitioning diagnostics
    gc = ss.graph_coloring
    num_partitions = gc.num_colors.numpy()[0] if hasattr(gc, 'num_colors') else -1
    print(f"\n  Graph coloring partitions: {num_partitions}")
    if hasattr(gc, 'partition_ends'):
        ends = gc.partition_ends.numpy()
        nz = ends[ends > 0]
        if len(nz) > 0:
            sizes = np.diff(np.concatenate([[0], nz]))
            print(f"  Partition sizes: {sizes.tolist()}")


if __name__ == "__main__":
    run_pyramid_drift_test()
