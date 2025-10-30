# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Compact procedural terrain generator for Newton physics examples.

Provides various terrain generation functions that output Newton-compatible triangle meshes.
Supports creating grids of terrain blocks with different procedural patterns.
"""

import numpy as np
import trimesh


# ============================================================================
# Primitive Terrain Functions
# ============================================================================


def flat_terrain(size, height=0.0):
    """Generate a flat plane terrain."""
    x0 = [size[0], size[1], height]
    x1 = [size[0], 0.0, height]
    x2 = [0.0, size[1], height]
    x3 = [0.0, 0.0, height]
    vertices = np.array([x0, x1, x2, x3], dtype=np.float32)
    faces = np.array([[1, 0, 2], [2, 3, 1]], dtype=np.int32)
    return vertices, faces.flatten()


def pyramid_stairs_terrain(size, step_width=0.5, step_height=0.1, platform_width=1.0):
    """Generate pyramid stairs terrain with steps converging to center platform."""
    meshes = []
    center = [size[0] / 2, size[1] / 2, 0.0]

    num_steps_x = int((size[0] - platform_width) / (2 * step_width))
    num_steps_y = int((size[1] - platform_width) / (2 * step_width))
    num_steps = min(num_steps_x, num_steps_y)

    # Add ground plane
    ground = trimesh.creation.box(
        (size[0], size[1], step_height),
        trimesh.transformations.translation_matrix([center[0], center[1], -step_height / 2]),
    )
    meshes.append(ground)

    # Create concentric rectangular steps (including final ring around platform)
    for k in range(num_steps + 1):
        box_size = (size[0] - 2 * k * step_width, size[1] - 2 * k * step_width)
        box_z = center[2] + (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        box_height = (k + 1) * step_height

        # Skip if this would be smaller than the platform
        if box_size[0] <= platform_width or box_size[1] <= platform_width:
            continue

        # Top/bottom/left/right boxes
        for dx, dy, sx, sy in [
            (0, size[1] / 2 - box_offset, box_size[0], step_width),  # top
            (0, -size[1] / 2 + box_offset, box_size[0], step_width),  # bottom
            (size[0] / 2 - box_offset, 0, step_width, box_size[1] - 2 * step_width),  # right
            (-size[0] / 2 + box_offset, 0, step_width, box_size[1] - 2 * step_width),  # left
        ]:
            pos = (center[0] + dx, center[1] + dy, box_z)
            mesh = trimesh.creation.box((sx, sy, box_height), trimesh.transformations.translation_matrix(pos))
            meshes.append(mesh)

    # Center platform (two steps higher than the last step ring)
    platform_height = (num_steps + 2) * step_height
    box_dims = (platform_width, platform_width, platform_height)
    box_pos = (center[0], center[1], center[2] + platform_height / 2)
    meshes.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    return _combine_meshes(meshes)


def random_grid_terrain(size, grid_width=0.5, grid_height_range=(-0.15, 0.15), platform_width=None, seed=None):
    """Generate terrain with randomized height grid cells."""
    rng = np.random.default_rng(seed)

    num_boxes_x = int(size[0] / grid_width)
    num_boxes_y = int(size[1] / grid_width)

    # Template box for a grid cell
    template = trimesh.creation.box((grid_width, grid_width, 1.0))
    vertices = template.vertices
    faces = template.faces

    # Create grid with random heights
    all_vertices = []
    all_faces = []
    vertex_count = 0

    for ix in range(num_boxes_x):
        for iy in range(num_boxes_y):
            # Position grid cells starting from (0, 0) with proper alignment
            x = ix * grid_width + grid_width / 2
            y = iy * grid_width + grid_width / 2
            h_noise = rng.uniform(*grid_height_range)

            # Offset vertices (trimesh box is centered at origin)
            v = vertices.copy()
            v[:, 0] += x
            v[:, 1] += y
            v[:, 2] -= 0.5
            v[v[:, 2] > -0.5, 2] += h_noise  # Only raise top vertices

            all_vertices.append(v)
            all_faces.append(faces + vertex_count)
            vertex_count += 8  # Each box has 8 vertices

    vertices = np.vstack(all_vertices).astype(np.float32)
    faces = np.vstack(all_faces).astype(np.int32)

    return vertices, faces.flatten()


def wave_terrain(size, wave_amplitude=0.3, wave_frequency=2.0, resolution=50):
    """Generate 2D sine wave terrain with zero boundaries."""
    x = np.linspace(0, size[0], resolution)
    y = np.linspace(0, size[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Create 2D sine pattern that is naturally zero at all boundaries
    # sin(n*pi*x/L) is zero at x=0 and x=L for integer n
    Z = wave_amplitude * np.sin(wave_frequency * np.pi * X / size[0]) * np.sin(wave_frequency * np.pi * Y / size[1])

    # Create vertices and faces
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)

    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = i * resolution + (j + 1)
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + (j + 1)
            # Counter-clockwise winding for upward-facing triangles
            faces.append([v0, v1, v2])
            faces.append([v2, v1, v3])

    return vertices, np.array(faces, dtype=np.int32).flatten()


def box_terrain(size, box_height=0.5, platform_width=1.5):
    """Generate terrain with a raised box platform in center."""
    meshes = []

    # Ground plane
    ground = trimesh.creation.box(
        (size[0], size[1], 1.0), trimesh.transformations.translation_matrix([size[0] / 2, size[1] / 2, -0.5])
    )
    meshes.append(ground)

    # Raised platform
    platform = trimesh.creation.box(
        (platform_width, platform_width, 1.0 + box_height),
        trimesh.transformations.translation_matrix([size[0] / 2, size[1] / 2, box_height / 2 - 0.5]),
    )
    meshes.append(platform)

    return _combine_meshes(meshes)


def gap_terrain(size, gap_width=0.8, platform_width=1.2):
    """Generate terrain with a gap around the center platform."""
    meshes = []
    center = (size[0] / 2, size[1] / 2, -0.5)

    # Outer border
    thickness_x = (size[0] - platform_width - 2 * gap_width) / 2
    thickness_y = (size[1] - platform_width - 2 * gap_width) / 2

    for dx, dy, sx, sy in [
        (0, (size[1] - thickness_y) / 2, size[0], thickness_y),  # top
        (0, -(size[1] - thickness_y) / 2, size[0], thickness_y),  # bottom
        ((size[0] - thickness_x) / 2, 0, thickness_x, platform_width + 2 * gap_width),  # right
        (-(size[0] - thickness_x) / 2, 0, thickness_x, platform_width + 2 * gap_width),  # left
    ]:
        pos = (center[0] + dx, center[1] + dy, center[2])
        meshes.append(trimesh.creation.box((sx, sy, 1.0), trimesh.transformations.translation_matrix(pos)))

    # Center platform
    meshes.append(
        trimesh.creation.box((platform_width, platform_width, 1.0), trimesh.transformations.translation_matrix(center))
    )

    return _combine_meshes(meshes)


# ============================================================================
# Terrain Grid Generator
# ============================================================================


def generate_terrain_grid(grid_size=(4, 4), block_size=(5.0, 5.0), terrain_types=None, terrain_params=None, seed=None):
    """Generate a grid of procedural terrain blocks.

    Args:
        grid_size: (rows, cols) number of terrain blocks
        block_size: (width, height) size of each terrain block in meters
        terrain_types: List of terrain type names or callable functions. If None, uses all types.
                      Available types: 'flat', 'pyramid_stairs', 'random_grid', 'wave', 'box', 'gap'
        terrain_params: Dictionary mapping terrain types to their parameter dicts
        seed: Random seed for reproducibility

    Returns:
        vertices: (N, 3) float32 array of vertex positions
        indices: (M,) int32 array of triangle indices
    """

    # Default terrain types
    if terrain_types is None:
        terrain_types = ["flat", "pyramid_stairs", "random_grid", "wave", "box", "gap"]

    terrain_funcs = {
        "flat": flat_terrain,
        "pyramid_stairs": pyramid_stairs_terrain,
        "random_grid": random_grid_terrain,
        "wave": wave_terrain,
        "box": box_terrain,
        "gap": gap_terrain,
    }

    if terrain_params is None:
        terrain_params = {}

    all_vertices = []
    all_indices = []
    vertex_offset = 0

    rows, cols = grid_size

    for row in range(rows):
        for col in range(cols):
            # Select terrain type (cycle or random)
            if isinstance(terrain_types, list):
                terrain_idx = (row * cols + col) % len(terrain_types)
                terrain_name = terrain_types[terrain_idx]
            else:
                terrain_name = terrain_types

            # Get terrain function
            if callable(terrain_name):
                terrain_func = terrain_name
            else:
                terrain_func = terrain_funcs[terrain_name]

            # Get parameters for this terrain type
            params = terrain_params.get(terrain_name, {})

            # Generate terrain block
            vertices, indices = terrain_func(block_size, **params)

            # Offset to grid position
            offset_x = col * block_size[0]
            offset_y = row * block_size[1]
            vertices[:, 0] += offset_x
            vertices[:, 1] += offset_y

            # Accumulate geometry
            all_vertices.append(vertices)
            all_indices.append(indices + vertex_offset)
            vertex_offset += len(vertices)

    # Combine all blocks
    vertices = np.vstack(all_vertices).astype(np.float32)
    indices = np.concatenate(all_indices).astype(np.int32)

    return vertices, indices


# ============================================================================
# Helper Functions
# ============================================================================


def _combine_meshes(meshes):
    """Combine multiple trimesh objects into a single (vertices, indices) tuple."""
    if len(meshes) == 1:
        mesh = meshes[0]
        return mesh.vertices.astype(np.float32), mesh.faces.flatten().astype(np.int32)

    combined = trimesh.util.concatenate(meshes)
    return combined.vertices.astype(np.float32), combined.faces.flatten().astype(np.int32)


def to_newton_mesh(vertices, indices):
    """Convert terrain geometry to Newton mesh format.

    This is a convenience function that ensures proper dtypes.

    Args:
        vertices: (N, 3) array of vertex positions
        indices: (M,) array of triangle indices (flattened)

    Returns:
        Tuple of (vertices, indices) with proper dtypes for Newton
    """
    return vertices.astype(np.float32), indices.astype(np.int32)
