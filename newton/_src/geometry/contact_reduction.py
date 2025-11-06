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


import math

import numpy as np
import warp as wp


def get_icosahedron_face_normals(subdivisions: int = 0) -> np.ndarray:
    """
    Generate face normals for an icosahedron with optional subdivisions.

    Args:
        subdivisions: Number of subdivision iterations to apply (0 = base icosahedron)

    Returns:
        np.ndarray: Array of face normal vectors, shape (num_faces, 3)
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Base icosahedron vertices (12 vertices)
    vertices = np.array(
        [
            # Rectangle in XY plane
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            # Rectangle in YZ plane
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            # Rectangle in XZ plane
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )

    # Normalize vertices to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Base icosahedron faces (20 triangular faces)
    faces = np.array(
        [
            # 5 faces around point 0
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            # 5 adjacent faces
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            # 5 faces around point 3
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            # 5 adjacent faces
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    # Apply subdivisions
    current_vertices = vertices.copy()
    current_faces = faces.copy()

    for _ in range(subdivisions):
        current_vertices, current_faces = _subdivide_icosahedron(current_vertices, current_faces)

    # Compute face normals
    face_normals = []
    for face in current_faces:
        v0, v1, v2 = current_vertices[face]
        # Compute normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        # Normalize
        normal = normal / np.linalg.norm(normal)
        face_normals.append(normal)

    return np.array(face_normals, dtype=np.float32)


def _subdivide_icosahedron(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Subdivide an icosahedron by splitting each triangle into 4 smaller triangles.

    Args:
        vertices: Current vertices array
        faces: Current faces array

    Returns:
        tuple: (new_vertices, new_faces)
    """
    # Dictionary to store midpoint vertices to avoid duplicates
    midpoint_cache = {}
    new_vertices = list(vertices)

    def get_midpoint(i: int, j: int) -> int:
        """Get or create midpoint vertex between vertices i and j"""
        # Use sorted indices as key to avoid duplicates
        key = tuple(sorted([i, j]))
        if key in midpoint_cache:
            return midpoint_cache[key]

        # Compute midpoint and project to unit sphere
        midpoint = (vertices[i] + vertices[j]) / 2.0
        midpoint = midpoint / np.linalg.norm(midpoint)

        # Add to vertices list
        vertex_index = len(new_vertices)
        new_vertices.append(midpoint)
        midpoint_cache[key] = vertex_index

        return vertex_index

    # Subdivide each face into 4 triangles
    new_faces = []
    for face in faces:
        v0, v1, v2 = face

        # Get midpoint vertices
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)

        # Create 4 new triangular faces
        new_faces.extend(
            [
                [v0, m01, m20],  # Corner triangle at v0
                [v1, m12, m01],  # Corner triangle at v1
                [v2, m20, m12],  # Corner triangle at v2
                [m01, m12, m20],  # Center triangle
            ]
        )

    return np.array(new_vertices, dtype=np.float32), np.array(new_faces, dtype=np.int32)


def get_binning_kernels(n_bin_dirs: int, num_normal_bins: int, num_betas: int, sticky_contacts: float = 1e-6):
    """
    Factory method for creating binning kernels that output contact indices.
    """

    @wp.kernel
    def compute_bin_scores(
        grid_size: int,
        contact_count: wp.array(dtype=int),
        contact_pair: wp.array(dtype=wp.vec2i),
        contact_normal: wp.array(dtype=wp.vec3),
        contact_position: wp.array(dtype=wp.vec3),
        contact_penetration: wp.array(dtype=float),
        contact_id: wp.array(dtype=wp.uint32),
        shape_pair_to_bin: wp.array(dtype=wp.int32),
        bin_normals: wp.array(dtype=wp.vec3),
        penetration_betas: wp.array(dtype=wp.float32),
        binned_contact_id_prev: wp.array(dtype=wp.uint32, ndim=3),
        shape_pair_to_bin_prev: wp.array(dtype=wp.int32),
        # outputs
        binned_dot_product: wp.array(dtype=wp.float32, ndim=3),
        bin_occupied: wp.array(dtype=wp.bool, ndim=2),
        contact_normal_bin_idx: wp.array(dtype=wp.int32),
        bin_to_shape_pair_idx: wp.array(dtype=wp.int32),
        contact_to_bin_idx: wp.array(dtype=wp.int32),
    ):
        offset = wp.tid()
        for tid in range(offset, contact_count[0], grid_size):
            shape_pair = contact_pair[tid]
            # Create a unique hash for the shape pair
            pair_hash = shape_pair[0] * 1000000 + shape_pair[1]
            bin_idx = shape_pair_to_bin[pair_hash]

            normal = contact_normal[tid]
            position = contact_position[tid]
            depth = -contact_penetration[tid]  # Convert penetration to depth (negative overlap)
            id = contact_id[tid]

            # Find the normal bin which is closest to the contact normal
            max_dot_product = wp.float32(-1e10)
            bin_normal_idx = wp.int32(-1)
            for n_idx in range(wp.static(num_normal_bins)):
                dp = wp.dot(normal, bin_normals[n_idx])
                dp -= float(n_idx) * 1e-6  # for breaking ties in symmetric shapes
                if dp > max_dot_product:
                    max_dot_product = dp
                    bin_normal_idx = n_idx

            bin_normal_dir = bin_normals[bin_normal_idx]

            bin_to_shape_pair_idx[bin_idx] = pair_hash
            bin_occupied[bin_idx, bin_normal_idx] = True
            contact_normal_bin_idx[tid] = bin_normal_idx
            contact_to_bin_idx[tid] = bin_idx

            # Project position to the plane corresponding to the bin normal
            temp_vec = wp.vec3(0.0, 1.0, 0.0)
            if wp.abs(wp.dot(bin_normal_dir, temp_vec)) > 0.95:
                temp_vec = wp.vec3(0.0, 0.0, 1.0)

            # Create orthogonal basis vectors in the plane
            plane_u = wp.normalize(wp.cross(bin_normal_dir, temp_vec))
            plane_v = wp.normalize(wp.cross(bin_normal_dir, plane_u))

            # Convert position to 2D coordinates in the plane basis
            position_2d = wp.vec2(wp.dot(position, plane_u), wp.dot(position, plane_v))

            angle_increment = wp.static(2.0 * math.pi * (1.0 / n_bin_dirs))
            bin_idx_prev = shape_pair_to_bin_prev[pair_hash]

            # Loop over bin_directions, store the max dot product for each direction
            for dir_idx in range(wp.static(n_bin_dirs)):
                angle = float(dir_idx) * angle_increment
                direction_2d = wp.vec2(wp.cos(angle), wp.sin(angle))
                spatial_dot_product = wp.dot(position_2d, direction_2d)
                for i in range(wp.static(num_betas)):
                    offset = i * n_bin_dirs
                    idx_last = dir_idx + offset
                    dp = spatial_dot_product + depth * penetration_betas[i]
                    if wp.static(sticky_contacts > 0.0):
                        id_prev = binned_contact_id_prev[bin_idx_prev, bin_normal_idx, idx_last]
                        if id_prev == id:
                            dp += wp.static(sticky_contacts)

                    wp.atomic_max(binned_dot_product, bin_idx, bin_normal_idx, idx_last, dp)

    @wp.kernel
    def assign_contacts_to_bins(
        grid_size: int,
        contact_count: wp.array(dtype=int),
        contact_pair: wp.array(dtype=wp.vec2i),
        contact_normal: wp.array(dtype=wp.vec3),
        contact_normal_bin_idx: wp.array(dtype=wp.int32),
        contact_position: wp.array(dtype=wp.vec3),
        contact_penetration: wp.array(dtype=float),
        contact_id: wp.array(dtype=wp.uint32),
        contact_to_bin_idx: wp.array(dtype=wp.int32),
        shape_pair_to_bin: wp.array(dtype=wp.int32),
        bin_normals: wp.array(dtype=wp.vec3),
        penetration_betas: wp.array(dtype=wp.float32),
        binned_dot_product: wp.array(dtype=wp.float32, ndim=3),
        binned_contact_id_prev: wp.array(dtype=wp.uint32, ndim=3),
        shape_pair_to_bin_prev: wp.array(dtype=wp.int32),
        # outputs
        binned_contact_idx: wp.array(dtype=wp.int32, ndim=3),
        binned_contact_id: wp.array(dtype=wp.uint32, ndim=3),
    ):
        offset = wp.tid()
        for tid in range(offset, contact_count[0], grid_size):
            shape_pair = contact_pair[tid]
            pair_hash = shape_pair[0] * 1000000 + shape_pair[1]
            bin_idx = contact_to_bin_idx[tid]
            bin_normal_idx = contact_normal_bin_idx[tid]

            bin_normal_dir = bin_normals[bin_normal_idx]
            position = contact_position[tid]
            depth = -contact_penetration[tid]
            id = contact_id[tid]

            # Project position to the plane corresponding to the bin normal
            temp_vec = wp.vec3(0.0, 1.0, 0.0)
            if wp.abs(wp.dot(bin_normal_dir, temp_vec)) > 0.95:
                temp_vec = wp.vec3(0.0, 0.0, 1.0)

            # Create orthogonal basis vectors in the plane
            plane_u = wp.normalize(wp.cross(bin_normal_dir, temp_vec))
            plane_v = wp.normalize(wp.cross(bin_normal_dir, plane_u))

            # Convert position to 2D coordinates in the plane basis
            position_2d = wp.vec2(wp.dot(position, plane_u), wp.dot(position, plane_v))

            angle_increment = wp.static(2.0 * math.pi * (1.0 / n_bin_dirs))
            bin_idx_prev = shape_pair_to_bin_prev[pair_hash]

            for dir_idx in range(wp.static(n_bin_dirs)):
                angle = float(dir_idx) * angle_increment
                direction_2d = wp.vec2(wp.cos(angle), wp.sin(angle))
                spatial_dot_product = wp.dot(position_2d, direction_2d)
                for i in range(wp.static(num_betas)):
                    offset = i * n_bin_dirs
                    idx_last = dir_idx + offset
                    dp = spatial_dot_product + depth * penetration_betas[i]
                    if wp.static(sticky_contacts > 0.0):
                        contact_idx_prev = binned_contact_idx_prev[bin_idx_prev, bin_normal_idx, idx_last]
                        if contact_idx_prev >= 0 and contact_id[contact_idx_prev] == id:
                            dp += wp.static(sticky_contacts)
                    max_dp = binned_dot_product[bin_idx, bin_normal_idx, idx_last]
                    if dp >= max_dp:
                        binned_contact_idx[bin_idx, bin_normal_idx, idx_last] = tid

    @wp.kernel(enable_backward=False)
    def extract_contacts_from_bins(
        binned_contact_idx: wp.array(dtype=wp.int32, ndim=3),
        bin_occupied: wp.array(dtype=wp.bool, ndim=2),
        # outputs
        kept_contact_indices: wp.array(dtype=wp.int32),
        kept_contact_count: wp.array(dtype=int),
    ):
        bin_idx, normal_bin_idx = wp.tid()
        if not bin_occupied[bin_idx, normal_bin_idx]:
            return

        # Deduplicate contacts based on contact index
        n_bins = wp.static(num_betas * n_bin_dirs)

        unique_indices = wp.zeros(shape=(n_bins,), dtype=wp.int32)
        num_unique_contacts = wp.int32(1)
        unique_indices[0] = binned_contact_idx[bin_idx, normal_bin_idx, 0]

        for i in range(1, n_bins):
            contact_idx_i = binned_contact_idx[bin_idx, normal_bin_idx, i]
            if contact_idx_i < 0:
                continue
            found_duplicate = wp.bool(False)
            for j in range(num_unique_contacts):
                if unique_indices[j] == contact_idx_i:
                    found_duplicate = True
                    break
            if not found_duplicate:
                unique_indices[num_unique_contacts] = contact_idx_i
                num_unique_contacts += 1

        # Atomic add to get output position and write unique contact indices
        output_idx = wp.atomic_add(kept_contact_count, 0, num_unique_contacts)

        for i in range(n_bins):
            if i >= num_unique_contacts:
                return
            contact_idx = unique_indices[i]
            if contact_idx >= 0:
                kept_contact_indices[output_idx + i] = contact_idx

    return compute_bin_scores, assign_contacts_to_bins, extract_contacts_from_bins


@wp.kernel
def build_shape_pair_mask(
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_count: wp.array(dtype=int),
    shape_pair_mask: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    shape_pair = contact_pair[tid]
    pair_hash = shape_pair[0] * 1000000 + shape_pair[1]
    shape_pair_mask[pair_hash] = 1


class ContactReduction:
    def __init__(
        self,
        max_shape_pairs: int = 100000,
        max_contacts: int = 1000000,
        normal_subdivisions: int = 0,
        betas: tuple = (10.0, 20.0, 1000.0),
        bin_directions: int = 7,
        sticky_contacts: float = 1e-6,
        device=None,
    ):
        """
        Initialize contact reduction.

        Args:
            max_shape_pairs: Maximum number of unique shape pairs
            max_contacts: Maximum number of contacts
            normal_subdivisions: Icosahedron subdivision level for normal binning
            betas: Tuple of penetration weighting factors
            bin_directions: Number of spatial directions per normal bin
            sticky_contacts: Bonus score for previously selected contacts (temporal coherence)
            device: Warp device to use
        """
        self.max_shape_pairs = max_shape_pairs
        self.max_contacts = max_contacts
        self.grid_size = 256 * 8 * 128

        with wp.ScopedDevice(device):
            self.bin_normals = wp.array(
                get_icosahedron_face_normals(subdivisions=normal_subdivisions),
                dtype=wp.vec3,
            )
            self.penetration_betas = wp.array(betas, dtype=wp.float32)
            self.num_betas = len(betas)
            self.bin_directions = bin_directions
            num_normal_bins = self.bin_normals.shape[0]

            # Shape pair tracking
            self.shape_pair_mask = wp.zeros(max_shape_pairs, dtype=wp.int32)
            self.shape_pair_to_bin = wp.zeros(max_shape_pairs, dtype=wp.int32)
            self.shape_pair_to_bin_prev = wp.clone(self.shape_pair_to_bin)
            self.bin_to_shape_pair_idx = wp.zeros(max_shape_pairs, dtype=wp.int32)

            # Binned data (stores contact indices instead of contact data)
            self.binned_dot_product = wp.zeros(
                (max_shape_pairs, num_normal_bins, self.num_betas * bin_directions), dtype=wp.float32
            )
            self.binned_contact_idx = wp.full(
                (max_shape_pairs, num_normal_bins, self.num_betas * bin_directions), dtype=wp.int32, value=-1
            )
            self.binned_contact_idx_prev = wp.clone(self.binned_contact_idx)

            self.bin_occupied = wp.zeros((max_shape_pairs, num_normal_bins), dtype=wp.bool)

            # Per-contact temporary data
            self.contact_normal_bin_idx = wp.empty(max_contacts, dtype=wp.int32)
            self.contact_to_bin_idx = wp.empty(max_contacts, dtype=wp.int32)

        self.compute_bin_scores, self.assign_contacts_to_bins, self.extract_contacts_from_bins = get_binning_kernels(
            bin_directions,
            num_normal_bins,
            self.num_betas,
            sticky_contacts,
        )

    def launch(
        self,
        # Input arrays from narrow phase
        contact_pair: wp.array(dtype=wp.vec2i),
        contact_position: wp.array(dtype=wp.vec3),
        contact_normal: wp.array(dtype=wp.vec3),
        contact_penetration: wp.array(dtype=float),
        contact_id: wp.array(dtype=wp.uint32),
        contact_count: wp.array(dtype=int),
        # Output arrays
        kept_contact_indices: wp.array(dtype=wp.int32),
        kept_contact_count: wp.array(dtype=int),
    ):
        """
        Perform contact reduction and output indices of contacts to keep.

        Args:
            contact_pair: Shape pair for each contact (vec2i)
            contact_position: Contact position in world space
            contact_normal: Contact normal in world space
            contact_penetration: Contact penetration depth (negative = overlap)
            contact_id: Contact feature ID for temporal coherence
            contact_count: Number of input contacts
            kept_contact_indices: Output array of contact indices to keep
            kept_contact_count: Output count of kept contacts
        """
        device = contact_count.device

        # Save previous state for temporal coherence
        wp.copy(self.binned_contact_idx_prev, self.binned_contact_idx)
        wp.copy(self.shape_pair_to_bin_prev, self.shape_pair_to_bin)

        # Reset arrays
        self.binned_dot_product.fill_(-1e10)
        self.binned_contact_idx.fill_(-1)
        self.bin_occupied.zero_()
        self.shape_pair_mask.zero_()
        self.bin_to_shape_pair_idx.fill_(-1)
        kept_contact_count.zero_()

        # Build shape pair mask and compute bin indices
        wp.launch(
            build_shape_pair_mask,
            dim=[self.max_contacts],
            inputs=[contact_pair, contact_count],
            outputs=[self.shape_pair_mask],
            device=device,
        )

        wp.utils.array_scan(self.shape_pair_mask, self.shape_pair_to_bin, inclusive=False)

        # Compute bin scores
        wp.launch(
            kernel=self.compute_bin_scores,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                contact_count,
                contact_pair,
                contact_normal,
                contact_position,
                contact_penetration,
                contact_id,
                self.shape_pair_to_bin,
                self.bin_normals,
                self.penetration_betas,
                self.binned_contact_idx_prev,
                self.shape_pair_to_bin_prev,
            ],
            outputs=[
                self.binned_dot_product,
                self.bin_occupied,
                self.contact_normal_bin_idx,
                self.bin_to_shape_pair_idx,
                self.contact_to_bin_idx,
            ],
            device=device,
        )

        # Assign contacts to bins
        wp.launch(
            kernel=self.assign_contacts_to_bins,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                contact_count,
                contact_pair,
                contact_normal,
                self.contact_normal_bin_idx,
                contact_position,
                contact_penetration,
                contact_id,
                self.contact_to_bin_idx,
                self.shape_pair_to_bin,
                self.bin_normals,
                self.penetration_betas,
                self.binned_dot_product,
                self.binned_contact_idx_prev,
                self.shape_pair_to_bin_prev,
            ],
            outputs=[
                self.binned_contact_idx,
            ],
            device=device,
        )

        # Extract unique contact indices from bins
        wp.launch(
            kernel=self.extract_contacts_from_bins,
            dim=[self.binned_contact_idx.shape[0], self.binned_contact_idx.shape[1]],
            inputs=[
                self.binned_contact_idx,
                self.bin_occupied,
            ],
            outputs=[
                kept_contact_indices,
                kept_contact_count,
            ],
            device=device,
        )
