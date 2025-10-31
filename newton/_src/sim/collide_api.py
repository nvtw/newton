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


from __future__ import annotations

import warp as wp

from ..core.types import Devicelike
from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.narrow_phase import NarrowPhase
from ..sim.contacts import Contacts
from ..sim.model import Model
from ..sim.state import State
from .collide_unified import BroadPhaseMode, compute_shape_aabbs, write_contact


@wp.kernel
def convert_narrow_phase_to_contacts_kernel(
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_position: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_penetration: wp.array(dtype=float),
    narrow_contact_count: wp.array(dtype=int),
    geom_data: wp.array(dtype=wp.vec4),  # Contains thickness in w component
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    rigid_contact_margin: float,
    contact_max: int,
    # Outputs (Contacts format)
    out_count: wp.array(dtype=int),
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_thickness0: wp.array(dtype=float),
    out_thickness1: wp.array(dtype=float),
    out_tids: wp.array(dtype=int),
):
    """
    Convert NarrowPhase output format to Contacts format using write_contact.

    NarrowPhase outputs:
    - contact_position: center point of contact
    - contact_normal: NEGATED normal (pointing from shape1 to shape0)
    - contact_penetration: d = distance - total_separation, negative if penetrating

    write_contact expects:
    - contact_normal_a_to_b: normal pointing from shape0 to shape1
    - contact_distance: distance such that d = contact_distance - (thickness_a + thickness_b)

    This kernel handles the conversion between these two formats.
    """
    idx = wp.tid()
    num_contacts = narrow_contact_count[0]

    if idx >= num_contacts:
        return

    # Get contact pair
    pair = contact_pair[idx]
    shape0 = pair[0]
    shape1 = pair[1]

    # Extract thickness values
    thickness_a = geom_data[shape0][3]
    thickness_b = geom_data[shape1][3]

    # Get contact data from narrow phase
    contact_point_center = contact_position[idx]
    # Narrow phase outputs negated normal (pointing B to A), but write_contact expects A to B
    contact_normal_a_to_b = contact_normal[idx]  # Undo the negation from narrow_phase
    # Narrow phase outputs penetration (negative when overlapping)
    # write_contact expects contact_distance such that when recomputed gives same d
    # Since d = distance - (radii + thickness), and we have d directly, we need to add back thickness
    contact_distance = contact_penetration[idx] + thickness_a + thickness_b

    # Get body inverse transforms
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]

    X_bw_a = wp.transform_identity() if body0 == -1 else wp.transform_inverse(body_q[body0])
    X_bw_b = wp.transform_identity() if body1 == -1 else wp.transform_inverse(body_q[body1])

    # Use write_contact to format the contact
    # radius_eff_a and radius_eff_b are 0 for most shapes (non-sphere-based)
    write_contact(
        contact_point_center,
        contact_normal_a_to_b,
        contact_distance,
        0.0,  # radius_eff_a
        0.0,  # radius_eff_b
        thickness_a,
        thickness_b,
        shape0,
        shape1,
        X_bw_a,
        X_bw_b,
        idx,
        rigid_contact_margin,
        contact_max,
        out_count,
        out_shape0,
        out_shape1,
        out_point0,
        out_point1,
        out_offset0,
        out_offset1,
        out_normal,
        out_thickness0,
        out_thickness1,
        out_tids,
    )


@wp.kernel
def prepare_geom_data_kernel(
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_thickness: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    # Outputs
    geom_data: wp.array(dtype=wp.vec4),  # scale xyz, thickness w
    geom_transform: wp.array(dtype=wp.transform),  # world space transform
):
    """Prepare geometry data arrays for NarrowPhase API."""
    idx = wp.tid()

    # Pack scale and thickness into geom_data
    scale = shape_scale[idx]
    thickness = shape_thickness[idx]
    geom_data[idx] = wp.vec4(scale[0], scale[1], scale[2], thickness)

    # Compute world space transform
    body_idx = shape_body[idx]
    if body_idx >= 0:
        geom_transform[idx] = wp.transform_multiply(body_q[body_idx], shape_transform[idx])
    else:
        geom_transform[idx] = shape_transform[idx]


class CollisionPipelineAPI:
    """
    Collision pipeline using NarrowPhase class for narrow phase collision detection.

    This is similar to CollisionPipelineUnified but uses the NarrowPhase API,
    mainly for testing purposes.
    """

    def __init__(
        self,
        shape_count: int,
        particle_count: int,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
        rigid_contact_max: int | None = None,
        rigid_contact_max_per_pair: int = 10,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        requires_grad: bool = False,
        device: Devicelike = None,
        broad_phase_mode: int = BroadPhaseMode.NXN,
    ):
        """
        Initialize the CollisionPipelineAPI.

        Args:
            shape_count: Number of shapes in the simulation
            particle_count: Number of particles in the simulation
            shape_pairs_filtered: Precomputed shape pairs for EXPLICIT broad phase mode
            rigid_contact_max: Maximum number of rigid contacts to allocate
            rigid_contact_max_per_pair: Maximum number of contact points per shape pair
            rigid_contact_margin: Margin for rigid contact generation (not used directly, but for cutoff)
            soft_contact_max: Maximum number of soft contacts to allocate
            soft_contact_margin: Margin for soft contact generation
            requires_grad: Whether to enable gradient computation
            device: The device on which to allocate arrays
            broad_phase_mode: Broad phase mode (NXN, SAP, or EXPLICIT)
        """
        self.contacts = None
        self.shape_count = shape_count
        self.broad_phase_mode = broad_phase_mode

        self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
        self.rigid_contact_margin = rigid_contact_margin

        if rigid_contact_max is not None:
            self.rigid_contact_max = rigid_contact_max
        else:
            self.rigid_contact_max = self.shape_pairs_max * rigid_contact_max_per_pair

        # Initialize broad phase
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            self.nxn_broadphase = BroadPhaseAllPairs()
            self.sap_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            max_num_negative_group_members = max(int(shape_count**0.5), 10)
            max_num_distinct_positive_groups = max(int(shape_count**0.5), 10)
            self.sap_broadphase = BroadPhaseSAP(
                max_broad_phase_elements=shape_count,
                max_num_distinct_positive_groups=max_num_distinct_positive_groups,
                max_num_negative_group_members=max_num_negative_group_members,
            )
            self.nxn_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        else:  # BroadPhaseMode.EXPLICIT
            if shape_pairs_filtered is None:
                raise ValueError("shape_pairs_filtered must be provided when using EXPLICIT mode")
            self.explicit_broadphase = BroadPhaseExplicit()
            self.nxn_broadphase = None
            self.sap_broadphase = None
            self.shape_pairs_filtered = shape_pairs_filtered
            self.shape_pairs_max = len(shape_pairs_filtered)

        # Initialize narrow phase
        self.narrow_phase = NarrowPhase()

        # Allocate buffers
        with wp.ScopedDevice(device):
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self.shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)

            # Narrow phase input/output arrays
            self.geom_data = wp.zeros(shape_count, dtype=wp.vec4, device=device)
            self.geom_transform = wp.zeros(shape_count, dtype=wp.transform, device=device)
            self.geom_cutoff = wp.full(shape_count, rigid_contact_margin, dtype=wp.float32, device=device)

            # Narrow phase output arrays
            self.narrow_contact_pair = wp.zeros(self.rigid_contact_max, dtype=wp.vec2i, device=device)
            self.narrow_contact_position = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=device)
            self.narrow_contact_normal = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=device)
            self.narrow_contact_penetration = wp.zeros(self.rigid_contact_max, dtype=wp.float32, device=device)
            self.narrow_contact_tangent = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=device)
            self.narrow_contact_count = wp.zeros(1, dtype=wp.int32, device=device)

        if soft_contact_max is None:
            soft_contact_max = shape_count * particle_count
        self.soft_contact_margin = soft_contact_margin
        self.soft_contact_max = soft_contact_max
        self.requires_grad = requires_grad

    def collide(self, model: Model, state: State) -> Contacts:
        """
        Run the collision pipeline using NarrowPhase.

        Args:
            model: The simulation model
            state: The current simulation state

        Returns:
            Contacts: The generated contacts
        """
        # Allocate or clear contacts
        if self.contacts is None or self.requires_grad:
            self.contacts = Contacts(
                self.rigid_contact_max,
                self.soft_contact_max,
                requires_grad=self.requires_grad,
                device=model.device,
            )
        else:
            self.contacts.clear()

        contacts = self.contacts

        # Clear counters
        self.broad_phase_pair_count.zero_()
        self.narrow_contact_count.zero_()
        contacts.rigid_contact_count.zero_()  # Clear since write_contact uses atomic_add

        # Compute AABBs for all shapes
        wp.launch(
            kernel=compute_shape_aabbs,
            dim=model.shape_count,
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_collision_radius,
                model.shape_source_ptr,
                self.rigid_contact_margin,
            ],
            outputs=[
                self.shape_aabb_lower,
                self.shape_aabb_upper,
            ],
            device=model.device,
        )

        # Run broad phase
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            self.nxn_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                model.shape_thickness,
                model.shape_collision_group,
                model.shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=model.device,
            )
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            self.sap_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                model.shape_thickness,
                model.shape_collision_group,
                model.shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=model.device,
            )
        else:  # BroadPhaseMode.EXPLICIT
            self.explicit_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                model.shape_thickness,  # Use thickness as cutoff
                self.shape_pairs_filtered,
                len(self.shape_pairs_filtered),
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=model.device,
            )

        # Prepare geometry data arrays for NarrowPhase API
        wp.launch(
            kernel=prepare_geom_data_kernel,
            dim=model.shape_count,
            inputs=[
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_thickness,
                state.body_q,
            ],
            outputs=[
                self.geom_data,
                self.geom_transform,
            ],
            device=model.device,
        )

        # Run narrow phase
        self.narrow_phase.launch(
            candidate_pair=self.broad_phase_shape_pairs,
            num_candidate_pair=self.broad_phase_pair_count,
            geom_types=model.shape_type,
            geom_data=self.geom_data,
            geom_transform=self.geom_transform,
            geom_source=model.shape_source_ptr,
            geom_cutoff=self.geom_cutoff,
            geom_collision_radius=model.shape_collision_radius,
            contact_pair=self.narrow_contact_pair,
            contact_position=self.narrow_contact_position,
            contact_normal=self.narrow_contact_normal,
            contact_penetration=self.narrow_contact_penetration,
            contact_tangent=self.narrow_contact_tangent,
            contact_count=self.narrow_contact_count,
            device=model.device,
            rigid_contact_margin=self.rigid_contact_margin,
        )

        # Convert NarrowPhase output to Contacts format using write_contact
        wp.launch(
            kernel=convert_narrow_phase_to_contacts_kernel,
            dim=self.rigid_contact_max,
            inputs=[
                self.narrow_contact_pair,
                self.narrow_contact_position,
                self.narrow_contact_normal,
                self.narrow_contact_penetration,
                self.narrow_contact_count,
                self.geom_data,
                state.body_q,
                model.shape_body,
                self.rigid_contact_margin,
                contacts.rigid_contact_max,
            ],
            outputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                contacts.rigid_contact_tids,
            ],
            device=model.device,
        )

        return contacts

    @classmethod
    def from_model(
        cls,
        model: Model,
        rigid_contact_max_per_pair: int | None = None,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        requires_grad: bool | None = None,
        broad_phase_mode: int = BroadPhaseMode.NXN,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
    ) -> CollisionPipelineAPI:
        """
        Create a CollisionPipelineAPI instance from a Model.

        Args:
            model: The simulation model
            rigid_contact_max_per_pair: Maximum number of contact points per shape pair
            rigid_contact_margin: Margin for rigid contact generation
            soft_contact_max: Maximum number of soft contacts to allocate
            soft_contact_margin: Margin for soft contact generation
            requires_grad: Whether to enable gradient computation
            broad_phase_mode: Broad phase mode
            shape_pairs_filtered: Precomputed shape pairs for EXPLICIT mode

        Returns:
            CollisionPipelineAPI: The constructed collision pipeline
        """
        rigid_contact_max = None
        if rigid_contact_max_per_pair is None:
            rigid_contact_max = model.rigid_contact_max
            rigid_contact_max_per_pair = 0
        if requires_grad is None:
            requires_grad = model.requires_grad

        if shape_pairs_filtered is None and broad_phase_mode == BroadPhaseMode.EXPLICIT:
            if hasattr(model, "shape_contact_pairs") and model.shape_contact_pairs is not None:
                shape_pairs_filtered = model.shape_contact_pairs
            else:
                shape_pairs_filtered = None

        return CollisionPipelineAPI(
            model.shape_count,
            model.particle_count,
            shape_pairs_filtered,
            rigid_contact_max,
            rigid_contact_max_per_pair,
            rigid_contact_margin,
            soft_contact_max,
            soft_contact_margin,
            requires_grad,
            model.device,
            broad_phase_mode,
        )
