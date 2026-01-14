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
from warp.context import Devicelike

# No kernel needed - we just zero counts!


class ContactDebugInfo:
    """
    Graph-capture-compatible debug information for contacts.

    This class stores contact counts in pinned memory that can be asynchronously
    copied from device memory without breaking CUDA graph capture.
    """

    def __init__(self, device: Devicelike = None):
        """Initialize debug info with pinned memory for graph-compatible copies.

        Args:
            device: Device to allocate arrays on.
        """
        with wp.ScopedDevice(device):
            # Device-side count (will be copied from contact arrays)
            self.rigid_count_device = wp.zeros(1, dtype=int, device=device)
            self.soft_count_device = wp.zeros(1, dtype=int, device=device)

            # Pinned host memory for async copy (graph-compatible)
            self.rigid_count_host = wp.zeros(1, dtype=int, pinned=True, device="cpu")
            self.soft_count_host = wp.zeros(1, dtype=int, pinned=True, device="cpu")

            self.device = device

    def copy_counts(self, rigid_contact_count: wp.array, soft_contact_count: wp.array):
        """Copy contact counts to pinned host memory (graph-compatible).

        Args:
            rigid_contact_count: Device array with rigid contact count.
            soft_contact_count: Device array with soft contact count.
        """
        # Copy to pinned host memory (async, graph-compatible)
        wp.copy(self.rigid_count_host, rigid_contact_count)
        wp.copy(self.soft_count_host, soft_contact_count)

    def get_rigid_count(self) -> int:
        """Get the latest rigid contact count (non-blocking read from pinned memory)."""
        return int(self.rigid_count_host.numpy()[0])

    def get_soft_count(self) -> int:
        """Get the latest soft contact count (non-blocking read from pinned memory)."""
        return int(self.soft_count_host.numpy()[0])


class Contacts:
    """
    Stores contact information for rigid and soft body collisions, to be consumed by a solver.

    This class manages buffers for contact data such as positions, normals, thicknesses, and shape indices
    for both rigid-rigid and soft-rigid contacts. The buffers are allocated on the specified device and can
    optionally require gradients for differentiable simulation.

    .. note::
        This class is a temporary solution and its interface may change in the future.
    """

    def __init__(
        self,
        rigid_contact_max: int,
        soft_contact_max: int,
        requires_grad: bool = False,
        device: Devicelike = None,
        per_contact_shape_properties: bool = False,
        enable_debug_info: bool = False,
    ):
        self.per_contact_shape_properties = per_contact_shape_properties
        with wp.ScopedDevice(device):
            # rigid contacts
            self.rigid_contact_count = wp.zeros(1, dtype=wp.int32)
            self.rigid_contact_point_id = wp.zeros(rigid_contact_max, dtype=wp.int32)
            self.rigid_contact_shape0 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_shape1 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_point0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_point1 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_offset0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_offset1 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_normal = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_thickness0 = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            self.rigid_contact_thickness1 = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            self.rigid_contact_tids = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            # to be filled by the solver (currently unused)
            self.rigid_contact_force = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)

            # contact stiffness/damping/friction (only allocated if per_contact_shape_properties is enabled)
            if self.per_contact_shape_properties:
                self.rigid_contact_stiffness = wp.zeros(
                    rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad
                )
                self.rigid_contact_damping = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
                self.rigid_contact_friction = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            else:
                self.rigid_contact_stiffness = None
                self.rigid_contact_damping = None
                self.rigid_contact_friction = None

            # soft contacts
            self.soft_contact_count = wp.zeros(1, dtype=wp.int32)
            self.soft_contact_particle = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_shape = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_body_pos = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_body_vel = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_normal = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_tids = wp.full(soft_contact_max, -1, dtype=int)

        self.requires_grad = requires_grad

        self.rigid_contact_max = rigid_contact_max
        self.soft_contact_max = soft_contact_max

        # Optional debug info for graph-compatible contact count tracking
        if enable_debug_info:
            self.debug_info = ContactDebugInfo(device=device)
        else:
            self.debug_info = None

    def update_debug_info(self):
        """Update debug info with current contact counts (graph-compatible)."""
        if self.debug_info is not None:
            self.debug_info.copy_counts(self.rigid_contact_count, self.soft_contact_count)

    def clear(self):
        """
        Clear contact data by resetting counts to zero.

        Note: We only zero the counts, not the actual arrays. All code that reads
        contact arrays checks the count first, so there's no need to clear/fill the
        arrays themselves. This eliminates expensive memset operations.
        """
        # Just zero the counts - that's all we need!
        self.rigid_contact_count.zero_()
        self.soft_contact_count.zero_()

    @property
    def device(self):
        """
        Returns the device on which the contact buffers are allocated.
        """
        return self.rigid_contact_count.device
