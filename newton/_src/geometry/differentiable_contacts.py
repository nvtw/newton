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

"""Post-processing kernel that augments rigid contacts with differentiable data.

The narrow-phase collision kernels use ``enable_backward=False`` so they are
never recorded on a :class:`wp.Tape`.  This module provides a lightweight
kernel that re-reads the frozen contact geometry (body-local points, world
normal, margins) produced by the narrow phase and reconstructs world-space
quantities through the *differentiable* body transforms ``body_q``.

The resulting arrays carry ``requires_grad=True`` and participate in autodiff,
giving first-order (tangent-plane) gradients of contact distance and world-space
contact points with respect to body poses.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def differentiable_contact_augment_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=float),
    contact_margin1: wp.array(dtype=float),
    # outputs
    out_distance: wp.array(dtype=float),
    out_normal: wp.array(dtype=wp.vec3),
    out_point0_world: wp.array(dtype=wp.vec3),
    out_point1_world: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    body_a = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    body_b = -1
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    bx_a = wp.transform_point(X_wb_a, contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, contact_point1[tid])

    n = contact_normal[tid]
    thickness = contact_margin0[tid] + contact_margin1[tid]
    d = wp.dot(n, bx_b - bx_a) - thickness

    out_distance[tid] = d
    out_normal[tid] = n
    out_point0_world[tid] = bx_a
    out_point1_world[tid] = bx_b


def launch_differentiable_contact_augment(
    contacts,
    body_q: wp.array,
    shape_body: wp.array,
    device=None,
):
    """Launch the differentiable contact augmentation kernel.

    Args:
        contacts: :class:`~newton.Contacts` instance with differentiable arrays allocated.
        body_q: Body transforms, shape ``(body_count,)``, dtype :class:`wp.transform`.
        shape_body: Per-shape body index, shape ``(shape_count,)``, dtype ``int``.
        device: Warp device.
    """
    wp.launch(
        kernel=differentiable_contact_augment_kernel,
        dim=contacts.rigid_contact_max,
        inputs=[
            body_q,
            shape_body,
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
        ],
        outputs=[
            contacts.rigid_contact_diff_distance,
            contacts.rigid_contact_diff_normal,
            contacts.rigid_contact_diff_point0_world,
            contacts.rigid_contact_diff_point1_world,
        ],
        device=device,
    )
