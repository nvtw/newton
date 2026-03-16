# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""``@wp.struct`` schemas used by the PhoenX solver.

These types define the per-row column layouts consumed by
:class:`~newton._src.solvers.phoenx.data_base.DataStore` and
:class:`~newton._src.solvers.phoenx.data_base.HandleStore`.
"""

import warp as wp

BODY_FLAG_STATIC = 1


@wp.struct
class RigidBodySchema:
    """Per-body data stored in a :class:`HandleStore`.

    Attributes:
        position: Centre-of-mass position [m].
        orientation: Body orientation (unit quaternion).
        velocity: Linear velocity [m/s].
        angular_velocity: Angular velocity [rad/s].
        inverse_mass: Reciprocal mass [1/kg].  Zero for static bodies.
        inverse_inertia_local: Body-frame inverse inertia tensor [1/(kg m^2)].
        inverse_inertia_world: World-frame inverse inertia (recomputed each frame).
        linear_damping: Per-step linear damping multiplier.
        angular_damping: Per-step angular damping multiplier.
        flags: Bit field -- bit 0 (:data:`BODY_FLAG_STATIC`).
    """

    position: wp.vec3
    orientation: wp.quat
    velocity: wp.vec3
    angular_velocity: wp.vec3
    inverse_mass: wp.float32
    inverse_inertia_local: wp.mat33
    inverse_inertia_world: wp.mat33
    linear_damping: wp.float32
    angular_damping: wp.float32
    flags: wp.int32


@wp.struct
class ContactPointSchema:
    """Per-contact-point data stored in a :class:`DataStore`.

    Rebuilt every frame from Newton's :class:`CollisionPipeline` output.
    Fields below the ``tangent1`` marker are recomputed each frame by
    the bundled prepare kernel.

    Attributes:
        shape0: Shape index for body 0.
        shape1: Shape index for body 1.
        body0: Body storage row for body 0.
        body1: Body storage row for body 1.
        normal: Contact normal (shape 0 -> shape 1) [unitless].
        offset0: Body-local contact offset on body 0 [m].
        offset1: Body-local contact offset on body 1 [m].
        margin0: Contact margin for shape 0 [m].
        margin1: Contact margin for shape 1 [m].
        accumulated_normal_impulse: Warm-started / solved normal impulse [N s].
        accumulated_tangent_impulse1: Solved tangent impulse component 1 [N s].
        accumulated_tangent_impulse2: Solved tangent impulse component 2 [N s].
        friction: Coulomb friction coefficient.
        tangent1: First tangent vector orthogonal to normal [unitless].
        rel_pos_world0: World-space contact offset on body 0 [m].
        rel_pos_world1: World-space contact offset on body 1 [m].
        effective_mass_n: Effective mass along normal [kg].
        effective_mass_t1: Effective mass along tangent 1 [kg].
        effective_mass_t2: Effective mass along tangent 2 [kg].
        bias: Baumgarte penetration-correction bias [m/s].
    """

    shape0: wp.int32
    shape1: wp.int32
    body0: wp.int32
    body1: wp.int32
    normal: wp.vec3
    offset0: wp.vec3
    offset1: wp.vec3
    margin0: wp.float32
    margin1: wp.float32
    accumulated_normal_impulse: wp.float32
    accumulated_tangent_impulse1: wp.float32
    accumulated_tangent_impulse2: wp.float32
    friction: wp.float32
    tangent1: wp.vec3
    rel_pos_world0: wp.vec3
    rel_pos_world1: wp.vec3
    effective_mass_n: wp.float32
    effective_mass_t1: wp.float32
    effective_mass_t2: wp.float32
    bias: wp.float32
