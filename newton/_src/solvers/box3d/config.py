# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Box3D TGS-Soft solver."""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass
class Softness:
    """Pre-computed TGS-Soft constraint parameters.

    Derived from a spring frequency (hertz) and damping ratio using the
    second-order system formulation from Box2D v3.  The three values
    parameterize the soft constraint impulse equation::

        impulse = -normal_mass * (mass_scale * v + bias) - impulse_scale * accumulated_impulse

    Attributes:
        bias_rate: Position-error correction rate [1/s].
        mass_scale: Scales the effective-mass term (0 for zero-hertz, 1 for infinite stiffness).
        impulse_scale: Scales the accumulated-impulse damping term.
    """

    bias_rate: float = 0.0
    mass_scale: float = 1.0
    impulse_scale: float = 0.0


def compute_softness(hertz: float, damping_ratio: float, h: float) -> Softness:
    """Compute TGS-Soft parameters from spring frequency and damping ratio.

    Follows the Box2D v3 formulation (``b2MakeSoft``).  When *hertz* is
    zero the constraint is fully rigid (``mass_scale = 0``,
    ``impulse_scale = 0``, ``bias_rate = 0``).

    Args:
        hertz: Spring natural frequency [Hz].
        damping_ratio: Damping ratio (1.0 = critically damped).
        h: Sub-step duration [s].
    """
    if hertz == 0.0:
        return Softness(bias_rate=0.0, mass_scale=0.0, impulse_scale=0.0)
    omega = 2.0 * math.pi * hertz
    zeta = damping_ratio
    a1 = 2.0 * zeta + h * omega
    a2 = h * omega * a1
    a3 = 1.0 / (1.0 + a2)
    return Softness(
        bias_rate=omega / a1,
        mass_scale=a2 * a3,
        impulse_scale=a3,
    )


@dataclasses.dataclass
class Box3DConfig:
    """Configuration for the :class:`SolverBox3D` TGS-Soft solver.

    Attributes:
        num_substeps: Number of sub-steps per simulation step.
        num_velocity_iters: Biased (position-correcting) iterations per sub-step.
        num_relaxation_iters: Relaxation (no-bias) iterations per sub-step.
        contact_hertz: Contact constraint spring frequency [Hz].
        contact_damping_ratio: Contact constraint damping ratio.
        joint_hertz: Joint constraint spring frequency [Hz].
        joint_damping_ratio: Joint constraint damping ratio.
        static_hertz_scale: Multiplier on *contact_hertz* for static/kinematic contacts.
        linear_damping: Linear velocity damping coefficient.
        angular_damping: Angular velocity damping coefficient.
        restitution_threshold: Minimum pre-solve relative normal velocity
            required for restitution to be applied [m/s].
        contact_speed: Maximum penetration-recovery speed [m/s].
        max_bodies_per_world: Maximum rigid bodies per world.
        max_contacts_per_world: Maximum contact points per world.
        max_joints_per_world: Maximum joints per world.
        max_colors: Maximum graph colors (contacts + joints).
        block_dim: Threads per CUDA thread block.
    """

    num_substeps: int = 4
    num_velocity_iters: int = 1
    num_relaxation_iters: int = 1
    contact_hertz: float = 30.0
    contact_damping_ratio: float = 1.0
    joint_hertz: float = 60.0
    joint_damping_ratio: float = 1.0
    static_hertz_scale: float = 2.0
    linear_damping: float = 0.0
    angular_damping: float = 0.05
    restitution_threshold: float = 1.0
    contact_speed: float = 10.0
    max_bodies_per_world: int = 1024
    max_contacts_per_world: int = 4096
    max_joints_per_world: int = 1024
    max_colors: int = 64
    block_dim: int = 128
    enable_graph: bool = False
    """Enable CUDA graph capture for the solver-internal kernels.

    When enabled, the first call to :meth:`~SolverBox3D.step` captures a
    CUDA graph of the solve loop (coloring, integration, contact/joint
    solve, impulse store).  Subsequent calls replay the graph with zero
    Python overhead.  Requires that ``dt`` stays constant between steps.
    """
