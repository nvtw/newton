# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelFlags, State
from ...utils.deprecation import deprecate_nonkeyword_arguments
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from . import kernels
from .kernels import (
    accumulate_weighted_contact_impulse,
    apply_body_delta_velocities,
    apply_body_deltas,
    apply_joint_forces,
    apply_particle_deltas,
    apply_particle_shape_restitution,
    apply_rigid_restitution,
    bending_constraint,
    convert_contact_impulse_to_force,
    convert_joint_impulse_to_parent_f,
    copy_kinematic_body_state_kernel,
    solve_body_contact_positions,
    solve_body_joints,
    solve_particle_particle_contacts,
    solve_particle_shape_contacts,
    # solve_simple_body_joints,
    solve_springs,
    solve_tetrahedra,
    update_body_velocities,
)
from .pbf_kernels import (
    accumulate_boundary_density,
    apply_damping,
    build_neighbor_list,
    build_sorted_order,
    gather_sorted_positions,
    apply_pbf_deltas,
    apply_vorticity,
    calculate_density,
    finalize_pbf_velocities,
    solve_density,
    vorticity_confinement,
)


def _calculate_rest_density(rest_distance: float, h: float) -> tuple[float, float, float]:
    """Compute rest density, density constraint scale, and surface constraint scale.

    Ported from PhysX ``CalculateRestDensity`` — generates a tight sphere packing
    within radius *h* at spacing *rest_distance*, then evaluates the SPH spiky
    kernel over those neighbors to obtain the rest density that the density
    constraint should target.

    Returns:
        ``(rest_density, density_constraint_scale, surface_constraint_scale)``
    """
    if rest_distance <= 0.0:
        return 0.0, 1.0, 1.0

    inv_h = 1.0 / h
    k_w = 15.0 / (math.pi * h * h * h)
    k_dw = 30.0 / (math.pi * h * h * h * h)

    sqrt_075 = math.sqrt(0.75)
    dim = int(math.ceil(h / rest_distance))

    rho = 0.0
    rho_deriv = 0.0
    a = 0.0
    b = 0.0

    for z in range(-dim, dim + 1):
        for y in range(-dim, dim + 1):
            for x in range(-dim, dim + 1):
                offset = rest_distance * 0.5 if ((y + z) & 1) else 0.0
                xpos = x * rest_distance + offset
                ypos = y * sqrt_075 * rest_distance
                zpos = z * sqrt_075 * rest_distance

                d_sq = xpos * xpos + ypos * ypos + zpos * zpos
                if d_sq == 0.0:
                    continue
                d = math.sqrt(d_sq)
                if d > h:
                    continue

                q = d * inv_h
                w = k_w * (1.0 - q) * (1.0 - q)
                dw = -k_dw * (1.0 - q)

                rho += w
                rho_deriv += dw * dw

                if ypos <= 0.0:
                    cos_theta = ypos / d
                    a += dw * cos_theta
                    b -= d * cos_theta

    surface_deriv = a / b if b != 0.0 else 1.0
    return rho, rho_deriv, surface_deriv


class SolverXPBD(SolverBase, CouplingInterface):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16). Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU, Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    After constructing :class:`Model`, :class:`State`, and :class:`Control` (optional) objects, this time-integrator
    may be used to advance the simulation state forward in time.

    Limitations:
        **Fluid incompressibility** -- The density constraint is solved by
        relaxation, so a resting column stays slightly compressed under its own
        weight and the residual is set by the iteration budget, not by the
        solver. Measured at the base of a 24-particle-deep column, density runs
        1.17x rest at 2 iterations, 1.09x at 4, 1.05x at 8, 1.02x at 16 and
        1.01x at 32, converging to rest density as expected. Substeps buy the
        same accuracy more cheaply than iterations: doubling substeps costs
        twice as much and reaches what quadrupling iterations does. Raise
        ``substeps`` first if a scene needs stiffer incompressibility.

        **Position-based fluids** -- The fluid solve currently supports one
        global fluid material. Multiphase fluids, diffuse particles, anisotropy,
        smoothing, and fluid-particle adhesion are not supported. Fluid-shape
        interaction uses the existing XPBD particle-shape contact model.
        Differentiable simulation is not supported for fluid particles.

        **Momentum conservation** -- When ``rigid_contact_con_weighting`` is
        enabled (the default), each body's positional correction is divided by
        its number of active contacts.  This improves convergence for stacking
        scenarios but means the solver does not conserve momentum at contacts.
        Reported per-contact forces (see :meth:`update_contacts`) are
        approximate: for contacts between two dynamic bodies the force is
        computed using the harmonic mean of the two bodies' contact counts,
        which is symmetric but not exact.

        **Reported parent-joint forces** (see :attr:`~newton.State.body_parent_f`,
        populated when the extended state attribute is requested) are
        approximate.  XPBD applies relaxation factors
        (``joint_linear_relaxation``, ``joint_angular_relaxation``) to each
        joint constraint correction, and with a finite ``iterations`` count
        residual constraint error remains at end-of-step, so the reported
        wrench is the *applied* constraint reaction rather than the exact
        wrench needed to enforce the joint perfectly.  The convention matches
        :class:`~newton.solvers.SolverFeatherstone` and
        :class:`~newton.solvers.SolverMuJoCo`: it is the spatial wrench
        transmitted from the parent through the inbound joint, in world frame
        at the child body's COM, **including** both the constraint reaction
        and the body-frame contribution of :attr:`~newton.Control.joint_f`.
        In equilibrium this wrench counters all applied forces (gravity,
        contacts, ``State.body_f``) by Newton's third law.

    Joint limitations:
        - Supported joint types: PRISMATIC, REVOLUTE, BALL, FIXED, FREE, DISTANCE, D6.
          CABLE joints are not supported.
        - :attr:`~newton.Model.joint_enabled`,
          :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd`, and
          :attr:`~newton.Control.joint_f` are supported.
          Joint limits are enforced as hard positional constraints (``joint_limit_ke``/``joint_limit_kd`` are not used).
        - :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_velocity_limit`,
          and :attr:`~newton.Model.joint_target_mode` are not supported.
        - Equality and mimic constraints are not supported.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverXPBD(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    @property
    def pbf_neighbor_overflow_count(self) -> int:
        """Fluid neighbors dropped so far because ``pbf_max_neighbors`` was exceeded.

        Nonzero means some particles saw a truncated neighborhood and read as
        less dense than they are. Raise ``pbf_max_neighbors`` if this grows.
        """
        if not getattr(self, "pbf_enabled", False):
            return 0
        return int(self._pbf_neighbor_overflow.numpy()[0])

    @deprecate_nonkeyword_arguments
    def __init__(
        self,
        model: Model,
        *,
        iterations: int = 2,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        joint_linear_relaxation: float = 0.7,
        joint_angular_relaxation: float = 0.4,
        joint_linear_compliance: float = 0.0,
        joint_angular_compliance: float = 0.0,
        rigid_contact_relaxation: float = 0.8,
        rigid_contact_con_weighting: bool = True,
        angular_damping: float = 0.0,
        enable_restitution: bool = False,
        pbf_particle_contact_distance: float | None = None,
        pbf_fluid_rest_distance: float | None = None,
        pbf_relaxation: float = 1.0,
        pbf_viscosity: float = 0.0,
        pbf_cohesion: float = 0.0,
        pbf_surface_tension: float = 0.0,
        pbf_vorticity_confinement: float = 0.0,
        pbf_cfl_coefficient: float = 1.0,
        pbf_damping: float = 0.0,
        pbf_boundary_density: bool = True,
        pbf_max_neighbors: int = 64,
        deterministic: wp.DeterministicMode | None = None,
    ):
        """Initialize the XPBD solver.

        Position-based fluids are enabled when ``pbf_particle_contact_distance``
        is provided. Particles carrying :attr:`~newton.ParticleFlags.FLUID`
        then use the density constraint; other particles retain the standard
        XPBD particle-contact path.

        Args:
            model: Model to simulate.
            iterations: Number of constraint iterations per step.
            soft_body_relaxation: Relaxation for particle constraints.
            soft_contact_relaxation: Relaxation for particle-shape contacts.
            joint_linear_relaxation: Relaxation for linear joint constraints.
            joint_angular_relaxation: Relaxation for angular joint constraints.
            joint_linear_compliance: Compliance of linear joint constraints.
            joint_angular_compliance: Compliance of angular joint constraints.
            rigid_contact_relaxation: Relaxation for rigid contacts.
            rigid_contact_con_weighting: Whether to average each body's rigid
                contact corrections.
            angular_damping: Rigid-body angular damping coefficient.
            enable_restitution: Whether to apply restitution after the
                positional solve.
            pbf_particle_contact_distance: Fluid neighbor radius [m]. ``None``
                disables position-based fluids.
            pbf_fluid_rest_distance: Fluid rest spacing [m]. Defaults to 60%
                of ``pbf_particle_contact_distance``.
            pbf_relaxation: Fluid Jacobi relaxation factor, scaling each
                density correction after it is averaged over the contributing
                neighbors. Values below 1 under-relax: smaller, more stable
                steps that converge more slowly. 1.0 matches PhysX.
            pbf_viscosity: Fluid viscosity coefficient.
            pbf_cohesion: Fluid cohesion coefficient.
            pbf_surface_tension: Fluid surface-tension coefficient.
            pbf_vorticity_confinement: Fluid vorticity-confinement coefficient.
            pbf_cfl_coefficient: Maximum relative normal displacement as a
                fraction of the fluid neighbor radius.
            pbf_damping: Fluid velocity damping coefficient.
            pbf_boundary_density: Whether solid boundaries contribute to the
                fluid density. Without it the density sum omits the part of a
                particle's kernel support occupied by a solid, under-estimating
                density near walls so the solver never pushes back and
                particles pile into a compressed layer against the surface.
                Requires particle-shape contacts generated out to the fluid
                neighbor radius; a smaller ``soft_contact_margin`` truncates
                the correction. Disable only to reproduce the prior behavior.
            pbf_max_neighbors: Capacity of the cached per-particle fluid
                neighbor list, built once per substep and replayed by the
                density, pressure and vorticity kernels. Costs
                ``4 * particle_count * pbf_max_neighbors`` bytes. Neighbors
                beyond the cap are dropped; see
                :attr:`pbf_neighbor_overflow_count` to detect it.
            deterministic: Opt-in determinism for this solver's atomic-emitting
                kernel module. Pass a :class:`warp.DeterministicMode`, or
                ``None`` (default) to inherit the current
                ``wp.config.deterministic`` mode.
        """
        super().__init__(model=model)
        effective_deterministic = deterministic if deterministic is not None else wp.config.deterministic
        self._set_module_options(
            {
                "deterministic": effective_deterministic,
                "deterministic_max_records": 0,
            },
            module=kernels,
        )

        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation

        self.joint_linear_relaxation = joint_linear_relaxation
        self.joint_angular_relaxation = joint_angular_relaxation
        self.joint_linear_compliance = joint_linear_compliance
        self.joint_angular_compliance = joint_angular_compliance

        self.rigid_contact_relaxation = rigid_contact_relaxation
        self.rigid_contact_con_weighting = rigid_contact_con_weighting

        self.angular_damping = angular_damping

        self.enable_restitution = enable_restitution

        self.compute_body_velocity_from_position_delta = False

        self._init_kinematic_state()

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        if model.particle_count > 1 and model.particle_grid is not None:
            # reserve space for the particle hash grid
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

        self.pbf_enabled = pbf_particle_contact_distance is not None
        if self.pbf_enabled:
            h = pbf_particle_contact_distance
            if h <= 0.0:
                raise ValueError("pbf_particle_contact_distance must be positive")
            if model.particle_count > 1 and model.particle_grid is None:
                raise ValueError("Position-based fluids require a particle hash grid")

            fluid_rest_dist = pbf_fluid_rest_distance if pbf_fluid_rest_distance is not None else h * 0.6
            if fluid_rest_dist <= 0.0 or fluid_rest_dist >= h:
                raise ValueError("pbf_fluid_rest_distance must be positive and less than the contact distance")
            if pbf_relaxation <= 0.0:
                raise ValueError("pbf_relaxation must be positive")
            if (
                min(
                    pbf_viscosity,
                    pbf_cohesion,
                    pbf_surface_tension,
                    pbf_vorticity_confinement,
                    pbf_cfl_coefficient,
                    pbf_damping,
                )
                < 0.0
            ):
                raise ValueError("Position-based fluid coefficients must be nonnegative")

            self.pbf_contact_distance_sq = h * h
            self.pbf_inv_radius = 1.0 / h
            self.pbf_spiky1 = 15.0 / (math.pi * h * h * h)
            self.pbf_spiky2 = 30.0 / (math.pi * h * h * h * h)

            rest_density, density_constraint_scale, surface_constraint_scale = _calculate_rest_density(
                fluid_rest_dist, h
            )
            if rest_density <= 0.0 or density_constraint_scale <= 0.0:
                raise ValueError("Fluid rest spacing does not produce a valid density constraint")
            self.pbf_rest_density = rest_density
            self.pbf_lambda_scale = 1.0 / density_constraint_scale

            self.pbf_boundary_density = pbf_boundary_density
            # Rebuilding the neighbor grid inside the iteration loop is a large
            # fraction of the fluid step. Warp's hash grid searches the 3x3x3
            # cell neighborhood around a query point, so a grid built at the
            # start of the substep already tolerates drift on the order of half
            # a cell -- more than a particle moves across one substep's
            # iterations. Kept as an attribute so the assumption can be
            # A/B-tested rather than taken on faith.
            self._pbf_rebuild_grid_per_iteration = False
            self.pbf_relaxation = pbf_relaxation
            self.pbf_viscosity = pbf_viscosity
            self.pbf_cfl_coefficient = pbf_cfl_coefficient
            self.pbf_particle_contact_distance = h
            self.pbf_damping = pbf_damping
            self.pbf_vorticity_confinement = pbf_vorticity_confinement

            inv_rest_density = 1.0 / rest_density if rest_density > 0.0 else 0.0

            # Derive surface tension and cohesion following PhysX exactly:
            #   surfaceTension = invRestDensity * mat.surfaceTension / surfaceConstraintScale
            #   cohesion = mat.cohesion * particleContactDistance
            if surface_constraint_scale != 0.0:
                self.pbf_surface_tension = inv_rest_density * pbf_surface_tension / surface_constraint_scale
            else:
                self.pbf_surface_tension = pbf_surface_tension
            self.pbf_cohesion = pbf_cohesion * h

            # Cohesion kernel coefficients from PhysX:
            # W_cohesion(d) = c1*(d/h)^3 + c2*(d/h)^2 - 1
            # With rest = fluidRestDistance / particleContactDistance:
            #   c1 = -(1 + rest) / rest^2
            #   c2 = (rest^2 + rest + 1) / rest^2
            rest_ratio = fluid_rest_dist / h if h > 0.0 else 1.0
            rest_sq = rest_ratio * rest_ratio
            if rest_sq > 0.0:
                self.pbf_cohesion1 = -(1.0 + rest_ratio) / rest_sq
                self.pbf_cohesion2 = (rest_sq + rest_ratio + 1.0) / rest_sq
            else:
                self.pbf_cohesion1 = -2.0
                self.pbf_cohesion2 = 3.0

            n = model.particle_count
            self._pbf_densities = wp.zeros(n, dtype=float, device=model.device)
            self._pbf_surface_normals = wp.zeros(n, dtype=wp.vec3, device=model.device)
            self._pbf_deltas = wp.zeros(n, dtype=wp.vec3, device=model.device)
            self._pbf_weights = wp.zeros(n, dtype=float, device=model.device)
            self._pbf_accum_delta = wp.zeros(n, dtype=wp.vec3, device=model.device)
            self._pbf_pos_lambda = wp.zeros(n, dtype=wp.vec4, device=model.device)
            # Cell-ordered view of the particles. The hash grid already sorts by
            # cell each substep, so reusing that ordering costs nothing and makes
            # neighbour gathers mostly local instead of scattered.
            self._pbf_sorted_to_orig = wp.zeros(n, dtype=wp.int32, device=model.device)
            self._pbf_orig_to_sorted = wp.zeros(n, dtype=wp.int32, device=model.device)
            self._pbf_pos_sorted = wp.zeros(n, dtype=wp.vec3, device=model.device)
            self._pbf_boundary_log = wp.zeros(n, dtype=float, device=model.device)
            self._pbf_boundary_grad = wp.zeros(n, dtype=wp.vec3, device=model.device)
            # A fluid at rest spacing holds ~26 neighbors within the support
            # radius; a violent dam break peaks near 38. 64 leaves ample
            # headroom, and overflow is counted rather than silently dropped.
            self._pbf_max_neighbors = pbf_max_neighbors
            self._pbf_neighbors = wp.zeros(n * pbf_max_neighbors, dtype=wp.int32, device=model.device)
            self._pbf_neighbor_counts = wp.zeros(n, dtype=wp.int32, device=model.device)
            self._pbf_neighbor_overflow = wp.zeros(1, dtype=wp.int32, device=model.device)
            self._pbf_curl = wp.zeros(n, dtype=wp.vec3, device=model.device)
            self._pbf_curl_mag = wp.zeros(n, dtype=float, device=model.device)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh cached body data after model properties change.

        Effective inverse masses and inertia tensors are refreshed when
        :attr:`~newton.ModelFlags.BODY_PROPERTIES` or
        :attr:`~newton.ModelFlags.BODY_INERTIAL_PROPERTIES` is set. Other flags are ignored.

        Args:
            flags: Bitmask of :class:`~newton.ModelFlags` or custom ``int`` bits indicating which model properties
                changed.
        """
        self._apply_module_options()
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()

    @override
    def coupling_supports_inertial_property_refresh(self) -> bool:
        """Return whether inertial properties can be refreshed during graph capture.

        Returns:
            ``True`` because :meth:`notify_model_changed` refreshes the derived inertial buffers with device work.
        """
        return True

    def copy_kinematic_body_state(self, model: Model, state_in: State, state_out: State):
        """Copy kinematic body poses and velocities from an input state to an output state.

        Args:
            model: Simulation model that owns the body data.
            state_in: State containing the source kinematic body poses and velocities.
            state_out: State that receives the kinematic body poses and velocities.
        """
        if model.body_count == 0:
            return
        wp.launch(
            kernel=copy_kinematic_body_state_kernel,
            dim=model.body_count,
            inputs=[model.body_flags, state_in.body_q, state_in.body_qd],
            outputs=[state_out.body_q, state_out.body_qd],
            device=model.device,
        )

    def _apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
    ):
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                particle_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        if state_in.requires_grad:
            state_out.particle_q = new_particle_q
            state_out.particle_qd = new_particle_qd

        return new_particle_q, new_particle_qd

    def _apply_body_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        body_deltas: wp.array,
        dt: float,
        rigid_contact_inv_weight: wp.array = None,
    ):
        with wp.ScopedTimer("apply_body_deltas", False):
            if state_in.requires_grad:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                new_body_q = wp.clone(body_q)
                new_body_qd = wp.clone(body_qd)
                self._body_delta_counter += 1
            else:
                if self._body_delta_counter == 0:
                    body_q = state_out.body_q
                    body_qd = state_out.body_qd
                    new_body_q = state_in.body_q
                    new_body_qd = state_in.body_qd
                else:
                    body_q = state_in.body_q
                    body_qd = state_in.body_qd
                    new_body_q = state_out.body_q
                    new_body_qd = state_out.body_qd
                self._body_delta_counter = 1 - self._body_delta_counter

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    body_deltas,
                    rigid_contact_inv_weight,
                    dt,
                ],
                outputs=[
                    new_body_q,
                    new_body_qd,
                ],
                device=model.device,
            )

            if state_in.requires_grad:
                state_out.body_q = new_body_q
                state_out.body_qd = new_body_qd

        return new_body_q, new_body_qd

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation state by one time step using XPBD.

        Args:
            state_in: State at the beginning of the time step.
            state_out: State that receives the simulation result.
            control: Control inputs. If ``None``, the model's default control values are used.
            contacts: Contact data populated by :meth:`~newton.CollisionPipeline.collide` and allocated with
                :meth:`~newton.CollisionPipeline.contacts`. If ``None``, rigid and particle-shape contact handling
                is skipped; particle-particle contacts and model constraints are still solved.
            dt: Time step size [s].
        """
        self._apply_module_options()
        requires_grad = state_in.requires_grad
        if self.pbf_enabled and requires_grad:
            raise RuntimeError("Position-based fluids do not support differentiable simulation")
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        model = self.model

        particle_q = None
        particle_qd = None
        particle_deltas = None

        body_q = None
        body_qd = None
        body_q_init = None
        body_qd_init = None
        body_deltas = None

        rigid_contact_inv_weight = None

        contact_impulse = None
        contact_impulse_iter = None

        if contacts:
            if self.rigid_contact_con_weighting:
                rigid_contact_inv_weight = wp.zeros(model.body_count, dtype=float, device=model.device)
            rigid_contact_inv_weight_init = None

            if contacts.force is not None:
                contact_impulse = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device)
                contact_impulse_iter = wp.zeros(
                    contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device
                )

        # Optional per-joint accumulated child-side spatial impulse, used to
        # populate ``state_out.body_parent_f`` after the iteration loop.
        joint_impulse = None
        if state_out.body_parent_f is not None and model.joint_count > 0:
            joint_impulse = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=model.device)

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd

                self.particle_q_init = wp.clone(state_in.particle_q)
                if self.enable_restitution:
                    self.particle_qd_init = wp.clone(state_in.particle_qd)
                particle_deltas = wp.empty_like(state_out.particle_qd)

                self.integrate_particles(model, state_in, state_out, dt)

                # Build/update the particle hash grid for particle-particle contact queries
                if model.particle_count > 1 and model.particle_grid is not None:
                    # Search radius must cover the maximum interaction distance used by the contact query
                    grid_search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
                    if self.pbf_enabled:
                        grid_search_radius = max(grid_search_radius, self.pbf_particle_contact_distance)
                    with wp.ScopedDevice(model.device):
                        model.particle_grid.build(state_out.particle_q, radius=grid_search_radius)

                if self.pbf_enabled:
                    self._pbf_accum_delta.zero_()
                    # Cache the fluid neighborhood once for the whole substep;
                    # the density, pressure and vorticity kernels all replay it.
                    if model.particle_grid is not None:
                        wp.launch(
                            kernel=build_sorted_order,
                            dim=model.particle_count,
                            inputs=[model.particle_grid.id],
                            outputs=[self._pbf_sorted_to_orig, self._pbf_orig_to_sorted],
                            device=model.device,
                        )
                        wp.launch(
                            kernel=build_neighbor_list,
                            dim=model.particle_count,
                            inputs=[
                                model.particle_grid.id,
                                state_out.particle_q,
                                model.particle_flags,
                                self._pbf_orig_to_sorted,
                                self.pbf_contact_distance_sq,
                                self.pbf_particle_contact_distance,
                                self._pbf_max_neighbors,
                                model.particle_count,
                            ],
                            outputs=[
                                self._pbf_neighbors,
                                self._pbf_neighbor_counts,
                                self._pbf_neighbor_overflow,
                            ],
                            device=model.device,
                        )

                    # Solid boundaries contribute no neighbors to the
                    # density sum; recover their contribution from the
                    # particle-shape contacts before accumulating.
                    self._pbf_boundary_log.zero_()
                    self._pbf_boundary_grad.zero_()
                    if self.pbf_boundary_density and model.shape_count and contacts is not None:
                        wp.launch(
                            kernel=accumulate_boundary_density,
                            # Real contacts are far fewer than the
                            # particle-times-shape slot count the buffer is
                            # sized for; grid-stride over the actual count.
                            dim=min(contacts.soft_contact_max, model.particle_count),
                            inputs=[
                                particle_q,
                                model.particle_flags,
                                state_out.body_q,
                                model.shape_body,
                                contacts.soft_contact_count,
                                contacts.soft_contact_particle,
                                contacts.soft_contact_shape,
                                contacts.soft_contact_body_pos,
                                contacts.soft_contact_normal,
                                contacts.soft_contact_max,
                                min(contacts.soft_contact_max, model.particle_count),
                                self.pbf_inv_radius,
                                self.pbf_rest_density,
                            ],
                            outputs=[self._pbf_boundary_log, self._pbf_boundary_grad],
                            device=model.device,
                        )

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                if self.compute_body_velocity_from_position_delta or self.enable_restitution:
                    body_q_init = wp.clone(state_in.body_q)
                    body_qd_init = wp.clone(state_in.body_qd)

                body_deltas = wp.empty_like(state_out.body_qd)

                body_f_tmp = state_in.body_f
                if model.joint_count:
                    # Avoid accumulating joint_f into the persistent state body_f buffer.
                    body_f_tmp = wp.clone(state_in.body_f)
                    # ``joint_impulse`` (may be ``None`` when ``body_parent_f``
                    # was not requested) accumulates both the joint_f wrench
                    # contribution recorded here and the constraint-correction
                    # contribution added by :func:`solve_body_joints` inside
                    # the iteration loop.  Together they recover the total
                    # wrench transmitted to the child body, matching the
                    # :attr:`State.body_parent_f` convention.
                    wp.launch(
                        kernel=apply_joint_forces,
                        dim=model.joint_count,
                        inputs=[
                            state_in.body_q,
                            model.body_com,
                            model.joint_type,
                            model.joint_enabled,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_X_p,
                            model.joint_X_c,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_axis,
                            control.joint_f,
                            dt,
                        ],
                        outputs=[body_f_tmp, joint_impulse],
                        device=model.device,
                    )

                if body_f_tmp is state_in.body_f:
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                else:
                    body_f_prev = state_in.body_f
                    state_in.body_f = body_f_tmp
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                    state_in.body_f = body_f_prev

            spring_constraint_lambdas = None
            if model.spring_count:
                spring_constraint_lambdas = wp.empty_like(model.spring_rest_length)
            edge_constraint_lambdas = None
            if model.edge_count:
                edge_constraint_lambdas = wp.empty_like(model.edge_rest_angle)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.body_count:
                        if requires_grad and i > 0:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                    if model.particle_count:
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas)
                        else:
                            particle_deltas.zero_()

                        # --- PBF density calculation ---
                        if self.pbf_enabled and model.particle_grid is not None:
                            if i > 0 and self._pbf_rebuild_grid_per_iteration:
                                with wp.ScopedDevice(model.device):
                                    model.particle_grid.build(particle_q, radius=grid_search_radius)

                            # Positions move every iteration; the ordering only
                            # changes per substep. Refreshing the slot-ordered
                            # copy is O(N) and saves O(N * neighbours) scattered
                            # reads in the two kernels below.
                            wp.launch(
                                kernel=gather_sorted_positions,
                                dim=model.particle_count,
                                inputs=[particle_q, self._pbf_sorted_to_orig],
                                outputs=[self._pbf_pos_sorted],
                                device=model.device,
                            )

                            wp.launch(
                                kernel=calculate_density,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    model.particle_flags,
                                    self._pbf_pos_sorted,
                                    self._pbf_neighbors,
                                    self._pbf_neighbor_counts,
                                    model.particle_count,
                                    self.pbf_contact_distance_sq,
                                    self.pbf_inv_radius,
                                    self.pbf_spiky1,
                                    self.pbf_spiky2,
                                    self.pbf_rest_density,
                                    self.pbf_lambda_scale,
                                    self.pbf_surface_tension,
                                    self._pbf_boundary_log,
                                ],
                                outputs=[
                                    self._pbf_densities,
                                    self._pbf_pos_lambda,
                                    self._pbf_surface_normals,
                                ],
                                device=model.device,
                            )

                            # Compute displacement from predicted position for viscosity/CFL
                            pbf_delta_pos = self._pbf_accum_delta

                            wp.launch(
                                kernel=solve_density,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    model.particle_flags,
                                    model.particle_inv_mass,
                                    self._pbf_neighbors,
                                    self._pbf_neighbor_counts,
                                    model.particle_count,
                                    self._pbf_densities,
                                    self._pbf_pos_lambda,
                                    self._pbf_surface_normals,
                                    self._pbf_boundary_grad,
                                    pbf_delta_pos,
                                    self.pbf_contact_distance_sq,
                                    self.pbf_inv_radius,
                                    self.pbf_spiky1,
                                    self.pbf_spiky2,
                                    self.pbf_viscosity,
                                    1.0 / self.pbf_rest_density,
                                    self.pbf_cohesion,
                                    self.pbf_cohesion1,
                                    self.pbf_cohesion2,
                                    self.pbf_surface_tension,
                                    self.pbf_cfl_coefficient,
                                    1.0,  # coefficient
                                    dt,
                                ],
                                outputs=[self._pbf_deltas, self._pbf_weights],
                                device=model.device,
                            )

                            wp.launch(
                                kernel=apply_pbf_deltas,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_flags,
                                    self._pbf_sorted_to_orig,
                                    self._pbf_deltas,
                                    self._pbf_weights,
                                    self.pbf_relaxation,
                                ],
                                outputs=[particle_q, self._pbf_accum_delta],
                                device=model.device,
                            )

                        # particle-rigid body contacts (besides ground plane)
                        if model.shape_count and contacts is not None:
                            contacts._assert_particle_only_soft_contacts("SolverXPBD")
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    body_q,
                                    body_qd,
                                    model.body_com,
                                    self.body_inv_mass_effective,
                                    self.body_inv_inertia_effective,
                                    model.body_flags,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                # outputs
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )

                        if model.particle_max_radius > 0.0 and model.particle_count > 1:
                            # assert model.particle_grid.reserved, "model.particle_grid must be built, see HashGrid.build()"
                            assert model.particle_grid is not None
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                    self.pbf_enabled,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # distance constraints
                        if model.spring_count:
                            spring_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=solve_springs,
                                dim=model.spring_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.spring_indices,
                                    model.spring_rest_length,
                                    model.spring_stiffness,
                                    model.spring_damping,
                                    dt,
                                    spring_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # bending constraints
                        if model.edge_count:
                            edge_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=bending_constraint,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    model.edge_rest_angle,
                                    model.edge_bending_properties,
                                    dt,
                                    edge_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # tetrahedral FEM
                        if model.tet_count:
                            wp.launch(
                                kernel=solve_tetrahedra,
                                dim=model.tet_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.tet_indices,
                                    model.tet_poses,
                                    control.tet_activations,
                                    model.tet_materials,
                                    dt,
                                    self.soft_body_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        particle_q, particle_qd = self._apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                    # handle rigid bodies
                    # ----------------------------

                    # Solve rigid contact constraints
                    if model.body_count and contacts is not None:
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight.zero_()

                        if contact_impulse_iter is not None:
                            contact_impulse_iter.zero_()

                        wp.launch(
                            kernel=solve_body_contact_positions,
                            dim=contacts.rigid_contact_max,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_flags,
                                model.body_com,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
                                model.shape_body,
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_point0,
                                contacts.rigid_contact_point1,
                                contacts.rigid_contact_offset0,
                                contacts.rigid_contact_offset1,
                                contacts.rigid_contact_normal,
                                contacts.rigid_contact_margin0,
                                contacts.rigid_contact_margin1,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                model.shape_material_mu,
                                model.shape_material_mu_torsional,
                                model.shape_material_mu_rolling,
                                self.rigid_contact_relaxation,
                                dt,
                            ],
                            outputs=[
                                body_deltas,
                                rigid_contact_inv_weight,
                                contact_impulse_iter,
                            ],
                            device=model.device,
                        )

                        if contact_impulse_iter is not None:
                            wp.launch(
                                kernel=accumulate_weighted_contact_impulse,
                                dim=contacts.rigid_contact_max,
                                inputs=[
                                    contacts.rigid_contact_count,
                                    contact_impulse_iter,
                                    contacts.rigid_contact_shape0,
                                    contacts.rigid_contact_shape1,
                                    model.shape_body,
                                    rigid_contact_inv_weight,
                                ],
                                outputs=[contact_impulse],
                                device=model.device,
                            )

                        # if model.rigid_contact_count.numpy()[0] > 0:
                        #     print("rigid_contact_count:", model.rigid_contact_count.numpy().flatten())
                        #     # print("rigid_active_contact_distance:", rigid_active_contact_distance.numpy().flatten())
                        #     # print("rigid_active_contact_point0:", rigid_active_contact_point0.numpy().flatten())
                        #     # print("rigid_active_contact_point1:", rigid_active_contact_point1.numpy().flatten())
                        #     print("body_deltas:", body_deltas.numpy().flatten())

                        # print(rigid_active_contact_distance.numpy().flatten())

                        if self.enable_restitution and i == 0:
                            # remember contact constraint weighting from the first iteration
                            if self.rigid_contact_con_weighting:
                                rigid_contact_inv_weight_init = wp.clone(rigid_contact_inv_weight)
                            else:
                                rigid_contact_inv_weight_init = None

                        body_q, body_qd = self._apply_body_deltas(
                            model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight
                        )

                    if model.joint_count:
                        if requires_grad:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                        wp.launch(
                            kernel=solve_body_joints,
                            dim=model.joint_count,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
                                model.joint_type,
                                model.joint_enabled,
                                model.joint_parent,
                                model.joint_child,
                                model.joint_X_p,
                                model.joint_X_c,
                                model.joint_limit_lower,
                                model.joint_limit_upper,
                                model.joint_qd_start,
                                model.joint_target_q_start,
                                model.joint_dof_dim,
                                model.joint_axis,
                                control.joint_target_q,
                                control.joint_target_qd,
                                model.joint_target_ke,
                                model.joint_target_kd,
                                self.joint_linear_compliance,
                                self.joint_angular_compliance,
                                self.joint_angular_relaxation,
                                self.joint_linear_relaxation,
                                dt,
                            ],
                            outputs=[body_deltas, joint_impulse],
                            device=model.device,
                        )

                        body_q, body_qd = self._apply_body_deltas(model, state_in, state_out, body_deltas, dt)

            # --- PBF post-iteration passes ---
            if self.pbf_enabled and model.particle_count and model.particle_grid is not None:
                # Rebuild grid for the post-iteration neighbor queries
                with wp.ScopedDevice(model.device):
                    model.particle_grid.build(particle_q, radius=grid_search_radius)

                # Fluid forces below modify the velocity inferred from the
                # completed positional solve, so initialize that velocity first.
                wp.launch(
                    kernel=finalize_pbf_velocities,
                    dim=model.particle_count,
                    inputs=[
                        particle_q,
                        self.particle_q_init,
                        model.particle_flags,
                        dt,
                        model.particle_max_velocity,
                    ],
                    outputs=[particle_qd],
                    device=model.device,
                )

                # Vorticity confinement (optional)
                if self.pbf_vorticity_confinement > 0.0:
                    self._pbf_curl.zero_()
                    self._pbf_curl_mag.zero_()
                    wp.launch(
                        kernel=vorticity_confinement,
                        dim=model.particle_count,
                        inputs=[
                            model.particle_grid.id,
                            particle_q,
                            particle_qd,
                            model.particle_flags,
                            self._pbf_pos_sorted,
                            self._pbf_sorted_to_orig,
                            self._pbf_neighbors,
                            self._pbf_neighbor_counts,
                            model.particle_count,
                            self.pbf_contact_distance_sq,
                            self.pbf_inv_radius,
                            self.pbf_spiky2,
                        ],
                        outputs=[self._pbf_curl, self._pbf_curl_mag],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_vorticity,
                        dim=model.particle_count,
                        inputs=[
                            model.particle_grid.id,
                            particle_q,
                            model.particle_flags,
                            self._pbf_curl,
                            self._pbf_curl_mag,
                            self._pbf_pos_sorted,
                            self._pbf_neighbors,
                            self._pbf_neighbor_counts,
                            model.particle_count,
                            self.pbf_contact_distance_sq,
                            self.pbf_inv_radius,
                            self.pbf_spiky2,
                            self.pbf_vorticity_confinement,
                            1.0 / self.pbf_rest_density,
                            dt,
                        ],
                        outputs=[particle_qd],
                        device=model.device,
                    )

                # Velocity damping (used e.g. for snow settling)
                if self.pbf_damping > 0.0:
                    wp.launch(
                        kernel=apply_damping,
                        dim=model.particle_count,
                        inputs=[
                            model.particle_flags,
                            self.pbf_damping,
                            dt,
                        ],
                        outputs=[particle_qd],
                        device=model.device,
                    )

            self._contact_impulse = contact_impulse
            self._contact_impulse_capacity = contacts.rigid_contact_max if contacts is not None else 0
            self._last_dt = dt

            # Populate optional ``state_out.body_parent_f`` (incoming joint
            # wrench per body) from the per-joint accumulated child-side
            # impulse.  Bodies without an inbound joint (roots / free bodies)
            # remain zero-initialized, matching MuJoCo's behavior.
            if state_out.body_parent_f is not None:
                state_out.body_parent_f.zero_()
                if joint_impulse is not None:
                    wp.launch(
                        kernel=convert_joint_impulse_to_parent_f,
                        dim=model.joint_count,
                        inputs=[
                            joint_impulse,
                            model.joint_enabled,
                            model.joint_type,
                            model.joint_child,
                            dt,
                        ],
                        outputs=[state_out.body_parent_f],
                        device=model.device,
                    )
            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            if model.body_count:
                if body_q.ptr != state_out.body_q.ptr:
                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)

            # update body velocities from position changes
            if self.compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
                # causes gradient issues (probably due to numerical problems
                # when computing velocities from position changes)
                if requires_grad:
                    out_body_qd = wp.clone(state_out.body_qd)
                else:
                    out_body_qd = state_out.body_qd

                # update body velocities
                wp.launch(
                    kernel=update_body_velocities,
                    dim=model.body_count,
                    inputs=[state_out.body_q, body_q_init, model.body_com, dt],
                    outputs=[out_body_qd],
                    device=model.device,
                )

            if self.enable_restitution and contacts is not None:
                if model.particle_count:
                    wp.launch(
                        kernel=apply_particle_shape_restitution,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            particle_qd,
                            self.particle_q_init,
                            self.particle_qd_init,
                            model.particle_radius,
                            model.particle_flags,
                            body_q,
                            body_q_init,
                            body_qd,
                            body_qd_init,
                            model.body_com,
                            model.shape_body,
                            model.particle_adhesion,
                            model.soft_contact_restitution,
                            contacts.soft_contact_count,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                            contacts.soft_contact_max,
                        ],
                        outputs=[state_out.particle_qd],
                        device=model.device,
                    )

                if model.body_count:
                    body_deltas.zero_()

                    wp.launch(
                        kernel=apply_rigid_restitution,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            body_q_init,
                            body_qd_init,
                            model.body_com,
                            self.body_inv_mass_effective,
                            self.body_inv_inertia_effective,
                            model.body_world,
                            model.shape_body,
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_material_restitution,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_offset0,
                            contacts.rigid_contact_offset1,
                            rigid_contact_inv_weight_init,
                            model.gravity,
                            dt,
                        ],
                        outputs=[
                            body_deltas,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[
                            body_deltas,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            if model.body_count:
                self.copy_kinematic_body_state(model, state_in, state_out)

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Populate ``contacts.force`` from XPBD contact impulses accumulated during the last :meth:`step`.

        Both force [N] and torque [N·m] components are written.  The torque
        includes torsional and rolling friction contributions that cannot be
        reconstructed from the linear force alone.

        When ``rigid_contact_con_weighting`` is enabled, the raw per-contact
        impulse is scaled to reflect the ``1/N`` correction that
        ``apply_body_deltas`` applies.  For contacts between a dynamic and a
        kinematic body, ``N`` is the dynamic body's contact count.  For
        contacts between two dynamic bodies, the harmonic mean
        ``2/(N_a + N_b)`` is used so that the reported force is symmetric with
        respect to body ordering.  This is an approximation -- the solver
        applies ``1/N_a`` and ``1/N_b`` independently to each side, so no
        single scalar can exactly represent both.

        Args:
            contacts: :class:`Contacts` object whose :attr:`~Contacts.force` buffer will be written.
                Must have been created with ``"force"`` in its requested attributes and must
                match the :class:`Contacts` instance (same ``rigid_contact_max``) passed to
                the preceding :meth:`step`.
            state: Unused (accepted for API compatibility with :class:`SolverBase`).

        Raises:
            ValueError: If ``contacts.force`` is ``None`` (not requested), if no step has been run yet,
                or if the contacts capacity does not match the one used in the last :meth:`step`.
        """
        self._apply_module_options()
        if contacts.force is None:
            raise ValueError(
                "contacts.force is not allocated. Call model.request_contact_attributes('force') "
                "before creating the Contacts object."
            )
        if not hasattr(self, "_contact_impulse") or self._contact_impulse is None:
            raise ValueError("No contact impulse data available. Call step() before update_contacts().")
        if contacts.rigid_contact_max != self._contact_impulse_capacity:
            raise ValueError(
                f"Contacts capacity mismatch: update_contacts() received rigid_contact_max="
                f"{contacts.rigid_contact_max}, but step() used {self._contact_impulse_capacity}. "
                f"Pass the same Contacts instance to both step() and update_contacts()."
            )

        contacts.force.zero_()

        wp.launch(
            kernel=convert_contact_impulse_to_force,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                self._contact_impulse,
                self._last_dt,
            ],
            outputs=[contacts.force],
            device=self.model.device,
        )
