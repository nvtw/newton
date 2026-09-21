# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""PhoenX solver wrapped in Newton's :class:`SolverBase` interface.

Drives :class:`PhoenXWorld` from Newton's Model/State/Control/Contacts. Per step:
import (Newton -> PhoenX body fields + joint_f), direct-drive target binding,
and export (PhoenX -> body_q / body_qd). Experimental maximal projectors also
mirror drive controls into their private compatibility columns.
PhoenX slot 0 is the static world anchor; pass ``substeps=1`` to substep outside.
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

import newton
from newton._src.sim import BodyFlags, CollisionPipeline, Contacts, Control, JointType, Model, ModelFlags, State
from newton._src.solvers.phoenx.adapter_kernels import (
    _apply_joint_drive_control_kernel,
    _apply_joint_forces_kernel,
    _contact_impulse_to_force_wrapper_kernel,
    _export_body_qdd_kernel,
    _export_body_state_avg_kernel,
    _export_body_state_fd_kernel,
    _export_body_state_kernel,
    _import_body_forces_kernel,
    _import_body_state_kernel,
    _init_phoenx_body_container_kernel,
    _seed_kinematic_initial_pose_kernel,
    _snapshot_pre_step_pose_kernel,
    _snapshot_pre_step_velocity_kernel,
)
from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem
from newton._src.solvers.phoenx.articulations.direct_contact_gs import DirectContactRunSchedule
from newton._src.solvers.phoenx.articulations.direct_contact_response import DirectContactResponse
from newton._src.solvers.phoenx.articulations.direct_equality import DirectEqualitySystem, _drive_dof_masks
from newton._src.solvers.phoenx.articulations.maximal_contact_gs import MaximalContactRunSchedule
from newton._src.solvers.phoenx.articulations.maximal_contact_response import MaximalContactResponse
from newton._src.solvers.phoenx.articulations.maximal_projector import (
    MaximalTreeProjector,
    find_full_coordinate_revolute_trees,
)
from newton._src.solvers.phoenx.articulations.maximal_projector_general import GeneralMaximalTreeProjector
from newton._src.solvers.phoenx.articulations.reduced import ReducedPhoenXArticulation, _get_reduced_model
from newton._src.solvers.phoenx.body import BodyContainer, body_container_zeros
from newton._src.solvers.phoenx.cloth_collision import (
    PhoenXClothShareVertexFilterData,
    build_phoenx_share_vertex_filter_data,
    phoenx_cloth_share_vertex_filter,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _OFF_DAMPING_DRIVE,
    _OFF_DRIVE_MODE,
    _OFF_MAX_FORCE_DRIVE,
    _OFF_STIFFNESS_DRIVE,
    _OFF_TARGET,
    _OFF_TARGET_VELOCITY,
)
from newton._src.solvers.phoenx.constraints.contact_tgs import allocate_contact_tgs, export_contact_wrenches
from newton._src.solvers.phoenx.constraints.d6_joint_data import build_d6_inequality_data
from newton._src.solvers.phoenx.materials import CombineMode, Material, material_table_from_list
from newton._src.solvers.phoenx.model_adapter import (
    JointInitArrays,
    build_joint_init_arrays,
)
from newton._src.solvers.phoenx.simulation import PhoenXWorld
from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING
from newton._src.solvers.solver import SolverBase

__all__ = ["SolverPhoenX"]


_AUTO_SINGLE_WORLD_MIN_BODIES = 2048


def _resolve_auto_step_layout(
    *,
    step_layout: str,
    num_worlds: int,
    body_count: int,
    has_joints: bool,
    has_deformables: bool,
    has_shapes: bool,
    contact_friction_model: str,
    articulation_mode: str,
) -> str:
    """Resolve the model-facing PhoenX scheduler policy."""
    if step_layout not in ("auto", "multi_world", "single_world"):
        raise ValueError("step_layout must be auto, multi_world, or single_world")

    large_rigid_contact_world = (
        num_worlds == 1
        and body_count >= _AUTO_SINGLE_WORLD_MIN_BODIES
        and not has_joints
        and not has_deformables
        and has_shapes
        and contact_friction_model == "point"
        and articulation_mode == "maximal"
    )
    if step_layout == "auto":
        return "single_world" if large_rigid_contact_world else "multi_world"
    return step_layout


def _can_combine_direct_prepare_projection(
    has_velocity_limits: bool, contact_friction_model: str, articulation_mode: str
) -> bool:
    """Return whether row preparation is independent of projected velocity."""
    return articulation_mode == "maximal" and contact_friction_model == "point" and not has_velocity_limits


def _estimate_rigid_contact_max_phoenx(model) -> int | None:
    """Tight rigid_contact_max from shape_contact_pair_count * 5 (CPP) * 2 (safety).
    None when pair count is unavailable (caller falls back to Newton's default)."""
    pair_count = int(getattr(model, "shape_contact_pair_count", 0) or 0)
    if pair_count <= 0:
        return None

    # GJK/MPR primitive contacts cap at ~5/pair; Newton's default 40 is for
    # opt-in hydroelastic and overshoots us.
    PRIMITIVE_CPP = 5
    SAFETY = 2
    return max(1000, pair_count * PRIMITIVE_CPP * SAFETY)


def _estimate_contact_column_max_phoenx(model: Model, rigid_contact_max: int) -> int:
    """Bound shape-pair contact columns independently from contact points."""
    pair_count = int(getattr(model, "shape_contact_pair_count", 0) or 0)
    if pair_count <= 0:
        return int(rigid_contact_max)
    return min(int(rigid_contact_max), max(1000, pair_count))


def _build_maximal_motor_body_inv_inertia(model: Model) -> wp.array[wp.mat33f]:
    """Build the legacy PGS/projector stator-rotor approximation.

    Each revolute DoF adds motor-side armature to its parent body and
    gear-reflected armature to its child body, both along the joint axis in
    the respective body frame. Internal motor forces therefore use the
    ordinary momentum-conserving rigid-body mass operator.
    """
    body_inv = np.asarray(model.body_inv_inertia.numpy(), dtype=np.float64)
    body_inertia = np.zeros_like(body_inv)
    dynamic = np.asarray(model.body_inv_mass.numpy()) > 0.0
    for body in np.flatnonzero(dynamic):
        body_inertia[body] = np.linalg.inv(body_inv[body])

    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_type = model.joint_type.numpy()
    joint_xform_parent = model.joint_X_p.numpy()
    joint_xform_child = model.joint_X_c.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()
    joint_axis = model.joint_axis.numpy()
    armature = model.joint_armature.numpy()
    gear = model.joint_gear.numpy() if model.joint_gear is not None else np.ones_like(armature)

    for joint in range(int(model.joint_count)):
        qd_start = int(joint_qd_start[joint])
        linear_count = int(joint_dof_dim[joint, 0])
        angular_count = int(joint_dof_dim[joint, 1])
        child = int(joint_child[joint])
        if np.any(armature[qd_start : qd_start + linear_count + angular_count] < 0.0):
            raise ValueError("joint_armature must be nonnegative")
        for offset in range(linear_count):
            dof = qd_start + offset
            if float(armature[dof]) > 0.0:
                raise NotImplementedError(
                    "Maximal PhoenX body-space armature currently supports rotational motor rotors only"
                )
        for offset in range(angular_count):
            dof = qd_start + linear_count + offset
            rotor_inertia = float(armature[dof])
            if rotor_inertia == 0.0:
                continue
            if int(joint_type[joint]) != int(JointType.REVOLUTE):
                raise NotImplementedError("Maximal PhoenX rotor-side armature currently supports revolute joints only")
            ratio = float(gear[dof])
            if not np.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(f"joint_gear[{dof}] must be finite and positive")
            axis_joint = np.asarray(joint_axis[dof], dtype=np.float64)
            axis_norm = float(np.linalg.norm(axis_joint))
            if axis_norm <= 1.0e-12:
                raise ValueError(f"joint_axis[{dof}] must be non-zero for motor armature")
            axis_joint /= axis_norm
            parent = int(joint_parent[joint])
            if parent >= 0 and dynamic[parent]:
                q_parent_joint = np.asarray(joint_xform_parent[joint, 3:7], dtype=np.float64)
                q_xyz = q_parent_joint[:3]
                axis_parent = axis_joint + 2.0 * np.cross(
                    q_xyz, np.cross(q_xyz, axis_joint) + q_parent_joint[3] * axis_joint
                )
                body_inertia[parent] += rotor_inertia * np.outer(axis_parent, axis_parent)
            if child >= 0 and dynamic[child]:
                q_child_joint = np.asarray(joint_xform_child[joint, 3:7], dtype=np.float64)
                q_xyz = q_child_joint[:3]
                axis_child = axis_joint + 2.0 * np.cross(
                    q_xyz, np.cross(q_xyz, axis_joint) + q_child_joint[3] * axis_joint
                )
                body_inertia[child] += (ratio * ratio * rotor_inertia) * np.outer(axis_child, axis_child)

    for body in np.flatnonzero(dynamic):
        body_inv[body] = np.linalg.inv(body_inertia[body])
    return wp.array(body_inv.astype(np.float32), dtype=wp.mat33f, device=model.device)


class _PhoenXCollisionPipelineAdapter:
    """Route explicit pipeline collisions through PhoenX deformable refresh."""

    def __init__(self, solver: SolverPhoenX, pipeline):
        self._solver = solver
        self._pipeline = pipeline

    def __getattr__(self, name: str):
        return getattr(self._pipeline, name)

    def contacts(self):
        return self._pipeline.contacts()

    def collide(self, state: State, contacts: Contacts, *, soft_contact_margin: float | None = None) -> None:
        self._solver.collide(state, contacts)


class SolverPhoenX(SolverBase):
    """Newton :class:`SolverBase` wrapper around :class:`PhoenXWorld`.

    Supports REVOLUTE / PRISMATIC (PD drive, position/velocity limit), BALL,
    FIXED, CABLE (soft fixed with PD bend/twist; stretch DoF is rigid), FREE
    (no column), DISTANCE bounds, and D6 bilateral lock patterns and drives.
    Common D6 inequality rows handle per-axis limits, velocity caps, and
    friction for maximal-coordinate joints. Reduced-coordinate ownership is
    available independently.

    Newton :class:`Picking` works out of the box: pick force/torque is
    added to ``state.body_f``, which :meth:`step` imports into PhoenX's
    force accumulators before integrating.
    """

    def __init__(
        self,
        model: Model,
        *,
        collision_pipeline: CollisionPipeline | None = None,
        substeps: int = 1,
        solver_iterations: int = 8,
        solver_scheme: str = "soft",
        velocity_iterations: int = 1,
        velocity_relaxation: str = "each_substep",
        joint_friction_model: str = "hard",
        contact_friction_model: str = "point",
        default_friction: float = 0.5,
        friction_combine_mode: str = "average",
        step_layout: str = "auto",
        threads_per_world: int | str = "auto",
        multi_world_scheduler: str = "auto",
        max_thread_blocks: int | None = None,
        velocity_readout: str = "substep_end",
        mass_splitting: bool = False,
        max_colored_partitions: int = 12,
        mass_splitting_batch_size: int = 8,
        mass_splitting_color_group_size: int = 0,
        joint_refinement_iterations: int = 0,
        direct_joint_projection_passes: int = 1,
        mass_splitting_unrolled: bool = False,
        partitioner_algorithm: str = "greedy",
        max_greedy_outer_iters: int | None = None,
        enable_warm_start_coloring: bool = True,
        enable_column_timers: bool = False,
        sor_boost: float = 1.0,
        sleeping_velocity_threshold: float = 0.0,
        sleeping_frames_required: int = 30,
        prepare_refresh_stride: int | str = "auto",
        parallel_contact_prepare: bool = False,
        contact_chunk_size: int = 0,
        enable_body_pair_grouping: bool | None = None,
        solver_flavor: str | None = None,
        jacobi_max_colors: int | None = None,
        joint_mode: str = "maximal_direct",
        reduced_articulation_path: str = "reference",
    ):
        """Build the PhoenX solver from ``model``.

        Args:
            collision_pipeline: Optional preconfigured collision pipeline.
                PhoenX reuses this pipeline and its contact-buffer sizing
                instead of creating a default sticky pipeline.
            substeps: PhoenX internal substeps per :meth:`step` call.
            solver_iterations: PGS iterations per substep.
            solver_scheme: ``"soft"`` preserves the existing solver. Experimental
                ``"tgs"`` uses persistent two-anchor friction patches, temporal
                joint springs and one external-force update per outer step.
                Requires CUDA, maximal rigid worlds, ``step_layout="single_world"``,
                ``joint_mode="maximal_pgs"``,
                mass splitting with color groups, one solver iteration, prepare
                stride 1, SOR 1, physical ``substep_end`` velocity readout,
                no contact chunks, sleeping, partition reuse or unrolled dispatch.
                Restitution, soft contacts, armature and bounded drives are unsupported.
                Requested contact forces include persistent-anchor friction and torque,
                averaged over the outer step.
                This is a runtime solver policy, not a Model or USD attribute.
                Anchor correlation and friction admission tolerances are currently
                fixed at 0.00025 m and 0.0004 m, respectively.
            velocity_iterations: TGS-soft relax sweeps at each selected relaxation phase.
            velocity_relaxation: ``"each_substep"`` relaxes after every temporal
                substep (the default). ``"final_substep"`` relaxes only after
                the final substep of each solver step.
            joint_friction_model: "hard" uses PhoenX Coulomb friction;
                "mujoco" maps MuJoCo solref/solimp friction metadata
                when available.
            contact_friction_model: "point" solves two tangent rows at
                every contact point. Experimental "patch" keeps every
                point normal but couples friction into one central 2D row for
                each convex shape pair. Raw meshes, heightfields, and compound
                body-pair columns retain point friction. Patch friction supports
                maximal and reference reduced articulations.
            prepare_refresh_stride: Refresh cached rigid contact/joint
                prepare data every N substeps. ``"auto"`` chooses a
                conservative stride from the substep count and falls back
                to ``1`` when cached prepare is unsupported. Pass ``1``
                to force exact per-substep rebuilds.
            mass_splitting_color_group_size: Experimental number of sequential
                colors sharing each mass copy. Zero preserves existing scheduling.
                Positive values use deterministic color groups for small CUDA
                rigid mechanisms in the single-world layout with block PGS or direct joints
                and point friction. Direct joints are projected exactly between grouped contact
                sweeps instead of coupling contacts through the complete mechanism response.
                Requires mass splitting and sor_boost=1.0; incompatible with sleeping,
                packed contacts, deformables, and unrolled mass splitting.
            joint_refinement_iterations: Additional joint-only biased sweeps after
                the mixed constraint iterations, reconciling mass copies after
                each sweep. Defaults to zero. Requires the soft scheme, grouped
                single-world mass splitting, and block PGS joints.
            direct_joint_projection_passes: Number of exact direct-joint projections
                distributed across each group of PGS contact iterations. Two passes
                feed contact impulses through the joint graph at mid-solve and again
                after the final iteration. Defaults to one final projection. Requires
                soft grouped single-world mass splitting with direct joints.
            parallel_contact_prepare: Experimental parallel geometry preparation
                for CUDA single-world maximal rigid point contacts. Preserves
                ordered contact warm starts, including mass-split copy states.
                Requires no deformables, sleeping, patch friction, or unrolled
                mass splitting. Defaults to False. Independent kernels may
                change floating-point rounding relative to serial preparation.
            contact_chunk_size: Experimental maximum contacts per solver column.
                Zero preserves existing grouping. Positive values preserve every
                contact row while splitting long columns for scheduling. Requires
                maximal rigid point contacts without deformables.
            enable_body_pair_grouping: Experimental contact-column policy. None
                automatically groups compound-body contacts in eligible single-scene
                layouts. True requests the grouped path, while False preserves
                separate shape-pair manifolds. Shape-pair manifolds are appropriate for
                disconnected or strongly nonconvex compound contact patches.
            default_friction: Fallback when Contacts/shapes carry no material.
            friction_combine_mode: Rule used to combine per-shape friction.
                Supported values are ``"average"``, ``"min"``, ``"multiply"``,
                and ``"max"``. Defaults to ``"average"``.
            step_layout: ``"auto"`` keeps the multi-world scheduler except
                for one rigid contact-only world with at least 2,048 bodies,
                where it selects ``"single_world"``. Explicit
                ``"multi_world"`` and ``"single_world"`` override the policy.
                The single-world layout colors all constraints together and can
                also process multiple independent model worlds, including with
                temporal color groups. Temporal sweeps use separate blocks per
                world when there are no shared global bodies. World collision
                isolation is unchanged.
            threads_per_world: ``"auto"`` / 32 / 16 / 8 (multi-world).
            multi_world_scheduler: Static multi-world scheduler policy.
                ``"auto"`` is the default performance policy and resolves
                before graph capture; ``"fast_tail"`` and
                ``"block_world[_32|_64|_128]"`` force a path for
                benchmarking.
            max_thread_blocks: Optional cap on the single-world PGS grid.
            velocity_readout: ``"substep_end"`` (default, bit-faithful),
                ``"finite_difference"``, or ``"substep_average"``.
            mass_splitting: Enable the graph-colored mass-splitting tail.
            max_colored_partitions: True GS colors retained before the
                mass-splitting tail. Defaults to 12.
            mass_splitting_unrolled: Use the fixed-launch mass-splitting
                dispatcher. This can reduce graph-control overhead on
                workloads whose overflow structure is stable.
            partitioner_algorithm: ``"greedy"`` (default),
                ``"endpoint_owner"`` (single-world mass splitting only), or
                ``"luby_fixed"`` (single-world only).
            max_greedy_outer_iters: Optional cap on greedy partitioner
                iterations. Remaining constraints enter the overflow
                partition.
            enable_warm_start_coloring: Reuse previous-frame colour
                assignments. No-op on multi-world.
            enable_column_timers: Collect per-constraint-column timing
                counters for diagnostics. Defaults to ``False``.
            sor_boost: Per-impulse SOR factor. 1.0 = vanilla PGS;
                1.1-1.5 typical; ``>= 2.0`` diverges.
            sleeping_velocity_threshold: Per-island sleep cutoff
                ``[m/s + rad/s * 0.5 * aabb_diag]``. ``0.0`` disables
                sleeping (no island build, no extra allocations).
                Sleeping bodies are dropped from coloring + the
                overflow partition and skip gravity / forces; a
                sleeping-aware broad-phase filter is auto-installed.
            sleeping_frames_required: Frames an island must stay below
                threshold before being flagged sleeping. Default 30
                (~0.5 s @ 60 Hz). Wake-up is always single-frame.
                ``0`` recovers single-frame sleep.
            joint_mode: Joint representation and solve strategy. ``"maximal_direct"``
                (default) keeps independent-body dynamics, solves structural
                joint rows as one sparse mechanism system, and leaves inequality
                rows in PhoenX PGS. ``"maximal_pgs"`` solves physical D6 joint
                blocks in the contact color sweeps; it requires maximal
                coordinates, rigid point contacts, and the single-world layout.
                ``"reduced"`` lets generalized coordinates own declared tree
                joints. The solver never changes this choice from model topology.
                With ``"maximal_direct"``, revolute and
                prismatic ``joint_armature`` use exact generalized dynamic
                rows, reflected through ``joint_gear`` squared, in each
                mechanism system. Experimental maximal-projector paths retain
                a revolute stator/rotor body-inertia approximation.
                ``"maximal_projected"`` additionally applies an exact
                mass-metric tree projection on supported CUDA robot forests.
                Experimental ``"maximal_articulated"`` also applies exact
                tree-constrained hard-contact responses for free-root
                revolute forests; sleeping, soft-PD contacts, deformables,
                and patch friction are not supported by this mode.
                Common anchored/floating mixed rigid joints use a general
                projector; other topologies use pure reduced ownership without
                redundant maximal joint rows. ``"hybrid"`` retains the
                articulated-body preconditioner.
            solver_flavor: Deprecated compatibility argument. Omit it or pass
                ``"standard"``. The experimental ``"simple"`` Jacobi solver
                moved out of production PhoenX; use PhoenX Mini under ``phoenx.experimental.mini`` for solver experiments.
            jacobi_max_colors: Deprecated compatibility argument with no effect.
            reduced_articulation_path: ``"reference"`` uses the established
                reduced solver. Experimental ``"persistent"`` enables
                topology-proven cross-phase articulation fusion on CUDA.
        """
        super().__init__(model)
        if solver_scheme not in ("soft", "tgs"):
            raise ValueError("solver_scheme must be 'soft' or 'tgs'")
        self.solver_scheme = solver_scheme
        if solver_flavor is not None:
            warnings.warn(
                "SolverPhoenX.solver_flavor is deprecated; omit it to use the production solver.",
                DeprecationWarning,
                stacklevel=2,
            )
            if solver_flavor != "standard":
                raise ValueError(
                    "solver_flavor='simple' was removed from production PhoenX; use PhoenX Mini under phoenx.experimental.mini"
                )
        if jacobi_max_colors is not None:
            warnings.warn(
                "SolverPhoenX.jacobi_max_colors is deprecated and has no effect.",
                DeprecationWarning,
                stacklevel=2,
            )
        joint_modes = {
            "maximal_direct": ("maximal", "direct"),
            "maximal_pgs": ("maximal", "block_pgs"),
            "reduced": ("reduced", "direct"),
            "maximal_projected": ("maximal_projected", "direct"),
            "maximal_articulated": ("maximal_articulated", "direct"),
            "hybrid": ("hybrid", "direct"),
        }
        if joint_mode not in joint_modes:
            raise ValueError(f"joint_mode must be one of {tuple(joint_modes)}, got {joint_mode!r}")
        articulation_mode, joint_solver = joint_modes[joint_mode]
        self.joint_mode = joint_mode
        self._joint_solver = joint_solver
        self._articulation_mode = articulation_mode
        if isinstance(contact_chunk_size, bool) or not isinstance(contact_chunk_size, int) or contact_chunk_size < 0:
            raise ValueError("contact_chunk_size must be a nonnegative integer")
        if enable_body_pair_grouping is not None and not isinstance(enable_body_pair_grouping, bool):
            raise TypeError("enable_body_pair_grouping must be a bool or None")
        if (
            isinstance(joint_refinement_iterations, bool)
            or not isinstance(joint_refinement_iterations, int)
            or joint_refinement_iterations < 0
        ):
            raise ValueError("joint_refinement_iterations must be a nonnegative integer")
        if (
            isinstance(direct_joint_projection_passes, bool)
            or not isinstance(direct_joint_projection_passes, int)
            or not 1 <= direct_joint_projection_passes <= solver_iterations
        ):
            raise ValueError("direct_joint_projection_passes must be an integer between 1 and solver_iterations")
        gravity_np = self._read_model_gravity_np(model)

        num_worlds = max(1, int(gravity_np.shape[0]))
        has_deformables = any(
            int(getattr(model, field, 0) or 0) > 0
            for field in ("particle_count", "tri_count", "edge_count", "tet_count")
        )
        joint_types = model.joint_type.numpy() if int(model.joint_count) > 0 else np.empty(0, dtype=np.int32)
        has_constraint_joints = bool(np.any(joint_types != int(JointType.FREE)))
        if contact_chunk_size > 0 and (
            has_deformables or articulation_mode != "maximal" or contact_friction_model != "point"
        ):
            raise ValueError("contact_chunk_size requires maximal rigid point contacts without deformables")

        if parallel_contact_prepare and (
            has_deformables or articulation_mode != "maximal" or contact_friction_model != "point"
        ):
            raise ValueError("parallel_contact_prepare requires maximal rigid point contacts without deformables")

        step_layout = _resolve_auto_step_layout(
            step_layout=step_layout,
            num_worlds=num_worlds,
            body_count=int(model.body_count),
            has_joints=has_constraint_joints,
            has_deformables=has_deformables,
            has_shapes=int(model.shape_count) > 0,
            contact_friction_model=contact_friction_model,
            articulation_mode=articulation_mode,
        )
        if joint_solver == "block_pgs" and (
            articulation_mode != "maximal"
            or has_deformables
            or contact_friction_model != "point"
            or step_layout != "single_world"
            or np.any(joint_types == int(JointType.ROD))
        ):
            raise ValueError(
                "joint_mode='maximal_pgs' requires single-world maximal rigid point contacts without rod joints"
            )
        if solver_scheme == "tgs":
            if (
                not model.device.is_cuda
                or step_layout != "single_world"
                or articulation_mode != "maximal"
                or joint_solver != "block_pgs"
                or has_deformables
                or contact_friction_model != "point"
                or not mass_splitting
                or mass_splitting_color_group_size <= 0
                or mass_splitting_unrolled
                or solver_iterations != 1
                or velocity_readout != "substep_end"
                or prepare_refresh_stride != 1
                or sor_boost != 1.0
                or sleeping_velocity_threshold != 0.0
                or contact_chunk_size != 0
            ):
                raise ValueError("solver_scheme='tgs' requires the documented temporal rigid color-group configuration")
            self._validate_temporal_model_properties()
        if mass_splitting_color_group_size and (
            not mass_splitting
            or articulation_mode != "maximal"
            or joint_solver not in ("block_pgs", "direct")
            or step_layout != "single_world"
            or has_deformables
            or contact_friction_model != "point"
        ):
            raise ValueError(
                "mass_splitting_color_group_size requires maximal rigid worlds in the single_world layout "
                "with mass splitting, point contacts, and a rigid joint solver"
            )
        if joint_refinement_iterations and (
            solver_scheme != "soft"
            or joint_solver != "block_pgs"
            or step_layout != "single_world"
            or not mass_splitting
            or mass_splitting_color_group_size <= 0
            or solver_iterations <= 0
        ):
            raise ValueError(
                "joint_refinement_iterations requires soft grouped single-world mass splitting "
                "with block PGS joints and positive solver_iterations"
            )
        if direct_joint_projection_passes > 1 and (
            solver_scheme != "soft"
            or joint_solver != "direct"
            or step_layout != "single_world"
            or not mass_splitting
            or mass_splitting_color_group_size <= 0
        ):
            raise ValueError(
                "multiple direct_joint_projection_passes require soft grouped "
                "single-world mass splitting with direct joints"
            )
        if (
            contact_chunk_size
            and has_constraint_joints
            and joint_solver != "block_pgs"
            and not (joint_solver == "direct" and mass_splitting_color_group_size)
        ):
            raise ValueError(
                "contact_chunk_size with joints requires joint_mode='maximal_pgs' or grouped maximal-direct contacts"
            )
        if reduced_articulation_path not in ("reference", "persistent"):
            raise ValueError(
                f"reduced_articulation_path must be 'reference' or 'persistent', got {reduced_articulation_path!r}"
            )
        if (
            articulation_mode in ("maximal", "maximal_projected", "maximal_articulated")
            and reduced_articulation_path != "reference"
        ):
            raise ValueError("reduced_articulation_path requires joint_mode='hybrid' or 'reduced'")
        valid_combine_modes = ("average", "min", "multiply", "max")
        if friction_combine_mode not in valid_combine_modes:
            raise ValueError(
                f"friction_combine_mode must be one of {valid_combine_modes}, got {friction_combine_mode!r}"
            )
        self.friction_combine_mode = friction_combine_mode
        if contact_friction_model not in ("point", "patch"):
            raise ValueError(f"contact_friction_model must be 'point' or 'patch', got {contact_friction_model!r}")
        if contact_friction_model == "patch" and articulation_mode not in ("maximal", "reduced"):
            raise ValueError("contact_friction_model='patch' requires a maximal or reduced joint_mode")
        if (
            contact_friction_model == "patch"
            and articulation_mode == "reduced"
            and reduced_articulation_path == "persistent"
        ):
            raise ValueError("contact_friction_model='patch' does not support the persistent reduced articulation path")
        if contact_friction_model == "patch" and mass_splitting:
            raise ValueError("contact_friction_model='patch' currently requires mass_splitting=False")
        if articulation_mode in ("maximal_projected", "maximal_articulated", "hybrid") and mass_splitting:
            raise ValueError("projected and hybrid articulation modes require mass_splitting=False")
        if articulation_mode in (
            "maximal_projected",
            "maximal_articulated",
            "hybrid",
            "reduced",
        ) and multi_world_scheduler.startswith("block_world"):
            raise ValueError("projected/hybrid/reduced articulation modes require the fast-tail multi-world scheduler")
        if (
            articulation_mode in ("maximal_projected", "maximal_articulated", "hybrid", "reduced")
            and multi_world_scheduler == "auto"
        ):
            multi_world_scheduler = "fast_tail"
        self.reduced_articulation_path = reduced_articulation_path
        self._reduced_articulation: ReducedPhoenXArticulation | None = None
        self._maximal_tree_projector: MaximalTreeProjector | GeneralMaximalTreeProjector | None = None
        self._maximal_contact_response: MaximalContactResponse | None = None
        self._maximal_contact_schedule: MaximalContactRunSchedule | None = None
        self._direct_contact_response: DirectContactResponse | None = None
        self._direct_contact_schedule: DirectContactRunSchedule | None = None
        self._direct_contact_active_mechanisms: tuple[bool, ...] | None = None
        self._maximal_tree_projector_cls: type[MaximalTreeProjector] | type[GeneralMaximalTreeProjector] | None = None
        shape_flags = model.shape_flags.numpy() if model.shape_flags is not None else np.empty(0, dtype=np.int32)
        has_rigid_collision_shapes = bool(
            np.any(np.asarray(shape_flags, dtype=np.int32) & int(newton.ShapeFlags.COLLIDE_SHAPES))
        )
        # Direct LLT remains the equality owner; an exact tree factor may
        # additionally supply constrained mobility to point-contact rows.
        # Eligibility is resolved from the enabled joint graph after joint constraint has
        # classified D6 joints; reduced-coordinate metadata is never an input.
        direct_tree_contact_candidate = bool(
            joint_solver == "direct"
            and articulation_mode == "maximal"
            and contact_friction_model == "point"
            and mass_splitting_color_group_size == 0
            and not has_deformables
            and has_rigid_collision_shapes
        )
        self._direct_tree_contacts = False
        if articulation_mode in ("maximal_projected", "maximal_articulated"):
            if MaximalTreeProjector.supports_model(model):
                self._maximal_tree_projector_cls = MaximalTreeProjector
            elif GeneralMaximalTreeProjector.supports_model(model):
                self._maximal_tree_projector_cls = GeneralMaximalTreeProjector
        self._uses_maximal_tree_projector = bool(
            articulation_mode in ("maximal_projected", "maximal_articulated")
            and self._maximal_tree_projector_cls is not None
        )
        if articulation_mode == "maximal_articulated" and self._maximal_tree_projector_cls is not MaximalTreeProjector:
            raise NotImplementedError("maximal_articulated currently requires free-root revolute articulation trees")
        self._uses_reduced_joint_ownership = articulation_mode == "reduced" or (
            articulation_mode == "maximal_projected" and not self._uses_maximal_tree_projector
        )
        self._phoenx_body_inv_inertia = (
            _build_maximal_motor_body_inv_inertia(model)
            if self._uses_maximal_tree_projector
            else model.body_inv_inertia
        )
        valid_readouts = ("substep_end", "finite_difference", "substep_average")
        if velocity_readout not in valid_readouts:
            raise ValueError(f"velocity_readout must be one of {valid_readouts}, got {velocity_readout!r}")
        self._velocity_readout = velocity_readout
        valid_friction_models = ("hard", "mujoco")
        if joint_friction_model not in valid_friction_models:
            raise ValueError(
                f"joint_friction_model must be one of {valid_friction_models}, got {joint_friction_model!r}"
            )
        self._joint_friction_model = joint_friction_model

        num_bodies_phoenx = int(model.body_count) + 1
        self.bodies: BodyContainer = body_container_zeros(num_bodies_phoenx, device=self.device)

        # FD/substep-avg readout buffers — always allocated so graph capture
        # has stable refs; only written when the corresponding readout fires.
        n_newton_bodies = int(model.body_count)
        self._fd_pos_prev = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)
        self._fd_orient_prev = wp.zeros(max(1, n_newton_bodies), dtype=wp.quatf, device=self.device)
        self._substep_vel_accum = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)
        self._substep_omega_accum = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)
        # body_qdd snapshot buffers (linear + angular velocity, pre-step).
        # Always allocated; the FD launch is gated on state.body_qdd being live.
        self._qdd_vel_prev = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)
        self._qdd_omega_prev = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)

        # Identity orientation everywhere so the first _update_inertia is well-defined.
        self.bodies.orientation.assign(np.tile([0.0, 0.0, 0.0, 1.0], (num_bodies_phoenx, 1)).astype(np.float32))

        if model.body_count:
            self._launch_init_phoenx_bodies(model)

        # Initialize authored generalized coordinates when the model declares
        # an articulation. Metadata-free full-coordinate models keep authored
        # body poses authoritative.
        has_articulation_metadata = (
            int(model.articulation_count) > 0
            and model.articulation_start is not None
            and model.joint_articulation is not None
        )
        if (
            int(model.body_count) > 0
            and int(model.joint_count) > 0
            and (articulation_mode != "maximal" or has_articulation_metadata)
        ):
            newton.eval_fk(model, model.joint_q, model.joint_qd, model)

        self._joint_constraints: JointInitArrays = build_joint_init_arrays(
            model,
            device=self.device,
            reduced_articulations=self._uses_reduced_joint_ownership,
        )
        num_joints = self._joint_constraints.num_joint_columns
        num_particles = int(getattr(model, "particle_count", 0) or 0)
        num_cloth_triangles = int(getattr(model, "tri_count", 0) or 0)
        num_cloth_bending = int(getattr(model, "edge_count", 0) or 0)
        num_soft_tetrahedra = int(getattr(model, "tet_count", 0) or 0)
        self._has_particles = num_particles > 0
        self._has_deformable_collision = num_cloth_triangles > 0 or num_soft_tetrahedra > 0
        self._particle_state_imported: State | None = None
        self._default_joint_gear = wp.ones(max(1, int(model.joint_dof_count)), dtype=wp.float32, device=self.device)

        if collision_pipeline is not None:
            if collision_pipeline.model is not model:
                raise ValueError("collision_pipeline must have been created for model")
            model._collision_pipeline = collision_pipeline

        # PhoenX's warm-start path needs contact_matching != "disabled".
        # Auto-attach a sticky pipeline so users don't have to size Contacts.
        self._sleeping_enabled: bool = float(sleeping_velocity_threshold) > 0.0
        if articulation_mode == "maximal_articulated" and self._sleeping_enabled:
            raise NotImplementedError(
                "maximal_articulated does not yet support sleeping; use maximal_projected or reduced"
            )
        if articulation_mode == "maximal_articulated" and (self._has_particles or self._has_deformable_collision):
            raise NotImplementedError(
                "maximal_articulated currently supports rigid contacts only; use maximal_projected or reduced"
            )
        if int(model.shape_count) > 0 and not self._has_deformable_collision:
            existing_cp = getattr(model, "_collision_pipeline", None)
            needs_new_cp = existing_cp is None or not getattr(existing_cp, "contact_matching", False)
            # When sleeping is on we need a broad-phase filter func wired
            # into the pipeline at construction time; force rebuild if
            # the existing pipeline doesn't carry one.
            if self._sleeping_enabled and not needs_new_cp:
                existing_filter = getattr(existing_cp, "_broad_phase_filter_func", None)
                if existing_filter is None:
                    needs_new_cp = True
            if needs_new_cp:
                # PhoenX-tight rigid_contact_max from shape_contact_pair_count;
                # Newton's default ignores COLLIDE_SHAPES filter and overshoots
                # ~15x with visual-only meshes.
                tight_rcm = _estimate_rigid_contact_max_phoenx(model)
                if tight_rcm is not None:
                    model.rigid_contact_max = 0  # bypass "already sized" short-circuit

                cp_kwargs = {
                    "contact_matching": PHOENX_CONTACT_MATCHING,
                    "rigid_contact_max": tight_rcm,
                }
                if self._sleeping_enabled:
                    cp_kwargs["broad_phase_filter"] = (
                        phoenx_cloth_share_vertex_filter,
                        PhoenXClothShareVertexFilterData,
                    )
                model._collision_pipeline = newton.CollisionPipeline(model, **cp_kwargs)
                model._collision_pipeline.contacts()  # forces buffer sizing
        if self._has_deformable_collision and int(model.rigid_contact_max) <= 0:
            deformable_shapes = num_cloth_triangles + num_soft_tetrahedra
            model.rigid_contact_max = max(1000, 8 * (int(model.shape_count) + deformable_shapes))
        rigid_contact_max = int(model.rigid_contact_max)

        gravity_tuples = [tuple(float(x) for x in row) for row in gravity_np]
        if len(gravity_tuples) == 1:
            gravity_arg = gravity_tuples[0]
        else:
            gravity_arg = gravity_tuples

        self._constraints: ConstraintContainer = PhoenXWorld.make_constraint_container(
            num_joints=num_joints,
            num_cloth_triangles=num_cloth_triangles,
            num_cloth_bending=num_cloth_bending,
            num_soft_tetrahedra=num_soft_tetrahedra,
            device=self.device,
        )

        # Body-pair grouping pays a sort/gather cost and reduces all shape-pair
        # patches between two bodies into one manifold. Keep an explicit opt-out
        # for disconnected or strongly nonconvex compound geometry.
        body_pair_grouping_eligible = step_layout == "single_world" or num_worlds == 1
        if enable_body_pair_grouping and not body_pair_grouping_eligible:
            raise ValueError("enable_body_pair_grouping requires a single-scene layout")
        if solver_scheme == "tgs" and enable_body_pair_grouping is False:
            raise ValueError("solver_scheme='tgs' requires body-pair contact grouping")

        has_compound_bodies = False
        if enable_body_pair_grouping is None and model.shape_body is not None and model.shape_count > 0:
            sb = model.shape_body.numpy()
            sb = sb[sb >= 0]
            if sb.size > 0:
                counts = np.bincount(sb, minlength=int(model.body_count))
                has_compound_bodies = bool((counts > 1).any())
        use_body_pair_grouping = body_pair_grouping_eligible and (
            solver_scheme == "tgs"
            or enable_body_pair_grouping is True
            or (enable_body_pair_grouping is None and has_compound_bodies)
        )

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self._constraints,
            substeps=int(substeps),
            solver_iterations=int(solver_iterations),
            velocity_iterations=int(velocity_iterations),
            velocity_relaxation=velocity_relaxation,
            gravity=gravity_arg,
            rigid_contact_max=rigid_contact_max,
            max_contact_columns=(
                rigid_contact_max
                if contact_chunk_size > 0
                else _estimate_contact_column_max_phoenx(model, rigid_contact_max)
            ),
            num_joints=num_joints,
            num_particles=num_particles,
            num_cloth_triangles=num_cloth_triangles,
            num_cloth_bending=num_cloth_bending,
            num_soft_tetrahedra=num_soft_tetrahedra,
            default_friction=float(default_friction),
            num_worlds=num_worlds,
            step_layout=step_layout,
            threads_per_world=threads_per_world,
            multi_world_scheduler=multi_world_scheduler,
            max_thread_blocks=max_thread_blocks,
            enable_body_pair_grouping=use_body_pair_grouping,
            mass_splitting=mass_splitting,
            max_colored_partitions=max_colored_partitions,
            contact_friction_model=contact_friction_model if articulation_mode == "maximal" else "point",
            combine_direct_prepare_projection=_can_combine_direct_prepare_projection(
                self._joint_constraints.has_velocity_limits, contact_friction_model, articulation_mode
            ),
            mass_splitting_batch_size=mass_splitting_batch_size,
            mass_splitting_color_group_size=mass_splitting_color_group_size,
            joint_refinement_iterations=joint_refinement_iterations,
            direct_joint_projection_passes=direct_joint_projection_passes,
            mass_splitting_unrolled=mass_splitting_unrolled,
            partitioner_algorithm=partitioner_algorithm,
            max_greedy_outer_iters=max_greedy_outer_iters,
            enable_warm_start_coloring=enable_warm_start_coloring,
            enable_column_timers=enable_column_timers,
            sor_boost=sor_boost,
            sleeping_velocity_threshold=float(sleeping_velocity_threshold),
            sleeping_frames_required=int(sleeping_frames_required),
            prepare_refresh_stride=prepare_refresh_stride,
            parallel_contact_prepare=parallel_contact_prepare,
            contact_chunk_size=contact_chunk_size,
            device=self.device,
        )

        # When sleeping is on (and not already wired by a downstream
        # ``setup_cloth_collision_pipeline``), bind the share-vertex
        # filter data with sleeping fields populated. The pipeline's
        # filter func is shared with cloth setups; cloth setup paths
        # call ``build_phoenx_share_vertex_filter_data`` themselves and
        # overwrite this binding without losing the sleeping fields.
        if self._sleeping_enabled and int(model.shape_count) > 0 and not self._has_deformable_collision:
            tri_sentinel = wp.zeros((1, 3), dtype=wp.int32, device=self.device)
            tet_sentinel = wp.zeros((1, 4), dtype=wp.int32, device=self.device)
            filter_data = build_phoenx_share_vertex_filter_data(
                num_rigid_shapes=int(model.shape_count),
                num_cloth_triangles=0,
                tri_indices=tri_sentinel,
                tet_indices=tet_sentinel,
                sleeping_enabled=True,
                phoenx_body_offset=1,
                shape_body=model.shape_body,
                body_island_root=self.bodies.island_root,
                body_motion_type=self.bodies.motion_type,
                device=self.device,
            )
            model._collision_pipeline.set_broad_phase_filter_data(filter_data)
            self._share_vertex_filter_data = filter_data
            self.world._share_vertex_filter_data = filter_data

        # Seed body pose BEFORE joint init — joint constraint init reads body positions to
        # snapshot body-local anchors. Without this, welds pull child to origin.
        if int(model.body_count) > 0:
            zero_wrench = wp.zeros(int(model.body_count), dtype=wp.spatial_vector, device=self.device)
            wp.launch(
                _import_body_state_kernel,
                dim=int(model.body_count),
                inputs=[
                    model.body_q,
                    model.body_qd,
                    zero_wrench,
                    model.body_com,
                    self.bodies,
                ],
                device=self.device,
            )
            wp.launch(
                _seed_kinematic_initial_pose_kernel,
                dim=int(model.body_count) + 1,  # +1 for slot 0 (world anchor)
                inputs=[self.bodies],
                device=self.device,
            )

        if num_joints > 0:
            joint_kwargs = self._joint_constraints.to_initialize_kwargs()
            self.world.initialize_joint_constraints(**joint_kwargs)

        joint_idx_to_cid = self._joint_constraints.joint_idx_to_cid.numpy()
        d6_data, d6_inequality_count = build_d6_inequality_data(model, joint_idx_to_cid, self._joint_friction_model)
        self.world.constraints.d6 = d6_data
        active_joint = joint_idx_to_cid >= 0
        full_coordinate_tree_joints: tuple[tuple[int, ...], ...] = ()
        if direct_tree_contact_candidate:
            full_coordinate_tree_joints = find_full_coordinate_revolute_trees(model)
            self._direct_tree_contacts = bool(full_coordinate_tree_joints)
            if self._direct_tree_contacts:
                self._maximal_tree_projector_cls = MaximalTreeProjector

        if self._uses_maximal_tree_projector or self._direct_tree_contacts:
            if self._direct_tree_contacts:
                self._maximal_tree_projector = MaximalTreeProjector(
                    model,
                    self._constraints,
                    self.bodies,
                    self._joint_constraints.joint_idx_to_cid,
                    joint_trees=full_coordinate_tree_joints,
                )
            else:
                assert self._maximal_tree_projector_cls is not None
                self._maximal_tree_projector = self._maximal_tree_projector_cls(
                    model,
                    self._constraints,
                    self.bodies,
                    self._joint_constraints.joint_idx_to_cid,
                )
            self.world._maximal_tree_projector = self._maximal_tree_projector
            if articulation_mode == "maximal_articulated" or self._direct_tree_contacts:
                self._maximal_contact_response = MaximalContactResponse(self._maximal_tree_projector)
                self._maximal_contact_schedule = MaximalContactRunSchedule(
                    self._maximal_contact_response,
                    self.world.max_contact_columns,
                    self.world.rigid_contact_max,
                )
                self.world._maximal_contact_response = self._maximal_contact_response
                self.world._maximal_contact_schedule = self._maximal_contact_schedule
                self.world._direct_tree_contacts = self._direct_tree_contacts
                if articulation_mode == "maximal_articulated":
                    joint_pgs_enabled = np.ones(num_joints, dtype=np.int32)
                    joint_idx_to_cid = self._joint_constraints.joint_idx_to_cid.numpy()
                    for joint in self._maximal_tree_projector.data.joint_index.numpy().ravel():
                        if joint >= 0:
                            cid = int(joint_idx_to_cid[int(joint)])
                            if cid >= 0:
                                # Prepare the structural row for projector geometry,
                                # but let the exact tree projection own its solve.
                                joint_pgs_enabled[cid] = 2
                    self.world.set_joint_pgs_ownership(joint_pgs_enabled)
        elif (
            self._articulation_mode in ("maximal_projected", "maximal_articulated", "hybrid", "reduced")
            and int(model.articulation_count) > 0
        ):
            reduced_model = _get_reduced_model(model)
            if reduced_model.articulation_count > 0:
                self._reduced_articulation = ReducedPhoenXArticulation(
                    reduced_model,
                    self.bodies,
                    execution_path=self.reduced_articulation_path,
                    contact_friction_model=contact_friction_model,
                )
                joint_idx_to_cid = self._joint_constraints.joint_idx_to_cid.numpy()
                joint_pgs_enabled = np.ones(num_joints, dtype=np.int32)
                if self._uses_reduced_joint_ownership:
                    for joint, owned in enumerate(self._reduced_articulation.owned_joint_mask_np):
                        cid = int(joint_idx_to_cid[joint])
                        if owned and cid >= 0:
                            joint_pgs_enabled[cid] = 0
                self.world.set_reduced_articulation(self._reduced_articulation, joint_pgs_enabled)

        self._direct_equality_system: DirectEqualitySystem | None = None
        # The experimental maximal tree projector already owns structural tree
        # equalities. Installing a second direct owner would apply incompatible
        # projections and overwrite its prepare-only PGS ownership state.
        if not self._uses_maximal_tree_projector:
            excluded_joint_mask = None
            if self._reduced_articulation is not None and self._uses_reduced_joint_ownership:
                excluded_joint_mask = self._reduced_articulation.owned_joint_mask_np
            effective_joint_dof_start = self._joint_constraints.joint_idx_to_dof_start.numpy().copy()
            effective_joint_target_start = self._joint_constraints.drive_target_q_index.numpy()
            joint_target_start = np.full(int(model.joint_count), -1, dtype=np.int32)
            drive_joint_mask = active_joint & (effective_joint_dof_start >= 0)
            joint_target_start[drive_joint_mask] = effective_joint_target_start
            material_joint = model.joint_type.numpy()[: int(model.joint_count)] == int(newton.JointType.ROD)
            if np.any(material_joint):
                effective_joint_dof_start[material_joint] = model.joint_qd_start.numpy()[: int(model.joint_count)][
                    material_joint
                ]
            equality_system_type = DirectEqualitySystem
            if joint_solver == "block_pgs":
                equality_system_type = BlockJointSystem
            self._direct_equality_system = equality_system_type(
                model,
                self.bodies,
                excluded_joint_mask=excluded_joint_mask,
                effective_joint_dof_start=effective_joint_dof_start,
                effective_joint_target_start=joint_target_start,
                direct_joint_friction=joint_solver == "direct" and self._joint_friction_model == "hard",
            )
            self.world._direct_equality_system = self._direct_equality_system
            self._direct_equality_system.bind_constraint_indices(self._joint_constraints.joint_idx_to_cid)
            if np.any(self._direct_equality_system.direct_friction_dof_mask):
                d6_data, d6_inequality_count = build_d6_inequality_data(
                    model,
                    joint_idx_to_cid,
                    self._joint_friction_model,
                    friction_dof_owned=self._direct_equality_system.direct_friction_dof_mask,
                )
                self.world.constraints.d6 = d6_data
            self._direct_equality_system.d6_inequality_count = d6_inequality_count
            if self._direct_equality_system.enabled:
                if self._direct_tree_contacts:
                    assert self._maximal_tree_projector is not None
                    topology = self._direct_equality_system.topology
                    self._maximal_tree_projector.bind_direct_dynamic_state(
                        topology.row_joint,
                        topology.row_dynamic,
                        self._direct_equality_system.dynamic_mass,
                        self._direct_equality_system.accumulated_impulse,
                    )
                # Equality- and direct-drive-only columns leave coloring.
                # Axial friction and limits retain the lean PGS iteration.
                self._direct_base_joint_pgs_enabled = self.world._joint_pgs_enabled.numpy()[:num_joints].copy()
                self._refresh_direct_joint_ownership()
                if joint_solver == "block_pgs":
                    self._direct_equality_system.bind_world(self.world, joint_idx_to_cid)

                if direct_tree_contact_candidate and self.world.max_contact_columns > 0:
                    direct_joint_mask = self._direct_equality_system.joint_mask
                    tree_joint_sets = {
                        frozenset(joint for joint in joints if direct_joint_mask[joint])
                        for joints in full_coordinate_tree_joints
                    }
                    topology = self._direct_equality_system.topology
                    active_mechanisms = tuple(
                        frozenset(
                            topology.row_joint[
                                topology.mechanism_row_start[mechanism] : topology.mechanism_row_start[mechanism + 1]
                            ]
                        )
                        not in tree_joint_sets
                        for mechanism in range(len(topology.dimensions))
                    )
                    if any(active_mechanisms):
                        self._direct_contact_active_mechanisms = active_mechanisms
                        self._direct_contact_response = DirectContactResponse(
                            self._direct_equality_system,
                            self.world.rigid_contact_max,
                            self.world.max_contact_columns,
                            active_mechanisms=active_mechanisms,
                        )
                        self._direct_contact_schedule = DirectContactRunSchedule(
                            self._direct_contact_response,
                            self.world.max_contact_columns,
                        )
                        self.world._direct_contact_response = self._direct_contact_response
                        self.world._direct_contact_schedule = self._direct_contact_schedule

        if num_cloth_triangles > 0:
            self.world.populate_cloth_triangles_from_model(model)
        if num_cloth_bending > 0:
            self.world.populate_cloth_bending_from_model(model)
        if num_soft_tetrahedra > 0:
            self.world.populate_soft_tetrahedra_from_model(model)
        if self._has_deformable_collision:
            pipeline = self.world.setup_cloth_collision_pipeline(model, rigid_contact_max=rigid_contact_max)
            model._collision_pipeline = _PhoenXCollisionPipelineAdapter(self, pipeline)
        self._collision_pipeline = getattr(model, "_collision_pipeline", None)

        if model.shape_material_mu is not None and model.shape_count > 0:
            self._install_shape_materials()

        # Newton shape_body uses -1 for world; PhoenX slot 0 is the world anchor.
        if self._has_deformable_collision and self.world._shape_body_internal is not None:
            self._shape_body = self.world._shape_body_internal
        elif model.shape_body is not None and model.shape_count > 0:
            shape_body_np = model.shape_body.numpy()
            shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
            self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)
        else:
            self._shape_body = None

        self._has_joint_forces = model.joint_dof_count > 0
        self._last_dt: float = 0.0

        if solver_scheme == "tgs":
            direct = self._direct_equality_system
            if direct is not None:
                if direct.has_bounded_drives:
                    raise ValueError("solver_scheme='tgs' requires unbounded joint drives")
                direct.set_temporal_substeps(self.world.substeps)
            world = self.world
            # Global bodies may couple otherwise separate worlds. Keep the
            # original ordered schedule for those models; global ground shapes
            # without a body do not couple the worlds.
            if np.all(model.body_world.numpy() >= 0):
                world._temporal_sweep_worlds = num_worlds
            world._temporal_contact_state = allocate_contact_tgs(
                world.rigid_contact_max, world.bodies.position.shape[0], world.substeps, world.device
            )
            world._temporal_static_heads = wp.full(world.num_bodies, -1, dtype=int, device=world.device)
            world._temporal_static_links = wp.full(
                world._contact_cols.data.shape[1], -1, dtype=int, device=world.device
            )
            world._temporal_contact_state.prepared_friction = 1
            world._temporal_joint_springs = True
            world._temporal_force_step = True

        # Placeholder for _contact_impulse_to_force_wrapper_kernel when grouping
        # is off (has_perm=0 makes the kernel ignore it).
        self._sort_perm_placeholder = wp.zeros(1, dtype=wp.int32, device=self.device)

    def _validate_temporal_model_properties(self) -> None:
        """Reject unsupported temporal properties before refreshing cached rows."""
        for field in ("joint_armature", "shape_material_restitution"):
            values = getattr(self.model, field, None)
            if values is not None and np.any(values.numpy()):
                raise ValueError(f"solver_scheme='tgs' requires zero {field}")
        if self.model.joint_dof_count and np.any(_drive_dof_masks(self.model)[1]):
            raise ValueError("solver_scheme='tgs' requires unbounded joint drives")

    def _install_shape_materials(self) -> None:
        """Stream Model's per-shape (mu_static, mu_dynamic, restitution) into
        PhoenX's material table; each shape gets its own material index."""

        mu_np = self.model.shape_material_mu.numpy()
        restitution = (
            self.model.shape_material_restitution.numpy()
            if self.model.shape_material_restitution is not None
            else np.zeros_like(mu_np)
        )
        combine_modes = {
            "average": CombineMode.AVERAGE,
            "min": CombineMode.MIN,
            "multiply": CombineMode.MULTIPLY,
            "max": CombineMode.MAX,
        }
        combine_mode = combine_modes[self.friction_combine_mode]
        materials = [
            Material(
                static_friction=float(mu_np[i]),
                dynamic_friction=float(mu_np[i]),
                restitution=float(restitution[i]),
                friction_combine_mode=combine_mode,
                restitution_combine_mode=CombineMode.AVERAGE,
            )
            for i in range(self.model.shape_count)
        ]
        material_data = material_table_from_list(materials, device=self.device)
        shape_material_idx = wp.array(
            np.arange(self.model.shape_count, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.world.set_materials(material_data, shape_material_idx)

    def _rebuild_direct_contact_response(self) -> None:
        """Rebind general contact mobility after a direct factor rebuild."""
        active_mechanisms = self._direct_contact_active_mechanisms
        direct = self._direct_equality_system
        if active_mechanisms is None or direct is None or not direct.enabled:
            return
        self._direct_contact_response = DirectContactResponse(
            direct,
            self.world.rigid_contact_max,
            self.world.max_contact_columns,
            active_mechanisms=active_mechanisms,
        )
        self._direct_contact_schedule = DirectContactRunSchedule(
            self._direct_contact_response,
            self.world.max_contact_columns,
        )
        self.world._direct_contact_response = self._direct_contact_response
        self.world._direct_contact_schedule = self._direct_contact_schedule

    def _refresh_direct_joint_ownership(self) -> None:
        """Reapply direct-drive and residual-PGS ownership after property edits."""
        direct = self._direct_equality_system
        if direct is None or not direct.enabled:
            return
        joint_idx_to_cid = self._joint_constraints.joint_idx_to_cid.numpy()
        if self._joint_solver == "block_pgs":
            self.world.set_joint_pgs_ownership(self._direct_base_joint_pgs_enabled.copy())
            return
        joint_pgs_enabled = self._direct_base_joint_pgs_enabled.copy()
        for joint in np.flatnonzero(direct.joint_mask):
            cid = int(joint_idx_to_cid[joint])
            if cid >= 0 and int(direct.d6_inequality_count[cid]) == 0:
                joint_pgs_enabled[cid] = 0
        self.world.set_joint_pgs_ownership(joint_pgs_enabled)

    def _apply_joint_control(self, control: Control) -> None:
        """Bind direct targets and update experimental projector drives."""
        model = self.model
        target_pos = (
            control.joint_target_q
            if control is not None and control.joint_target_q is not None
            else model.joint_target_q
        )
        target_vel = (
            control.joint_target_qd
            if control is not None and control.joint_target_qd is not None
            else model.joint_target_qd
        )
        if target_pos is None or target_vel is None or model.joint_target_mode is None:
            return  # no per-DOF drive configured
        if self._direct_equality_system is not None:
            self._direct_equality_system.set_control_targets(target_pos, target_vel)
        if not self._uses_maximal_tree_projector or self._joint_constraints.num_drive_columns == 0:
            return
        wp.launch(
            _apply_joint_drive_control_kernel,
            dim=int(self._joint_constraints.num_drive_columns),
            inputs=[
                self._joint_constraints.drive_cid,
                self._joint_constraints.drive_dof_start,
                self._joint_constraints.drive_target_q_index,
                self._joint_constraints.drive_q_at_init,
                model.joint_target_mode,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_effort_limit,
                self._joint_gear_array(),
                target_pos,
                target_vel,
                wp.int32(0),  # DRIVE_MODE_OFF
                wp.int32(1),  # DRIVE_MODE_POSITION
                wp.int32(2),  # DRIVE_MODE_VELOCITY
                wp.int32(int(newton.JointTargetMode.POSITION)),
                wp.int32(int(newton.JointTargetMode.VELOCITY)),
                wp.int32(int(newton.JointTargetMode.POSITION_VELOCITY)),
                wp.int32(int(_OFF_DRIVE_MODE)),
                wp.int32(int(_OFF_TARGET)),
                wp.int32(int(_OFF_TARGET_VELOCITY)),
                wp.int32(int(_OFF_STIFFNESS_DRIVE)),
                wp.int32(int(_OFF_DAMPING_DRIVE)),
                wp.int32(int(_OFF_MAX_FORCE_DRIVE)),
                self._constraints,
            ],
            device=self.device,
        )

    def _joint_gear_array(self) -> wp.array[wp.float32]:
        if self.model.joint_gear is not None:
            return self.model.joint_gear
        n = max(1, int(self.model.joint_dof_count))
        if self._default_joint_gear.shape[0] != n:
            self._default_joint_gear = wp.ones(n, dtype=wp.float32, device=self.device)
        return self._default_joint_gear

    def _accumulate_joint_forces(self, state_in: State, control: Control, dt: float) -> None:
        """Fold ``control.joint_f`` into ``state_in.body_f`` (Newton's EFFORT path)."""
        if control is None or control.joint_f is None:
            return
        if not self._has_joint_forces:
            return
        model = self.model
        if model.joint_count == 0:
            return
        wp.launch(
            _apply_joint_forces_kernel,
            dim=int(model.joint_count),
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
                state_in.body_f,
            ],
            device=self.device,
        )

    def _import_body_state(self, state_in: State) -> None:
        """Pull state_in into the PhoenX body container (slot i+1).
        Kinematic bodies go to kinematic_target_*; dynamic/static go direct."""
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            _import_body_state_kernel,
            dim=n,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_in.body_f,
                self.model.body_com,
                self.bodies,
            ],
            device=self.device,
        )

    def _import_particle_state(self, state_in: State, *, force: bool = False) -> None:
        """Pull Newton particle state into PhoenX."""
        if not self._has_particles:
            return
        particles = self.world.particles
        if particles is None:
            return
        if not force and state_in is self._particle_state_imported:
            return
        if state_in.particle_q is None or state_in.particle_qd is None:
            raise ValueError("SolverPhoenX requires particle_q and particle_qd for particle models.")
        wp.copy(particles.position, state_in.particle_q)
        wp.copy(particles.velocity, state_in.particle_qd)
        self._particle_state_imported = state_in

    def _export_particle_state(self, state_out: State) -> None:
        """Push PhoenX particle state to Newton."""
        if not self._has_particles:
            return
        particles = self.world.particles
        if particles is None:
            return
        if state_out.particle_q is None or state_out.particle_qd is None:
            raise ValueError("SolverPhoenX requires particle_q and particle_qd for particle models.")
        wp.copy(state_out.particle_q, particles.position)
        wp.copy(state_out.particle_qd, particles.velocity)

    def collide(self, state: State, contacts: Contacts) -> None:
        """Run PhoenX deformable-aware collision."""
        if not self._has_deformable_collision:
            if self._collision_pipeline is None:
                raise RuntimeError("SolverPhoenX.collide() requires a model with collision shapes.")
            self._collision_pipeline.collide(state, contacts)
            return
        self._import_particle_state(state, force=True)
        self.world.collide(state, contacts)

    def _snapshot_pre_step_pose(self) -> None:
        """Snapshot pre-step COM-in-world pose for the FD readout."""
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            _snapshot_pre_step_pose_kernel,
            dim=n,
            inputs=[self.bodies.position, self.bodies.orientation],
            outputs=[self._fd_pos_prev, self._fd_orient_prev],
            device=self.device,
        )

    def _snapshot_pre_step_velocity(self) -> None:
        """Snapshot pre-step linear + angular velocity for the body_qdd readout.
        Captured after :meth:`_import_body_state` so the FD covers the outer dt."""
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            _snapshot_pre_step_velocity_kernel,
            dim=n,
            inputs=[self.bodies.velocity, self.bodies.angular_velocity],
            outputs=[self._qdd_vel_prev, self._qdd_omega_prev],
            device=self.device,
        )

    def _export_body_qdd(self, state_out: State, dt: float) -> None:
        """Pack post-step ``body_qdd`` into ``state_out`` as a finite-difference
        of (post-step - pre-step) velocity over the outer dt. Newton convention:
        ``spatial_top`` is linear acceleration (world frame, includes
        gravity-induced terms), ``spatial_bottom`` is angular acceleration
        (world frame). Matches what :class:`~newton.sensors.SensorIMU` consumes."""
        n = int(self.model.body_count)
        if n == 0:
            return
        inv_dt = 1.0 / float(dt) if dt > 0.0 else 0.0
        wp.launch(
            _export_body_qdd_kernel,
            dim=n,
            inputs=[
                self.bodies.velocity,
                self.bodies.angular_velocity,
                self._qdd_vel_prev,
                self._qdd_omega_prev,
                wp.float32(inv_dt),
            ],
            outputs=[state_out.body_qdd],
            device=self.device,
        )

    def _export_body_state(self, state_out: State, dt: float) -> None:
        """Pack PhoenX body state back into state_out.body_q / body_qd, switching
        on ``self._velocity_readout``."""
        n = int(self.model.body_count)
        if n == 0:
            return
        if self._velocity_readout == "finite_difference":
            inv_dt = 1.0 / float(dt) if dt > 0.0 else 0.0
            wp.launch(
                _export_body_state_fd_kernel,
                dim=n,
                inputs=[
                    self.bodies.position,
                    self.bodies.orientation,
                    self.model.body_com,
                    self._fd_pos_prev,
                    self._fd_orient_prev,
                    wp.float32(inv_dt),
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=self.device,
            )
        elif self._velocity_readout == "substep_average":
            inv_dt = 1.0 / float(dt) if dt > 0.0 else 0.0
            wp.launch(
                _export_body_state_avg_kernel,
                dim=n,
                inputs=[
                    self.bodies.position,
                    self.bodies.orientation,
                    self.model.body_com,
                    self._substep_vel_accum,
                    self._substep_omega_accum,
                    wp.float32(inv_dt),
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=self.device,
            )
        else:
            wp.launch(
                _export_body_state_kernel,
                dim=n,
                inputs=[
                    self.bodies.position,
                    self.bodies.orientation,
                    self.bodies.velocity,
                    self.bodies.angular_velocity,
                    self.model.body_com,
                ],
                outputs=[
                    state_out.body_q,
                    state_out.body_qd,
                ],
                device=self.device,
            )

    def wake_on_external_input(self, state_in: State) -> None:
        """Wake every sleeping island whose bodies carry an external
        force or torque set in ``state_in.body_f``, *before* the host
        calls ``CollisionPipeline.collide(...)``.

        The per-step sleeping pass inside :meth:`step` cannot drive
        broad-phase decisions on the wake frame: by the time it clears
        ``island_root`` for a body that picking just pushed, the
        sleep-aware broad-phase filter has already dropped that body's
        contact pairs and the substep solve sees an empty stack. Call
        sequence on the host side::

            state.body_f.assign(...)  # picking / wrenches
            solver.wake_on_external_input(state)  # propagate wake
            collision_pipeline.collide(state, contacts)  # broad-phase keeps pairs
            solver.step(state, state_out, ...)

        Imports ``state_in.body_f`` into PhoenX's force accumulators
        first so the wake pass reads the user-applied wrench rather
        than the post-clear zeroes that :meth:`step` would otherwise
        re-load it from. A no-op when the sleeping pipeline is
        disabled.
        """
        if not self._sleeping_enabled:
            return
        self._import_body_state(state_in)
        self.world.wake_on_external_input()

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        state_is_continuation: bool = False,
        state_kinematics_valid: bool = False,
    ) -> None:
        """Advance the simulation state by one time step.

        Args:
            state_in: Input simulation state.
            state_out: Output simulation state.
            control: Joint controls, or ``None`` to use model controls.
            contacts: Active contacts, or ``None``.
            dt: Step duration [s].
            state_is_continuation: Whether the pose, velocity, and generalized
                arrays in ``state_in`` are the unmodified ``state_out`` from
                the immediately preceding PhoenX step. Reduced-coordinate
                solvers then preserve their authoritative state while still
                importing forces and controls. Defaults to ``False``; use
                ``False`` after changing any kinematic state array.
            state_kinematics_valid: Whether ``state_in.body_q`` and
                ``state_in.body_qd`` already match its generalized state.
                Defaults to ``False``.
        """
        if self.solver_scheme == "tgs" and bool(getattr(self, "reuse_partition", False)):
            raise ValueError("solver_scheme='tgs' requires contact ingestion on every outer step")
        if control is None:
            # Alias Model per-DOF arrays (no clone). Matches XPBD/Featherstone.
            control = self.model.control(clone_variables=False)

        self._apply_joint_control(control)
        if self._reduced_articulation is None:
            self._accumulate_joint_forces(state_in, control, dt)
        continue_reduced_state = state_is_continuation and self._reduced_articulation is not None
        if continue_reduced_state:
            wp.launch(
                _import_body_forces_kernel,
                dim=int(self.model.body_count),
                inputs=[state_in.body_f],
                outputs=[self.bodies],
                device=self.device,
            )
        else:
            self._import_body_state(state_in)
        self._import_particle_state(state_in)
        if self._reduced_articulation is not None:
            self._reduced_articulation.import_step(
                state_in,
                control,
                state_is_continuation=continue_reduced_state,
                state_kinematics_valid=state_kinematics_valid,
            )

        # FD readout snapshots the imported (state_in-aligned) pose so the
        # post-step delta covers the full outer dt.
        if self._velocity_readout == "finite_difference":
            self._snapshot_pre_step_pose()

        # body_qdd readout: snapshot pre-step velocity so FD covers the outer dt.
        # Gated on state_out.body_qdd allocation (stable across graph capture
        # since the user requests the attribute on the Model before allocating
        # State).
        want_body_qdd = state_out.body_qdd is not None
        if want_body_qdd:
            self._snapshot_pre_step_velocity()

        if self._velocity_readout == "substep_average":
            self._substep_vel_accum.zero_()
            self._substep_omega_accum.zero_()
            world_vel_accum = self._substep_vel_accum
            world_omega_accum = self._substep_omega_accum
        else:
            world_vel_accum = None
            world_omega_accum = None

        # When sleeping is enabled, hand the broad-phase per-shape AABB
        # arrays to the world so it can compute body diagonals for the
        # spin-velocity term of the sleep score. ``narrow_phase`` is
        # populated by ``CollisionPipeline.collide(...)`` before stepping.
        # before ``solver.step(...)``.
        shape_aabb_lower = None
        shape_aabb_upper = None
        if self._sleeping_enabled:
            cp = getattr(self.model, "_collision_pipeline", None)
            np_ = getattr(cp, "narrow_phase", None) if cp is not None else None
            shape_aabb_lower = getattr(np_, "shape_aabb_lower", None)
            shape_aabb_upper = getattr(np_, "shape_aabb_upper", None)

        if self.solver_scheme == "tgs":
            self.world._temporal_contact_state.record_wrenches = int(
                contacts is not None and contacts.force is not None
            )
        self.world.step(
            dt=float(dt),
            contacts=contacts if self._shape_body is not None else None,
            shape_body=self._shape_body,
            shape_type=self.model.shape_type,
            vel_accum=world_vel_accum,
            omega_accum=world_omega_accum,
            shape_aabb_lower=shape_aabb_lower,
            shape_aabb_upper=shape_aabb_upper,
            # Opt-in (set by callers that substep against a fixed contact set,
            # e.g. the Anymal env's collide-once path): reuse the prior step's
            # graph colouring instead of re-colouring an unchanged graph.
            reuse_partition=bool(getattr(self, "reuse_partition", False)),
        )
        self._last_dt = float(dt) if self.solver_scheme == "tgs" else float(dt) / max(1, self.world.substeps)

        self._export_body_state(state_out, dt=float(dt))
        self._export_particle_state(state_out)
        self._particle_state_imported = None
        if want_body_qdd:
            self._export_body_qdd(state_out, dt=float(dt))
        # Reduced coordinates are authoritative; maximal mode reconstructs them by IK.
        if self._reduced_articulation is not None:
            self._reduced_articulation.export_step(state_out)
        elif state_out.joint_q is not None and state_out.joint_qd is not None and int(self.model.joint_count) > 0:
            newton.eval_ik(self.model, state_out, state_out.joint_q, state_out.joint_qd)

    @staticmethod
    def _read_model_gravity_np(model) -> np.ndarray:
        """Host-side ``model.gravity`` array, defaulting to single-world
        ``[0, 0, -9.81]``. Shared by ``__init__`` (consumed as tuples for
        the PhoenXWorld ctor) and :meth:`notify_model_changed` (consumed
        as a ``wp.array``)."""
        if model.gravity is not None:
            return model.gravity.numpy()
        return np.asarray([[0.0, 0.0, -9.81]], dtype=np.float32)

    def _launch_init_phoenx_bodies(self, model) -> None:
        """Refresh the PhoenX :class:`BodyContainer` from a Newton
        :class:`Model`. Same kernel + inputs/outputs at ``__init__`` and
        :meth:`notify_model_changed`'s body-property refresh path."""
        wp.launch(
            _init_phoenx_body_container_kernel,
            dim=int(model.body_count) + 1,
            inputs=[
                model.body_inv_mass,
                self._phoenx_body_inv_inertia,
                model.body_com,
                model.body_flags,
                model.body_world,
                wp.int32(int(BodyFlags.KINEMATIC)),
            ],
            outputs=[
                self.bodies.inverse_mass,
                self.bodies.inverse_inertia,
                self.bodies.inverse_inertia_world,
                self.bodies.body_com,
                self.bodies.affected_by_gravity,
                self.bodies.motion_type,
                self.bodies.world_id,
                self.bodies.linear_damping,
                self.bodies.angular_damping,
            ],
            device=self.device,
        )

    def notify_model_changed(self, flags: int) -> None:
        """Refresh state on Model edits. Joint-property changes rebuild the joint constraint
        init arrays from scratch; gravity is reread from ``model.gravity``."""
        if self.solver_scheme == "tgs":
            self._validate_temporal_model_properties()
        joint_props_changed = bool(flags & (int(ModelFlags.JOINT_PROPERTIES) | int(ModelFlags.JOINT_DOF_PROPERTIES)))
        if joint_props_changed:
            self._joint_constraints = build_joint_init_arrays(
                self.model,
                device=self.device,
                reduced_articulations=self._uses_reduced_joint_ownership,
            )
            self.world._combine_direct_prepare_projection = _can_combine_direct_prepare_projection(
                self._joint_constraints.has_velocity_limits,
                self.world.contact_friction_model,
                self._articulation_mode,
            )
            if self._joint_constraints.num_joint_columns > 0:
                self.world.initialize_joint_constraints(**self._joint_constraints.to_initialize_kwargs())
            if self._direct_equality_system is not None:
                previous_direct_solver = getattr(self._direct_equality_system, "solver", None)
                self._direct_equality_system.refresh_joint_properties()
                self._direct_equality_system.bind_constraint_indices(self._joint_constraints.joint_idx_to_cid)
                friction_dof_owned = self._direct_equality_system.direct_friction_dof_mask
            else:
                friction_dof_owned = None
            d6_data, d6_inequality_count = build_d6_inequality_data(
                self.model,
                self._joint_constraints.joint_idx_to_cid.numpy(),
                self._joint_friction_model,
                friction_dof_owned=friction_dof_owned,
            )
            self.world.constraints.d6 = d6_data
            if self._direct_equality_system is not None:
                self._direct_equality_system.d6_inequality_count = d6_inequality_count
                if previous_direct_solver is not getattr(self._direct_equality_system, "solver", None):
                    self._rebuild_direct_contact_response()
                self._refresh_direct_joint_ownership()
        cable_rest_changed = bool(flags & int(ModelFlags.JOINT_PROPERTIES | ModelFlags.BODY_PROPERTIES))
        if cable_rest_changed and self._direct_equality_system is not None:
            self._direct_equality_system.refresh_cable_rest_state()
        if flags & int(ModelFlags.MODEL_PROPERTIES):
            self.world.gravity = wp.array(self._read_model_gravity_np(self.model), dtype=wp.vec3f, device=self.device)
        # Single body refresh kernel covers both BODY_PROPERTIES and
        # BODY_INERTIAL_PROPERTIES. Maximal-coordinate armature lives in the
        # rebuilt joint rows above and does not mutate body inertia.
        body_refresh_mask = int(ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.BODY_PROPERTIES)
        body_properties_changed = bool(flags & body_refresh_mask)
        uses_maximal_mass = self._articulation_mode == "maximal" or self._uses_maximal_tree_projector
        maximal_joint_properties_changed = joint_props_changed and uses_maximal_mass
        reduced_joint_properties_changed = joint_props_changed and self._reduced_articulation is not None
        uses_legacy_body_armature = self._uses_maximal_tree_projector
        if uses_maximal_mass and (body_properties_changed or joint_props_changed):
            self._phoenx_body_inv_inertia = (
                _build_maximal_motor_body_inv_inertia(self.model)
                if uses_legacy_body_armature
                else self.model.body_inv_inertia
            )
        if body_properties_changed or maximal_joint_properties_changed or reduced_joint_properties_changed:
            if self.model.body_count > 0:
                self._launch_init_phoenx_bodies(self.model)
                if self._reduced_articulation is not None:
                    self._reduced_articulation.system.refresh_inertial_properties()
                    self._reduced_articulation.invalidate_kinematics()
        if joint_props_changed and self._uses_maximal_tree_projector:
            assert self._maximal_tree_projector_cls is not None
            if not self._maximal_tree_projector_cls.supports_model(self.model):
                raise RuntimeError(
                    "joint topology no longer supports the active maximal tree projector; rebuild SolverPhoenX"
                )
            self._maximal_tree_projector = self._maximal_tree_projector_cls(
                self.model,
                self._constraints,
                self.bodies,
                self._joint_constraints.joint_idx_to_cid,
            )
            self.world._maximal_tree_projector = self._maximal_tree_projector
        if flags & int(ModelFlags.SHAPE_PROPERTIES):
            if self.model.shape_material_mu is not None and self.model.shape_count > 0:
                self._install_shape_materials()

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Write per-contact wrenches back to
        :attr:`Contacts.force` if the user opted in via
        :meth:`Model.request_contact_attributes('force')`.

        Temporal forces are outer-step average world-frame wrenches on body0,
        about its current COM. Each patch's anchor friction is assigned to its
        first contact row; its moment includes the true anchor application
        points throughout the step. Thus body-pair wrench sums are preserved,
        while the per-point friction distribution is a reporting convention.
        Request forces before stepping to enable this optional accounting.

        The legacy soft scheme reports point forces with zero torque. Both
        schemes restore the original collision-pipeline contact ordering.
        """
        if contacts.force is None:
            raise ValueError(
                "contacts.force is not allocated. Call model.request_contact_attributes('force') "
                "before creating the Contacts object."
            )
        if self._last_dt <= 0.0 or self._shape_body is None or contacts.rigid_contact_max == 0:
            contacts.force.zero_()
            return

        if self.solver_scheme == "tgs":
            temporal = self.world._temporal_contact_state
            if not temporal.record_wrenches:
                raise ValueError("Request contact force before stepping so temporal impulse wrenches are recorded")
            contacts.force.zero_()
            wp.launch(
                export_contact_wrenches,
                dim=int(contacts.rigid_contact_max),
                inputs=[
                    contacts.rigid_contact_count,
                    self.world._cid_of_contact_cur,
                    temporal,
                    self.bodies,
                    self._shape_body,
                    contacts.rigid_contact_shape0,
                    self.world._ingest_scratch.sorted_shape0,
                    self.world._ingest_scratch.sort_perm,
                    1.0 / self._last_dt,
                ],
                outputs=[contacts.force],
                device=self.device,
            )
            return

        cc = self.world._contact_container
        # Body-pair grouping keys cc by sorted_k; sort_perm maps to newton_k.
        scratch = self.world._ingest_scratch
        if scratch is not None and scratch.sort_perm is not None:
            sort_perm = scratch.sort_perm
            sorted_shape0 = scratch.sorted_shape0
            has_perm = wp.int32(1)
        else:
            sort_perm = self._sort_perm_placeholder
            sorted_shape0 = contacts.rigid_contact_shape0
            has_perm = wp.int32(0)
        contacts.force.zero_()
        wp.launch(
            _contact_impulse_to_force_wrapper_kernel,
            dim=int(contacts.rigid_contact_max),
            inputs=[
                contacts.rigid_contact_count,
                self.world._cid_of_contact_cur,
                cc,
                wp.float32(1.0 / self._last_dt),
                sort_perm,
                has_perm,
                contacts.rigid_contact_shape0,
                sorted_shape0,
            ],
            outputs=[contacts.force],
            device=self.device,
        )

    def step_report(self) -> PhoenXWorld.StepReport:
        """Diagnostic snapshot. Forwards to :meth:`PhoenXWorld.step_report`.
        Triggers D2H copies — not graph-capture safe."""
        return self.world.step_report()
