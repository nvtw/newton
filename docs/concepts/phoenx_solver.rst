.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _PhoenX Solver:

PhoenX Solver
=============

:class:`~newton.solvers.SolverPhoenX` is a maximal-coordinate, GPU-resident
rigid-body solver built on a **substepped Projected Gauss-Seidel (PGS) +
TGS-soft relax** loop with a deterministic graph-coloured constraint
schedule and a fast-tail multi-world dispatch designed for fleets of small,
independent simulations (locomotion, manipulation, RL training).

The solver is a Warp port of the C# PhoenX engine; the contact
backbone closely follows Box2D v3 (``b2SolveOverflowContacts`` /
``b2SolveContactsTask``).

.. contents::
    :local:
    :depth: 2


When to use PhoenX
------------------

PhoenX is the right choice when you need:

- **Many small worlds in parallel** — RL fleets, batched manipulation,
  domain randomisation. The fast-tail multi-world kernels run one warp
  per world; throughput scales near-linearly into the thousands.
- **Deterministic replay** — every step is byte-identical across runs
  on the same device and Warp version. No floating-point atomics, no
  scheduling-dependent constraint ordering.
- **CUDA-graph capture** — :meth:`step` performs no host syncs and
  allocates no Python objects. Capture once, replay millions of times.
- **Stable resting stacks of rigid bodies** — speculative-contact
  handling and TGS-soft relax keep block towers and dense rabbit
  piles from inflating.

It is **not** the right choice for:

- Generalised-coordinate articulation work that needs the kinematic
  tree (use :class:`~newton.solvers.SolverFeatherstone` or
  :class:`~newton.solvers.SolverMuJoCo`).
- Equality / mimic constraints, ``D6``, or ``DISTANCE`` joints
  through Newton's standard :class:`~newton.ModelBuilder` flow (use
  :class:`~newton.solvers.SolverMuJoCo` or
  :class:`~newton.solvers.SolverXPBD`). ``CABLE`` joints *are*
  supported, with the caveat that PhoenX has no axial-length
  compliance — see :ref:`phoenx-cable-joints` below.
- Cloth, particles, soft bodies, MPM (use the dedicated solvers).
- Differentiable simulation (PhoenX kernels are not authored under
  ``wp.Tape``).


Quickstart
----------

.. code-block:: python

    import newton

    builder = newton.ModelBuilder()
    # ... populate the builder ...
    model = builder.finalize()

    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=2,
        solver_iterations=8,
        velocity_iterations=1,
    )

    state_in, state_out = model.state(), model.state()
    contacts = model.contacts()
    control = model.control()

    for _ in range(num_steps):
        contacts = model.collide(state_in)
        solver.step(state_in, state_out, control, contacts, dt=1.0 / 60.0)
        state_in, state_out = state_out, state_in

The constructor self-sizes contacts: if the model does not already
have a :class:`~newton.CollisionPipeline` with contact matching
enabled, ``SolverPhoenX`` attaches one. PhoenX **requires** matched
contacts for warm-starting and stacking stability.


Algorithmic overview
--------------------

Each :meth:`SolverPhoenX.step` call executes the following pipeline:

.. mermaid::
   :config: {"theme": "forest", "themeVariables": {"lineColor": "#76b900"}}

   flowchart TB
     A["Apply Control<br/>→ joint drives + body forces"] --> B
     B["Import State<br/>→ PhoenX BodyContainer"] --> C
     C["Ingest Contacts<br/>+ Warm-start λ"] --> D
     D["Rebuild element view<br/>+ Jones-Plassmann colouring"] --> E
     subgraph Substep ["For each substep (× substeps)"]
       direction TB
       E["Apply forces + gravity"] --> F
       F["Main PGS sweep<br/>(solver_iterations, bias=ON)"] --> G
       G["Semi-implicit position integration"] --> H
       H["TGS-soft relax<br/>(velocity_iterations, bias=OFF)"] --> I
       I["Kinematic interp<br/>+ global damping"]
     end
     I --> J["Update inertia<br/>+ clear forces"]
     J --> K["Export State<br/>(eval_ik for joint_q/qd)"]


Substepping
~~~~~~~~~~~

PhoenX runs ``substeps`` *internal* substeps per :meth:`step` call.
Each substep advances the world by ``dt / substeps`` seconds and
re-runs the full main solve + relax loop with the same colouring
(constraint topology is constant within a step). Contact ingest and
warm-start happen once per outer step.

Outer substepping (calling :meth:`step` ``N`` times with ``dt/N``)
re-ingests contacts every substep, which is more expensive but lets
the broad/narrow phase keep up with fast-moving geometry. Pass
``substeps=1`` if you prefer to substep externally.

Typical settings:

==========================  =============== ====================
Scenario                    ``substeps``    ``solver_iterations``
==========================  =============== ====================
Locomotion (humanoid)       1               8
Stacking (towers, piles)    2-4             8-16
Manipulation (slow)         1               4
==========================  =============== ====================


Projected Gauss-Seidel main solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main solve runs ``solver_iterations`` PGS sweeps per substep with
**positional bias enabled**. Each sweep walks the entire constraint
graph in graph-coloured order: for every active constraint ``c``,
solve

.. math::

    \Delta\lambda_c = -\frac{J_c \, v + b_c}{m_c^{\text{eff}}}

then project (clamp) the accumulated impulse into the constraint's
admissible set:

- **Contact normal** (1 row): ``λ_n ≥ 0`` (no pulling).
- **Contact friction** (2 tangent rows): pyramidal Coulomb,
  ``|λ_t| ≤ μ · λ_n`` per tangent direction, evaluated against the
  current normal accumulator.
- **Joint rows**: hard equality (``λ`` unbounded) for ball-socket
  point lock and weld; soft PD limit/drive rows are clamped to the
  drive's ``effort_limit`` per substep.

Impulses accumulate across iterations within the substep (warm-start
between iterations). The Baumgarte positional bias
``b = c_error / dt`` drives positional error toward zero across the
sweeps; without it the solve only enforces ``Jv = 0`` and penetration
would persist.

PhoenX uses **rigid Box2D v3 PGS coefficients** for separated /
speculative contacts (``mass_coeff = 1``, ``impulse_coeff = 0``) and
**soft Baumgarte coefficients** for penetrating contacts. The
speculative branch caps closing displacement at ``gap / dt`` so
contacts brake before impact instead of relying on penetration
recovery — the same construction Box2D v3 uses.


TGS-soft relax (velocity iterations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After position integration, ``velocity_iterations`` *additional*
PGS sweeps run with **bias disabled**. These sweeps remove the drift
velocity that the positional bias injected during the main solve —
without them, tall stacks visibly creep upward (the Baumgarte spring
keeps depositing kinetic energy at every substep).

``velocity_iterations = 0`` recovers the raw PhoenX behaviour.
``velocity_iterations = 1`` is the shipped default and good for most
scenes; set ``2`` for very tall stacks or hard-impact scenarios.


Friction model
~~~~~~~~~~~~~~

Pyramidal (two-tangent) Coulomb friction. Each contact has three
constraint rows: one normal (with non-negativity projection) and two
orthogonal tangent rows (each clamped against the current normal
accumulator). This is the same "simple friction" Box2D v3 ships and is
slightly looser than the exact circular cone; in practice it is
indistinguishable for contact manifolds whose tangent basis aligns
with motion (the common case).

Friction comes from a per-shape PhysX-style material table:

.. code-block:: python

    # mu lives on Model.shape_material_mu and is installed automatically
    # by the SolverPhoenX constructor.
    mu_static = mu_dynamic = model.shape_material_mu

Per-pair resolution combines the two shapes' coefficients with the
**stricter** combine mode (``max(mode_a, mode_b)``):
``AVERAGE < MIN < MULTIPLY < MAX``.

Per-contact overrides are supported via
:attr:`Contacts.rigid_contact_friction` (set via
:meth:`Model.request_contact_attributes`) when the soft-contact
path is enabled.

.. note::

    Restitution is **not** consumed by the contact constraint -- all
    contacts are perfectly inelastic. The materials API exposes a
    restitution coefficient for forward compatibility, but the
    constraint kernels currently ignore it. This matches
    :class:`~newton.solvers.SolverMuJoCo`, which models contacts as
    soft constraints via ``solref`` / ``solimp`` instead of a
    classical coefficient of restitution and likewise does not
    consume :attr:`Model.shape_material_restitution`;
    :class:`~newton.solvers.SolverXPBD` is the only Newton rigid-body
    solver that does (opt-in via ``enable_restitution=True``).
    To get bounce in PhoenX, drive bodies through a joint or apply
    explicit forces.


Warm starting
~~~~~~~~~~~~~

PhoenX maintains per-contact ``(λ_n, λ_t1, λ_t2)`` impulses across
frames. On ingest, each contact that the
:class:`~newton.CollisionPipeline` matched to a previous-frame contact
(via :attr:`Contacts.rigid_contact_match_index`) inherits all three
impulses **at full strength**, plus the matched contact's tangent
basis when it is still geometrically representative. New / unmatched
contacts initialise to zero impulse and a tangent direction derived
from the relative velocity. This dramatically reduces the iteration
count needed to re-converge a settled stack.

Contact matching is **mandatory**: the constructor raises if the
:class:`~newton.CollisionPipeline` was built with
``contact_matching=DISABLED``. The recommended mode for production is
``"sticky"`` (replays previous-frame anchors and normals over matched
contacts to suppress geometry jitter); ``"latest"`` is the default
and exists primarily to diagnose whether anchor pinning is causing a
regression.


Determinism
~~~~~~~~~~~

PhoenX produces **bit-identical** output across re-runs on the same
device + Warp version, by construction:

- **Constraint ordering** comes from a Jones-Plassmann maximal
  independent set over the constraint graph. Element priorities are
  drawn once at solver construction from a seeded RNG and never
  change; intra-colour slot assignment uses
  ``wp.tile_scan_exclusive`` (deterministic prefix scan).
- **Per-world bucketing** uses ``wp.utils.radix_sort_pairs`` (stable
  sort by ``world_id``); the sorted output depends only on the
  inputs.
- **No floating-point atomics** in the hot path. Constraint impulses
  are applied per-cid (not per-body atomic add); body-velocity
  updates happen at colour boundaries with ``__syncthreads`` ordering.

Nothing in :meth:`step` calls ``.numpy()`` or otherwise issues a
device-to-host copy, so determinism survives CUDA graph capture.


Graph colouring & multi-world dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every step rebuilds an *element view* (``constraint id → (body₀,
body₁, world)``) and runs Jones-Plassmann colouring to partition the
constraint graph into independent sets. Within a colour, every
constraint touches a disjoint pair of bodies, so PGS sweeps can run
fully in parallel without write conflicts.

Two dispatch layouts are available via the ``step_layout`` constructor
argument:

``"multi_world"`` (default)
    Per-world Jones-Plassmann run on independent sub-graphs; one
    block per world solves its local colouring, then a single
    *fast-tail* PGS kernel iterates every world's colours in
    lockstep, one warp per world. This is the right layout when you
    have many small worlds and want to fill the GPU through
    concurrency rather than per-world parallelism.

    The adaptive ``threads_per_world`` knob picks ``32`` (one warp
    per world), ``16`` (two worlds per warp), or ``8`` (four) per
    step from the colour-size histogram. ``"auto"`` (default) decides
    on the GPU at every step; this is captured into the CUDA graph.

``"single_world"``
    Global Jones-Plassmann across all bodies; per-colour persistent-
    grid PGS launches via ``wp.capture_while``. A *fused-tail*
    optimisation drains small final colours in a single block with
    ``__syncthreads`` between colours, saving
    ``O(remaining_colours)`` kernel launches per sweep. Use this
    layout when you have one or a few very large worlds.

The two layouts produce identical physics; they trade off
launch-overhead vs. per-colour parallelism.


Joint support
~~~~~~~~~~~~~

PhoenX supports five Newton :class:`~newton.JointType` values via a
single unified *actuated double-ball-socket* (ADBS) constraint
column:

============== =============================================== =================
Joint type     Behaviour                                       Drive / limit
============== =============================================== =================
``REVOLUTE``   1-DoF hinge (5 constrained rows)                ✅ both
``PRISMATIC``  1-DoF slider (5 constrained rows)               ✅ both
``BALL``       3-DoF ball-socket (3 point-lock rows)           ❌ neither
``FIXED``      6-DoF weld (3 point + 3 angular)                ❌ neither
``CABLE``      Rigid ball-socket + 2 bend + 1 twist soft rows  PD on bend/twist
``FREE``       Free-floating; no constraint column             —
============== =============================================== =================

``DISTANCE`` and ``D6`` joints raise at construction.

Drives (``REVOLUTE`` / ``PRISMATIC``) are PD with per-DoF
``joint_target_ke`` (stiffness), ``joint_target_kd`` (damping), and
``joint_effort_limit`` (saturation). Drive mode follows
:attr:`Model.joint_target_mode`: ``POSITION``, ``VELOCITY``, or
``POSITION_VELOCITY``.

Joint limits for ``REVOLUTE`` / ``PRISMATIC`` use
``joint_limit_lower`` / ``joint_limit_upper`` with the same PD-style
``ke``/``kd`` as drives.

Feed-forward joint efforts (``Control.joint_f``) are converted to body
wrenches via the stock :func:`apply_joint_forces` kernel and folded
into the per-body force accumulator before the PGS solve.

**Joint armature** (:attr:`Model.joint_armature`) is supported on
``REVOLUTE`` and ``PRISMATIC`` joints. Because PhoenX is
maximal-coordinate, armature is implemented by **baking the rotor
inertia into both attached bodies along the joint axis** at solver
construction (and on
:meth:`~SolverPhoenX.notify_model_changed` for ``BODY_INERTIAL_PROPERTIES``
or ``JOINT_PROPERTIES``). This is how the constraint kernel sees
``M_eff = M_chain + armature`` along the joint axis without any
hot-path overhead. Armature on ``BALL`` / ``FIXED`` / ``FREE``
joints is ignored (no axial DoF). Critical for stability of
high-stiffness PD drives on chains where an intermediate link has
near-zero inertia about the joint axis (e.g. humanoid waist links
of <0.1 kg with ``target_ke = 300`` N·m/rad).


.. _phoenx-cable-joints:

Cable joints (bend + twist stiffness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actuated double-ball-socket constraint also carries a **cable
mode** — a rigid ball-socket at ``anchor1`` plus three soft angular
rows formulated against the joint's finalize-time rest pose:

- Two **bend** rows, with stiffness ``k_bend`` [N·m/rad] and damping
  ``d_bend`` [N·m·s/rad], penalise rotation about the two axes
  perpendicular to the cable's reference axis ``anchor1 → anchor2``.
- One **twist** row, with stiffness ``k_twist`` and damping
  ``d_twist`` (same units), penalises rotation about the reference
  axis itself.

The angular rows are integrated as PD constraints inside the same
PGS sweep that handles contacts and rigid joints, so chained cable
segments converge under the standard ``solver_iterations``.

**Newton API mapping** (``ModelBuilder.add_joint_cable`` /
:attr:`~newton.JointType.CABLE`): the parent attachment in world
becomes ``anchor1``, the child attachment becomes ``anchor2``, and
the angular DoF's ``target_ke`` / ``target_kd`` are written to both
the bend and twist slots (Newton's bend/twist is isotropic; PhoenX's
two bend axes plus the twist axis all get the same gain).

There is **one semantic gap** to be aware of: PhoenX's cable mode
has a rigid ball-socket at the parent attachment, whereas Newton's
cable joint also models the linear stretch as a 1-DoF spring
(``stretch_stiffness``). PhoenX has no axial-length compliance, so
the stretch is treated as rigid (Newton's default
``stretch_stiffness = 1e9`` makes this a tight approximation in
practice). If you need a genuinely stretchable cable, use
:class:`~newton.solvers.SolverVBD`, which has a soft 1-DoF stretch
DoF for cables.

For low-level builder access — chained cable patterns, per-segment
twist gains different from bend, etc. — call PhoenX's standalone
``WorldBuilder.add_joint(mode=JointMode.CABLE, ...)`` directly
instead of going through Newton's ``add_joint_cable``.


Contacts
~~~~~~~~

Contact ingest packs Newton's flat ``Contacts`` buffer into
**contact columns**, one per ``(shape_a, shape_b)`` pair. A column
covers an arbitrary number of contacts between the same pair (dense
mesh manifolds are fine); each contact is iterated serially within
its column (Gauss-Seidel within the pair).

Contact features:

- **Speculative contacts** (``gap > 0``): rigid PGS coefficients with
  ``bias = gap * inv_dt``. The row only fires when predicted closing
  motion would exceed the gap — closing bodies brake before impact.
- **Penetrating contacts** (``gap < 0``): soft Baumgarte with the
  contact stiffness/damping from the material; bias=0 in the relax
  pass.
- **Per-contact overrides**: pass
  ``Contacts.rigid_contact_stiffness`` / ``rigid_contact_damping`` /
  ``rigid_contact_friction`` (request via
  :meth:`Model.request_contact_attributes` and
  :meth:`CollisionPipeline.contacts(per_contact_shape_properties=True)`).
- **Pairwise contact filter**: register
  ``(body_a, body_b)`` pairs whose contacts the solver will silently
  drop on ingest.
- **Warm-started normal impulses** (see above).

To read back per-contact spatial forces (``[N, N·m]`` at the contact
point in world frame), call:

.. code-block:: python

    model.request_contact_attributes("force")
    contacts = model.contacts()  # now has contacts.force allocated
    # ... step ...
    solver.update_contacts(contacts)

The torque is always zero (a per-point force has no torque about its
own application point).


Kinematic bodies
~~~~~~~~~~~~~~~~

Bodies flagged :attr:`BodyFlags.KINEMATIC` follow scripted poses.
On import, ``state_in.body_q`` is routed to a kinematic-target slot;
the solver infers the per-step linear and angular velocity from the
target/current pose delta and **interpolates** the pose across
substeps with lerp/slerp. Contacts prepared at substep ``k`` therefore
see the pose at fractional time ``(k+1)/substeps`` rather than a
discrete jump at substep ``substeps`` — important for fast-moving
floors, conveyors, and tracked-object grippers.


Materials
~~~~~~~~~

Friction is per-shape. The constructor reads
``Model.shape_material_mu`` and stamps a one-material-per-shape PhysX-
style table. Shape index ``-1`` (Newton's "world" sentinel) maps to
PhoenX slot ``0`` (the static world anchor).

To rebuild the table after editing ``shape_material_mu``, call
:meth:`SolverPhoenX.notify_model_changed` with
``SolverNotifyFlags.SHAPE_PROPERTIES``.


Picking
~~~~~~~

Newton's interactive picking spring works without a PhoenX-specific
shim. The viewer's ``apply_forces(state)`` atomically adds the
spring's force/torque to ``state.body_f``; :meth:`step` imports
``state.body_f`` into PhoenX's per-body force accumulator before
integrating, so picks land uniformly across substeps.


Constructor reference
---------------------

.. code-block:: python

    SolverPhoenX(
        model,
        *,
        substeps=1,
        solver_iterations=8,
        velocity_iterations=1,
        default_friction=0.5,
        step_layout="multi_world",
        threads_per_world="auto",
    )

``model``
    A finalised :class:`~newton.Model`. Mass, inertia, COM, world id,
    and body flags are copied at construction; joint and contact
    arrays are read every step.

``substeps`` *(int, default 1)*
    Internal substeps per :meth:`step` call. Set ``1`` and substep
    externally if you prefer to re-ingest contacts every substep.

``solver_iterations`` *(int, default 8)*
    PGS sweeps per substep, **bias on**. Higher = better penetration
    recovery; cost is linear in this knob.

``velocity_iterations`` *(int, default 1)*
    TGS-soft relax sweeps per substep, **bias off**. Removes
    Baumgarte drift; ``0`` recovers raw PhoenX, ``2`` for very tall
    stacks.

``default_friction`` *(float, default 0.5)*
    Fallback μ when no per-shape material is registered.

``step_layout`` *(str, default ``"multi_world"``)*
    ``"multi_world"`` for many small worlds (one warp per world);
    ``"single_world"`` for one or a few very large worlds.

``threads_per_world`` *(int | str, default ``"auto"``)*
    Effective threads-per-world for the multi-world fast-tail
    kernels. ``"auto"`` picks per-step from the colour-size
    histogram; ``32`` / ``16`` / ``8`` pin a value. Below ``8 ×
    sm_count`` worlds the picker is short-circuited host-side to
    save ~10 µs/step.


Tuning constants
----------------

Module-level knobs in :mod:`newton._src.solvers.phoenx.solver_config`:

.. list-table::
    :header-rows: 1
    :widths: 30 15 55

    * - Constant
      - Default
      - Notes
    * - ``PHOENX_CONTACT_MATCHING``
      - ``"latest"``
      - ``"sticky"`` for stable stacking; ``"disabled"`` forbidden.
    * - ``NUM_INNER_WHILE_ITERATIONS``
      - ``8``
      - ``wp.capture_while`` body unroll; amortises edge-traversal cost.
    * - ``FUSE_TAIL_MAX_COLOR_SIZE``
      - ``256``
      - Hand-off threshold for the single-block fused tail kernel
        (single-world only).
    * - ``FUSE_TAIL_BLOCK_DIM``
      - ``256``
      - Block width of the fused tail kernel; must be
        ``≥ FUSE_TAIL_MAX_COLOR_SIZE``.


Notify flags
------------

:meth:`SolverPhoenX.notify_model_changed` re-syncs internal state
when you mutate :class:`~newton.Model` fields after construction:

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - Flag
      - Effect
    * - ``JOINT_PROPERTIES`` / ``JOINT_DOF_PROPERTIES``
      - Rebuild ADBS init arrays + re-stamp joint columns.
    * - ``MODEL_PROPERTIES``
      - Refresh per-world ``gravity``.
    * - ``BODY_PROPERTIES`` / ``BODY_INERTIAL_PROPERTIES``
      - Re-derive ``motion_type``, ``affected_by_gravity``, mass,
        inertia, COM.
    * - ``SHAPE_PROPERTIES``
      - Rebuild per-shape friction / restitution material table.


PhoenX vs MuJoCo Warp
---------------------

Both PhoenX and :class:`~newton.solvers.SolverMuJoCo` (which dispatches
to ``mujoco_warp``) target rigid-body simulation on the GPU; they
make different trade-offs and excel at different things.

**Pick PhoenX when**

- Your model fits the supported joint set (``REVOLUTE`` / ``PRISMATIC``
  / ``BALL`` / ``FIXED`` / ``FREE``) and you don't need equality,
  mimic, or ``D6`` constraints.
- You're running many small worlds in parallel (RL training,
  domain randomisation). PhoenX's fast-tail multi-world kernels scale
  well into the thousands of worlds.
- You need bit-identical replay (training reproducibility, regression
  testing, debugging non-determinism).
- You're throughput-bound on contact-heavy scenes (large stacks, dense
  mesh manifolds, big rabbit piles).

**Pick MuJoCo Warp when**

- You need the full joint vocabulary: ``D6``, ``DISTANCE``, equality
  constraints (``CONNECT`` / ``WELD`` / ``JOINT``), mimic constraints,
  friction-loss, velocity limits.
- You're running on generalised-coordinate articulations and want
  reduced-coordinate dynamics rather than a maximal-coordinate
  formulation.
- You want to mix Newton bodies with MuJoCo's hand-tuned contact
  dynamics (soft constraints with ``solref`` / ``solimp``, multi-CCD).
- You need integration schemes beyond semi-implicit Euler (RK4,
  implicit-velocity).

**Honest weaknesses**

PhoenX:

- No contact restitution today (the materials API exposes the
  coefficient but the constraint kernels ignore it).
  :class:`~newton.solvers.SolverMuJoCo` also doesn't consume
  ``shape_material_restitution`` -- MuJoCo models contacts as soft
  constraints via ``solref`` / ``solimp`` rather than a classical
  coefficient of restitution -- so this isn't a behavioural
  divergence between the two solvers; it's a Newton ecosystem-wide
  gap. :class:`~newton.solvers.SolverXPBD` (opt-in
  ``enable_restitution=True``) is currently the only rigid-body
  Newton solver that consumes the field directly.
- Smaller joint vocabulary on Newton's standard ``ModelBuilder`` path;
  no equality / mimic constraints, no ``D6``, no ``DISTANCE``.
  ``CABLE`` is supported but only with rigid stretch (see
  :ref:`phoenx-cable-joints`).
- Pyramidal Coulomb friction, not the exact circular cone (loose by
  a few percent in pathological tangent-misaligned cases).
- Tall stacks of contacting bodies may need extra ``substeps`` and
  ``solver_iterations`` to converge cleanly under PGS — pure
  Gauss-Seidel without mass-splitting biases toward the bottom of
  the stack.
- The Newton port is younger than MuJoCo's decade of physics tuning;
  expect rougher edges in less-travelled corners.

MuJoCo Warp:

- LCP / CG-style solver iterates more on contact-dominated scenes;
  PhoenX's coloured PGS is often faster wall-clock on dense stacks
  in like-for-like benchmarks, but the comparison is workload-
  dependent — measure before betting on it.
- Multi-world dispatch is built around one large kernel grid rather
  than per-world parallelism; very small worlds tend to be launch-
  overhead bound.
- Determinism is best-effort, not by-construction (floating-point
  atomics in some paths).
- More state is implicit (``mjModel`` / ``mjData``), which makes
  hot-path re-tuning (e.g., changing friction live) more
  ceremonious than swapping a few floats.

Neither solver is universally better. PhoenX prioritises **speed and
determinism on the modal Newton workload** (locomotion / manipulation
fleets); MuJoCo Warp prioritises **feature completeness and physics
fidelity** for the workloads MuJoCo has historically served.


Examples
--------

A growing subset of the standard Newton examples accepts
``--solver phoenx``:

.. code-block:: bash

    uv run -m newton.examples basic_pendulum --solver phoenx
    uv run -m newton.examples basic_pendulum --solver phoenx --benchmark

A larger catalogue of PhoenX-specific scenes — Box2D v3 ports
(stacking, friction, ragdoll), articulation rigs (humanoid pyramids,
motors), mesh contacts (nut/bolt, bunny-on-plane), and many-world
scaling stress tests (Kapla arena, Kapla tower) — lives in the
solver's source tree alongside the kernels and is primarily aimed at
contributors hardening the solver. A throughput regression suite is
maintained next to it for catching kernel-level performance and
memory regressions.


See also
--------

- :doc:`newton.solvers <../api/newton_solvers>` — solver feature
  matrix and joint-support tables across all Newton solvers, plus
  the generated :class:`~newton.solvers.SolverPhoenX` API reference.
- :class:`~newton.solvers.SolverMuJoCo` — generalised-coordinate
  rigid-body / articulation solver with the full MuJoCo feature set.
- :class:`~newton.solvers.SolverFeatherstone` — articulation solver
  in generalised coordinates.
- :class:`~newton.solvers.SolverXPBD` — maximal-coordinate solver
  with broader joint support but no contact warm-starting.
- :ref:`Collisions` — collision pipeline that produces the
  :class:`~newton.Contacts` buffer PhoenX consumes.
