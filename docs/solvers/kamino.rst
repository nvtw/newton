.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Kamino
======

:class:`~newton.solvers.SolverKamino` simulates constrained rigid multi-body
systems in maximal coordinates. It is designed for mechanical assemblies with
kinematic loops, under- or overactuation, joint limits, hard frictional
contacts, and restitutive impacts.

Unlike the other maximal-coordinate solvers, Kamino focuses on constrained
rigid mechanical assemblies rather than particle or deformable simulation.
Kamino is currently in BETA 1, and Newton users are discouraged from depending
on it. Evaluate it only when kinematic loops and hard contact constraints are
primary requirements and an experimental solver is acceptable.

.. experimental::

   :class:`~newton.solvers.SolverKamino` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

See the :class:`~newton.solvers.SolverKamino` API reference for construction
and configuration details. Runnable workflows are available in the
`Kamino examples <https://github.com/newton-physics/newton/tree/main/newton/examples/kamino>`_.

Choosing a dynamics solver
--------------------------

Kamino provides two forward-dynamics backends:

* ``"padmm"`` is the default. It uses proximal ADMM, dense Jacobians and
  dynamics by default, and the Euler integrator.
* ``"dvi"`` is an opt-in projected solver. It defaults to matrix-free sparse
  dynamics, sparse Jacobians, and the Moreau integrator. DVI does not support
  dual preconditioning.

DVI has been tuned primarily for contact-heavy rigid mechanisms. PADMM remains
the more broadly validated choice. Select the backend when constructing the
configuration so its dependent defaults are initialized consistently:

.. code-block:: python

   config = newton.solvers.SolverKamino.Config(dynamics_solver="dvi")
   config.dvi.tolerance = 1.0e-4
   solver = newton.solvers.SolverKamino(model, config=config)

Set both ``sparse_jacobian=False`` and ``sparse_dynamics=False`` to select the
dense DVI path. DVI requires ``config.dynamics.preconditioning=False``.

Warm-starting and diagnostics
-----------------------------

Both backends support cold starts, internally cached solutions, and
container-based warm-starting. DVI defaults to contact matching with a net-force
fallback to preserve individual resting-contact impulses.

Set ``SolverKamino.Config.collect_solver_info=True`` to retain terminal
per-world convergence diagnostics. DVI reports bilateral velocity, primal and
dual cone-feasibility, and complementarity residuals; these are DVI residuals
and should not be interpreted as PADMM primal and dual iteration residuals.
