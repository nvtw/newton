.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.phoenx
=====================

Runtime creation and removal of bodies, shapes, joints, and articulations in PhoenX.

Reserve rows with :meth:`~newton.ModelBuilder.reserve_dynamic_capacity`, then pass
``solver.dynamics.data`` to your kernels and call the ``add_*`` and ``remove_*`` functions
there. Changes take effect at the next :meth:`~newton.solvers.SolverPhoenX.step`, so creation
and removal can be captured in a CUDA graph together with the simulation. Every function
returns ``-1`` or ``False`` when a request cannot be met and counts it in
:attr:`Dynamics.error_count`.

.. experimental::

    Runtime creation may change without a deprecation period.

.. note::

   This page documents helper functions exposed through the ``newton.solvers.phoenx`` attribute.
   Because ``newton.solvers`` is a module rather than a package, use
   ``from newton.solvers import phoenx`` instead of ``import newton.solvers.phoenx``.

.. py:module:: newton.solvers.phoenx
.. currentmodule:: newton.solvers.phoenx

.. rubric:: Classes

.. autoclass:: BodyDesc

.. autoclass:: Dynamics

.. autoclass:: DynamicsData

.. autoclass:: JointDesc

.. autoclass:: ShapeDesc


.. rubric:: Functions

.. autofunction:: add_articulation

.. autofunction:: add_body

.. autofunction:: add_joint

.. autofunction:: add_shape

.. autofunction:: articulation_body

.. autofunction:: articulation_joint

.. autofunction:: body_desc

.. autofunction:: is_articulation_active

.. autofunction:: is_body_active

.. autofunction:: joint_desc

.. autofunction:: joint_wrench

.. autofunction:: remove_articulation

.. autofunction:: remove_body

.. autofunction:: remove_joint

.. autofunction:: remove_shape

.. autofunction:: shape_desc

