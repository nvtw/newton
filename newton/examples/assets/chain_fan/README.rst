.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: Apache-2.0

Chain fan meshes
================

Project-authored watertight triangle meshes for ``phoenx_chain_fan``.
The example scales model units by 0.042 m and builds a shared mesh SDF
for each asset. No CAD package is needed to run the example.

``chain_link.obj`` is round wire of radius 0.06 swept around a stadium
centerline in the XY plane: bend radius 0.2, straight half-length 0.15,
long axis X. It has 768 triangles.

``chain_pendant.obj`` is a pentagonal plate in the XY plane, of thickness
0.16, with outline (0, -0.25), (0.45, -0.25), (0.75, 0), (0.45, 0.25),
(0, 0.25). Its hole has radius 0.1 and center (0.15, 0). The example
centers the mesh and rotates it to thread through either link parity.

These assets use the repository's Apache-2.0 license.
