# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Neutral Hoberman sphere data assembled from separate generated modules."""

from newton._src.solvers.phoenx.examples.hoberman_sphere_body_data import BODIES
from newton._src.solvers.phoenx.examples.hoberman_sphere_joint_data import ARTICULATION_JOINT_COUNT, JOINTS
from newton._src.solvers.phoenx.examples.hoberman_sphere_tile_data import TILES

__all__ = ["ARTICULATION_JOINT_COUNT", "BODIES", "JOINTS", "TILES"]
