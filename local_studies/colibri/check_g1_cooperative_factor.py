# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Try the existing cooperative factor kernel for one G1, without changing equations."""

import runpy

from newton._src.solvers.phoenx.articulations import reduced

reduced._WARP_FACTOR_MIN_ARTICULATIONS = 1
runpy.run_module("local_studies.colibri.check_g1_policy", run_name="__main__")
