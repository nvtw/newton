# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hand-triggered PhoenX vs. MuJoCo-Warp benchmark harness.

Sits next to the PhoenX solver (not next to Newton's shared tests) so
iteration doesn't risk flakiness in the main test gate. Single-GPU
local runs only. Produces ``results/points.jsonl`` compatible with a
trimmed-down subset of Dylan Turpin's nightly JSONL schema plus a
minimal Chart.js dashboard in :mod:`dashboard`. Nothing in this
package is imported by the solver itself or exercised on CI.
"""
