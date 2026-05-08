# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Wrapper to expose ``newton._src.solvers.phoenx.tests.test_armature``
test classes to the top-level ``newton.tests`` discovery path
(``newton/tests/``). The actual test logic lives in the phoenx tests
package alongside the rest of the solver-internal tests; this stub
re-exports the public test classes so the parallel test runner picks
them up.
"""

from newton._src.solvers.phoenx.tests.test_armature import (
    TestArmatureMatchesMuJoCo,
    TestArmatureNoOpAtZero,
    TestExactModeChainComposition,
    TestExactModeIsolatedJoints,
    TestExactModeMixedArmatureLimitation,
    TestOffModeIgnoresArmature,
    TestPendulumPeriod,
    TestPrismaticTwoBodyPeriod,
    TestSkinnyChainStability,
    TestSymmetricTwoBodyPeriod,
    TestThreeBodyChain,
)

__all__ = [
    "TestArmatureMatchesMuJoCo",
    "TestArmatureNoOpAtZero",
    "TestExactModeChainComposition",
    "TestExactModeIsolatedJoints",
    "TestExactModeMixedArmatureLimitation",
    "TestOffModeIgnoresArmature",
    "TestPendulumPeriod",
    "TestPrismaticTwoBodyPeriod",
    "TestSkinnyChainStability",
    "TestSymmetricTwoBodyPeriod",
    "TestThreeBodyChain",
]
