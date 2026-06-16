# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Robot DR Legs (PhoenX DVI)
#
# Single-scene entry point for running the Disney Research DR Legs asset with
# SolverPhoenX's full-coordinate DVI articulation solve enabled. This reuses
# the validated PhoenX Dr Legs setup and only changes the defaults: one visible
# world, and DVI owning the articulation tree joints by default.
#
# Command: python -m newton.examples robot_dr_legs_phoenx_dvi
#
###########################################################################

from __future__ import annotations

import newton.examples
from newton.examples.robot.example_robot_dr_legs_phoenx import Example as _BaseExample


class Example(_BaseExample):
    @staticmethod
    def create_parser():
        parser = _BaseExample.create_parser()
        parser.set_defaults(
            articulation_dvi=True,
            articulation_dvi_solver="device_block_sparse",
            world_count=1,
            visible_world_count=1,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    viewer.show_visual = False
    viewer.show_inertia_boxes = True

    example = Example(viewer, args)

    newton.examples.run(example, args)
