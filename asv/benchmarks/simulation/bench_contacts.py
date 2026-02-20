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

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

import newton
import newton.examples
from newton.examples.contacts.example_nut_bolt_hydro import Example as ExampleHydroWorking
from newton.examples.contacts.example_nut_bolt_sdf import Example as ExampleSdf
from newton.viewer import ViewerNull

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
ISAACGYM_GEARS_FOLDER = "assets/factory/mesh/factory_gears"


class FastExampleContactSdfDefaults:
    repeat = 2
    number = 1

    def setup_cache(self):
        newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_GEARS_FOLDER)

    def setup(self):
        self.num_frames = 20
        self.example = ExampleSdf(
            viewer=ViewerNull(num_frames=self.num_frames),
            world_count=100,
            num_per_world=1,
            scene="nut_bolt",
            solver="mujoco",
            test_mode=False,
        )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactHydroWorkingDefaults:
    repeat = 2
    number = 1

    def setup_cache(self):
        newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_GEARS_FOLDER)

    def setup(self):
        self.num_frames = 20
        self.example = ExampleHydroWorking(
            viewer=ViewerNull(num_frames=self.num_frames),
            num_worlds=20,
            num_per_world=1,
            scene="nut_bolt",
            solver="mujoco",
            test_mode=False,
        )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleContactSdfDefaults": FastExampleContactSdfDefaults,
        "FastExampleContactHydroWorkingDefaults": FastExampleContactHydroWorkingDefaults,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
