# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Compatibility entrypoint; forwards to example_optix_pathtracing_build_steps."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_forwarded_module():
    """Run the canonical module entrypoint in a clean subprocess."""
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    cmd = [sys.executable, "-m", "newton._src.viewer.optix.example_optix_pathtracing_build_steps", *sys.argv[1:]]
    completed = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(_run_forwarded_module())
