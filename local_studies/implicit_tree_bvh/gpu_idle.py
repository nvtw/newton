# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Wait until this process is the only GPU user (other agents share the GPU)."""

import os
import subprocess
import time

stats = {"waited_s": 0.0}


def wait_gpu_idle(poll_s: float = 2.0, max_wait_s: float = 1800.0) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < max_wait_s:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        others = [p for p in out.split() if p.strip() and int(p) != os.getpid()]
        if not others:
            break
        time.sleep(poll_s)
    stats["waited_s"] += time.perf_counter() - start
