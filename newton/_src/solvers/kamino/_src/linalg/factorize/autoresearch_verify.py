# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-number verification wrapper for /autoresearch on the blocked LLT kernels.

Runs ``benchmark_llt_blocked.py`` over a fixed size sweep, weights factor + solve
(solve is called many times per factor in production, so it dominates the cost),
and prints one floating-point number to stdout. On any failure (benchmark crash,
unparseable output, or numerical-accuracy regression) it exits non-zero so
autoresearch treats the iteration as a crash.

Usage:

    uv run python -m newton._src.solvers.kamino._src.linalg.factorize.autoresearch_verify

Exit code 0 + single number on stdout → successful iteration.
Non-zero exit → crash (autoresearch will revert the change).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

###
# Config
###

# Sizes to benchmark. Cover small/medium/large and unaligned cases so the
# optimizer can't game padding-aligned sizes alone.
SIZES = (32, 70, 128, 192, 257, 320, 401, 512, 768, 1000)

# How many solves run per factor in production. The solve is called many times
# per factor by the PGS/PADMM solver; weight it heavily so the metric tracks
# the real cost.
SOLVE_WEIGHT = 10.0

# Accuracy tripwire — chol_err already scales roughly linearly with n at fp32.
# If a change pushes error above this envelope, treat the iteration as a crash
# (the tests are the primary correctness gate, but this catches silent drift
# before tests run).
MAX_CHOL_ERR = 5e-2
MAX_RES_NORM = 1e-3

# Best-of-N to reduce GPU clock-noise.
NUM_ITERS = 50


###
# Parsing
###

# Benchmark data row:
#      n   factor [us]   solve [us]   solve_ip [us]   chol_err   res_norm
#      32          7.2           8.9           11.6   4.68e-05   1.53e-06
_ROW = re.compile(
    r"^\s*(\d+)\s+"
    r"([\d.]+)\s+"  # factor_us
    r"([\d.]+)\s+"  # solve_us
    r"([\d.]+)\s+"  # solve_ip_us
    r"([\d.eE+-]+)\s+"  # chol_err
    r"([\d.eE+-]+)\s*$"  # res_norm
)


def _run_benchmark() -> str:
    repo_root = Path(__file__).resolve().parents[6]
    bench = Path(__file__).with_name("benchmark_llt_blocked.py")
    cmd = [
        "uv",
        "run",
        "python",
        str(bench),
        "--sizes",
        *(str(s) for s in SIZES),
        "--num-iters",
        str(NUM_ITERS),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=repo_root)
    if proc.returncode != 0:
        sys.stderr.write(f"benchmark failed (rc={proc.returncode})\n{proc.stderr}\n")
        sys.exit(2)
    return proc.stdout


def _score(stdout: str) -> float:
    sizes_seen: set[int] = set()
    total = 0.0
    for line in stdout.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        n = int(m.group(1))
        factor_us = float(m.group(2))
        solve_us = float(m.group(3))
        chol_err = float(m.group(5))
        res_norm = float(m.group(6))

        if chol_err > MAX_CHOL_ERR or res_norm > MAX_RES_NORM:
            sys.stderr.write(
                f"accuracy regression at n={n}: chol_err={chol_err:.2e} res_norm={res_norm:.2e}\n"
            )
            sys.exit(3)

        sizes_seen.add(n)
        total += factor_us + SOLVE_WEIGHT * solve_us

    missing = set(SIZES) - sizes_seen
    if missing:
        sys.stderr.write(f"missing sizes in benchmark output: {sorted(missing)}\n")
        sys.exit(4)

    return total


def main() -> None:
    stdout = _run_benchmark()
    score = _score(stdout)
    print(f"{score:.2f}")


if __name__ == "__main__":
    main()
