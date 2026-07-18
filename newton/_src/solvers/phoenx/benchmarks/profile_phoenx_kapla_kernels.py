# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Per-kernel GPU-time breakdown for the Kapla tower via nsys.

Captured CUDA graphs hide per-kernel timing from Warp's event API, so
we shell out to ``nsys profile --cuda-graph-trace=node`` and parse its
SQLite report. The decomposed kernel timings (graph nodes, decoded
back into individual launch events) drop into a ranked table.

Three modes:

* default — run with one config, dump top-25 kernels.
* ``--diff`` — run two configs (default: mass_splitting on vs off) and
  print a side-by-side delta table.
* ``--frames N`` — control sample size; default 100.

Requirements: ``nsys`` on PATH. The Kapla tower scene compiles a lot
of CUDA on first run (~60-90 s); the second run uses the kernel cache.

Examples::

    # Single config (defaults: 100 frames, mass splitting on).
    python -m newton._src.solvers.phoenx.benchmarks.profile_phoenx_kapla_kernels

    # Diff mass-splitting off vs on.
    python -m newton._src.solvers.phoenx.benchmarks.profile_phoenx_kapla_kernels --diff

    # Diff a different pair, e.g. before vs after an optimisation.
    python -m newton._src.solvers.phoenx.benchmarks.profile_phoenx_kapla_kernels \\
        --diff --label-a "before" --label-b "after"
"""

from __future__ import annotations

import argparse
import csv
import io
import pathlib
import shutil
import subprocess
import sys
import tempfile

_TOP_N = 25


def _check_nsys() -> str:
    path = shutil.which("nsys")
    if path is None:
        raise SystemExit("[profile_phoenx_kapla_kernels] nsys not on PATH; install NVIDIA Nsight Systems")
    return path


def _run_worker_script(
    out_dir: pathlib.Path,
    mass_splitting: bool,
    frames: int,
    grid_side: int,
) -> pathlib.Path:
    """Write a tiny worker that runs Kapla for ``frames`` captured-graph steps,
    then nsys-profile that worker. Returns the .nsys-rep path."""
    nsys = _check_nsys()
    worker = out_dir / "run_kapla_worker.py"
    label = "on" if mass_splitting else "off"
    worker.write_text(
        "import ctypes\n"
        "import warp as wp\n"
        "wp.init()\n"
        "from newton._src.solvers.phoenx.examples import example_kapla_tower as ek\n"
        "\n"
        "class V:\n"
        "    def __init__(self):\n"
        "        from types import SimpleNamespace\n"
        "        self.camera = SimpleNamespace(pos=SimpleNamespace(x=1.2, y=0.0, z=0.4))\n"
        "    def set_model(self, m): pass\n"
        "    def set_camera(self, **kw): pass\n"
        "    def begin_frame(self, t): pass\n"
        "    def log_state(self, s): pass\n"
        "    def log_contacts(self, c, s): pass\n"
        "    def end_frame(self): pass\n"
        "\n"
        "class A: pass\n"
        "\n"
        f"ek.ENABLE_MASS_SPLITTING = {mass_splitting!r}\n"
        f"ek.USE_COLORED_CONTACT_HEADERS = {mass_splitting!r}\n"
        f"ek.USE_COLORED_CONTACT_ROWS = {mass_splitting!r}\n"
        f"ek.TOWER_GRID_DIMS = ({grid_side}, {grid_side})\n"
        "ex = ek.Example(V(), A())\n"
        "# Warmup: damping released after WARMUP_FRAMES, then a few more so\n"
        "# the steady state stabilises.\n"
        "for _ in range(40):\n"
        "    ex.step()\n"
        "wp.synchronize_device(ex.device)\n"
        'cudart = ctypes.CDLL("libcudart.so")\n'
        "cudart.cudaProfilerStart()\n"
        f"for _ in range({frames}):\n"
        "    ex.step()\n"
        "wp.synchronize_device(ex.device)\n"
        "cudart.cudaProfilerStop()\n"
    )
    rep = out_dir / f"kapla_{label}"
    cmd = [
        nsys,
        "profile",
        "--trace=cuda",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--cuda-graph-trace=node",
        f"--output={rep}",
        "--force-overwrite=true",
        sys.executable,
        str(worker),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return rep.with_suffix(".nsys-rep")


def _parse_kernel_stats(rep: pathlib.Path) -> list[dict]:
    """Run ``nsys stats --report cuda_gpu_kern_sum`` and parse the CSV."""
    nsys = _check_nsys()
    proc = subprocess.run(
        [nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(rep)],
        check=True,
        capture_output=True,
        text=True,
    )
    # The output has a few header lines followed by CSV. Find the
    # ``Time (%),Total Time (ns),...`` row and parse from there.
    lines = proc.stdout.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Time (%)")), None)
    if header_idx is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    rows: list[dict] = []
    for row in reader:
        rows.append(
            {
                "pct": float(row["Time (%)"]),
                "total_ns": int(row["Total Time (ns)"]),
                "instances": int(row["Instances"]),
                "avg_ns": float(row["Avg (ns)"]),
                "name": _short_name(row["Name"]),
            }
        )
    return rows


def _short_name(name: str) -> str:
    """Strip Warp's per-build hash suffix so before/after diffs match."""
    name = name.strip().strip('"')
    # Warp kernel mangling: ``foo_kernel_<hex>_cuda_kernel_forward``.
    for suffix in ("_cuda_kernel_forward", "_cuda_kernel_backward"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Strip trailing 8+ hex chars after the last underscore.
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) >= 6 and all(c in "0123456789abcdef" for c in parts[1]):
        name = parts[0]
    return name


def _print_table(rows: list[dict], title: str, top_n: int) -> None:
    print(f"\n=== {title} — top {top_n} kernels ===")
    print(f"  {'rank':>4}  {'pct':>5}  {'tot_ms':>9}  {'calls':>7}  {'avg_us':>7}  name")
    for i, row in enumerate(rows[:top_n], 1):
        total_ms = row["total_ns"] / 1e6
        avg_us = row["avg_ns"] / 1e3
        short = row["name"][-80:] if len(row["name"]) > 80 else row["name"]
        print(f"  {i:>4}  {row['pct']:>4.1f}%  {total_ms:>9.2f}  {row['instances']:>7}  {avg_us:>7.2f}  {short}")


def _print_diff(rows_a: list[dict], rows_b: list[dict], label_a: str, label_b: str, top_n: int) -> None:
    by_name_a = {r["name"]: r for r in rows_a}
    by_name_b = {r["name"]: r for r in rows_b}
    all_names = set(by_name_a) | set(by_name_b)
    diffs: list[tuple[float, dict, dict, str]] = []
    for name in all_names:
        a = by_name_a.get(name)
        b = by_name_b.get(name)
        ms_a = (a["total_ns"] / 1e6) if a else 0.0
        ms_b = (b["total_ns"] / 1e6) if b else 0.0
        diffs.append((ms_b - ms_a, a or {"total_ns": 0}, b or {"total_ns": 0}, name))
    diffs.sort(key=lambda x: abs(x[0]), reverse=True)

    total_a = sum(r["total_ns"] for r in rows_a) / 1e6
    total_b = sum(r["total_ns"] for r in rows_b) / 1e6
    print(f"\n=== {label_b} vs {label_a} — kernels by absolute GPU-time delta (top {top_n}) ===")
    print(f"  TOTAL: {label_a}={total_a:.2f}ms  {label_b}={total_b:.2f}ms  Δ={total_b - total_a:+.2f}ms")
    print(f"  {'delta(ms)':>10}  {label_a:>10}  {label_b:>10}  name")
    for d, a, b, name in diffs[:top_n]:
        ms_a = a["total_ns"] / 1e6
        ms_b = b["total_ns"] / 1e6
        short = name[-80:] if len(name) > 80 else name
        print(f"  {d:+10.2f}  {ms_a:>10.2f}  {ms_b:>10.2f}  {short}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=100, help="Steady-state frames after warmup (default 100).")
    p.add_argument("--grid-side", type=int, default=1, help="Square tower-grid side length (default 1).")
    p.add_argument("--top-n", type=int, default=_TOP_N, help="Top-N kernels to print (default %(default)s).")
    p.add_argument("--mass-splitting", choices=["on", "off"], default="on", help="Default config when not diffing.")
    p.add_argument("--diff", action="store_true", help="Run mass-splitting on AND off, print a delta table.")
    p.add_argument("--label-a", default="off", help="Label for the first config in diff mode.")
    p.add_argument("--label-b", default="on", help="Label for the second config in diff mode.")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="phoenx_profile_") as tmp:
        tmp_path = pathlib.Path(tmp)
        if args.diff:
            print(f"[profile] {args.label_a}: running {args.frames} steady-state frames...")
            rep_a = _run_worker_script(tmp_path, mass_splitting=False, frames=args.frames, grid_side=args.grid_side)
            print(f"[profile] {args.label_b}: running {args.frames} steady-state frames...")
            rep_b = _run_worker_script(tmp_path, mass_splitting=True, frames=args.frames, grid_side=args.grid_side)
            rows_a = _parse_kernel_stats(rep_a)
            rows_b = _parse_kernel_stats(rep_b)
            _print_table(rows_a, args.label_a, args.top_n)
            _print_table(rows_b, args.label_b, args.top_n)
            _print_diff(rows_a, rows_b, args.label_a, args.label_b, args.top_n)
        else:
            ms = args.mass_splitting == "on"
            label = args.mass_splitting
            print(f"[profile] {label}: running {args.frames} steady-state frames...")
            rep = _run_worker_script(tmp_path, mass_splitting=ms, frames=args.frames, grid_side=args.grid_side)
            rows = _parse_kernel_stats(rep)
            _print_table(rows, label, args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
