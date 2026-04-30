# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render per-scene comparison plots from ``results/points.jsonl``.

Offline counterpart to the Chart.js dashboard. Produces one PNG per
scenario with three side-by-side panels (env_fps, ms/world-step,
GPU used) and every solver overlaid as a separate line. Useful for
pasting into issues / chat / slides without running the local
HTTP server.

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.plot_bench

Defaults read ``results/points.jsonl`` and write ``results/plots/``
next to it. Both are overridable with CLI flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib

    # ``Agg`` is the headless backend; works in a pure-CLI session
    # without a display. Set before ``pyplot`` imports.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover -- install-time hint
    print(
        "[plot_bench] matplotlib is not installed. Install with: uv add matplotlib",
        file=sys.stderr,
    )
    raise

# Keep PhoenX / MuJoCo colours consistent with the Chart.js dashboard so
# panels copy-pasted across the two media read the same way.
DEFAULT_COLORS: dict[str, str] = {
    "phoenx": "#0f8c7a",
    "mujoco": "#2563eb",
}

_DEFAULT_POINTS = Path(__file__).parent / "results" / "points.jsonl"
_DEFAULT_OUT = Path(__file__).parent / "results" / "plots"


def _load_points(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} doesn't exist yet. Run the benchmark first:\n"
            f"  python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks"
        )
    rows: list[dict] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _plot_scene(
    scene: str,
    rows: list[dict],
    *,
    colors: dict[str, str],
    log_y_fps: bool = True,
    log_y_ms: bool = True,
) -> plt.Figure:
    """Build one figure for ``scene``: three panels, one line per
    solver. Only rows with ``ok=True`` are plotted; failed rows land
    in the JSONL but shouldn't distort the curves."""
    ok_rows = [r for r in rows if r["scenario"] == scene and r.get("ok", True)]
    solvers = sorted({r["solver"] for r in ok_rows})

    gpu_label = ok_rows[0].get("gpu", "unknown GPU") if ok_rows else ""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"PhoenX vs MuJoCo-Warp — {scene}  ({gpu_label})", fontsize=11)

    panels = [
        (axes[0], "env_fps", "env_fps (higher is better)", log_y_fps),
        (axes[1], "ms_per_step", "ms / world-step (lower is better)", log_y_ms),
        (axes[2], "gpu_used_gb", "GPU used (GB)", False),
    ]
    for ax, ykey, ylabel, log_y in panels:
        for solver in solvers:
            pts = sorted(
                (r for r in ok_rows if r["solver"] == solver),
                key=lambda r: r["num_worlds"],
            )
            if not pts:
                continue
            xs = [r["num_worlds"] for r in pts]
            ys = [r.get(ykey, 0.0) for r in pts]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=colors.get(solver, "grey"),
                label=solver,
            )
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("num_worlds")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if solvers:
            ax.legend()

    fig.tight_layout()
    return fig


def render(
    points_path: Path = _DEFAULT_POINTS,
    out_dir: Path = _DEFAULT_OUT,
    *,
    colors: dict[str, str] | None = None,
) -> list[Path]:
    """Render one PNG per scenario found in ``points_path`` into
    ``out_dir``; return the written paths."""
    colors = colors if colors is not None else DEFAULT_COLORS
    rows = _load_points(points_path)
    scenes = sorted({r["scenario"] for r in rows})
    if not scenes:
        print(f"[plot_bench] {points_path} has no rows", file=sys.stderr)
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for scene in scenes:
        fig = _plot_scene(scene, rows, colors=colors)
        out_path = out_dir / f"{scene}.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
        print(f"[plot_bench] wrote {out_path}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render PhoenX vs MuJoCo-Warp benchmark plots.")
    parser.add_argument(
        "--points",
        type=Path,
        default=_DEFAULT_POINTS,
        help="Path to points.jsonl (default: results/points.jsonl).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output directory for PNGs (default: results/plots/).",
    )
    args = parser.parse_args(argv)
    render(args.points, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
