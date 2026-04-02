# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live training dashboard for the Newton OpenGL viewer.

Displays PPO training curves (reward, loss, entropy coefficient) as
scrolling line plots inside the viewer's ImGui overlay.

Usage::

    from newton._src.training_monitor import TrainingMonitor

    monitor = TrainingMonitor(viewer)

    # In training loop, after trainer.update():
    monitor.log(trainer.get_stats())
"""

from __future__ import annotations

from typing import Any

import numpy as np


class _RingPlot:
    """Fixed-size ring buffer with cached plot data."""

    __slots__ = ("_buf", "_n", "_head", "_dirty", "_plot", "_lo", "_hi", "_last")

    def __init__(self, n: int):
        self._buf = np.zeros(n, dtype=np.float32)
        self._n = n
        self._head = 0
        self._dirty = False
        self._plot = np.zeros(n, dtype=np.float32)
        self._lo = 0.0
        self._hi = 1e-8
        self._last = 0.0

    def push(self, v: float) -> None:
        self._buf[self._head % self._n] = v
        self._head += 1
        self._last = v
        self._dirty = True

    def get_plot(self) -> tuple[np.ndarray, float, float, float]:
        """Return (array, lo, hi, last) -- recomputes only when dirty."""
        if self._dirty:
            n = min(self._head, self._n)
            if self._head <= self._n:
                self._plot[:n] = self._buf[:n]
                if n < self._n:
                    self._plot[n:] = 0.0
            else:
                start = self._head % self._n
                self._plot[:] = np.roll(self._buf, -start)
            self._lo = float(self._plot[:n].min())
            self._hi = float(self._plot[:n].max()) + 1e-8
            self._dirty = False
        return self._plot, self._lo, self._hi, self._last


class TrainingMonitor:
    """Live training dashboard rendered in the Newton OpenGL viewer.

    Call :meth:`log` after each PPO update with the dict from
    ``PPOTrainer.get_stats()``.

    Args:
        viewer: A Newton viewer instance.  If ``None``, rendering is skipped.
        history_len: Number of data points to keep per metric.
    """

    def __init__(self, viewer: Any = None, history_len: int = 500):
        self._viewer = viewer
        self._update_count = 0
        self._reward = _RingPlot(history_len)
        self._loss = _RingPlot(history_len)
        self._alpha = _RingPlot(history_len)

        if viewer is not None and hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._render, "free")

    def log(self, stats: dict[str, float]) -> None:
        """Record one training update's statistics."""
        self._reward.push(stats.get("mean_reward", 0.0))
        self._loss.push(stats.get("loss", 0.0))
        self._alpha.push(stats.get("alpha", 0.0))
        self._update_count += 1

    def _render(self, imgui: Any) -> None:
        if self._viewer is None or not hasattr(self._viewer, "ui") or not self._viewer.ui.is_available:
            return

        io = self._viewer.ui.io
        w, h = 380, 520
        imgui.set_next_window_pos(imgui.ImVec2(io.display_size[0] - w - 10, 10))
        imgui.set_next_window_size(imgui.ImVec2(w, h))

        if not imgui.begin(f"Training  [update {self._update_count}]", flags=imgui.WindowFlags_.no_resize.value):
            imgui.end()
            return

        if self._update_count == 0:
            imgui.text("Waiting for first update...")
            imgui.end()
            return

        avail_w = imgui.get_content_region_avail().x
        plot_h = 120

        arr, lo, hi, last = self._reward.get_plot()
        imgui.text(f"Mean Reward:  {last:.3f}")
        imgui.plot_lines("##reward", arr, graph_size=(avail_w, plot_h), scale_min=lo, scale_max=hi)

        imgui.spacing()

        arr, lo, hi, last = self._loss.get_plot()
        imgui.text(f"Loss:  {last:.4f}")
        imgui.plot_lines("##loss", arr, graph_size=(avail_w, plot_h), scale_min=0.0, scale_max=hi)

        imgui.spacing()

        arr, lo, hi, last = self._alpha.get_plot()
        imgui.text(f"Alpha:  {last:.5f}")
        imgui.plot_lines("##alpha", arr, graph_size=(avail_w, plot_h), scale_min=0.0, scale_max=hi)

        imgui.end()
