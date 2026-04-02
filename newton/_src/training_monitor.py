# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live training dashboard for the Newton OpenGL viewer.

Displays PPO training curves (reward, loss, entropy coefficient) as
scrolling line plots inside the viewer's ImGui overlay.  Inspired by
TensorBoard but zero-dependency -- uses only ImGui's built-in
``plot_lines``.

Usage::

    from newton._src.training_monitor import TrainingMonitor

    monitor = TrainingMonitor(viewer)

    # In training loop, after trainer.update():
    monitor.log(trainer.get_stats())
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TrainingMonitor:
    """Live training dashboard rendered in the Newton OpenGL viewer.

    Call :meth:`log` after each PPO update with the dict from
    ``PPOTrainer.get_stats()``.  The monitor auto-registers itself as
    a viewer UI callback.

    Args:
        viewer: A Newton viewer instance (``ViewerGL`` or similar).
            If ``None`` or if the viewer has no UI, rendering is silently
            skipped.
        history_len: Number of data points to keep per metric.
        update_title: Show update counter and steps in the title bar.
    """

    def __init__(self, viewer: Any = None, history_len: int = 500):
        self._viewer = viewer
        self._n = history_len
        self._update_count = 0

        # Ring buffers for each metric
        self._reward = np.zeros(history_len, dtype=np.float32)
        self._loss = np.zeros(history_len, dtype=np.float32)
        self._alpha = np.zeros(history_len, dtype=np.float32)
        self._head = 0  # next write position

        # Register with viewer
        if viewer is not None and hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._render, "free")

    def log(self, stats: dict[str, float]) -> None:
        """Record one training update's statistics.

        Args:
            stats: Dict from ``PPOTrainer.get_stats()``.  Expected keys:
                ``mean_reward``, ``loss``, ``alpha``.
        """
        i = self._head % self._n
        self._reward[i] = stats.get("mean_reward", 0.0)
        self._loss[i] = stats.get("loss", 0.0)
        self._alpha[i] = stats.get("alpha", 0.0)
        self._head += 1
        self._update_count += 1

    def _render(self, imgui: Any) -> None:
        """ImGui callback -- draws the dashboard window."""
        if self._viewer is None or not hasattr(self._viewer, "ui") or not self._viewer.ui.is_available:
            return

        io = self._viewer.ui.io
        w = 380
        h = 520
        imgui.set_next_window_pos(imgui.ImVec2(io.display_size[0] - w - 10, 10))
        imgui.set_next_window_size(imgui.ImVec2(w, h))

        flags = imgui.WindowFlags_.no_resize.value
        if not imgui.begin(f"Training  [update {self._update_count}]", flags=flags):
            imgui.end()
            return

        n = min(self._head, self._n)
        if n == 0:
            imgui.text("Waiting for first update...")
            imgui.end()
            return

        # Get the data in chronological order
        if self._head <= self._n:
            reward = self._reward[:n]
            loss = self._loss[:n]
            alpha = self._alpha[:n]
        else:
            start = self._head % self._n
            reward = np.roll(self._reward, -start)
            loss = np.roll(self._loss, -start)
            alpha = np.roll(self._alpha, -start)

        avail_w = imgui.get_content_region_avail().x
        plot_h = 120

        # -- Reward --
        imgui.text(f"Mean Reward:  {reward[-1]:.3f}")
        imgui.plot_lines(
            "##reward",
            reward,
            graph_size=(avail_w, plot_h),
            scale_min=float(np.min(reward)),
            scale_max=float(np.max(reward)) + 1e-8,
        )

        imgui.spacing()

        # -- Loss --
        imgui.text(f"Loss:  {loss[-1]:.4f}")
        imgui.plot_lines(
            "##loss",
            loss,
            graph_size=(avail_w, plot_h),
            scale_min=0.0,
            scale_max=float(np.max(loss)) + 1e-8,
        )

        imgui.spacing()

        # -- Alpha (entropy coefficient) --
        imgui.text(f"Alpha:  {alpha[-1]:.5f}")
        imgui.plot_lines(
            "##alpha",
            alpha,
            graph_size=(avail_w, plot_h),
            scale_min=0.0,
            scale_max=float(np.max(alpha)) + 1e-8,
        )

        imgui.end()
