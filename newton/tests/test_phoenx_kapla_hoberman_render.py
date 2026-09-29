# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

from newton._src.solvers.phoenx.examples.example_kapla_hoberman import Example


class _RecordingViewer:
    def __init__(self, calls):
        self.calls = calls

    def begin_frame(self, time):
        self.calls.append("begin")

    def log_state(self, state):
        self.calls.append("log")

    def end_frame(self):
        self.calls.append("end")


class TestKaplaHobermanRender(unittest.TestCase):
    def test_snapshot_released_after_end_frame(self):
        """Keep the snapshot live until end_frame has enqueued its GPU reads."""
        calls = []
        example = Example.__new__(Example)
        example.viewer = _RecordingViewer(calls)
        example._render_states = (object(), object())
        example._render_state_index = 0
        example._render_state_prepared = True
        example._render_state_done = (object(), object())
        example._render_time = 0.0
        example.state = object()

        with mock.patch(
            "newton._src.solvers.phoenx.examples.example_kapla_hoberman.wp.record_event",
            side_effect=lambda event: calls.append("release"),
        ):
            example.render()

        self.assertEqual(calls, ["begin", "log", "end", "release"])
