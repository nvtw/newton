# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest import mock

from newton.examples._render_state import RenderStateBuffer
from newton.examples.phoenx.example_phoenx_ragdoll_pile import Example


class TestExampleRenderState(unittest.TestCase):
    def test_release_after_frame_reads(self):
        """A snapshot cannot be reused before end_frame submits GPU reads."""
        calls = []
        snapshots = [SimpleNamespace(body_q=object()), SimpleNamespace(body_q=object())]
        model = SimpleNamespace(device=object(), state=mock.Mock(side_effect=snapshots))
        source = SimpleNamespace(body_q=SimpleNamespace(shape=(1,)), particle_q=None)
        with mock.patch("newton.examples._render_state.wp.Event", side_effect=["event0", "event1"]):
            buffer = RenderStateBuffer(model, enabled=True)
        with (
            mock.patch("newton.examples._render_state.wp.wait_event", side_effect=calls.append),
            mock.patch("newton.examples._render_state.wp.copy", side_effect=lambda dst, src: calls.append("copy")),
            mock.patch(
                "newton.examples._render_state.wp.record_event", side_effect=lambda event: calls.append("release")
            ),
        ):
            buffer.prepare(source, 1.0)
            example = Example.__new__(Example)
            example.render_state = buffer
            example.state = source
            example.sim_time = 2.0
            example.viewer = SimpleNamespace(
                begin_frame=lambda time: calls.append(("begin", time)),
                log_state=lambda state: calls.append(("log", state)),
                end_frame=lambda: calls.append("end"),
            )
            example.render()
        self.assertEqual(calls, ["event1", "copy", ("begin", 1.0), ("log", snapshots[1]), "end", "release"])
        self.assertEqual(buffer.get(source, 2.0), (source, 2.0))

    def test_disabled_snapshots_keep_synchronous_state(self):
        model = SimpleNamespace(state=mock.Mock())
        buffer = RenderStateBuffer(model, enabled=False)
        state = object()
        self.assertEqual(buffer.get(state, 1.0), (state, 1.0))
        buffer.release()
        model.state.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            buffer.prepare(state, 1.0)


if __name__ == "__main__":
    unittest.main()
