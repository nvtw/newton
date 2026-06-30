# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ViewerGL live MP4 recording.

Tests that require an actual ``ffmpeg`` binary are skipped when neither
``imageio-ffmpeg`` nor a system ``ffmpeg`` is available.
"""

from __future__ import annotations

import logging
import queue as _queue
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import newton._src.viewer.recording as recording_mod
from newton._src.viewer.recording import LiveMp4Recorder, _resolve_ffmpeg_executable
from newton._src.viewer.viewer import ViewerBase
from newton._src.viewer.viewer_gl import ViewerGL


def _ffmpeg_available() -> bool:
    return _resolve_ffmpeg_executable() is not None


class TestQualityClamping(unittest.TestCase):
    """Quality setter must clamp to [0, 100]."""

    def test_clamping_and_roundtrip(self):
        rec = LiveMp4Recorder()
        self.assertEqual(rec.quality, 90.0)
        rec.set_quality(50.0)
        self.assertEqual(rec.quality, 50.0)
        rec.set_quality(-5.0)
        self.assertEqual(rec.quality, 0.0)
        rec.set_quality(120.0)
        self.assertEqual(rec.quality, 100.0)


class TestSuggestedOutputPath(unittest.TestCase):
    """Default output path uses the configured prefix and ends in ``.mp4``."""

    def test_prefix_used_in_suggested_path(self):
        rec = LiveMp4Recorder()
        rec.set_filename_prefix("unit_test_recording")
        path = rec.suggested_output_path()
        self.assertEqual(path.suffix, ".mp4")
        self.assertTrue(path.name.startswith("unit_test_recording_"))


class TestWriteFrameWithoutStart(unittest.TestCase):
    """Calling ``write_frame`` before ``start`` must be a silent no-op."""

    def test_write_without_start_is_noop(self):
        rec = LiveMp4Recorder()
        rec.write_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertFalse(rec.is_recording)
        self.assertIsNone(rec.stop())


class TestEncoderSelection(unittest.TestCase):
    def test_unavailable_hardware_encoder_falls_back(self):
        rec = LiveMp4Recorder()
        listing = Mock(stdout="h264_nvenc h264_qsv libx264", returncode=0)
        with (
            patch.object(recording_mod.subprocess, "run", return_value=listing),
            patch.object(rec, "_probe_encoder", side_effect=[False, True]) as probe,
        ):
            self.assertEqual(rec._pick_encoder("ffmpeg"), "h264_qsv")
        self.assertEqual([call.args[1] for call in probe.call_args_list], ["h264_nvenc", "h264_qsv"])

    def test_no_usable_h264_encoder(self):
        rec = LiveMp4Recorder()
        listing = Mock(stdout="h264_nvenc", returncode=0)
        with (
            patch.object(recording_mod.subprocess, "run", return_value=listing),
            patch.object(rec, "_probe_encoder", return_value=False),
        ):
            self.assertIsNone(rec._pick_encoder("ffmpeg"))

    def test_encoder_probe_timeout(self):
        with patch.object(
            recording_mod.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5.0),
        ):
            self.assertFalse(LiveMp4Recorder._probe_encoder("ffmpeg", "h264_nvenc"))


class TestViewerGLRecordingTiming(unittest.TestCase):
    def _make_viewer(self):
        recorder = SimpleNamespace(is_recording=True, frames=[])
        recorder.write_frame_bytes = lambda data, width, height: recorder.frames.append((data, width, height))
        frame = SimpleNamespace(shape=(2, 3, 3), numpy=lambda: np.zeros((2, 3, 3), dtype=np.uint8))

        viewer = ViewerGL.__new__(ViewerGL)
        viewer._recorder = recorder
        viewer._record_next_sim_t = 0.0
        viewer._record_fps = 2.0
        viewer._record_frame_gpu = frame
        viewer._record_last_frame_bytes = None
        viewer.renderer = SimpleNamespace(_screen_width=3, _screen_height=2)
        viewer.get_frame = Mock(return_value=frame)
        viewer.time = 0.0
        return viewer, recorder

    def test_capture_is_paced_by_simulation_time(self):
        viewer, recorder = self._make_viewer()

        viewer._record_frame_if_needed()
        self.assertEqual(len(recorder.frames), 1)
        self.assertEqual(viewer.get_frame.call_count, 1)

        viewer.time = 0.2
        viewer._record_frame_if_needed()
        self.assertEqual(len(recorder.frames), 1)
        self.assertEqual(viewer.get_frame.call_count, 1)

        viewer.time = 0.5
        viewer._record_frame_if_needed()
        self.assertEqual(len(recorder.frames), 2)
        self.assertEqual(viewer.get_frame.call_count, 2)

    def test_large_time_jump_drops_excess_backlog(self):
        viewer, recorder = self._make_viewer()
        viewer.time = 10.0

        viewer._record_frame_if_needed()

        self.assertEqual(len(recorder.frames), 8)
        self.assertGreater(viewer._record_next_sim_t, viewer.time)

    def test_catch_up_holds_previous_frame_until_current_time(self):
        viewer, recorder = self._make_viewer()
        old_frame = SimpleNamespace(shape=(2, 3, 3), numpy=lambda: np.zeros((2, 3, 3), dtype=np.uint8))
        new_frame = SimpleNamespace(shape=(2, 3, 3), numpy=lambda: np.ones((2, 3, 3), dtype=np.uint8))
        viewer.get_frame.side_effect = [old_frame, new_frame]

        viewer._record_frame_if_needed()
        viewer.time = 1.0
        viewer._record_frame_if_needed()

        self.assertEqual([frame[0][0] for frame in recorder.frames], [0, 0, 1])

    def test_initializing_layer_does_not_replace_recorder(self):
        viewer = ViewerGL.__new__(ViewerGL)
        recorder = object()
        viewer._recorder = recorder
        layer = SimpleNamespace()

        with patch.object(ViewerBase, "_init_extra_layer_state"):
            viewer._init_extra_layer_state(layer)

        self.assertIs(viewer._recorder, recorder)


@unittest.skipUnless(_ffmpeg_available(), "ffmpeg not available (install imageio-ffmpeg or system ffmpeg)")
class TestLiveMp4Recording(unittest.TestCase):
    """End-to-end recording into a temporary directory.

    Pinned to ``libx264`` so the tests do not depend on the host having a
    working NVENC/QSV/AMF driver (e.g. CPU-only CI runners).
    """

    def setUp(self):
        self._encoder_patch = patch.object(LiveMp4Recorder, "_pick_encoder", return_value="libx264")
        self._encoder_patch.start()

    def tearDown(self):
        self._encoder_patch.stop()

    def test_record_synthetic_frames(self):
        width, height, n_frames = 256, 256, 24
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.mp4"
            rec = LiveMp4Recorder()
            rec.set_quality(50.0)
            self.assertTrue(
                rec.start(width=width, height=height, fps=30.0, output_path=out),
                "Recorder failed to start with ffmpeg available",
            )
            try:
                self.assertTrue(rec.is_recording)
                for i in range(n_frames):
                    frame = np.full((height, width, 3), i * 20 % 255, dtype=np.uint8)
                    rec.write_frame(frame)
            finally:
                stopped = rec.stop()

            self.assertEqual(stopped, out)
            self.assertFalse(rec.is_recording)
            self.assertTrue(out.exists(), f"Expected MP4 output at {out}")
            self.assertGreater(out.stat().st_size, 0, "MP4 output should be non-empty")

    def test_rejects_wrong_shape_and_dtype(self):
        size = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "shape.mp4"
            rec = LiveMp4Recorder()
            self.assertTrue(rec.start(size, size, fps=15.0, output_path=out))
            try:
                # Wrong dtype: silently dropped.
                rec.write_frame(np.zeros((size, size, 3), dtype=np.float32))
                # Wrong channel count.
                rec.write_frame(np.zeros((size, size, 4), dtype=np.uint8))
                # Wrong size.
                rec.write_frame(np.zeros((size // 2, size // 2, 3), dtype=np.uint8))
                # A handful of valid frames so the encoder has something to flush.
                for _ in range(8):
                    rec.write_frame(np.zeros((size, size, 3), dtype=np.uint8))
            finally:
                rec.stop()
            self.assertTrue(out.exists())

    def test_odd_frame_dimensions_are_padded(self):
        width, height = 255, 257
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "odd_dimensions.mp4"
            rec = LiveMp4Recorder()
            self.assertTrue(rec.start(width, height, fps=15.0, output_path=out))
            try:
                for _ in range(4):
                    rec.write_frame(np.zeros((height, width, 3), dtype=np.uint8))
            finally:
                rec.stop()
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


class TestQueueDropping(unittest.TestCase):
    """Pushing more than ``maxsize`` frames without a worker must not raise."""

    def test_drops_excess_frames_silently(self):
        rec = LiveMp4Recorder()

        # Simulate an active recording with a real queue but no live ffmpeg
        # subprocess: we manipulate internals directly to avoid spawning ffmpeg.
        # ``write_frame`` only cares about ``is_recording`` and ``self._queue``.

        class _DummyProc:
            pass

        rec._proc = _DummyProc()
        rec._queue = _queue.Queue(maxsize=4)
        rec._width = 16
        rec._height = 16
        try:
            for _ in range(64):
                rec.write_frame(np.zeros((16, 16, 3), dtype=np.uint8))
            self.assertLessEqual(rec._queue.qsize(), 4)
        finally:
            # Manual cleanup to avoid touching the (non-existent) worker thread.
            rec._proc = None
            rec._queue = None


@unittest.skipUnless(_ffmpeg_available(), "ffmpeg not available")
class TestDoubleStartIsIdempotent(unittest.TestCase):
    """Calling ``start`` twice is a no-op while already recording."""

    def setUp(self):
        self._encoder_patch = patch.object(LiveMp4Recorder, "_pick_encoder", return_value="libx264")
        self._encoder_patch.start()

    def tearDown(self):
        self._encoder_patch.stop()

    def test_double_start(self):
        size = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "double.mp4"
            rec = LiveMp4Recorder()
            self.assertTrue(rec.start(size, size, fps=15.0, output_path=out))
            try:
                # Second start should report success but not respawn ffmpeg.
                self.assertTrue(rec.start(size * 2, size * 2, fps=30.0, output_path=Path(tmpdir) / "other.mp4"))
                self.assertEqual(rec.output_path, out)
                for _ in range(4):
                    rec.write_frame(np.zeros((size, size, 3), dtype=np.uint8))
            finally:
                rec.stop()
            self.assertTrue(out.exists())
            # Confirm the second path was not created.
            self.assertFalse((Path(tmpdir) / "other.mp4").exists())


class TestFfmpegFallback(unittest.TestCase):
    """When ffmpeg is truly unavailable, start() must return False (not raise)."""

    def test_start_returns_false_when_ffmpeg_missing(self):
        rec = LiveMp4Recorder()

        with patch.object(recording_mod, "_resolve_ffmpeg_executable", return_value=None):
            self.assertFalse(rec.start(width=16, height=16, fps=30.0))
            self.assertFalse(rec.is_recording)
        # Ensure no lingering subprocess handle.
        self.assertIsNone(rec.stop())


if __name__ == "__main__":
    # Surface log output during local debugging runs.
    logging.basicConfig(level=logging.INFO)
    unittest.main()
