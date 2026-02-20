# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Optional live MP4 recording helper for the OptiX viewer."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LiveMp4Recorder:
    """Record RGB frames to MP4 via an ffmpeg subprocess."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[bytes | None] | None = None
        self._worker: threading.Thread | None = None
        self._output_path: Path | None = None
        self._width = 0
        self._height = 0

    @property
    def is_recording(self) -> bool:
        return self._proc is not None

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def default_output_directory(self) -> Path:
        """Return a cross-platform default directory for recordings."""
        videos_dir = Path.home() / "Videos"
        if videos_dir.is_dir():
            return videos_dir / "NewtonRecordings"
        return Path.home() / "NewtonRecordings"

    def suggested_output_path(self) -> Path:
        """Return a timestamped default output path."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.default_output_directory() / f"optix_recording_{stamp}.mp4"

    def _pick_encoder(self, ffmpeg_exe: str) -> str:
        preferred = ("h264_nvenc", "h264_qsv", "h264_amf", "libx264")
        try:
            result = subprocess.run(
                [ffmpeg_exe, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
            )
            out = result.stdout or ""
            for codec in preferred:
                if codec in out:
                    return codec
        except Exception:
            pass
        return "libx264"

    def _writer_loop(self):
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._queue is not None
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                try:
                    self._proc.stdin.write(item)
                except BrokenPipeError:
                    break
        finally:
            try:
                self._proc.stdin.close()
            except Exception:
                pass

    def start(self, width: int, height: int, fps: float = 60.0, output_path: str | Path | None = None) -> bool:
        """Start recording. Returns False when optional deps are unavailable."""
        if self.is_recording:
            return True

        try:
            import imageio_ffmpeg  # noqa: PLC0415
        except Exception:
            logger.info("imageio-ffmpeg unavailable; recording is disabled.")
            return False

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self._width = int(width)
        self._height = int(height)
        fps = max(1.0, float(fps))

        if output_path is None:
            output_path = self.suggested_output_path()
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = self._pick_encoder(ffmpeg_exe)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            f"{fps:.3f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            codec,
        ]
        if codec == "h264_nvenc":
            # High-quality VBR mode for NVENC.
            cmd += ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "16", "-b:v", "0"]
        elif codec in {"h264_qsv", "h264_amf"}:
            # Use a high target bitrate on other HW encoders to reduce visible artifacts.
            cmd += ["-b:v", "50M", "-maxrate", "100M", "-bufsize", "150M"]
        elif codec == "libx264":
            # Stronger quality setting while keeping H.264 web compatibility.
            cmd += ["-preset", "slow", "-crf", "16"]
        cmd += ["-vf", "vflip", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._queue = queue.Queue(maxsize=8)
        self._worker = threading.Thread(target=self._writer_loop, name="optix-mp4-writer", daemon=True)
        self._worker.start()
        self._output_path = output_path
        logger.info("Started recording to %s using encoder %s.", output_path, codec)
        return True

    def write_frame(self, frame_rgb: np.ndarray):
        """Queue a frame for encoding. Frame must be HxWx3 uint8."""
        if not self.is_recording or self._queue is None:
            return
        frame = np.asarray(frame_rgb)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            return
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        try:
            self._queue.put_nowait(frame.tobytes())
        except queue.Full:
            # Keep renderer real-time by dropping frames under sustained pressure.
            pass

    def stop(self) -> Path | None:
        """Stop recording and flush the output file."""
        if not self.is_recording:
            return self._output_path
        assert self._proc is not None

        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None

        try:
            self._proc.wait(timeout=10.0)
        except Exception:
            self._proc.kill()

        stopped_path = self._output_path
        logger.info("Stopped recording: %s", stopped_path)
        self._proc = None
        self._queue = None
        return stopped_path

