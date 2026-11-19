# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live MP4 recording for the OptiX viewer (R or Shift+R starts, T stops).

Frames never stall rendering: a kernel packs the tonemapped frame into RGB bytes on the
render stream, a separate copy stream moves them into one of a few pinned host buffers while
the next frames render, and an ffmpeg writer thread takes each buffer once its copy has
completed. Frames are dropped
rather than slowing the viewer when the encoder falls behind.

The video runs at a fixed frame rate in simulation time, so it plays back smoothly at the
simulated speed however fast the window renders, and pauses add no frozen frames. The H.264
High-profile, yuv420p, fast-start MP4 output plays inline in Slack and browsers.
"""

from __future__ import annotations

import collections
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import warp as wp

OUTPUT_DIRECTORY = Path.home() / "Videos" / "NewtonRecordings"
_SLOTS = 4  # frames in flight between the GPU and the encoder
_LATENCY = 2  # frames a copy gets before the writer waits on it


class _FpsOverlay:
    def __init__(self, height: int):
        from PIL import ImageFont

        size = max(12, round(height / 45))
        self.font = ImageFont.load_default(size=size)
        self.margin = max(2, size // 3)
        self.label = None
        self.badge = None

    def draw(self, frame: np.ndarray, fps: float) -> None:
        label = f"{fps:.1f} FPS" if fps > 0.0 else "— FPS"
        if label != self.label:
            from PIL import Image, ImageDraw

            left, top, right, bottom = self.font.getbbox(label)
            pad = max(2, self.margin // 2)
            badge = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), color=28)
            ImageDraw.Draw(badge).text((pad - left, pad - top), label, font=self.font, fill=255)
            self.badge = np.asarray(badge)
            self.label = label
        h, w = self.badge.shape
        patch = frame[self.margin : self.margin + h, self.margin : self.margin + w]
        patch[:] = self.badge[: patch.shape[0], : patch.shape[1], None]


@wp.kernel(enable_backward=False)
def _pack_rgb8_kernel(ldr: wp.array2d[wp.vec4], rgb: wp.array3d[wp.uint8]):
    """Linear-to-byte RGB with the top row first (the renderer's rows start at the bottom)."""
    i, j = wp.tid()
    c = ldr[ldr.shape[0] - 1 - i, j]
    for k in range(3):
        rgb[i, j, k] = wp.uint8(wp.clamp(c[k], 0.0, 1.0) * 255.0 + 0.5)


class _Slot:
    def __init__(self, height: int, width: int, device):
        self.gpu = wp.empty((height, width, 3), dtype=wp.uint8, device=device)
        self.host = wp.empty((height, width, 3), dtype=wp.uint8, device="cpu", pinned=True)
        self.repeats = 1  # video frames this rendered frame fills
        self.viewer_fps = 0.0
        self.packed = wp.Event(device)
        self.copied = wp.Event(device)


class LiveMp4Recorder:
    """Record the OptiX viewer's tonemapped frames to MP4 through ffmpeg."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._output_path: Path | None = None
        self._slots: list[_Slot] = []
        self._free: queue.Queue[_Slot] = queue.Queue()
        self._ready: queue.Queue[_Slot | None] = queue.Queue()
        self._in_flight: collections.deque[_Slot] = collections.deque()
        self._writer: threading.Thread | None = None
        self._shape = (0, 0)
        self._copy_stream: wp.Stream | None = None
        self._fps = 60.0
        self._fps_overlay: _FpsOverlay | None = None
        self._start_time: float | None = None
        self._next_frame = 0
        self.dropped = 0

    @property
    def is_recording(self) -> bool:
        return self._proc is not None

    @staticmethod
    def _encoder_args(ffmpeg_exe: str) -> list[str]:
        """High-quality H.264: NVENC constant quality when available, otherwise x264 CRF."""
        try:
            out = subprocess.run([ffmpeg_exe, "-hide_banner", "-encoders"], check=False, capture_output=True, text=True)
            if "h264_nvenc" in (out.stdout or ""):
                return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0"]
        except Exception:
            pass
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]

    def start(self, width: int, height: int, fps: float = 60.0, *, show_fps: bool = False) -> Path | None:
        """Start recording to a timestamped file in :data:`OUTPUT_DIRECTORY`.

        Returns the output path, or ``None`` without imageio-ffmpeg.
        """
        if self.is_recording:
            return self._output_path
        try:
            import imageio_ffmpeg  # noqa: PLC0415
        except Exception:
            return None
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self._fps_overlay = _FpsOverlay(height) if show_fps else None
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIRECTORY / f"optix_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self._fps = max(1.0, float(fps))
        cmd = [
            ffmpeg_exe, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{int(width)}x{int(height)}", "-r", f"{self._fps:.3f}", "-i", "-", "-an",
            # yuv420p needs even dimensions; High profile with fast start plays inline in Slack.
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", *self._encoder_args(ffmpeg_exe),
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ]  # fmt: skip
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._shape = (int(height), int(width))
        self._output_path = output_path
        self._start_time = None
        self._next_frame = 0
        self.dropped = 0
        self._writer = threading.Thread(target=self._writer_loop, name="optix-mp4-writer", daemon=True)
        self._writer.start()
        return output_path

    def _writer_loop(self):
        while (slot := self._ready.get()) is not None:
            try:
                pixels = slot.host.numpy()
                if self._fps_overlay is not None:
                    self._fps_overlay.draw(pixels, slot.viewer_fps)
                frame = memoryview(pixels)
                for _ in range(slot.repeats):
                    self._proc.stdin.write(frame)
            except (BrokenPipeError, ValueError):
                pass
            self._free.put(slot)

    def capture(self, ldr: wp.array2d[wp.vec4], stream: wp.Stream, time: float, *, viewer_fps: float = 0.0) -> None:
        """Queue the tonemapped frame at simulation ``time`` [s]; never waits for the GPU."""
        if not self.is_recording or tuple(ldr.shape) != self._shape:
            return
        # Fill the video frames up to this simulation time; skip frames of an unchanged time.
        if self._start_time is None:
            self._start_time = time
        frame = round((time - self._start_time) * self._fps)
        if frame < self._next_frame:
            return
        repeats = min(frame - self._next_frame + 1, 8)
        self._next_frame = frame + 1
        if not self._slots:
            self._copy_stream = wp.Stream(ldr.device)
            self._slots = [_Slot(*self._shape, ldr.device) for _ in range(_SLOTS)]
            for slot in self._slots:
                self._free.put(slot)
        # Hand frames whose copies have long finished to the writer.
        while len(self._in_flight) >= _LATENCY:
            self._release_oldest()
        try:
            slot = self._free.get_nowait()
        except queue.Empty:
            self.dropped += 1  # the encoder is behind; drop this frame
            return
        slot.repeats = repeats
        slot.viewer_fps = viewer_fps
        # Only the cheap pack runs on the render stream; the next frame may overwrite ``ldr``.
        with wp.ScopedStream(stream):
            wp.launch(_pack_rgb8_kernel, dim=self._shape, inputs=[ldr], outputs=[slot.gpu])
        stream.record_event(slot.packed)
        self._copy_stream.wait_event(slot.packed)
        wp.copy(slot.host, slot.gpu, stream=self._copy_stream)
        self._copy_stream.record_event(slot.copied)
        self._in_flight.append(slot)

    def _release_oldest(self):
        slot = self._in_flight.popleft()
        wp.synchronize_event(slot.copied)
        self._ready.put(slot)

    def stop(self) -> Path | None:
        """Write the queued frames, finish the file, and return its path, or ``None``."""
        if not self.is_recording:
            return None
        while self._in_flight:
            self._release_oldest()
        self._ready.put(None)
        self._writer.join(timeout=30.0)
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=30.0)
        except Exception:
            self._proc.kill()
        self._proc = None
        self._writer = None
        self._slots = []
        self._free = queue.Queue()
        self._ready = queue.Queue()
        return self._output_path
