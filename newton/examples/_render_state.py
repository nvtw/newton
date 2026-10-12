# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pose snapshots for examples that overlap simulation and rendering."""

import warp as wp


class RenderStateBuffer:
    """Retain two pose snapshots until the renderer has finished reading them."""

    def __init__(self, model, *, enabled):
        self._states = (model.state(), model.state()) if enabled else None
        self._done = (wp.Event(model.device), wp.Event(model.device)) if enabled else None
        self._index = 0
        self._prepared = False
        self._time = 0.0

    def prepare(self, state, time):
        """Copy on the simulation stream before the next physics step."""
        if self._states is None:
            raise RuntimeError("render snapshots are disabled")
        self._index = 1 - self._index
        wp.wait_event(self._done[self._index])
        target = self._states[self._index]
        for name in ("body_q", "particle_q"):
            source_array = getattr(state, name, None)
            if source_array is not None and source_array.shape[0] > 0:
                wp.copy(getattr(target, name), source_array)
        self._time = time
        self._prepared = True

    def get(self, state, time):
        """Return the prepared snapshot or the synchronous simulation state."""
        return (self._states[self._index], self._time) if self._prepared else (state, time)

    def release(self):
        """Record on the render stream after end_frame has enqueued its reads."""
        if self._prepared:
            wp.record_event(self._done[self._index])
        self._prepared = False
