# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""FP32 reference targets for a persistent, certified planar material face.

This CPU study computes geometry/history only. It neither pools force capacity
nor changes any contact's force application point. The caller must certify that
face_token identifies one connected coplanar material face and that each new
witness lies on that actual face (including its holes), not only its plane.
"""

import numpy as np


class PlanarPatchReference:
    """Retain a rigid material reference inside the patch's birth support domain."""

    def __init__(self, ra, ta, rb, tb, support_polygon_b, normal_b, *, face_token, normal_dot_min):
        f = np.float32
        ra, ta, rb, tb = [np.asarray(x, dtype=f) for x in (ra, ta, rb, tb)]
        self.rotation = ra.T @ rb
        self.translation = ra.T @ (tb - ta)
        self.polygon = np.asarray(support_polygon_b, dtype=f)
        self.normal = np.asarray(normal_b, dtype=f)
        self.face_token = face_token
        self.normal_dot_min = f(normal_dot_min)
        self.active = True
        if len(self.polygon) < 3 or self.polygon.shape[1] != 3:
            raise ValueError("A certified counterclockwise planar support polygon is required")
        if not np.isfinite(self.polygon).all() or not np.isfinite(self.normal).all():
            raise ValueError("Nonfinite material geometry")
        if abs(self.normal @ self.normal - np.float32(1)) > np.float32(8) * np.finfo(np.float32).eps:
            raise ValueError("A unit material normal is required")

    def invalidate(self):
        """Require a new birth reference after complete contact loss or physical slip."""
        self.active = False

    def contains(self, point, normal, *, face_token):
        """Accept only the same material face inside its original support polygon."""
        if not self.active or face_token != self.face_token:
            return False
        p = np.asarray(point, dtype=np.float32)
        n = np.asarray(normal, dtype=np.float32)
        if not np.isfinite(p).all() or not np.isfinite(n).all() or n @ self.normal < self.normal_dot_min:
            return False
        if abs(n @ n - np.float32(1)) > np.float32(8) * np.finfo(np.float32).eps:
            return False
        # Conservative arithmetic error envelopes, not physical gap tolerances.
        eps = np.float32(8) * np.finfo(np.float32).eps
        displacement = p - self.polygon[0]
        plane_error = abs(displacement @ self.normal)
        plane_bound = eps * ((abs(p) + abs(self.polygon[0])) @ abs(self.normal))
        if plane_error > plane_bound:
            return False
        for a, b in zip(self.polygon, np.roll(self.polygon, -1, axis=0), strict=True):
            edge = b - a
            delta = p - a
            signed = np.cross(edge, delta) @ self.normal
            error = eps * (np.linalg.norm(edge) * np.linalg.norm(delta))
            if signed < -error:
                return False
        return True

    def displacement(self, point_b, normal_b, ra, ta, rb, tb, *, face_token):
        """Evaluate original B material against its A material anchor at patch birth."""
        if not self.contains(point_b, normal_b, face_token=face_token):
            return None
        f = np.float32
        ra, ta, rb, tb = [np.asarray(x, dtype=f) for x in (ra, ta, rb, tb)]
        b = np.asarray(point_b, dtype=f)
        anchor_a = self.rotation @ b + self.translation
        return (rb @ b + tb) - (ra @ anchor_a + ta)
