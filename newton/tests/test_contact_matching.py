# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for frame-to-frame contact matching via ContactMatcher."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_match import ContactMatcher  # noqa: F401
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestContactMatching(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_simple_scene(device):
    """Build a scene with spheres resting on a ground plane.

    Returns (model, state) with 3 spheres at different positions on the plane.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    for x in (-0.5, 0.0, 0.5):
        b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1)))
        builder.add_shape_sphere(body=b, radius=0.1)

    model = builder.finalize(device=device)
    state = model.state()
    return model, state


def _collide_once(pipeline, state, contacts):
    """Clear and collide, returning the contact count on host."""
    contacts.clear()
    pipeline.collide(state, contacts)
    return contacts.rigid_contact_count.numpy()[0]


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_first_frame_all_not_found(test, device):
    """First frame: no previous data, all contacts should be MATCH_NOT_FOUND."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
        contacts = pipeline.contacts()

        count = _collide_once(pipeline, state, contacts)
        test.assertGreater(count, 0, "Expected contacts between spheres and ground plane")

        match_idx = contacts.rigid_contact_match_index.numpy()[:count]
        test.assertTrue(
            np.all(match_idx == -1),
            f"First frame should have all MATCH_NOT_FOUND, got unique values: {np.unique(match_idx)}",
        )


def test_stable_scene_all_matched(test, device):
    """Stable scene: same state across two frames, all contacts should match."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
        contacts = pipeline.contacts()

        # Frame 1: populate previous-frame data.
        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Frame 2: same state, should match all contacts.
        count2 = _collide_once(pipeline, state, contacts)
        test.assertEqual(count1, count2, "Contact count should be stable between identical frames")

        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]
        matched = match_idx >= 0
        test.assertTrue(
            np.all(matched),
            f"Stable scene should match all contacts. Unmatched count: {np.sum(~matched)}, "
            f"unique values: {np.unique(match_idx[~matched])}",
        )


def test_new_contact_detection(test, device):
    """Adding a new sphere produces new (unmatched) contacts."""
    with wp.ScopedDevice(device):
        # Start with 2 spheres resting on the ground plane.
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for x in (-0.5, 0.5):
            b = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1)))
            builder.add_shape_sphere(body=b, radius=0.1)
        # Third sphere starts far away (no contacts).
        b3 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 10.0)))
        builder.add_shape_sphere(body=b3, radius=0.1)

        model = builder.finalize(device=device)
        state = model.state()

        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
        contacts = pipeline.contacts()

        # Frame 1: only 2 spheres in contact.
        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Move third sphere down to touch the ground plane for frame 2.
        q = state.body_q.numpy()
        q[2][0:3] = [0.0, 0.0, 0.1]
        state.body_q = wp.array(q, dtype=wp.transform, device=device)

        count2 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count2, count1, "More contacts expected with third sphere on ground")

        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]
        has_new = np.any(match_idx == -1)
        test.assertTrue(has_new, "New sphere should produce MATCH_NOT_FOUND contacts")

        has_matched = np.any(match_idx >= 0)
        test.assertTrue(has_matched, "Original spheres should still have matched contacts")


def test_broken_contact_pos_threshold(test, device):
    """Moving a body far between frames should produce MATCH_BROKEN contacts."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            contact_matching=True,
            contact_matching_pos_threshold=0.001,  # very tight threshold
        )
        contacts = pipeline.contacts()

        # Frame 1.
        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        # Nudge all bodies slightly (more than 0.001 m threshold).
        q = state.body_q.numpy()
        for i in range(len(q)):
            q[i][0] += 0.01  # 1 cm shift in x
        state.body_q = wp.array(q, dtype=wp.transform, device=device)

        # Frame 2.
        count2 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count2, 0)

        match_idx = contacts.rigid_contact_match_index.numpy()[:count2]
        has_broken = np.any(match_idx == -2)
        test.assertTrue(
            has_broken,
            f"Expected MATCH_BROKEN with tight threshold. Unique values: {np.unique(match_idx)}",
        )


def test_contact_report(test, device):
    """Contact report should list new and broken contact indices."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True, contact_report=True)
        contacts = pipeline.contacts()

        # Frame 1: all contacts are new.
        count1 = _collide_once(pipeline, state, contacts)
        test.assertGreater(count1, 0)

        matcher = pipeline.contact_matcher
        new_count = matcher.new_contact_count.numpy()[0]
        test.assertEqual(new_count, count1, "First frame: all contacts should be reported as new")

        # Frame 2: stable scene, no new or broken contacts.
        _collide_once(pipeline, state, contacts)
        new_count2 = matcher.new_contact_count.numpy()[0]
        broken_count2 = matcher.broken_contact_count.numpy()[0]
        test.assertEqual(new_count2, 0, f"Stable scene: expected 0 new contacts, got {new_count2}")
        test.assertEqual(broken_count2, 0, f"Stable scene: expected 0 broken contacts, got {broken_count2}")


def test_deterministic_implied(test, device):
    """contact_matching=True should imply deterministic=True."""
    with wp.ScopedDevice(device):
        model, _state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
        test.assertTrue(pipeline.deterministic, "contact_matching should imply deterministic")
        test.assertIsNotNone(pipeline.contact_matcher)


def test_match_index_survives_sort(test, device):
    """match_index values should be correctly permuted during sorting."""
    with wp.ScopedDevice(device):
        model, state = _build_simple_scene(device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching=True)
        contacts = pipeline.contacts()

        # Frame 1: populate.
        _collide_once(pipeline, state, contacts)

        # Frame 2: check that match indices are valid.
        count = _collide_once(pipeline, state, contacts)
        match_idx = contacts.rigid_contact_match_index.numpy()[:count]

        # All matched indices should be in valid range.
        matched_mask = match_idx >= 0
        if np.any(matched_mask):
            matched_vals = match_idx[matched_mask]
            test.assertTrue(
                np.all(matched_vals < count),
                f"Matched indices should be < contact count ({count}), max found: {matched_vals.max()}",
            )
            # No duplicates among matched indices (each old contact matches at most one new).
            unique_matched = np.unique(matched_vals)
            test.assertEqual(
                len(unique_matched),
                len(matched_vals),
                "Matched indices should be unique (no two contacts match same old contact)",
            )


# ---------------------------------------------------------------------------
# Register tests
# ---------------------------------------------------------------------------

devices = get_cuda_test_devices()

add_function_test(
    TestContactMatching, "test_first_frame_all_not_found", test_first_frame_all_not_found, devices=devices
)
add_function_test(TestContactMatching, "test_stable_scene_all_matched", test_stable_scene_all_matched, devices=devices)
add_function_test(TestContactMatching, "test_new_contact_detection", test_new_contact_detection, devices=devices)
add_function_test(
    TestContactMatching, "test_broken_contact_pos_threshold", test_broken_contact_pos_threshold, devices=devices
)
add_function_test(TestContactMatching, "test_contact_report", test_contact_report, devices=devices)
add_function_test(TestContactMatching, "test_deterministic_implied", test_deterministic_implied, devices=devices)
add_function_test(
    TestContactMatching, "test_match_index_survives_sort", test_match_index_survives_sort, devices=devices
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
