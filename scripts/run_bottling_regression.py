# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the bottling plant's physics regression suite and print one verdict per test.

Each test reproduces one failure mode seen in long bottling-plant runs on a
small scene with the plant's colliders, materials and solver settings, so a
solver or collision change can be checked in minutes instead of an 1100 s
factory run. Tests run one at a time, each in its own process; with
``--lock`` every test holds that file lock (``flock``) while it runs, so other
GPU jobs sharing the lock interleave between tests.

Run ``uv run scripts/run_bottling_regression.py --lock /tmp/gpu.lock``. Pass
``-k`` substrings to run a subset and ``--list`` to print the suite.
"""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Failure mode, then the unittest id. New tests live in TestBottlingRegression;
# the others are earlier regressions of the same modes, collected here.
_PLANT = "newton.tests.phoenx.test_bottling_plant.TestBottlingPlant."
_SUITE = "newton.tests.phoenx.test_bottling_regression.TestBottlingRegression."
TESTS = [
    ("pallet on belt seam", _SUITE + "test_loaded_pallet_crosses_belt_seam"),
    ("heel on deck seams", _SUITE + "test_bottle_heels_hold_on_deck_seams"),
    (
        "heel on deck seams",
        " ".join(
            f"newton.tests.test_convex_refinement_regressions.TestConvexRefinementRegressions.{name}_cuda_0"
            for name in (
                "test_refinement_triangle_hull_seam_support",
                "test_refinement_triangle_hull_tilted_seam_support",
                "test_refinement_triangle_hull_chamfer_seam",
                "test_refinement_triangle_hull_rim_on_seam",
                "test_refinement_triangle_capsule_sunk_across_seam",
            )
        ),
    ),
    ("abreast bottles", _PLANT + "test_abreast_plant_bottles_touch_on_their_sides"),
    (
        "pressed field rest",
        "newton._src.solvers.phoenx.tests.test_contact_rest_stability.TestContactRestStability."
        "test_pressed_cylinder_field_comes_to_rest",
    ),
    (
        "squat cylinders",
        " ".join(
            f"newton.tests.test_box_contact_normal.TestBoxContactNormal.{name}_cuda_0"
            for name in (
                "test_squat_cylinder_side_contact",
                "test_nearly_parallel_cylinder_line_contact",
            )
        ),
    ),
    ("pressed neck", _SUITE + "test_pressed_bottles_do_not_climb_at_neck"),
    ("pressed neck", _PLANT + "test_wet_end_tables_carry_packed_beds"),
    ("outfeed single file", _PLANT + "test_outfeed_conveyor_carries_plant_bottles"),
    ("outfeed single file", _SUITE + "test_pasteuriser_infeed_spin_is_bounded"),
    ("belt transfers", _PLANT + "test_abutting_belt_pieces_slide_as_one_surface"),
    ("belt transfers", _PLANT + "test_filling_line_delivers_rated_output"),
    (
        "robot drives",
        "newton._src.solvers.phoenx.tests.test_direct_drive.TestDirectDriveAnalytical."
        "test_moving_setpoint_tracks_ramp_without_saturating "
        "newton.tests.test_joint_trajectory.TestJointTrajectory",
    ),
    ("forklift carry", "newton.tests.phoenx.test_vibration_creep.TestVibrationCreep"),
    ("case floor", _SUITE + "test_cased_bottles_ride_a_conveyor"),
]


def _name(test):
    ids = test.split()
    name = ids[0].rsplit(".", 1)[-1].removesuffix("_cuda_0")
    return f"{name} (+{len(ids) - 1})" if len(ids) > 1 else name


def _run(test, timeout, python):
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [python, "-m", "unittest", "-q", *test.split()],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.perf_counter() - start, ""
    elapsed = time.perf_counter() - start
    output = result.stdout + result.stderr
    if result.returncode != 0:
        verdict = "FAIL"
    elif "skipped" in output.splitlines()[-1] if output.strip() else False:
        verdict = "SKIP"
    else:
        verdict = "PASS"
    return verdict, elapsed, output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", action="append", default=[], help="Run only tests whose id contains this substring.")
    parser.add_argument("--lock", help="Hold this file lock (flock) while each test runs.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Wall-clock limit per test [s].")
    parser.add_argument("--list", action="store_true", help="Print the suite and exit.")
    parser.add_argument("--verbose", action="store_true", help="Print the output of failing tests.")
    args = parser.parse_args()

    tests = [(mode, test) for mode, test in TESTS if not args.k or any(k in test or k in mode for k in args.k)]
    if args.list:
        for mode, test in tests:
            print(f"{mode:22s} {test}")
        return 0
    failed, total = 0, 0.0
    for mode, test in tests:
        lock = open(args.lock, "a") if args.lock else nullcontext()
        with lock:
            if args.lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
            verdict, elapsed, output = _run(test, args.timeout, sys.executable)
        total += elapsed
        failed += verdict in ("FAIL", "TIMEOUT")
        print(f"{verdict:7s} {elapsed:6.1f} s  {mode:22s} {_name(test)}", flush=True)
        if args.verbose and verdict in ("FAIL", "TIMEOUT"):
            print(output[-4000:], flush=True)
    print(f"{'FAILED' if failed else 'OK':7s} {total:6.1f} s  {failed} of {len(tests)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
