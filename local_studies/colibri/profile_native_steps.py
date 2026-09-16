"""Capture warmed simulation steps with Nsight's CUDA profiler API range."""

import ctypes
import os
import runpy

import newton.examples


def main():
    """Profile frames 31 through 60 after construction and warmup."""
    cuda = ctypes.CDLL("libcuda.so.1")
    original_run = newton.examples.run

    def run(example, args):
        step = example.step
        count = 0

        def profiled_step():
            nonlocal count
            if count == 30:
                assert cuda.cuProfilerStart() == 0
            step()
            count += 1
            if count == 60:
                assert cuda.cuProfilerStop() == 0

        example.step = profiled_step
        try:
            return original_run(example, args)
        finally:
            example.step = step

    newton.examples.run = run
    try:
        runpy.run_module(
            os.environ.get("COLIBRI_PROFILE_RUNNER", "local_studies.colibri.check_public_analytic_gradient"),
            run_name="__main__",
        )
    finally:
        newton.examples.run = original_run


if __name__ == "__main__":
    main()
