"""Profile two warmed G1 policy steps, including captured CUDA graph nodes."""

import ctypes

import warp as wp

from newton.examples.robot.example_robot_policy import Example
from newton.viewer import ViewerNull

wp.init()
example = Example(ViewerNull(), Example.create_parser().parse_args(["--robot", "g1_29dof", "--solver", "phoenx"]))
for _ in range(20):
    example.step()
driver = ctypes.CDLL("libcuda.so.1")
driver.cuProfilerStart.restype = ctypes.c_int
driver.cuProfilerStop.restype = ctypes.c_int
wp.synchronize_device(example.model.device)
if driver.cuProfilerStart():
    raise RuntimeError("cuProfilerStart failed")
for _ in range(2):
    example.step()
wp.synchronize_device(example.model.device)
if driver.cuProfilerStop():
    raise RuntimeError("cuProfilerStop failed")
print("PROFILE_CAPTURED_TWO_POLICY_STEPS")
