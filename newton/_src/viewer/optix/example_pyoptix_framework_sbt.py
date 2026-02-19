"""Advanced OptiX sample using translated Python helper framework.

This sample mirrors the C# MiniOptixScene flow more closely:
- Build PTX through Warp
- Create explicit program groups via SbtKernelManager
- Build SBT from manager
- Launch a headless OptiX render loop
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import warp as wp
from newton._src.viewer.optix.example_pyoptix_empty_buffer import (
    _build_optix_ptx_from_warp,
    _create_cube_gas,
    _create_optix_context,
    _require_optix,
)
from newton._src.viewer.optix.launch_params import build_launch_params_dtype
from newton._src.viewer.optix.sbt_helpers import SbtKernelManager


def _create_pipeline_with_manager(optix, ctx, ptx: bytes):
    kwargs = {
        "usesMotionBlur": False,
        "traversableGraphFlags": int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
        "numPayloadValues": 1,
        "numAttributeValues": 2,
        "exceptionFlags": int(optix.EXCEPTION_FLAG_NONE),
        "pipelineLaunchParamsVariableName": "params",
    }
    if optix.version()[1] >= 2:
        kwargs["usesPrimitiveTypeFlags"] = int(optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
    pipeline_options = optix.PipelineCompileOptions(**kwargs)
    module_options = optix.ModuleCompileOptions(
        maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
    )
    module, log = ctx.moduleCreate(module_options, pipeline_options, ptx)
    if log:
        print(f"Module create log:\n{log}")

    manager = SbtKernelManager(optix, ctx, module, num_ray_subtypes=1)
    manager.set_raygen_kernel("__raygen__cube")
    manager.add_miss_kernels(["__miss__cube"])
    manager.register_hit_shader_type("__closesthit__cube")

    link_options = optix.PipelineLinkOptions()
    link_options.maxTraceDepth = 1
    groups = manager.get_all_program_groups()
    pipeline = ctx.pipelineCreate(pipeline_options, link_options, groups, "")
    pipeline.setStackSize(2 * 1024, 2 * 1024, 2 * 1024, 1)
    sbt = manager.build_sbt(device="cuda")
    return pipeline, sbt, manager, module


def _run_headless(optix, pipeline, sbt, gas_handle: int, width: int, height: int, frames: int):
    params_dtype = build_launch_params_dtype()
    params_host = np.zeros(1, dtype=params_dtype)
    params_device = wp.empty(params_dtype.itemsize, dtype=wp.uint8, device="cuda")
    image = wp.empty(width * height, dtype=wp.uint32, device="cuda")
    start_time = time.perf_counter()

    for _ in range(max(frames, 1)):
        params_host["image"][0] = image.ptr
        params_host["width"][0] = width
        params_host["height"][0] = height
        params_host["time_sec"][0] = np.float32(time.perf_counter() - start_time)
        params_host["trav_handle"][0] = np.uint64(gas_handle)
        params_bytes = params_host.view(np.uint8).reshape(-1)
        wp.copy(params_device, wp.array(params_bytes, dtype=wp.uint8, device="cpu", copy=False))

        optix.launch(
            pipeline,
            0,
            params_device.ptr,
            params_dtype.itemsize,
            sbt.sbt,
            width,
            height,
            1,
        )
        wp.synchronize_device("cuda")

    # Return a small checksum so users can compare runs.
    arr = image.numpy()
    return int(np.bitwise_xor.reduce(arr))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    optix = _require_optix()
    wp.init()
    with wp.ScopedDevice(args.device):
        if not wp.get_device().is_cuda:
            raise RuntimeError("This sample requires CUDA.")
        print(f"PyOptiX: OptiX SDK {optix.version()}")

        ptx = _build_optix_ptx_from_warp(trace_mode="trace0", device=args.device)
        wp_device = wp.get_device(args.device)
        cu_context = wp_device.context.value if hasattr(wp_device.context, "value") else int(wp_device.context)
        ctx, _logger = _create_optix_context(optix, int(cu_context))
        gas_handle, _gas_keepalive = _create_cube_gas(optix, ctx)
        pipeline, sbt, _manager, _module = _create_pipeline_with_manager(optix, ctx, ptx)

        checksum = _run_headless(optix, pipeline, sbt, gas_handle, args.width, args.height, args.frames)
        print(f"Headless frame checksum: {checksum}")


if __name__ == "__main__":
    main()
