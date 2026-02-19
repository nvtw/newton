from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import warp as wp

from .hit_kernels import HitKernel, HitKernelManager


def _round_up(v: int, a: int) -> int:
    return v if (v % a) == 0 else v + a - (v % a)


def _aligned_record_dtype(optix, payload_dtype: np.dtype | None = None) -> np.dtype:
    header_format = f"{optix.SBT_RECORD_HEADER_SIZE}B"
    payload_size = 0 if payload_dtype is None else payload_dtype.itemsize
    itemsize = _round_up(optix.SBT_RECORD_HEADER_SIZE + payload_size, optix.SBT_RECORD_ALIGNMENT)
    if payload_dtype is None:
        names = ["header"]
        formats = [header_format]
    else:
        names = ["header", "data"]
        formats = [header_format, payload_dtype]
    return np.dtype({"names": names, "formats": formats, "itemsize": itemsize, "align": True})


def _pack_headers(optix, program_groups, record_dtype: np.dtype) -> np.ndarray:
    h = np.zeros(len(program_groups), dtype=record_dtype)
    for i, pg in enumerate(program_groups):
        optix.sbtRecordPackHeader(pg, h[i : i + 1])
    return h


def _to_device_bytes(host_records: np.ndarray, device: str) -> wp.array:
    host_bytes = np.ascontiguousarray(host_records).view(np.uint8).reshape(-1)
    return wp.array(host_bytes, dtype=wp.uint8, device=device)


@dataclass
class SbtResources:
    sbt: object
    keepalive: dict


class SbtKernelManager:
    """SBT helper with similar role to MiniOptixScene.SbtKernelManager."""

    def __init__(self, optix, ctx, module, num_ray_subtypes: int = 1) -> None:
        self.optix = optix
        self.ctx = ctx
        self.module = module
        self.raygen_group = None
        self.miss_groups = []
        self.hit_kernels = HitKernelManager(optix, ctx, module, num_ray_subtypes)

    def set_raygen_kernel(self, kernel_name: str) -> None:
        desc = self.optix.ProgramGroupDesc()
        desc.raygenModule = self.module
        desc.raygenEntryFunctionName = kernel_name
        if self.optix.version()[1] >= 4:
            self.raygen_group = self.ctx.programGroupCreate([desc], self.optix.ProgramGroupOptions())[0][0]
        else:
            self.raygen_group = self.ctx.programGroupCreate([desc])[0][0]

    def add_miss_kernels(self, kernel_names: list[str]) -> None:
        for name in kernel_names:
            desc = self.optix.ProgramGroupDesc()
            desc.missModule = self.module
            desc.missEntryFunctionName = name
            if self.optix.version()[1] >= 4:
                pg = self.ctx.programGroupCreate([desc], self.optix.ProgramGroupOptions())[0][0]
            else:
                pg = self.ctx.programGroupCreate([desc])[0][0]
            self.miss_groups.append(pg)

    def register_hit_shader_type(self, *kernel_names: str | HitKernel):
        return self.hit_kernels.register_hit_shader_type(*kernel_names)

    def get_all_program_groups(self):
        groups = []
        if self.raygen_group is not None:
            groups.append(self.raygen_group)
        groups.extend(self.miss_groups)
        groups.extend(self.hit_kernels.get_list())
        return groups

    def build_sbt(self, device: str = "cuda") -> SbtResources:
        if self.raygen_group is None:
            raise RuntimeError("Raygen kernel not set")
        if not self.miss_groups:
            raise RuntimeError("At least one miss kernel is required")

        record_dtype = _aligned_record_dtype(self.optix)
        h_rg = _pack_headers(self.optix, [self.raygen_group], record_dtype)
        h_ms = _pack_headers(self.optix, self.miss_groups, record_dtype)
        hit_groups = self.hit_kernels.get_list()
        h_hg = _pack_headers(self.optix, hit_groups, record_dtype) if hit_groups else np.zeros(0, dtype=record_dtype)

        d_rg = _to_device_bytes(h_rg, device=device)
        d_ms = _to_device_bytes(h_ms, device=device)
        d_hg = _to_device_bytes(h_hg, device=device) if len(h_hg) > 0 else None

        sbt = self.optix.ShaderBindingTable()
        sbt.raygenRecord = d_rg.ptr
        sbt.missRecordBase = d_ms.ptr
        sbt.missRecordStrideInBytes = h_ms.dtype.itemsize
        sbt.missRecordCount = len(h_ms)
        if d_hg is not None:
            sbt.hitgroupRecordBase = d_hg.ptr
            sbt.hitgroupRecordStrideInBytes = h_hg.dtype.itemsize
            sbt.hitgroupRecordCount = len(h_hg)
        else:
            sbt.hitgroupRecordBase = 0
            sbt.hitgroupRecordStrideInBytes = 0
            sbt.hitgroupRecordCount = 0

        keepalive = {"d_rg": d_rg, "d_ms": d_ms, "d_hg": d_hg}
        return SbtResources(sbt=sbt, keepalive=keepalive)
