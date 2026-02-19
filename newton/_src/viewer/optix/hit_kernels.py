from __future__ import annotations

from dataclasses import dataclass

from .handles import Handle, HandleBuffer


@dataclass
class HitKernel:
    closest_hit: str
    any_hit: str | None = None
    intersection: str | None = None


class HitKernelManager:
    """Python translation of MiniOptixScene.HitKernelManager."""

    def __init__(self, optix, ctx, module, num_ray_subtypes: int) -> None:
        self.optix = optix
        self.ctx = ctx
        self.module = module
        self.num_ray_types_per_intersection_type = int(num_ray_subtypes)
        self._handle_to_offset = HandleBuffer[int]()
        self._hit_shaders = []

    @property
    def count(self) -> int:
        return self._handle_to_offset.count

    def register_hit_shader_type(self, *kernel_names: str | HitKernel) -> Handle:
        kernels: list[HitKernel] = []
        for k in kernel_names:
            if isinstance(k, HitKernel):
                kernels.append(k)
            else:
                kernels.append(HitKernel(str(k)))

        if len(kernels) != self.num_ray_types_per_intersection_type:
            raise ValueError(f"Expected {self.num_ray_types_per_intersection_type} kernels, got {len(kernels)}")

        index = len(self._hit_shaders)
        handle = self._handle_to_offset.add(index)
        for k in kernels:
            desc = self.optix.ProgramGroupDesc()
            desc.hitgroupModuleCH = self.module
            desc.hitgroupEntryFunctionNameCH = k.closest_hit
            if k.any_hit:
                desc.hitgroupModuleAH = self.module
                desc.hitgroupEntryFunctionNameAH = k.any_hit
            if k.intersection:
                desc.hitgroupModuleIS = self.module
                desc.hitgroupEntryFunctionNameIS = k.intersection

            if self.optix.version()[1] >= 4:
                pg_options = self.optix.ProgramGroupOptions()
                pg = self.ctx.programGroupCreate([desc], pg_options)[0][0]
            else:
                pg = self.ctx.programGroupCreate([desc])[0][0]
            self._hit_shaders.append(pg)
        return handle

    def get_sbt_offset(self, handle: Handle) -> int:
        ok, offset = self._handle_to_offset.try_get_value(handle)
        if not ok or offset is None:
            raise KeyError(f"Handle {handle.value} not found")
        return int(offset)

    def get_list(self):
        return list(self._hit_shaders)
