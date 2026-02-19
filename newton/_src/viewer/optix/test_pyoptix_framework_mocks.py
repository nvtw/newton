import unittest

import numpy as np

import warp as wp
from newton._src.viewer.optix.hit_kernels import HitKernelManager
from newton._src.viewer.optix.sbt_helpers import SbtKernelManager


class _DummyPg:
    def __init__(self, name):
        self.name = name


class _FakeDesc:
    pass


class _FakeOptix:
    SBT_RECORD_HEADER_SIZE = 32
    SBT_RECORD_ALIGNMENT = 16

    def version(self):
        return (9, 0, 0)

    class ProgramGroupOptions:
        pass

    def ProgramGroupDesc(self):
        return _FakeDesc()

    class ShaderBindingTable:
        pass

    def sbtRecordPackHeader(self, pg, arr):
        arr["header"][0][0] = hash(pg.name) & 0xFF


class _FakeCtx:
    def __init__(self):
        self._count = 0

    def programGroupCreate(self, descs, *args):
        out = []
        for _ in descs:
            out.append(_DummyPg(f"pg_{self._count}"))
            self._count += 1
        return [out, ""]


class TestFrameworkMocks(unittest.TestCase):
    def test_hit_kernel_offsets(self):
        optix = _FakeOptix()
        ctx = _FakeCtx()
        hk = HitKernelManager(optix, ctx, module="mod", num_ray_subtypes=2)
        h0 = hk.register_hit_shader_type("__closesthit__a", "__closesthit__b")
        h1 = hk.register_hit_shader_type("__closesthit__c", "__closesthit__d")
        self.assertEqual(hk.get_sbt_offset(h0), 0)
        self.assertEqual(hk.get_sbt_offset(h1), 2)
        self.assertEqual(len(hk.get_list()), 4)

    def test_sbt_manager_build_cpu_device(self):
        wp.init()
        optix = _FakeOptix()
        ctx = _FakeCtx()
        sbtm = SbtKernelManager(optix, ctx, module="mod", num_ray_subtypes=1)
        sbtm.set_raygen_kernel("__raygen__x")
        sbtm.add_miss_kernels(["__miss__x"])
        sbtm.register_hit_shader_type("__closesthit__x")
        resources = sbtm.build_sbt(device="cpu")
        self.assertTrue(hasattr(resources.sbt, "raygenRecord"))
        self.assertGreater(resources.sbt.missRecordCount, 0)
        self.assertGreater(resources.sbt.hitgroupRecordCount, 0)

    def test_sbt_alignment_dtype_shape(self):
        wp.init()
        optix = _FakeOptix()
        ctx = _FakeCtx()
        sbtm = SbtKernelManager(optix, ctx, module="mod", num_ray_subtypes=1)
        sbtm.set_raygen_kernel("__raygen__x")
        sbtm.add_miss_kernels(["__miss__x"])
        sbtm.register_hit_shader_type("__closesthit__x")
        res = sbtm.build_sbt(device="cpu")
        self.assertIsInstance(res.keepalive["d_rg"].numpy(), np.ndarray)


if __name__ == "__main__":
    unittest.main()
