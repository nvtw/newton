"""Replay the missing joint-cache bug without modifying the working adapter."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

source = Path("local_studies/colibri/physical_head_overflow.py").read_text()
start = source.index("        if cid < contact_offset:")
end = source.index("\n\n\ndef _factor_kernel", start)
source = (
    source[:start]
    + """        if tid < starts[cap]:
            _cache_slots_for_partition(cid, wp.int32(-1), copies, constraints, columns, contact_offset)
"""
    + source[end:]
)
path = Path("/tmp/physical_head_missing_joint_cache_control.py")
path.write_text(source)
name = "physical_head_missing_joint_cache_control"
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
os.environ["PHYSICAL_HEAD_PROTOTYPE"] = name
suite = unittest.defaultTestLoader.loadTestsFromName(
    "local_studies.colibri.test_physical_head_physics.TestPhysicalHeadPhysics.test_overflow_joint_counts_and_warm_momentum"
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
