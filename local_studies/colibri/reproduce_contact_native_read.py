# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Recreate the prepared-row native read-after-write failure; expect test failure."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from local_studies.colibri import prototype_normal_first, test_normal_first

_NATIVE_READ_BLOCK = '                old_n = cc_get_normal_lambda(cc, k)\n                if wp.static(has_soft_contact_pd):\n                    contact_project_normal_velocity_update(\n                        cc,\n                        k,\n                        n,\n                        jv_n,\n                        eff_n,\n                        bias_val,\n                        mass_coeff_n,\n                        impulse_coeff_n,\n                        sor_boost,\n                        pd_eff_soft_n,\n                        pd_gamma_n,\n                        pd_bias_n,\n                    )\n                else:\n                    contact_project_normal_velocity_update_no_soft_pd(\n                        cc,\n                        k,\n                        n,\n                        jv_n,\n                        eff_n,\n                        bias_val,\n                        mass_coeff_n,\n                        impulse_coeff_n,\n                        sor_boost,\n                        pd_eff_soft_n,\n                        pd_gamma_n,\n                        pd_bias_n,\n                    )\n                new_n = cc_get_normal_lambda(cc, k)\n                normal_delta = new_n - old_n\n                load = new_n\n                if wp.static(has_soft_contact_pd):\n                    if pd_eff_soft_n <= wp.float32(0.0):\n                        load = _friction_normal_lambda(new_n, eff_n, bias_val, mass_coeff_n, sor_boost)\n                else:\n                    load = _friction_normal_lambda(new_n, eff_n, bias_val, mass_coeff_n, sor_boost)\n'

source = Path(prototype_normal_first.__file__).read_text()
branch = source.index("elif wp.static(not cloth_support) and not use_patch:")
start = source.index("                if wp.static(has_soft_contact_pd):", branch)
end = source.index("                tangent_delta = wp.vec2f(0.0)", start)
source = source[:start] + _NATIVE_READ_BLOCK + source[end:]
path = Path(tempfile.gettempdir()) / "phoenx_native_read_prepared_repro.py"
path.write_text(source)
spec = importlib.util.spec_from_file_location("phoenx_native_read_prepared_repro", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
test_normal_first._make_normal_first = module._make_contact_iterate_at

if __name__ == "__main__":
    unittest.main(module=test_normal_first, failfast=True)
