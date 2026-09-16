"""Install the bounded adaptive reference after the friction overlays."""

import runpy

from local_studies.colibri import two_body_condensed_gpu
from local_studies.colibri.two_body_condensed_adaptive import solve

two_body_condensed_gpu.solve = solve
runpy.run_module("local_studies.colibri.two_body_coupled_live", run_name="__main__")
