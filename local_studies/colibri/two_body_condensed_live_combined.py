"""Install the local combined kernel after friction overlays are installed."""

import runpy

from local_studies.colibri import two_body_condensed_gpu
from local_studies.colibri.two_body_condensed_combined import solve

two_body_condensed_gpu.solve = solve
runpy.run_module("local_studies.colibri.two_body_coupled_live", run_name="__main__")
