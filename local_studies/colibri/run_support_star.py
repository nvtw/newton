"""Run a local diagnostic with the existing support-star scheduler."""

import runpy

from local_studies.colibri.support_star_groups import install

install()
runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
