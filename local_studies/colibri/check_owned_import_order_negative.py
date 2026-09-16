"""Reproduce prior late finder placement; require rejection before model creation."""

import sys

from local_studies.colibri import combined_gap_matching
from local_studies.colibri import native_conditioned_two_body as runner

active = [False]
find = runner.OwnedFinder.find_spec
install = combined_gap_matching.install


def late_find(self, fullname, path=None, target=None):
    return find(self, fullname, path, target) if active[0] else None


def late_install():
    result = install()
    active[0] = True
    return result


runner.OwnedFinder.find_spec = late_find
combined_gap_matching.install = late_install
sys.argv = [
    "negative",
    "--native-path",
    "direct",
    "--body-count",
    "2",
    "--frames",
    "0",
    "--output",
    "/tmp/colibri_owned_negative.json",
]
try:
    runner.main()
except AssertionError as error:
    assert "articulations." in str(error) and "/tmp/newton-colibri-phoenx/newton/" in str(error), str(error)
    print("PASS: prior late finder rejected before model construction:", error)
else:
    raise AssertionError("Prior late import incorrectly accepted")
