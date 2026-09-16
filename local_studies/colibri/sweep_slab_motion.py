# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Save drive motion while retaining every slab physical check."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np

from local_studies.colibri import check_slab_colibri as slab
from local_studies.colibri.drive_motion import measure

original = slab.SlabExample.test_post_step
history = []


def checked(self):
    history.append(self.state_0.body_q.numpy())
    return original(self)


slab.SlabExample.test_post_step = checked
try:
    runpy.run_module("local_studies.colibri.check_public_slab_noenvelope", run_name="__main__")
finally:
    if history and slab.instances:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        labels = slab.instances[0].model.body_label
        np.savez_compressed(output.with_suffix(".motion.npz"), q_history=np.asarray(history), labels=np.asarray(labels))
        if len(history) > 1:
            output.with_suffix(".motion.json").write_text(json.dumps(measure(history, labels, fps=60), indent=2))
