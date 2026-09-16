# Current-source joint phase capture

The improved capture_support_relax.py records before-relax physical arrays,
pre-average copy velocities and after-relax physical arrays in one run. It
records490 solver/example Python source hashes and rejects source changes
during capture. --snapshot chooses a separate diagnostic artifact; it cannot
overwrite the public trajectory snapshot.

The330-frame current-source run passed all checks. All ten trajectory arrays
are byte-identical to the canonical cooperative-forward smoke reference.
All31 arrays from the earlier colibri_support_relax330.npz are also
byte-identical. This confirms that the earlier frozen operator and drive
accounting findings reproduce with current source, despite the old capture's
missing source manifest. No numerical solver behavior changed.

Artifacts:

- /tmp/colibri_current_joint_phases330.npz
- /tmp/colibri_current_joint_phases330.capture.json
- /tmp/colibri_current_joint_phases330_validation.json

Reproduce:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python \
  -m local_studies.colibri.capture_support_relax \
  --snapshot /tmp/colibri_current_joint_phases330.npz \
  --frames 330 --substeps 30 --save-history \
  --output /tmp/colibri_current_joint_phases330_trajectory.json
```

Capture copies add GPU work. Its timing is not an uninstrumented FPS result.
