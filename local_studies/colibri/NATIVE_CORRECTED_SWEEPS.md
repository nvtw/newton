# Corrected native owned-contact sweep controls

These controls supersede the earlier sweep comparison for claims about the
combined friction fix. The earlier wrapper installed OwnedFinder after Newton
had imported the owned callback modules. Its path hashes did not prove those
callbacks executed.

The rebuilt native_conditioned_contact_sweeps.py installs the finder first and
retains the corrected native wrapper's actual module paths, solver-kernel object
identity, metric-helper identity, executed callback source semantics and CPU
break-state probe. audit_corrected_native_sweeps.py additionally requires all
production/asset hashes and actual callback metadata to match the corrected N1
reference across runs.

## Results

All configurations use a free dynamic base, the original native spring,
normalized FP32 material reference, unsplit direct joint conditioning, SOR 1,
30 substeps per 120 Hz collision interval. Both biased contact iterations and
final-substep velocity iterations are N. This increases work to 60N biased plus
2N final-relax contact sweeps per rendered frame.

| N | Base-origin drift 5–10 s (µm/s) | x/y displacement (µm) | Hinge at 10 s (deg) | FPS |
|---|---:|---|---:|---:|
| 1 | 2.445929 | +2.344 / +12.003 | 18.207416 | 20.4515 |
| 2 | 9.775499 | -2.858 / -48.794 | 18.207649 | 12.2091 |
| 4 | 2.036435 | -0.365 / -10.176 | 18.208392 | 6.8667 |

All existing physical geometry/joint bounds and source guards pass. N1 timing
uses its existing 60-second run; N2/N4 use separate isolated 10-second runs.
Drift comparison uses exactly the same 5–10-second interval. These are live
different trajectories, not convergence measurements at one frozen state.

Neither extra-sweep control demonstrates acceptable stationarity. N4 provides
only a modest reduction in translation magnitude with reversed direction and
roughly triple the cost. No 60-second extension or default change was made.

Final loaded, strictly interior contact slip:
N2 18.953 µm/s (4 interior rows); N4 5.587 µm/s (5 interior rows).
These instantaneous final-relax readouts are not an integration/work ledger.

## Evidence and reproduction

- N1: /tmp/colibri_native_direct_owned_fixed3600.*
- N2: /tmp/colibri_native_owned_fixed_sweeps2_600.*
- N4: /tmp/colibri_native_owned_fixed_sweeps4_600.*
- Cross-run audit: /tmp/colibri_native_owned_fixed_sweep_comparison.json

For each N=2 or N=4, set both arguments consistently:

    COLIBRI_DIFFERENTIAL_REFERENCE=1 COLIBRI_DIFFERENTIAL_NORMALIZED=1 uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.native_conditioned_contact_sweeps --native-path direct --contact-sweeps N --body-count 2 --frames 600 --substeps 30 --iterations N --save-history --output /tmp/colibri_native_owned_fixed_sweepsN_600.json

    uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.audit_corrected_native_sweeps
